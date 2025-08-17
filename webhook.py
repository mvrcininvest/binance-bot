"""
Webhook endpoint for TradingView alerts v9.1
Handles incoming alerts with validation, parsing, and routing
"""

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from config import Config
from database import Session, AlertHistory, get_setting, check_idempotency
from binance_handler import binance_handler
from discord_notifications import DiscordNotifier
from mode_manager import ModeManager

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Binance Trading Bot Webhook",
    version="9.1",
    description="Webhook endpoint for TradingView alerts with v9.1 indicator support"
)

# Discord notifier
discord = DiscordNotifier()

# Mode manager
mode_manager = ModeManager()


class AlertPayloadV91(BaseModel):
    """v9.1 Alert payload schema"""
    
    # Required fields
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    action: str = Field(..., description="Action: buy/sell/long/short/close")
    indicator_version: str = Field(..., description="Indicator version (must be 9.1)")
    timestamp: str = Field(..., description="Alert timestamp")
    
    # Signal tier and strength
    tier: str = Field("Standard", description="Signal tier: Platinum/Premium/Standard/Quick/Emergency")
    strength: float = Field(0.0, description="Signal strength 0-100")
    
    # Price levels
    price: float = Field(..., description="Current price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit_1: Optional[float] = Field(None, description="Take profit 1")
    take_profit_2: Optional[float] = Field(None, description="Take profit 2")
    take_profit_3: Optional[float] = Field(None, description="Take profit 3")
    
    # Leverage
    leverage: Optional[int] = Field(None, description="Suggested leverage")
    
    # v9.1 Enhanced fields
    institutional_flow: float = Field(0.0, description="Institutional flow strength")
    retest_confidence: float = Field(0.0, description="Retest pattern confidence")
    fake_breakout_detected: bool = Field(False, description="Fake breakout detected")
    fake_breakout_penalty: float = Field(1.0, description="Fake breakout penalty multiplier")
    regime: str = Field("NEUTRAL", description="Market regime")
    regime_confidence: float = Field(0.0, description="Regime confidence")
    mtf_agreement: float = Field(0.0, description="Multi-timeframe agreement ratio")
    
    # Context
    timeframe: str = Field("15m", description="Signal timeframe")
    session: Optional[str] = Field(None, description="Trading session")
    bar_close_time: Optional[str] = Field(None, description="Bar close time")
    volume_context: Optional[Dict] = Field(None, description="Volume analysis context")
    
    # Optional metadata
    comment: Optional[str] = Field(None, description="Additional comment")
    alert_id: Optional[str] = Field(None, description="Unique alert ID from TradingView")
    
    @validator('indicator_version')
    def validate_version(cls, v):
        """Ensure we only accept v9.1 alerts"""
        if v not in ['9.1', '9.0']:  # Accept 9.0 for backward compatibility during transition
            raise ValueError(f"Unsupported indicator version: {v}")
        return v
    
    @validator('tier')
    def validate_tier(cls, v):
        """Validate tier is in allowed list"""
        allowed_tiers = ['Platinum', 'Premium', 'Standard', 'Quick', 'Emergency']
        if v not in allowed_tiers:
            raise ValueError(f"Invalid tier: {v}. Must be one of {allowed_tiers}")
        return v
    
    @validator('action')
    def normalize_action(cls, v):
        """Normalize action to uppercase"""
        return v.upper()
    
    @validator('symbol')
    def normalize_symbol(cls, v):
        """Normalize symbol to uppercase"""
        return v.upper()


class WebhookResponse(BaseModel):
    """Standard webhook response"""
    status: str
    message: str
    data: Optional[Dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify webhook signature if configured"""
    if not Config.WEBHOOK_SECRET:
        return True  # No secret configured, skip verification
    
    expected_signature = hmac.new(
        Config.WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


def generate_idempotency_key(alert: AlertPayloadV91) -> str:
    """Generate idempotency key for alert deduplication"""
    key_data = f"{alert.symbol}_{alert.action}_{alert.timestamp}_{alert.price}"
    if alert.alert_id:
        key_data += f"_{alert.alert_id}"
    
    return hashlib.sha256(key_data.encode()).hexdigest()


def check_alert_age(alert: AlertPayloadV91) -> Tuple[bool, int]:
    """Check if alert is too old"""
    try:
        alert_time = datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00'))
        current_time = datetime.utcnow()
        age_seconds = (current_time - alert_time).total_seconds()
        
        max_age = Config.ALERT_MAX_AGE_SECONDS
        is_valid = age_seconds <= max_age
        
        return is_valid, int(age_seconds)
        
    except Exception as e:
        logger.error(f"Failed to parse alert timestamp: {e}")
        return False, 0


def validate_alert_conditions(alert: AlertPayloadV91) -> Tuple[bool, str]:
    """Validate alert meets all conditions"""
    
    # Check indicator version
    if alert.indicator_version != Config.INDICATOR_VERSION_REQUIRED:
        return False, f"Wrong indicator version: {alert.indicator_version}"
    
    # Check alert age
    age_valid, age_seconds = check_alert_age(alert)
    if not age_valid:
        return False, f"Alert too old: {age_seconds}s"
    
    # Check tier if in Emergency mode
    with Session() as session:
        if get_setting(session, "emergency_enabled", False):
            if alert.tier != "Emergency":
                return False, "Only Emergency tier accepted in emergency mode"
    
    # Check fake breakout penalty
    if alert.fake_breakout_detected and alert.fake_breakout_penalty < 0.5:
        return False, f"Fake breakout penalty too high: {alert.fake_breakout_penalty}"
    
    # Check signal strength for tier
    min_strengths = {
        "Platinum": 80,
        "Premium": 60,
        "Standard": 40,
        "Quick": 30,
        "Emergency": 0
    }
    
    min_strength = min_strengths.get(alert.tier, 40)
    if alert.strength < min_strength:
        return False, f"Signal strength {alert.strength} below minimum {min_strength} for {alert.tier}"
    
    return True, "Valid"


@app.get("/")
async def root():
    """Health check endpoint"""
    return WebhookResponse(
        status="ok",
        message="Binance Trading Bot Webhook v9.1 is running"
    )


@app.get("/health")
async def health_check():
    """Detailed health check"""
    with Session() as session:
        try:
            # Check database
            db_healthy = session.execute("SELECT 1").scalar() == 1
            
            # Check Binance connection
            binance_healthy = binance_handler.validate_api_credentials()
            
            # Get bot status
            is_paused = get_setting(session, "is_paused", False)
            current_mode = get_setting(session, "mode", Config.DEFAULT_MODE)
            emergency_enabled = get_setting(session, "emergency_enabled", False)
            
            # Get open positions count
            from database import get_open_positions
            open_positions = len(get_open_positions(session))
            
            return {
                "status": "healthy" if (db_healthy and binance_healthy) else "degraded",
                "version": "9.1",
                "components": {
                    "database": "healthy" if db_healthy else "unhealthy",
                    "binance": "healthy" if binance_healthy else "unhealthy",
                    "discord": "healthy"  # Assume Discord is always healthy
                },
                "bot_status": {
                    "paused": is_paused,
                    "mode": current_mode,
                    "emergency": emergency_enabled,
                    "open_positions": open_positions
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/webhook/tradingview")
async def tradingview_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_signature: Optional[str] = Header(None)
):
    """Main webhook endpoint for TradingView alerts"""
    
    # Get raw payload
    raw_payload = await request.body()
    
    # Verify signature if configured
    if Config.WEBHOOK_SECRET and x_signature:
        if not verify_signature(raw_payload, x_signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        # Parse JSON payload
        payload_dict = json.loads(raw_payload)
        
        # Validate against schema
        alert = AlertPayloadV91(**payload_dict)
        
        # Generate idempotency key
        idempotency_key = generate_idempotency_key(alert)
        
        with Session() as session:
            # Check for duplicate
            if check_idempotency(session, idempotency_key):
                logger.info(f"Duplicate alert rejected: {idempotency_key}")
                return WebhookResponse(
                    status="rejected",
                    message="Duplicate alert",
                    data={"idempotency_key": idempotency_key}
                )
            
            # Record alert receipt
            alert_record = AlertHistory(
                idempotency_key=idempotency_key,
                symbol=alert.symbol,
                action=alert.action,
                tier=alert.tier,
                processed=False,
                raw_payload=payload_dict,
                signature_valid=bool(x_signature),
                schema_valid=True
            )
            session.add(alert_record)
            session.commit()
        
        # Validate alert conditions
        is_valid, validation_message = validate_alert_conditions(alert)
        
        if not is_valid:
            logger.info(f"Alert validation failed: {validation_message}")
            
            # Update alert record
            with Session() as session:
                alert_record = session.query(AlertHistory).filter_by(
                    idempotency_key=idempotency_key
                ).first()
                if alert_record:
                    alert_record.error = validation_message
                    session.commit()
            
            return WebhookResponse(
                status="rejected",
                message=validation_message,
                data={"symbol": alert.symbol, "tier": alert.tier}
            )
        
        # Process alert in background
        background_tasks.add_task(process_alert_async, alert, idempotency_key)
        
        return WebhookResponse(
            status="accepted",
            message="Alert accepted for processing",
            data={
                "symbol": alert.symbol,
                "action": alert.action,
                "tier": alert.tier,
                "idempotency_key": idempotency_key
            }
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_alert_async(alert: AlertPayloadV91, idempotency_key: str):
    """Process alert asynchronously"""
    start_time = time.time()
    
    try:
        # Convert Pydantic model to dict
        alert_dict = alert.dict()
        
        # Add processing metadata
        alert_dict['idempotency_key'] = idempotency_key
        alert_dict['received_at'] = datetime.utcnow().isoformat()
        
        # Check mode-specific rules
        mode_decision = await mode_manager.evaluate_signal(alert_dict)
        
        if not mode_decision['accept']:
            logger.info(f"Signal rejected by mode manager: {mode_decision['reason']}")
            
            # Update alert record
            with Session() as session:
                alert_record = session.query(AlertHistory).filter_by(
                    idempotency_key=idempotency_key
                ).first()
                if alert_record:
                    alert_record.processed = True
                    alert_record.error = mode_decision['reason']
                    session.commit()
            
            # Send Discord notification
            await discord.send_signal_decision_notification(
                alert_dict, 
                accepted=False, 
                reason=mode_decision['reason']
            )
            return
        
        # Apply mode adjustments
        if 'adjustments' in mode_decision:
            alert_dict.update(mode_decision['adjustments'])
        
        # Process with Binance handler
        result = await binance_handler.process_signal(alert_dict)
        
        # Update alert record
        with Session() as session:
            alert_record = session.query(AlertHistory).filter_by(
                idempotency_key=idempotency_key
            ).first()
            if alert_record:
                alert_record.processed = True
                alert_record.error = result.get('reason') if result['status'] != 'success' else None
                session.commit()
        
        # Send Discord notification
        if result['status'] == 'success':
            await discord.send_signal_decision_notification(
                alert_dict,
                accepted=True,
                reason="Signal executed successfully",
                trade_id=result.get('trade_id')
            )
        else:
            await discord.send_signal_decision_notification(
                alert_dict,
                accepted=False,
                reason=result.get('reason', 'Unknown error')
            )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Alert processed in {processing_time:.2f}ms: {result['status']}")
        
    except Exception as e:
        logger.error(f"Failed to process alert: {e}", exc_info=True)
        
        # Update alert record
        with Session() as session:
            alert_record = session.query(AlertHistory).filter_by(
                idempotency_key=idempotency_key
            ).first()
            if alert_record:
                alert_record.processed = True
                alert_record.error = str(e)
                session.commit()
        
        # Send error notification
        await discord.send_error_notification(f"Alert processing failed: {e}")


@app.post("/webhook/test")
async def test_webhook():
    """Test endpoint for webhook validation"""
    
    # Generate test alert
    test_alert = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "indicator_version": "9.1",
        "timestamp": datetime.utcnow().isoformat(),
        "tier": "Standard",
        "strength": 65.0,
        "price": 50000.0,
        "stop_loss": 49000.0,
        "take_profit_1": 51000.0,
        "leverage": 10,
        "institutional_flow": 0.5,
        "retest_confidence": 0.7,
        "fake_breakout_detected": False,
        "regime": "BULLISH",
        "regime_confidence": 0.8,
        "mtf_agreement": 0.75,
        "timeframe": "15m",
        "comment": "Test alert - DO NOT EXECUTE"
    }
    
    try:
        # Validate schema
        alert = AlertPayloadV91(**test_alert)
        
        return WebhookResponse(
            status="success",
            message="Test alert validated successfully",
            data=test_alert
        )
        
    except Exception as e:
        return WebhookResponse(
            status="error",
            message=f"Test alert validation failed: {e}",
            data=test_alert
        )


@app.post("/control/pause")
async def pause_bot():
    """Pause bot trading"""
    with Session() as session:
        from database import set_setting
        set_setting(session, "is_paused", True, "bool", "api")
        
        await discord.send_control_notification("Bot PAUSED via API")
        
        return WebhookResponse(
            status="success",
            message="Bot paused"
        )


@app.post("/control/resume")
async def resume_bot():
    """Resume bot trading"""
    with Session() as session:
        from database import set_setting
        set_setting(session, "is_paused", False, "bool", "api")
        
        await discord.send_control_notification("Bot RESUMED via API")
        
        return WebhookResponse(
            status="success",
            message="Bot resumed"
        )


@app.post("/control/emergency")
async def trigger_emergency():
    """Trigger emergency mode"""
    try:
        results = await binance_handler.emergency_close_all("API trigger")
        
        return WebhookResponse(
            status="success",
            message="Emergency mode activated",
            data={"closed_positions": results}
        )
        
    except Exception as e:
        logger.error(f"Emergency trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/control/mode/{mode}")
async def change_mode(mode: str):
    """Change trading mode"""
    if mode not in Config.MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    
    with Session() as session:
        from database import set_setting
        set_setting(session, "mode", mode, "string", "api")
        
        await discord.send_control_notification(f"Mode changed to: {mode.upper()}")
        
        return WebhookResponse(
            status="success",
            message=f"Mode changed to {mode}",
            data={"mode": mode, "config": Config.MODES[mode]}
        )


@app.get("/stats/positions")
async def get_positions():
    """Get current open positions"""
    with Session() as session:
        from database import get_open_positions
        positions = get_open_positions(session)
        
        return {
            "count": len(positions),
            "positions": [p.to_dict() for p in positions]
        }


@app.get("/stats/performance/{period}")
async def get_performance(period: str = "daily"):
    """Get performance statistics"""
    if period not in ["daily", "weekly", "monthly"]:
        raise HTTPException(status_code=400, detail="Invalid period")
    
    with Session() as session:
        from database import get_performance_stats
        stats = get_performance_stats(session, period)
        
        return stats


@app.get("/stats/tiers")
async def get_tier_performance():
    """Get performance by tier"""
    with Session() as session:
        from database import get_tier_performance
        stats = get_tier_performance(session)
        
        return stats


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Webhook server starting up...")
    
    # Initialize async sessions
    await binance_handler.init_async()
    
    # Validate configuration
    if not Config.BINANCE_API_KEY or not Config.BINANCE_API_SECRET:
        logger.error("Binance API credentials not configured!")
    
    # Test database connection
    with Session() as session:
        try:
            session.execute("SELECT 1")
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
    
    logger.info(f"Webhook server ready (v9.1, Mode: {Config.DEFAULT_MODE})")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Webhook server shutting down...")
    
    # Close async sessions
    await binance_handler.close_async()
    
    logger.info("Webhook server stopped")


if __name__ == "__main__":
    # Run with uvicorn for development
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.WEBHOOK_PORT,
        log_level="info",
        access_log=True
    )
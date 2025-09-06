"""
Webhook endpoint for TradingView alerts v9.1
Enhanced with comprehensive validation, security, and bot integration
"""

import hashlib
import hmac
import json
import logging
logger = logging.getLogger(__name__)
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
import uvicorn

from config import Config
from database import( 
    Session,
    AlertHistory,
    get_setting,
    check_idempotency,
    set_setting,
    finalize_alert_processing,
    create_execution_trace,
    complete_execution_trace,
    create_decision_trace,
    update_decision_trace,
    log_pine_health,
    create_pattern_alert
)
from discord_notifications import discord_notifier
from database import log_execution_trace, complete_execution_trace
from bot import get_bot

# v9.1 Enhanced nested models
class VolumeContext(BaseModel):
    institutional_flow: float | None = None
    retest_confidence: float | None = None
    climax_detected: bool | None = None
    emergency_bypass: bool | None = None

class FakeBreakout(BaseModel):
    detected: bool | None = None
    penalty_multiplier: float | None = None
    emergency_bypass: bool | None = None

class V91Enhancements(BaseModel):
    volume_context: VolumeContext | None = None
    fake_breakout: FakeBreakout | None = None

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Binance Trading Bot Webhook v9.1",
    version="9.1",
    description="Enhanced webhook endpoint for TradingView alerts with v9.1 indicator support"
)

def get_bot():
    """Get bot instance from FastAPI app state"""
    return getattr(app.state, 'bot', None)

class AlertPayloadV91(BaseModel):
    action: str = Field(..., description="buy/sell/long/short/emergency_buy/emergency_sell/close/emergency_close")
    symbol: str = Field(..., description="e.g. BINANCE:BTCUSDT or BTCUSDT")
    price: float
    sl: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    break_even: float | None = None
    strength: float = 0.0
    tier: str = "Standard"
    position_size_multiplier: float = 1.0
    pair_tier: int | None = None
    leverage: int | None = None
    session: str | None = None
    timeframe: str | None = None

    # zagnieżdżone v9.1:
    v91_enhancements: V91Enhancements | None = None
    emergency_conditions: dict | None = None

    # alias na version z Pine:
    indicator_version: str = Field(alias="version")

    # nowe – do opóźnień:
    tv_ts: int | None = None             # ms since epoch (z Pine: time_close)
    timestamp: str | None = None         # opcjonalny ISO8601 fallback
    alert_id: str | None = None

    @field_validator("indicator_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
    # Akceptuj wszystkie wersje wskaźnika - nie blokuj na podstawie wersji
        if isinstance(v, str) and len(v) > 0:
            # Normalizuj do 9.1 jeśli zaczyna się od 9
            if v.startswith("9"):
                return "9.1"
            # Akceptuj inne wersje też
            return v
        return "9.1"  # Domyślna wersjaversion: {v}")

    @field_validator("action")
    @classmethod
    def normalize_action(cls, v: str) -> str:
        v = v.lower().strip()
        mapping = {
            "long": "buy",
            "short": "sell",
        }
        return mapping.get(v, v)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        # BINANCE:ETHUSDT -> ETHUSDT
        return v.split(":")[-1].upper()

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        allowed = ["Emergency", "Platinum", "Premium", "Standard", "Quick"]
        if v not in allowed:
            raise ValueError(f"Invalid tier: {v}. Must be one of {allowed}")
        return v


class WebhookResponse(BaseModel):
    """Standard webhook response"""
    status: str
    message: str
    data: Optional[Dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_time_ms: Optional[int] = None


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify webhook signature if configured - v9.1 ENHANCED"""
    if not Config.WEBHOOK_SECRET:
        return True  # No secret configured, skip verification
    
    try:
        # Support multiple signature formats
        if signature.startswith('sha256='):
            signature = signature[7:]
        
        expected_signature = hmac.new(
            Config.WEBHOOK_SECRET.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        return False


def generate_idempotency_key(alert: AlertPayloadV91) -> str:
    """Generate idempotency key for alert deduplication - v9.1 ENHANCED"""
    key_components = [
        alert.symbol,
        alert.action,
        str(alert.timestamp or ""),  # ← ZMIANA: konwertuj None na pusty string
        str(alert.price),
        alert.tier,
        str(alert.strength)
    ]

    if alert.alert_id:
        key_components.append(alert.alert_id)

    key_data = "_".join(key_components)
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def check_alert_age(alert: AlertPayloadV91) -> tuple[bool, int]:
    """
    True + sekundy wieku jeśli OK lub brak danych; False jeśli za stare.
    Priorytetem jest tv_ts (timestamp w ms).
    """
    try:
        now_utc = datetime.utcnow()
        age_sec = -1

        if alert.tv_ts and isinstance(alert.tv_ts, int):
            alert_time_utc = datetime.utcfromtimestamp(alert.tv_ts / 1000)
            age_sec = int((now_utc - alert_time_utc).total_seconds())
        elif alert.timestamp and isinstance(alert.timestamp, str):
            ts = alert.timestamp
            if ts.endswith('Z'):
                ts = ts[:-1] + '+00:00'
            alert_time_utc = datetime.fromisoformat(ts).replace(tzinfo=None)
            age_sec = int((now_utc - alert_time_utc).total_seconds())
        else:
            logger.info("Alert nie zawierał ani 'tv_ts', ani 'timestamp'. Nie można zweryfikować wieku.")
            return True, 0

        max_age = Config.ALERT_MAX_AGE_SEC
        if age_sec > max_age:
            logger.warning(f"Alert odrzucony jako przestarzały: {age_sec}s > {max_age}s")
            return False, age_sec
        
        return True, age_sec
    except Exception as e:
        logger.error(f"Krytyczny błąd podczas parsowania wieku alertu: {e}", exc_info=True)
        return True, 0 # W razie błędu parsowania, nie blokuj - na wypadek nietypowego formatu.

def validate_alert_conditions(alert: AlertPayloadV91) -> tuple[bool, str]:
    # wiek
    ok, age = check_alert_age(alert)
    if not ok:
        return False, f"Alert too old: {age}s > {Config.ALERT_MAX_AGE_SEC}s"

    # Akcje – akceptuj również emergency_* i close
    action = alert.action
    if action not in ["buy", "sell", "emergency_buy", "emergency_sell", "close", "emergency_close"]:
        return False, f"Unsupported action: {action}"

    # Akceptuj wszystkie alerty bez dodatkowych filtrów (siła/tier/SL/TP)
    return True, "Valid"


# Rate limiting storage
request_counts = {}
RATE_LIMIT_WINDOW = 60  # 1 minute
RATE_LIMIT_MAX_REQUESTS = 100  # Max requests per minute


def check_rate_limit(client_ip: str) -> bool:
    """Check rate limiting - v9.1 SECURITY"""
    current_time = time.time()
    
    # Clean old entries
    expired_ips = [
        ip for ip, data in request_counts.items()
        if current_time - data['window_start'] > RATE_LIMIT_WINDOW
    ]
    for ip in expired_ips:
        del request_counts[ip]
    
    # Check current IP
    if client_ip not in request_counts:
        request_counts[client_ip] = {
            'count': 1,
            'window_start': current_time
        }
        return True
    
    ip_data = request_counts[client_ip]
    
    # Reset window if expired
    if current_time - ip_data['window_start'] > RATE_LIMIT_WINDOW:
        ip_data['count'] = 1
        ip_data['window_start'] = current_time
        return True
    
    # Check limit
    if ip_data['count'] >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    ip_data['count'] += 1
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware - v9.1 SECURITY"""
    client_ip = request.client.host
    
    if not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "message": "Rate limit exceeded",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    response = await call_next(request)
    return response


@app.get("/")
async def root():
    """Health check endpoint"""
    return WebhookResponse(
        status="ok",
        message="Binance Trading Bot Webhook v9.1 is running"
    )


@app.get("/health")
async def health_check():
    """Detailed health check - v9.1 ENHANCED"""
    with Session() as session:
        try:
            # Check database
            db_healthy = session.execute("SELECT 1").scalar() == 1

            # Check bot instance
            bot = get_bot()
            bot_healthy = bot is not None and getattr(bot, "running", False)
            bot_running = bot_healthy

            # Try to read status z bota (bez twardych zależności)
            is_paused = False
            current_mode = Config.DEFAULT_MODE
            emergency_enabled = False
            try:
                if bot and hasattr(bot, "get_status"):
                    st = bot.get_status() or {}
                    is_paused = bool(st.get("paused", False))
                    current_mode = st.get("mode", current_mode)
                    emergency_enabled = bool(st.get("emergency_mode", False))
            except Exception:
                pass

            # Get open positions count
            from database import get_open_positions
            open_positions = len(get_open_positions(session))

            return {
                "status": "healthy" if (db_healthy and bot_running) else "degraded",
                "version": "9.1",
                "components": {
                    "database": "healthy" if db_healthy else "unhealthy",
                    "bot": "healthy" if bot_running else "unhealthy",
                    "discord": "healthy"
                },
                "bot_status": {
                    "running": bot_running,
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
    x_signature: Optional[str] = Header(None, alias="X-Signature"),
    x_tradingview_signature: Optional[str] = Header(None, alias="X-TradingView-Signature")
):
    """Main webhook endpoint for TradingView alerts - v9.1 ENHANCED"""
    start_time = time.time()
    
    # ===== DODAJ RATE LIMITING =====
    client_ip = request.client.host
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        logger.warning(f"🚫 Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Loguj podejrzane IP
    if client_ip not in ["127.0.0.1", "172.21.0.5", "172.21.0.1"]:
        logger.info(f"🔍 Webhook from IP: {client_ip}")
    # ===== KONIEC RATE LIMITING =====
    
    # Get raw payload
    raw_payload = await request.body()
    
    # Use either signature header
    signature = x_signature or x_tradingview_signature
    
    # Verify signature if configured and required
    if Config.REQUIRE_HMAC_SIGNATURE and Config.WEBHOOK_SECRET and signature:
        if not verify_signature(raw_payload, signature):
            logger.warning(f"Invalid webhook signature from {request.client.host}")
            raise HTTPException(status_code=401, detail="Invalid signature")
    elif Config.REQUIRE_HMAC_SIGNATURE and not signature:
        logger.warning(f"Missing webhook signature from {request.client.host}")
        raise HTTPException(status_code=401, detail="Missing signature")
    
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
                    data={"idempotency_key": idempotency_key},
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Record alert receipt with headers
            request_headers = dict(request.headers)
            alert_record = AlertHistory(
                idempotency_key=idempotency_key,
                symbol=alert.symbol,
                action=alert.action,
                tier=alert.tier,
                processed=False,
                raw_payload=payload_dict,
                headers=request_headers,
                signature_valid=bool(signature),
                schema_valid=True,
                received_at=datetime.utcnow()
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
                    alert_record.processed = True
                    session.commit()
    
            # Send immediate Discord notification for validation failure
            try:
                await discord_notifier.send_signal_notification(
                    alert.dict(),
                    accepted=False,
                    reason=validation_message
                )
            except Exception as e:
                logger.error(f"Failed to send Discord notification: {e}")
    
            return WebhookResponse(
                status="rejected",
                message=validation_message,
                data={"symbol": alert.symbol, "tier": alert.tier},
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Process alert in background
        background_tasks.add_task(process_alert_async, alert, idempotency_key, start_time)
        
        return WebhookResponse(
            status="accepted",
            message="Alert accepted for processing",
            data={
                "symbol": alert.symbol,
                "action": alert.action,
                "tier": alert.tier,
                "strength": alert.strength,
                "idempotency_key": idempotency_key
            },
            processing_time_ms=int((time.time() - start_time) * 1000)
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

async def process_alert_async(self, alert_data: dict) -> dict:
    """Process alert with diagnostics"""
    import uuid
    from datetime import datetime
    import time
    
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    idempotency_key = alert_data.get('idempotency_key', str(uuid.uuid4()))
    
    try:
        # Log execution trace start
        with Session() as session:
            trace_data = {
                'trace_id': trace_id,
                'symbol': alert_data.get('symbol', 'UNKNOWN'),
                'action': alert_data.get('action', 'UNKNOWN'),
                'stage': 'received',
                'status': 'running',
                'message': 'Alert received for processing',
                'context_data': {'alert_data': alert_data}
            }
            create_execution_trace(session, trace_data)
        
        # Wyciągnij diagnostykę
        diagnostics = alert_data.get('diagnostics', {})
        
        if diagnostics:
            # Przetwórz diagnostykę Pine Script
            pine_results = await self.process_pine_diagnostics(alert_data)
            
            # Zapisz do bazy
            with Session() as session:
                # Decision trace dla śledzenia
                decision_trace_data = {
                    'trace_id': trace_id,
                    'symbol': alert_data.get('symbol'),
                    'action': alert_data.get('action'),
                    'tier': alert_data.get('tier'),
                    'alert_timestamp': datetime.utcnow(),
                    'processing_stage': 'received',
                    'final_decision': 'PENDING',
                    'pine_health_score': diagnostics.get('health_score', 0.5),
                    'atr_percentile': diagnostics.get('atr_percentile'),
                    'adx_strength': diagnostics.get('adx'),
                    'volume_profile_score': diagnostics.get('volume_quality'),
                    'raw_alert_data': alert_data,
                    'decision_context': {'diagnostics': diagnostics, 'pine_results': pine_results}
                }
                create_decision_trace(session, decision_trace_data)
        
        # v9.1 NEW: Calculate latency metrics if tv_ts available
        latency_metrics = {}
        if alert_data.get('tv_ts'):
            webhook_received_ms = int(time.time() * 1000)
            tradingview_to_webhook_latency = webhook_received_ms - alert_data['tv_ts']
            latency_metrics = {
                "tv_ts": alert_data['tv_ts'],
                "webhook_received_ms": webhook_received_ms,
                "tradingview_to_webhook_latency_ms": tradingview_to_webhook_latency
            }
            
            # Log latency warnings
            if tradingview_to_webhook_latency > 1000:
                logger.warning(f"⚠️ High latency: {tradingview_to_webhook_latency}ms for {alert_data.get('symbol')}")
            elif tradingview_to_webhook_latency > 2000:
                logger.error(f"🚨 CRITICAL latency: {tradingview_to_webhook_latency}ms for {alert_data.get('symbol')}")
        
        # Przygotuj alert_dict z metadanymi
        alert_dict = dict(alert_data)  # Kopia alert_data
        alert_dict['idempotency_key'] = idempotency_key
        alert_dict['received_at'] = datetime.utcnow().isoformat()
        alert_dict['latency_metrics'] = latency_metrics
        alert_dict['trace_id'] = trace_id
        
        # Pobierz instancję bota
        from bot import get_bot
        bot = get_bot()
        
        if not bot:
            raise Exception("Bot instance not available")
        
        # v9.1 CORE: Process signal through bot
        success, message = await bot.process_signal(alert_dict)
        
        # Update alert record
        with Session() as session:
            finalize_alert_processing(session, idempotency_key, success=bool(success), error=None if success else message)
            
            # Update decision trace
            update_decision_trace(session, trace_id, {
                'processing_stage': 'completed',
                'final_decision': 'ACCEPTED' if success else 'REJECTED',
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'completed_at': datetime.utcnow()
            })
        
        # Send Discord notification
        await discord_notifier.send_signal_notification(
            alert_dict,
            accepted=success,
            reason=message,
            trade_id=trace_id if success else None
        )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Alert processed in {processing_time:.2f}ms: {'SUCCESS' if success else 'FAILED'} - {message}")
        
        # Complete execution trace
        complete_execution_trace(trace_id, success, int(processing_time), message)
        
        return {
            'success': success,
            'message': message,
            'trace_id': trace_id,
            'processing_time_ms': processing_time
        }
    
    except Exception as e:
        logger.error(f"Failed to process alert: {e}", exc_info=True)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Complete execution trace with error
        complete_execution_trace(trace_id, False, processing_time_ms, f"Processing error: {str(e)}")
        
        # Update alert record
        with Session() as session:
            finalize_alert_processing(session, idempotency_key, success=False, error=str(e))
            
            # Update decision trace if exists
            update_decision_trace(session, trace_id, {
                'processing_stage': 'failed',
                'final_decision': 'ERROR',
                'processing_time_ms': processing_time_ms,
                'completed_at': datetime.utcnow(),
                'error': str(e)
            })
        
        # Send error notification
        await discord_notifier.send_error_notification(
            f"Failed to process alert: {str(e)}",
            title="Alert Processing Error",
            critical=True
        )
        
        return {
            'success': False,
            'message': str(e),
            'trace_id': trace_id,
            'processing_time_ms': processing_time_ms
        }


@app.post("/webhook/test")
async def test_webhook():
    """Test endpoint for webhook validation - v9.1 ENHANCED"""
    
    # Generate comprehensive test alert
    test_alert = {
        "symbol": "BTCUSDT",
        "action": "buy",
        "indicator_version": "9.1",
        "timestamp": datetime.utcnow().isoformat(),
        "tier": "Standard",
        "strength": 0.65,
        "price": 50000.0,
        "sl": 49000.0,
        "tp1": 51000.0,
        "tp2": 52000.0,
        "tp3": 53000.0,
        "leverage": 10,
        "position_size_multiplier": 1.0,
        
        # v9.1 fields
        "institutional_flow": 0.5,
        "retest_confidence": 0.7,
        "fake_breakout_detected": False,
        "fake_breakout_penalty": 1.0,
        "enhanced_regime": "TRENDING_UP",
        "regime_confidence": 0.8,
        "mtf_agreement_ratio": 0.75,
        
        # Technical indicators
        "mfi": 65.0,
        "adx": 30.0,
        "rsi": 60.0,
        "htf_trend": "bullish",
        "btc_correlation": 0.8,
        
        # Volume data
        "volume_spike": True,
        "volume_ratio": 2.5,
        "institutional_volume": 10000,
        "retail_volume": 50000,
        
        # Structure
        "near_key_level": True,
        "structure_break": True,
        "order_block_retest": True,
        
        # Context
        "timeframe": "15m",
        "session": "London",
        "market_condition": "NORMAL",
        "confidence_penalty": 0.0,
        
        "comment": "TEST ALERT - DO NOT EXECUTE"
    }
    
    try:
        # Validate schema
        alert = AlertPayloadV91(**test_alert)
        
        # Validate conditions
        is_valid, validation_message = validate_alert_conditions(alert)
        
        return WebhookResponse(
            status="success" if is_valid else "validation_failed",
            message=f"Test alert {'validated successfully' if is_valid else f'validation failed: {validation_message}'}",
            data=test_alert
        )
    
    except Exception as e:
        return WebhookResponse(
            status="error",
            message=f"Test alert validation failed: {e}",
            data=test_alert
        )


# v9.1 CORE: Control endpoints
@app.post("/control/pause")
async def pause_bot():
    """Pause bot trading - v9.1 ENHANCED"""
    try:
        bot = get_bot()
        if not bot:
            raise HTTPException(status_code=503, detail="Bot not available")
        bot.pause_trading()
        return WebhookResponse(status="success", message="Bot paused successfully")
    except Exception as e:
        logger.error(f"Failed to pause bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/control/resume")
async def resume_bot():
    """Resume bot trading - v9.1 ENHANCED"""
    try:
        bot = get_bot()
        if not bot:
            raise HTTPException(status_code=503, detail="Bot not available")
        bot.resume_trading()
        return WebhookResponse(status="success", message="Bot resumed successfully")
    except Exception as e:
        logger.error(f"Failed to resume bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/control/mode/{mode}")
async def change_mode(mode: str):
    """Change trading mode - v9.1 ENHANCED"""
    try:
        bot = get_bot()
        if not bot:
            raise HTTPException(status_code=503, detail="Bot not available")

        ok = False
        msg = "Mode change failed"
        if hasattr(bot, "mode_manager") and hasattr(bot.mode_manager, "set_mode"):
            ok = bot.mode_manager.set_mode(mode, reason="Manual API")
            msg = f"Mode changed to {mode}" if ok else f"Invalid mode: {mode}"

        return WebhookResponse(
            status="success" if ok else "error",
            message=msg,
            data={"mode": mode} if ok else None
        )

    except Exception as e:
        logger.error(f"Failed to change mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/control/emergency")
async def toggle_emergency():
    """Toggle emergency mode - v9.1 ENHANCED"""
    try:
        bot = get_bot()
        if not bot:
            raise HTTPException(status_code=503, detail="Bot not available")

        if getattr(bot, "emergency_mode", False):
            bot.disable_emergency_mode()
            message = "Emergency mode disabled"
        else:
            bot.enable_emergency_mode()
            message = "Emergency mode enabled"

        return WebhookResponse(status="success", message=message)

    except Exception as e:
        logger.error(f"Failed to toggle emergency mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_bot_status():
    """Get comprehensive bot status - v9.1 FEATURE"""
    try:
        bot = get_bot()
        if not bot:
            raise HTTPException(status_code=503, detail="Bot not available")
        
        status = bot.get_status()
        
        return WebhookResponse(
            status="success",
            message="Bot status retrieved",
            data=status
        )
    
    except Exception as e:
        logger.error(f"Failed to get bot status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint - v9.1 MONITORING"""
    try:
        # Import here to avoid circular imports
        from main import bot_instance
        
        # Basic metrics
        metrics = []
        
        # Bot status metrics
        bot_running = 1 if bot_instance and bot_instance.running else 0
        bot_paused = 1 if bot_instance and bot_instance.paused else 0
        bot_emergency = 1 if bot_instance and bot_instance.emergency_mode else 0
        
        metrics.append(f'trading_bot_running{{version="9.1"}} {bot_running}')
        metrics.append(f'trading_bot_paused{{version="9.1"}} {bot_paused}')
        metrics.append(f'trading_bot_emergency{{version="9.1"}} {bot_emergency}')
        
        if bot_instance:
            # Position metrics
            metrics.append(f'trading_bot_active_positions{{version="9.1"}} {len(bot_instance.active_positions)}')
            metrics.append(f'trading_bot_daily_pnl{{version="9.1"}} {bot_instance.daily_pnl}')
            metrics.append(f'trading_bot_daily_trades{{version="9.1"}} {bot_instance.daily_trades}')
            
            # Performance metrics
            perf = bot_instance.performance_metrics
            metrics.append(f'trading_bot_total_signals{{version="9.1"}} {perf.get("total_signals", 0)}')
            metrics.append(f'trading_bot_signals_taken{{version="9.1"}} {perf.get("signals_taken", 0)}')
            metrics.append(f'trading_bot_signals_rejected{{version="9.1"}} {perf.get("signals_rejected", 0)}')
            metrics.append(f'trading_bot_ml_rejections{{version="9.1"}} {perf.get("ml_rejections", 0)}')
            metrics.append(f'trading_bot_fake_breakout_detections{{version="9.1"}} {perf.get("fake_breakout_detections", 0)}')
            
            # Position guards
            metrics.append(f'trading_bot_position_guards{{version="9.1"}} {len(bot_instance.position_guards)}')
        
        # Database metrics
        try:
            with Session() as session:
                from database import Position, Trade
                open_positions = session.query(Position).filter(Position.status == 'open').count()
                total_trades = session.query(Trade).count()
                
                metrics.append(f'trading_bot_db_open_positions{{version="9.1"}} {open_positions}')
                metrics.append(f'trading_bot_db_total_trades{{version="9.1"}} {total_trades}')
        except Exception as e:
            logger.warning(f"Could not get database metrics: {e}")
        
        # System metrics
        try:
            import psutil
            metrics.append(f'trading_bot_cpu_percent{{version="9.1"}} {psutil.cpu_percent()}')
            metrics.append(f'trading_bot_memory_percent{{version="9.1"}} {psutil.virtual_memory().percent}')
        except ImportError:
            # psutil not available, skip system metrics
            pass
        
        return "\n".join(metrics) + "\n"
    
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return f"# Error generating metrics: {e}\n"


@app.get("/stats/positions")
async def get_positions():
    """Get current open positions - v9.1 ENHANCED"""
    with Session() as session:
        try:
            from database import get_open_positions
            positions = get_open_positions(session)
            
            return WebhookResponse(
                status="success",
                message=f"Retrieved {len(positions)} open positions",
                data={
                    "count": len(positions),
                    "positions": [
                        {
                            "id": p.id,
                            "symbol": p.symbol,
                            "side": p.side,
                            "status": p.status,
                            "entry_price": float(p.entry_price) if p.entry_price else None,
                            "current_pnl": float(p.pnl_usdt) if p.pnl_usdt is not None else None,
                            "entry_time": p.entry_time.isoformat() if p.entry_time else None
                        } for p in positions
                    ]
                }
            )
        
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions - v9.1 ENHANCED"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions - v9.1 ENHANCED"""
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize on startup - v9.1 ENHANCED"""
    logger.info("🚀 Webhook server v9.1 starting up...")
    
    # Validate configuration (uwzględnia testnet)
    api_key = Config.get_api_key()
    api_secret = Config.get_api_secret()
    if not api_key or not api_secret:
        logger.error(f"❌ Binance API credentials not configured! (IS_TESTNET={Config.IS_TESTNET})")
    else:
        logger.info(f"✅ Binance credentials detected (IS_TESTNET={Config.IS_TESTNET})")
    
    # Test database connection
    with Session() as session:
        try:
            session.execute("SELECT 1")
            logger.info("✅ Database connection successful")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
    
    # Check bot availability
    bot = get_bot()
    if bot:
        logger.info("✅ Bot instance available")
    else:
        logger.warning("⚠️ Bot instance not available - some endpoints will be disabled")
    
    logger.info(f"✅ Webhook server v9.1 ready (Mode: {Config.DEFAULT_MODE})")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown - v9.1 ENHANCED"""
    logger.info("🛑 Webhook server shutting down...")
    logger.info("✅ Webhook server stopped")

@app.get("/diagnostics")
async def get_diagnostics():
    """Get comprehensive system diagnostics - v9.1 HYBRID ULTRA-DIAGNOSTICS"""
    try:
        start_time = time.time()
        
        # Log execution trace
        trace_id = log_execution_trace("webhook_diagnostics", {"endpoint": "/diagnostics"})
        
        bot = get_bot()
        diagnostics = {
            "webhook_status": {
                "version": "9.1",
                "uptime_seconds": int(time.time() - start_time),
                "rate_limit_active": len(request_counts) > 0,
                "active_connections": len(request_counts)
            },
            "bot_status": {
                "available": bot is not None,
                "running": getattr(bot, "running", False) if bot else False,
                "paused": getattr(bot, "paused", False) if bot else False,
                "emergency_mode": getattr(bot, "emergency_mode", False) if bot else False
            },
            "database_status": {},
            "recent_alerts": [],
            "performance_metrics": {}
        }
        
        # Database diagnostics
        with Session() as session:
            try:
                session.execute("SELECT 1")
                diagnostics["database_status"]["connected"] = True
                
                # Recent alerts count
                from datetime import timedelta
                recent_cutoff = datetime.utcnow() - timedelta(hours=1)
                recent_alerts = session.query(AlertHistory).filter(
                    AlertHistory.received_at >= recent_cutoff
                ).count()
                
                diagnostics["recent_alerts"] = {
                    "last_hour": recent_alerts,
                    "rate_per_minute": recent_alerts / 60.0
                }
                
            except Exception as e:
                diagnostics["database_status"]["connected"] = False
                diagnostics["database_status"]["error"] = str(e)
        
        # Bot diagnostics
        if bot and hasattr(bot, 'get_diagnostics'):
            try:
                bot_diagnostics = await bot.get_diagnostics()
                diagnostics["bot_diagnostics"] = bot_diagnostics
            except Exception as e:
                diagnostics["bot_diagnostics"] = {"error": str(e)}
        
        # Performance metrics
        processing_time = int((time.time() - start_time) * 1000)
        diagnostics["performance_metrics"] = {
            "diagnostics_processing_time_ms": processing_time,
            "webhook_response_time": "< 100ms" if processing_time < 100 else "slow"
        }
        
        # Complete execution trace
        complete_execution_trace(trace_id, True, processing_time, "Diagnostics completed")
        
        return WebhookResponse(
            status="success",
            message="Diagnostics retrieved successfully",
            data=diagnostics,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Failed to get diagnostics: {e}")
        if 'trace_id' in locals():
            complete_execution_trace(trace_id, False, int((time.time() - start_time) * 1000), f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/diagnostics/alerts")
async def get_alert_diagnostics():
    """Get detailed alert processing diagnostics - v9.1 DIAGNOSTICS"""
    try:
        with Session() as session:
            # Recent alerts analysis
            from datetime import timedelta
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            
            recent_alerts = session.query(AlertHistory).filter(
                AlertHistory.received_at >= recent_cutoff
            ).all()
            
            # Analyze alerts
            total_alerts = len(recent_alerts)
            processed_alerts = len([a for a in recent_alerts if a.processed])
            successful_alerts = len([a for a in recent_alerts if a.processed and not a.error])
            
            # Group by tier
            tier_stats = {}
            for alert in recent_alerts:
                tier = alert.tier or "Unknown"
                if tier not in tier_stats:
                    tier_stats[tier] = {"total": 0, "processed": 0, "successful": 0}
                
                tier_stats[tier]["total"] += 1
                if alert.processed:
                    tier_stats[tier]["processed"] += 1
                    if not alert.error:
                        tier_stats[tier]["successful"] += 1
            
            # Common errors
            error_counts = {}
            for alert in recent_alerts:
                if alert.error:
                    error = alert.error[:100]  # Truncate long errors
                    error_counts[error] = error_counts.get(error, 0) + 1
            
            diagnostics = {
                "time_period": "Last 24 hours",
                "summary": {
                    "total_alerts": total_alerts,
                    "processed_alerts": processed_alerts,
                    "successful_alerts": successful_alerts,
                    "processing_rate": processed_alerts / max(1, total_alerts),
                    "success_rate": successful_alerts / max(1, processed_alerts)
                },
                "tier_breakdown": tier_stats,
                "common_errors": dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                "recent_failures": [
                    {
                        "symbol": alert.symbol,
                        "tier": alert.tier,
                        "error": alert.error,
                        "received_at": alert.received_at.isoformat()
                    }
                    for alert in recent_alerts[-10:] if alert.error
                ]
            }
            
            return WebhookResponse(
                status="success",
                message="Alert diagnostics retrieved",
                data=diagnostics
            )
            
    except Exception as e:
        logger.error(f"Failed to get alert diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Custom logging config to reduce spam
    import logging
    
    # Filtr dla health/metrics
    class HealthMetricsFilter(logging.Filter):
        def filter(self, record):
            if hasattr(record, 'getMessage'):
                message = record.getMessage()
                # Filtruj health/metrics/status
                if any(endpoint in message for endpoint in ['/health', '/metrics', '/status']):
                    return False
            return True
    
    # Dodaj filtr do uvicorn access log
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.addFilter(HealthMetricsFilter())
    
    # Run with uvicorn for development
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.WEBHOOK_PORT,
        log_level="info",
        access_log=True
    )
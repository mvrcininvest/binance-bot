"""
Webhook endpoint for TradingView alerts v9.1
Enhanced with comprehensive validation, security, and bot integration
"""

import hashlib
import hmac
import json
import logging
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
    create_pattern_alert,
    record_alert
)
from discord_notifications import discord_notifier
from database import log_execution_trace, complete_execution_trace


# Wklej ten kod na górze pliku webhook.py, pod importami

from typing import Optional, Any, List
from pydantic import BaseModel, Field, field_validator
import logging
logger = logging.getLogger(__name__) # Upewnij się, że ta linia jest na górze pliku

app = FastAPI(
    title="Binance Trading Bot Webhook v9.1",
    version="9.1",
    description="Enhanced webhook endpoint for TradingView alerts with v9.1 indicator support"
)

class AlertDiagnostics(BaseModel):
    timestamp: int
    health: float
    regime: str
    regime_conf: float
    buy_str: float
    sell_str: float
    zones: int
    inst_flow: float
    acc_ratio: float
    fake_break: float
    mtf_agree: float

class AlertPayloadV91(BaseModel):
    # --- Pola z alertu Pine Script ---
    symbol: str
    action: str
    tier: str
    strength: float
    price: float
    atr: float
    volume_ratio: float
    session: str
    regime: str
    regime_confidence: float
    mtf_agreement: float
    leverage: int
    version: str # <--- POPRAWKA: Zmieniono 'indicator_version' na 'version'
    diagnostics: AlertDiagnostics
    in_ob: bool
    in_fvg: bool
    ob_score: float
    fvg_score: float
    institutional_flow: float
    accumulation: float
    volume_climax: bool
    tv_ts: int

    # --- Opcjonalne pola dla logiki hybrydowej i przyszłych wersji wskaźnika ---
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    break_even: Optional[float] = None
    position_size_multiplier: float = 1.0
    pair_tier: Optional[int] = None
    timeframe: Optional[str] = None
    # <--- POPRAWKA: Usunięto 'timestamp' i 'alert_id', które nie są wysyłane z Pine Script

    @field_validator("version", mode='before')
    @classmethod
    def validate_version(cls, v: Any) -> str:
        # <--- POPRAWKA: Uproszczony i bardziej odporny walidator
        v_str = str(v)
        if v_str.startswith("9"):
            return v_str
        else:
            logger.warning(f"Unexpected indicator version: {v_str}")
            return v_str

    @field_validator("action")
    @classmethod
    def normalize_action(cls, v: str) -> str:
        v = v.lower().strip()
        mapping = {"long": "buy", "short": "sell"}
        return mapping.get(v, v)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        if v.startswith('BINANCE:'):
            v = v[8:]
        if v.endswith('.P'):
            v = v[:-2]
        return v.upper()

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        allowed = ["Emergency", "Platinum", "Premium", "Standard", "Quick"]
        if v not in allowed:
            logger.warning(f"Invalid tier received: '{v}'. Defaulting to 'Standard'.")
            return "Standard"
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
    # --- POPRAWKA: Używamy tv_ts (timestamp z TradingView), który jest zawsze obecny ---
    key_components = [
        alert.symbol,
        alert.action,
        str(alert.tv_ts), # Używamy tv_ts zamiast nieistniejącego timestamp
        str(alert.price),
        alert.tier,
        str(alert.strength)
    ]

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
        if age_sec < 0:
             logger.warning(f"Obliczono ujemny wiek alertu ({age_sec}s). Sprawdź synchronizację czasu na serwerze i w TradingView.")
             return True, age_sec # Nie odrzucaj, ale zaloguj

        if age_sec > max_age:
            logger.warning(f"Alert odrzucony jako przestarzały: {age_sec}s > {max_age}s")
            return False, age_sec
        
        return True, age_sec
    except Exception as e:
        logger.error(f"Krytyczny błąd podczas parsowania wieku alertu: {e}", exc_info=True)
        return True, 0

def validate_alert_conditions(alert: AlertPayloadV91) -> tuple[bool, str]:
    # 1. Sprawdzenie wieku alertu (krytyczne)
    is_ok, age_or_reason = check_alert_age(alert)
    if not is_ok:
        # Zmieniamy age_or_reason na string, aby uniknąć błędów typu
        return False, f"Alert too old: {age_or_reason}s > {Config.ALERT_MAX_AGE_SEC}s"

    # 2. Sprawdzenie akcji
    action = alert.action
    allowed_actions = ["buy", "sell", "emergency_buy", "emergency_sell", "close", "emergency_close"]
    if action not in allowed_actions:
        return False, f"Unsupported action: {action}"

    # 3. Sprawdzenie ceny
    if alert.price <= 0:
        return False, f"Invalid price: {alert.price}"

    # Walidacja przeszła pomyślnie
    return True, "Alert validation passed"


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
    """Sprawdza podstawowy stan serwera webhook i połączenia z bazą danych."""
    try:
        # Sprawdzamy, czy serwer jest w stanie wykonać prostą operację na bazie danych.
        # To potwierdza, że zarówno proces webowy, jak i połączenie z DB działają.
        with Session() as session:
            session.execute("SELECT 1")

        return {
            "status": "healthy",
            "version": "9.1",
            "message": "Webhook service is running and database is connected.",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Jeśli wystąpi błąd, zwracamy kod 503 (Service Unavailable)
        raise HTTPException(status_code=503, detail="Service unhealthy: Database connection failed.")


@app.post("/webhook/tradingview")
async def tradingview_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    payload: AlertPayloadV91,
    x_signature: Optional[str] = Header(None, alias="X-Signature"),
    x_tradingview_signature: Optional[str] = Header(None, alias="X-TradingView-Signature")
):
    """Main webhook endpoint for TradingView alerts - v9.1 ENHANCED"""
    start_time = time.time()

    # --- OSTATECZNA POPRAWKA: Pobieramy instancję bota bezpośrednio ze stanu aplikacji FastAPI ---
    bot = request.app.state.bot

    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        logger.warning(f"🚫 Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    raw_payload = await request.body()
    signature = x_signature or x_tradingview_signature

    if Config.REQUIRE_HMAC_SIGNATURE and Config.WEBHOOK_SECRET and signature:
        if not verify_signature(raw_payload, signature):
            logger.warning(f"Invalid webhook signature from {request.client.host}")
            raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        idempotency_key = generate_idempotency_key(payload)

        with Session() as session:
            if check_idempotency(session, idempotency_key):
                logger.info(f"Duplicate alert rejected: {idempotency_key}")
                return JSONResponse(status_code=200, content={"status": "rejected", "reason": "Duplicate alert"})

            latency_metrics = {
                "tv_ts": payload.tv_ts,
                "tradingview_to_webhook_latency_ms": int((time.time() * 1000) - payload.tv_ts) if payload.tv_ts else None
            }
            record_alert(
                session=session,
                payload=payload.model_dump(),
                headers=dict(request.headers),
                idempotency_key=idempotency_key,
                processed=False,
                latency_metrics=latency_metrics,
                signature_valid=bool(signature)
            )

        is_valid, validation_message = validate_alert_conditions(payload)

        if not is_valid:
            logger.info(f"Alert validation failed: {validation_message}")
            with Session() as session:
                finalize_alert_processing(session, idempotency_key, success=False, error=validation_message)
            await discord_notifier.send_signal_notification(
                payload.model_dump(),
                accepted=False,
                reason=validation_message
            )
            return JSONResponse(status_code=200, content={"status": "rejected", "reason": validation_message})

        # --- OSTATECZNA POPRAWKA: Sprawdzamy, czy instancja bota jest dostępna ---
        if not bot:
            logger.error("FATAL: Bot instance not available for webhook processing.")
            raise HTTPException(status_code=503, detail="Bot instance not available")

        alert_dict = payload.model_dump()
        background_tasks.add_task(bot.handle_signal, alert_dict)

        return WebhookResponse(
            status="accepted",
            message="Alert accepted for asynchronous processing",
            data={
                "symbol": payload.symbol, "action": payload.action, "tier": payload.tier,
                "strength": payload.strength, "idempotency_key": idempotency_key
            },
            processing_time_ms=int((time.time() - start_time) * 1000)
        )

    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")




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
    
@app.post("/webhook")
async def webhook_fallback(request: Request, background_tasks: BackgroundTasks):
    """Fallback webhook endpoint - redirects to main handler"""
    # Parse raw JSON
    raw_body = await request.body()
    try:
        payload_dict = json.loads(raw_body)
        payload = AlertPayloadV91(**payload_dict)
        return await tradingview_webhook(request, background_tasks, payload)
    except Exception as e:
        logger.error(f"Webhook fallback error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

def get_bot():
    """Get bot instance from app state"""
    try:
        from main import bot_instance
        return bot_instance
    except ImportError:
        logger.error("Cannot import bot_instance from main")
        return None

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

    # W webhook.py, dodaj ten kod pod istniejącymi klasami Pydantic

# --- Modele dla nowego API Dashboardu ---

class StatsResponse(BaseModel):
    totalPnL: float
    todayPnL: float
    winRate: float
    totalTrades: int
    activePositions: int

class StatusResponse(BaseModel):
    status: str
    botStatus: str
    tradingViewWebhook: str
    lastHeartbeat: Optional[str] = None

class Position(BaseModel):
    symbol: str
    side: str
    size: float
    entryPrice: float
    markPrice: float
    pnl: float
    pnlPercentage: float

class PositionsResponse(BaseModel):
    positions: List[Position]

class Signal(BaseModel):
    id: str
    symbol: str
    type: str
    price: float
    timestamp: str
    status: str

class SignalsResponse(BaseModel):
    signals: List[Signal]

class CommandRequest(BaseModel):
    command: str
    params: Optional[Dict[str, Any]] = None

# W webhook.py, wklej ten kod na samym końcu pliku

# --- API Endpoints dla Dashboardu ---

@app.get("/api/stats", response_model=StatsResponse, tags=["Dashboard API"])
async def get_stats():
    """Zwraca kluczowe statystyki wydajności bota."""
    bot = get_bot()
    if not bot:
        raise HTTPException(status_code=503, detail="Bot instance not available")
    
    try:
        # Używamy istniejącej funkcji do pobierania statystyk
        perf = await asyncio.to_thread(get_profile_performance, days=90)
        
        # Pobieramy PnL z dzisiaj
        today_perf = await asyncio.to_thread(get_profile_performance, days=1)

        return StatsResponse(
            totalPnL=perf.get("total_pnl", 0.0),
            todayPnL=today_perf.get("total_pnl", 0.0),
            winRate=perf.get("win_rate", 0.0),
            totalTrades=perf.get("total_trades", 0),
            activePositions=len(bot.active_positions)
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve bot statistics.")

@app.get("/api/status", response_model=StatusResponse, tags=["Dashboard API"])
async def get_status():
    """Zwraca status operacyjny bota."""
    bot = get_bot()
    if not bot:
        return StatusResponse(
            status="offline",
            botStatus="stopped",
            tradingViewWebhook="unknown",
            lastHeartbeat=None
        )

    bot_state = bot.get_status()
    bot_status_str = "active"
    if bot_state.get("emergency_mode"):
        bot_status_str = "emergency"
    elif bot_state.get("paused"):
        bot_status_str = "paused"

    return StatusResponse(
        status="online",
        botStatus=bot_status_str,
        tradingViewWebhook="accessible", # Zakładamy, że jest ok, skoro API odpowiada
        lastHeartbeat=bot_state.get("last_heartbeat")
    )

@app.get("/api/positions", response_model=PositionsResponse, tags=["Dashboard API"])
async def get_positions():
    """Zwraca listę aktywnych pozycji z giełdy."""
    try:
        # [cite_start]Używamy binance_handler do pobrania danych na żywo [cite: 1]
        raw_positions = await asyncio.to_thread(binance_handler.check_positions)
        
        positions_list = []
        for p in raw_positions:
            entry_price = float(p.get("entryPrice", 0))
            mark_price = float(p.get("markPrice", 0))
            pnl = float(p.get("unRealizedProfit", 0))
            
            pnl_percentage = (pnl / (float(p.get("positionAmt", 0)) * entry_price)) * 100 * float(p.get("leverage", 1)) if entry_price > 0 and float(p.get("positionAmt", 0)) != 0 else 0

            positions_list.append(Position(
                symbol=p.get("symbol"),
                side="LONG" if float(p.get("positionAmt", 0)) > 0 else "SHORT",
                size=abs(float(p.get("positionAmt", 0))),
                entryPrice=entry_price,
                markPrice=mark_price,
                pnl=pnl,
                pnlPercentage=pnl_percentage
            ))
        return PositionsResponse(positions=positions_list)
    except Exception as e:
        logger.error(f"Error getting positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve positions.")

@app.get("/api/signals", response_model=List[Signal], tags=["Dashboard API"])
async def get_signals():
    """Zwraca listę ostatnich 10 sygnałów z historii."""
    bot = get_bot()
    if not bot:
        return []

    # Używamy historii sygnałów przechowywanej w pamięci bota
    recent_signals = bot.signal_history[-10:] # Ostatnie 10
    
    signals_list = []
    for s in reversed(recent_signals): # Najnowsze na górze
        decision = s.get('decision', {})
        status = "executed" if (decision.get('execution_result') or {}).get('status') == 'success' else decision.get('reason', 'rejected')

        signals_list.append(Signal(
            id=str(s.get('trace_id', s['received_at'])),
            symbol=s.get('symbol'),
            type=s.get('action').upper(),
            price=s.get('price'),
            timestamp=s.get('received_at'),
            status=status
        ))
    return signals_list

@app.post("/api/command", tags=["Dashboard API"])
async def post_command(command_req: CommandRequest):
    """Przyjmuje i wykonuje komendy dla bota."""
    bot = get_bot()
    if not bot:
        raise HTTPException(status_code=503, detail="Bot is not running, cannot execute command.")

    command = command_req.command
    logger.info(f"Received command from API: {command}")

    if command == "pause-bot":
        await bot.pause()
        return {"status": "success", "message": "Bot paused."}
    elif command == "resume-bot":
        await bot.resume()
        return {"status": "success", "message": "Bot resumed."}
    elif command == "restart-bot":
        # Uwaga: Prawdziwy restart jest skomplikowany w Dockerze.
        # Ta komenda na razie tylko zasygnalizuje potrzebę restartu.
        logger.warning("API triggered a restart request. Manual restart is required.")
        # W przyszłości można zaimplementować mechanizm, który zakończy proces bota,
        # a Docker Compose go automatycznie podniesie.
        # asyncio.create_task(bot.stop()) # To by zatrzymało bota
        return {"status": "pending", "message": "Restart request logged. Manual restart required."}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {command}")
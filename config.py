"""
Binance Trading Bot Configuration
Version: 9.1
"""

import os
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def _to_bool(value: Any, default: bool = False) -> bool:
    """Convert string to boolean safely"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert string to float safely"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    """Convert string to int safely"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


class Config:
    """Main configuration class for the trading bot"""

    # ==== CORE SETTINGS ====
    # Bot identification
    BOT_NAME = os.getenv("BOT_NAME", "BinanceBot")
    BOT_VERSION = "9.1"

    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    IS_TESTNET = _to_bool(os.getenv("BINANCE_TESTNET", os.getenv("IS_TESTNET", "False")))
    DRY_RUN = _to_bool(os.getenv("DRY_RUN", "False"))
    DEBUG = _to_bool(os.getenv("DEBUG", "False"))

    # Machine Learning and Signal Intelligence
    ML_ENABLED = _to_bool(os.getenv("ML_ENABLED", "False"))
    SIGNAL_INTELLIGENCE_ENABLED = _to_bool(os.getenv("SIGNAL_INTELLIGENCE_ENABLED", "True"))
    USE_ML_FOR_SIZING = _to_bool(os.getenv("USE_ML_FOR_SIZING", "False"))

    # ==== API CREDENTIALS ====
    # Binance API
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()
    BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "").strip()
    BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "").strip()

    # Discord
    DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()
    DISCORD_CHANNEL_ID = _to_int(os.getenv("DISCORD_CHANNEL_ID", "0"))
    DISCORD_GUILD_ID = _to_int(os.getenv("DISCORD_GUILD_ID", "0"))
    
    # Discord Webhooks - Specialized channels
    DISCORD_ALERTS_WEBHOOK = os.getenv("DISCORD_ALERTS_WEBHOOK", "").strip()
    DISCORD_TRADES_WEBHOOK = os.getenv("DISCORD_TRADE_ENTRIES_WEBHOOK", "").strip()
    DISCORD_ERRORS_WEBHOOK = os.getenv("DISCORD_LOGS_WEBHOOK", "").strip()
    DISCORD_ANALYTICS_WEBHOOK = os.getenv("DISCORD_PERFORMANCE_WEBHOOK", "").strip()
    DISCORD_ADMIN_IDS = [
        int(id.strip())
        for id in os.getenv("DISCORD_ADMIN_IDS", "").split(",")
        if id.strip()
    ]

    # ==== DATABASE ====
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///trading_bot.db")
    DB_POOL_SIZE = _to_int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW = _to_int(os.getenv("DB_MAX_OVERFLOW", "20"))
    DB_POOL_TIMEOUT = _to_int(os.getenv("DB_POOL_TIMEOUT", "30"))

    # ==== WEBHOOK SECURITY ====
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
    REQUIRE_HMAC_SIGNATURE = _to_bool(os.getenv("REQUIRE_HMAC_SIGNATURE", "False"))
    WEBHOOK_PORT = _to_int(os.getenv("WEBHOOK_PORT", "80"))
    WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
    WEBHOOK_MAX_SIZE_MB = _to_int(os.getenv("WEBHOOK_MAX_SIZE_MB", "1"))

    # Rate limiting
    WEBHOOK_RATE_LIMIT_PER_MINUTE = _to_int(
        os.getenv("WEBHOOK_RATE_LIMIT_PER_MINUTE", "100")
    )
    WEBHOOK_RATE_LIMIT_PER_SECOND = _to_int(
        os.getenv("WEBHOOK_RATE_LIMIT_PER_SECOND", "10")
    )

    # ==== V9.1 FEATURE FLAGS ====
    # Leverage management
    USE_INDICATOR_LEVERAGE = _to_bool(os.getenv("USE_INDICATOR_LEVERAGE", "False"))
    USE_INDICATOR_SIZE_MULTIPLIER = _to_bool(
        os.getenv("USE_INDICATOR_SIZE_MULTIPLIER", "True")
    )

    # Position management
    MOVE_SL_TO_BE_AT_TP1 = _to_bool(os.getenv("MOVE_SL_TO_BE_AT_TP1", "True"))
    USE_TRAILING_AFTER_TP2 = _to_bool(os.getenv("USE_TRAILING_AFTER_TP2", "True"))
    TRAILING_STOP_PERCENTAGE = _to_float(os.getenv("TRAILING_STOP_PERCENTAGE", "0.5"))

    # Emergency mode
    EMERGENCY_ENABLED = _to_bool(os.getenv("EMERGENCY_ENABLED", "True"))
    EMERGENCY_CLOSE_THRESHOLD = _to_float(os.getenv("EMERGENCY_CLOSE_THRESHOLD", "5.0"))
    EMERGENCY_CLOSE_AND_REVERSE = _to_bool(
        os.getenv("EMERGENCY_CLOSE_AND_REVERSE", "False")
    )

    # ==== POSITION GUARDS ====
    # Alert validation
    ALERT_MAX_AGE_SEC = _to_int(os.getenv("ALERT_MAX_AGE_SEC", "30"))
    MAX_PRICE_DRIFT_PCT = _to_float(os.getenv("MAX_PRICE_DRIFT_PCT", "0.6"))

    # Position limits
    SINGLE_POSITION_PER_SYMBOL = _to_bool(
        os.getenv("SINGLE_POSITION_PER_SYMBOL", "True")
    )
    MAX_POSITIONS = _to_int(os.getenv("MAX_POSITIONS", "10"))
    MAX_CONCURRENT_SLOTS = _to_int(
        os.getenv("MAX_CONCURRENT_SLOTS", "10")
    )  # Maximum concurrent positions

    # ==== PORTFOLIO LIMITS ====
    MAX_EXPOSURE_PER_SYMBOL_PCT = _to_float(
        os.getenv("MAX_EXPOSURE_PER_SYMBOL_PCT", "10.0")
    )
    MAX_TOTAL_EXPOSURE_PCT = _to_float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "30.0"))
    MIN_BALANCE_USDT = _to_float(os.getenv("MIN_BALANCE_USDT", "100.0"))

    # ==== TIER SYSTEM ====
    # Tier configuration
    TIER_MINIMUM = os.getenv(
        "TIER_MINIMUM", "Emergency"
    )  # Akceptuj wszystkie tiery włącznie z Emergency
    TIER_HIERARCHY = ["Emergency", "Platinum", "Premium", "Standard", "Quick"]

    # Tier-specific leverage limits
    TIER_LEVERAGE_LIMITS = {
        "Platinum": _to_int(os.getenv("TIER_PLATINUM_MAX_LEVERAGE", "50")),
        "Premium": _to_int(os.getenv("TIER_PREMIUM_MAX_LEVERAGE", "40")),
        "Standard": _to_int(os.getenv("TIER_STANDARD_MAX_LEVERAGE", "30")),
        "Quick": _to_int(os.getenv("TIER_QUICK_MAX_LEVERAGE", "20")),
        "Emergency": _to_int(os.getenv("TIER_EMERGENCY_MAX_LEVERAGE", "50")),
    }

    # Tier-specific risk multipliers - ZWIĘKSZONE dla wyższej jakości sygnałów
    TIER_RISK_MULTIPLIERS = {
        "Platinum": _to_float(os.getenv("TIER_PLATINUM_RISK_MULT", "2.0")),  # Zwiększone z 1.5
        "Premium": _to_float(os.getenv("TIER_PREMIUM_RISK_MULT", "1.5")),   # Zwiększone z 1.2
        "Standard": _to_float(os.getenv("TIER_STANDARD_RISK_MULT", "1.2")), # Zwiększone z 1.0
        "Quick": _to_float(os.getenv("TIER_QUICK_RISK_MULT", "1.0")),       # Zwiększone z 0.8
        "Emergency": _to_float(os.getenv("TIER_EMERGENCY_RISK_MULT", "1.5")), # Obniżone z 2.0 - traktuj jak Standard
    }

    # Tier-specific leverage multipliers - ZWIĘKSZONA DŹWIGNIA dla lepszych sygnałów
    TIER_LEVERAGE_MULTIPLIERS = {
        "Platinum": _to_float(os.getenv("TIER_PLATINUM_LEVERAGE_MULT", "3")),  # 1.5x bazowej dźwigni
        "Premium": _to_float(os.getenv("TIER_PREMIUM_LEVERAGE_MULT", "2")),   # 1.3x bazowej dźwigni
        "Standard": _to_float(os.getenv("TIER_STANDARD_LEVERAGE_MULT", "1.0")), # Bazowa dźwignia
        "Quick": _to_float(os.getenv("TIER_QUICK_LEVERAGE_MULT", "1.0")),       # Bazowa dźwignia
        "Emergency": _to_float(os.getenv("TIER_EMERGENCY_LEVERAGE_MULT", "1.2")), # 1.2x bazowej dźwigni
    }

    # ==== SYMBOL-SPECIFIC OVERRIDES ====
    # Maximum leverage per symbol (Binance limits)
    SYMBOL_MAX_LEVERAGE = {
        "BTCUSDT": 125,
        "ETHUSDT": 100,
        "BNBUSDT": 75,
        "SOLUSDT": 50,
        "XRPUSDT": 75,
        "DOGEUSDT": 75,
        "ADAUSDT": 75,
        "MATICUSDT": 50,
        "DOTUSDT": 50,
        "AVAXUSDT": 50,
        "LINKUSDT": 50,
        "ATOMUSDT": 50,
        "LTCUSDT": 75,
        "ETCUSDT": 50,
        "NEARUSDT": 50,
        "ARBUSDT": 50,
        "OPUSDT": 50,
        # Add more as needed - these are examples
    }

    # Symbol-specific minimum quantities
    SYMBOL_MIN_QTY = {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.001,
        "BNBUSDT": 0.01,
        "SOLUSDT": 0.1,
        "XRPUSDT": 1.0,
        "DOGEUSDT": 1.0,
        # Add more as needed
    }

    # ==== TRADING PARAMETERS ====
    # Risk management
    DEFAULT_RISK_PERCENT = _to_float(os.getenv("DEFAULT_RISK_PERCENT", "1.0"))
    MAX_RISK_PERCENT = _to_float(os.getenv("MAX_RISK_PERCENT", "2.0"))
    MIN_RISK_PERCENT = _to_float(os.getenv("MIN_RISK_PERCENT", "0.5"))
    RISK_PER_TRADE = _to_float(
        os.getenv("RISK_PER_TRADE", "1.0")
    )  # Default risk per trade

    # Leverage
    DEFAULT_LEVERAGE = _to_int(os.getenv("DEFAULT_LEVERAGE", "10"))
    MAX_LEVERAGE = _to_int(os.getenv("MAX_LEVERAGE", "50"))
    MIN_LEVERAGE = _to_int(os.getenv("MIN_LEVERAGE", "1"))
    DEFAULT_MARGIN_TYPE = os.getenv(
        "DEFAULT_MARGIN_TYPE", "ISOLATED"
    )  # ISOLATED or CROSSED

    # Stop Loss and Take Profit
    DEFAULT_SL_PERCENT = _to_float(os.getenv("DEFAULT_SL_PERCENT", "2.0"))
    DEFAULT_TP1_PERCENT = _to_float(os.getenv("DEFAULT_TP1_PERCENT", "2.0"))
    DEFAULT_TP2_PERCENT = _to_float(os.getenv("DEFAULT_TP2_PERCENT", "4.0"))
    DEFAULT_TP3_PERCENT = _to_float(os.getenv("DEFAULT_TP3_PERCENT", "6.0"))

    # TP split percentages (must sum to 100)
    TP1_SIZE_PERCENT = _to_float(os.getenv("TP1_SIZE_PERCENT", "40"))
    TP2_SIZE_PERCENT = _to_float(os.getenv("TP2_SIZE_PERCENT", "30"))
    TP3_SIZE_PERCENT = _to_float(os.getenv("TP3_SIZE_PERCENT", "30"))

    # Take profit risk-reward levels
    TP_RR_LEVELS = [1.5, 3.0, 5.0]  # Default risk-reward ratios for TP levels
    USE_ALERT_LEVELS = _to_bool(
        os.getenv("USE_ALERT_LEVELS", "True")
    )  # Use alert-based TP levels

    # ==== MODE MANAGER ====
    # Trading modes
    DEFAULT_MODE = os.getenv("DEFAULT_MODE", "normal")
    AVAILABLE_MODES = ["conservative", "normal", "aggressive", "scalping", "emergency"]

    # Mode configurations
    MODE_CONFIGS = {
        "conservative": {
            "risk_multiplier": 0.5,
            "max_leverage": 10,
            "max_positions": 3,
            "tier_minimum": "Premium",
            "require_confirmation": True,
        },
        "normal": {
            "risk_multiplier": 1.0,
            "max_leverage": 20,
            "max_positions": 5,
            "tier_minimum": "Standard",
            "require_confirmation": False,
        },
        "aggressive": {
            "risk_multiplier": 1.5,
            "max_leverage": 30,
            "max_positions": 8,
            "tier_minimum": "Quick",
            "require_confirmation": False,
        },
        "scalping": {
            "risk_multiplier": 0.8,
            "max_leverage": 25,
            "max_positions": 10,
            "tier_minimum": "Quick",
            "require_confirmation": False,
        },
        "emergency": {
            "risk_multiplier": 2.0,
            "max_leverage": 50,
            "max_positions": 1,
            "tier_minimum": "Emergency",
            "require_confirmation": False,
        },
    }

    # ==== SIGNAL PROCESSING ====
    # Signal validation
    REQUIRE_ALL_TP_LEVELS = _to_bool(os.getenv("REQUIRE_ALL_TP_LEVELS", "False"))
    MIN_SIGNAL_STRENGTH = _to_float(os.getenv("MIN_SIGNAL_STRENGTH", "0.3"))
    MAX_SIGNAL_STRENGTH = _to_float(os.getenv("MAX_SIGNAL_STRENGTH", "1.0"))

    # Signal cooldown
    SIGNAL_COOLDOWN_SECONDS = _to_int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "60"))
    ONE_TRADE_PER_BAR = _to_bool(os.getenv("ONE_TRADE_PER_BAR", "True"))

    # ==== MONITORING & ANALYTICS ====
    # Metrics collection
    ENABLE_METRICS = _to_bool(os.getenv("ENABLE_METRICS", "True"))
    METRICS_INTERVAL_SECONDS = _to_int(os.getenv("METRICS_INTERVAL_SECONDS", "300"))

    # Performance tracking
    TRACK_TIER_PERFORMANCE = _to_bool(os.getenv("TRACK_TIER_PERFORMANCE", "True"))
    TRACK_LEVERAGE_PERFORMANCE = _to_bool(
        os.getenv("TRACK_LEVERAGE_PERFORMANCE", "True")
    )

    # Alerts
    ALERT_ON_LOW_BALANCE = _to_bool(os.getenv("ALERT_ON_LOW_BALANCE", "True"))
    ALERT_ON_HIGH_DRAWDOWN = _to_bool(os.getenv("ALERT_ON_HIGH_DRAWDOWN", "True"))
    MAX_DRAWDOWN_PCT = _to_float(os.getenv("MAX_DRAWDOWN_PCT", "20.0"))

    # ==== ADAPTIVE RISK MANAGER ====
    # Simple adaptive risk (light version)
    ENABLE_ADAPTIVE_RISK = _to_bool(os.getenv("ENABLE_ADAPTIVE_RISK", "False"))
    ADAPTIVE_LOOKBACK_TRADES = _to_int(os.getenv("ADAPTIVE_LOOKBACK_TRADES", "10"))
    ADAPTIVE_MIN_MULTIPLIER = _to_float(os.getenv("ADAPTIVE_MIN_MULTIPLIER", "0.7"))
    ADAPTIVE_MAX_MULTIPLIER = _to_float(os.getenv("ADAPTIVE_MAX_MULTIPLIER", "1.3"))
    ADAPTIVE_WIN_RATE_THRESHOLD = _to_float(
        os.getenv("ADAPTIVE_WIN_RATE_THRESHOLD", "0.6")
    )

    # ==== MACHINE LEARNING ====
    # ML-based decision making
    USE_ML_FOR_DECISION = _to_bool(os.getenv("USE_ML_FOR_DECISION", "False"))
    ML_MIN_WIN_PROB = _to_float(
        os.getenv("ML_MIN_WIN_PROB", "0.6")
    )  # Minimum win probability for ML trades

    # ==== BINANCE API SETTINGS ====
    # API endpoints
    BINANCE_BASE_URL = "https://api.binance.com"
    BINANCE_TESTNET_BASE_URL = "https://testnet.binancefuture.com"
    BINANCE_WS_URL = "wss://fstream.binance.com"
    BINANCE_TESTNET_WS_URL = "wss://stream.binancefuture.com"

    # API limits and retries
    API_RETRY_COUNT = _to_int(os.getenv("API_RETRY_COUNT", "3"))
    API_RETRY_DELAY = _to_float(os.getenv("API_RETRY_DELAY", "1.0"))
    API_TIMEOUT = _to_int(os.getenv("API_TIMEOUT", "30"))

    # Rate limiting
    RATE_LIMIT_BUFFER = _to_float(os.getenv("RATE_LIMIT_BUFFER", "0.9"))
    ORDER_RATE_LIMIT = _to_int(os.getenv("ORDER_RATE_LIMIT", "50"))  # per 10 seconds

    # ==== WEBSOCKET SETTINGS ====
    WS_RECONNECT_DELAY = _to_int(os.getenv("WS_RECONNECT_DELAY", "5"))
    WS_MAX_RECONNECT_ATTEMPTS = _to_int(os.getenv("WS_MAX_RECONNECT_ATTEMPTS", "10"))
    WS_PING_INTERVAL = _to_int(os.getenv("WS_PING_INTERVAL", "180"))
    WS_HEARTBEAT_INTERVAL = _to_int(os.getenv("WS_HEARTBEAT_INTERVAL", "30"))

    # ==== LOGGING ====
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    LOG_FILE = os.getenv("LOG_FILE", "bot.log")
    LOG_MAX_SIZE_MB = _to_int(os.getenv("LOG_MAX_SIZE_MB", "100"))
    LOG_BACKUP_COUNT = _to_int(os.getenv("LOG_BACKUP_COUNT", "5"))

    # ==== NOTIFICATIONS ====
    # Discord notification settings
    NOTIFY_ON_ENTRY = _to_bool(os.getenv("NOTIFY_ON_ENTRY", "True"))
    NOTIFY_ON_EXIT = _to_bool(os.getenv("NOTIFY_ON_EXIT", "True"))
    NOTIFY_ON_ERROR = _to_bool(os.getenv("NOTIFY_ON_ERROR", "True"))
    NOTIFY_ON_BE = _to_bool(os.getenv("NOTIFY_ON_BE", "True"))
    NOTIFY_ON_TRAILING = _to_bool(os.getenv("NOTIFY_ON_TRAILING", "False"))

    # Notification grouping
    GROUP_EXIT_NOTIFICATIONS = _to_bool(os.getenv("GROUP_EXIT_NOTIFICATIONS", "True"))
    NOTIFICATION_BATCH_SECONDS = _to_int(os.getenv("NOTIFICATION_BATCH_SECONDS", "5"))

    # ==== CACHE SETTINGS ====
    # Leverage cache
    LEVERAGE_CACHE_TTL = _to_int(os.getenv("LEVERAGE_CACHE_TTL", "300"))  # 5 minutes

    # Price cache
    PRICE_CACHE_TTL = _to_int(os.getenv("PRICE_CACHE_TTL", "2"))  # 2 seconds

    # Balance cache
    BALANCE_CACHE_TTL = _to_int(os.getenv("BALANCE_CACHE_TTL", "10"))  # 10 seconds
        # ==== HYBRID ULTRA-DIAGNOSTICS ====
    # General diagnostic settings
    DIAGNOSTICS_ENABLED = _to_bool(os.getenv("DIAGNOSTICS_ENABLED", "True"))
    DIAGNOSTICS_LOG_LEVEL = os.getenv("DIAGNOSTICS_LOG_LEVEL", "INFO")
    DIAGNOSTICS_DATA_RETENTION_DAYS = _to_int(os.getenv("DIAGNOSTICS_DATA_RETENTION_DAYS", "30"))
    
    # Health monitoring
    HEALTH_CHECK_INTERVAL = _to_int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))  # seconds
    HEALTH_CHECK_TIMEOUT = _to_int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))   # seconds
    HEALTH_METRICS_RETENTION_HOURS = _to_int(os.getenv("HEALTH_METRICS_RETENTION_HOURS", "168"))  # 7 days
    
    # Health thresholds
    CPU_WARNING_THRESHOLD = _to_float(os.getenv("CPU_WARNING_THRESHOLD", "70.0"))
    CPU_CRITICAL_THRESHOLD = _to_float(os.getenv("CPU_CRITICAL_THRESHOLD", "85.0"))
    MEMORY_WARNING_THRESHOLD = _to_float(os.getenv("MEMORY_WARNING_THRESHOLD", "75.0"))
    MEMORY_CRITICAL_THRESHOLD = _to_float(os.getenv("MEMORY_CRITICAL_THRESHOLD", "90.0"))
    DISK_WARNING_THRESHOLD = _to_float(os.getenv("DISK_WARNING_THRESHOLD", "80.0"))
    DISK_CRITICAL_THRESHOLD = _to_float(os.getenv("DISK_CRITICAL_THRESHOLD", "95.0"))
    
    # Network thresholds
    API_LATENCY_WARNING_MS = _to_int(os.getenv("API_LATENCY_WARNING_MS", "1000"))
    API_LATENCY_CRITICAL_MS = _to_int(os.getenv("API_LATENCY_CRITICAL_MS", "3000"))
    API_ERROR_RATE_WARNING = _to_float(os.getenv("API_ERROR_RATE_WARNING", "5.0"))  # %
    API_ERROR_RATE_CRITICAL = _to_float(os.getenv("API_ERROR_RATE_CRITICAL", "15.0"))  # %
    
    # Performance monitoring
    PERFORMANCE_MONITORING_ENABLED = _to_bool(os.getenv("PERFORMANCE_MONITORING_ENABLED", "True"))
    PERFORMANCE_CHECK_INTERVAL = _to_int(os.getenv("PERFORMANCE_CHECK_INTERVAL", "300"))  # 5 minutes
    PERFORMANCE_METRICS_RETENTION_HOURS = _to_int(os.getenv("PERFORMANCE_METRICS_RETENTION_HOURS", "720"))  # 30 days
    
    # Performance thresholds
    TRADE_EXECUTION_WARNING_MS = _to_int(os.getenv("TRADE_EXECUTION_WARNING_MS", "2000"))
    TRADE_EXECUTION_CRITICAL_MS = _to_int(os.getenv("TRADE_EXECUTION_CRITICAL_MS", "5000"))
    ORDER_FILL_RATE_WARNING = _to_float(os.getenv("ORDER_FILL_RATE_WARNING", "95.0"))  # %
    ORDER_FILL_RATE_CRITICAL = _to_float(os.getenv("ORDER_FILL_RATE_CRITICAL", "85.0"))  # %
    
    # Alert management
    ALERT_MANAGEMENT_ENABLED = _to_bool(os.getenv("ALERT_MANAGEMENT_ENABLED", "True"))
    ALERT_RETENTION_DAYS = _to_int(os.getenv("ALERT_RETENTION_DAYS", "90"))
    ALERT_RATE_LIMIT_PER_HOUR = _to_int(os.getenv("ALERT_RATE_LIMIT_PER_HOUR", "100"))
    ALERT_DUPLICATE_WINDOW_MINUTES = _to_int(os.getenv("ALERT_DUPLICATE_WINDOW_MINUTES", "15"))
    
    # Alert severity escalation
    ALERT_ESCALATION_ENABLED = _to_bool(os.getenv("ALERT_ESCALATION_ENABLED", "True"))
    ALERT_ESCALATION_THRESHOLD_MINUTES = _to_int(os.getenv("ALERT_ESCALATION_THRESHOLD_MINUTES", "30"))
    CRITICAL_ALERT_IMMEDIATE_NOTIFY = _to_bool(os.getenv("CRITICAL_ALERT_IMMEDIATE_NOTIFY", "True"))
    
    # Decision engine
    DECISION_ENGINE_ENABLED = _to_bool(os.getenv("DECISION_ENGINE_ENABLED", "True"))
    DECISION_LOGGING_ENABLED = _to_bool(os.getenv("DECISION_LOGGING_ENABLED", "True"))
    DECISION_EXPLANATION_ENABLED = _to_bool(os.getenv("DECISION_EXPLANATION_ENABLED", "True"))
    DECISION_RETENTION_DAYS = _to_int(os.getenv("DECISION_RETENTION_DAYS", "60"))
    
    # ML Explainability (SHAP)
    SHAP_EXPLANATIONS_ENABLED = _to_bool(os.getenv("SHAP_EXPLANATIONS_ENABLED", "True"))
    SHAP_SAMPLE_SIZE = _to_int(os.getenv("SHAP_SAMPLE_SIZE", "100"))
    SHAP_CACHE_EXPLANATIONS = _to_bool(os.getenv("SHAP_CACHE_EXPLANATIONS", "True"))
    SHAP_EXPLANATION_THRESHOLD = _to_float(os.getenv("SHAP_EXPLANATION_THRESHOLD", "0.1"))
    
    # Pattern detection
    PATTERN_DETECTION_ENABLED = _to_bool(os.getenv("PATTERN_DETECTION_ENABLED", "True"))
    PATTERN_ANALYSIS_INTERVAL = _to_int(os.getenv("PATTERN_ANALYSIS_INTERVAL", "3600"))  # 1 hour
    PATTERN_LOOKBACK_DAYS = _to_int(os.getenv("PATTERN_LOOKBACK_DAYS", "30"))
    ANOMALY_DETECTION_SENSITIVITY = _to_float(os.getenv("ANOMALY_DETECTION_SENSITIVITY", "0.05"))
    
    # Pine Script health monitoring
    PINE_HEALTH_MONITORING_ENABLED = _to_bool(os.getenv("PINE_HEALTH_MONITORING_ENABLED", "True"))
    PINE_SIGNAL_TIMEOUT_MINUTES = _to_int(os.getenv("PINE_SIGNAL_TIMEOUT_MINUTES", "10"))
    PINE_HEARTBEAT_INTERVAL_MINUTES = _to_int(os.getenv("PINE_HEARTBEAT_INTERVAL_MINUTES", "5"))
    PINE_ALERT_VALIDATION_ENABLED = _to_bool(os.getenv("PINE_ALERT_VALIDATION_ENABLED", "True"))
    
    # Diagnostic API
    DIAGNOSTIC_API_ENABLED = _to_bool(os.getenv("DIAGNOSTIC_API_ENABLED", "True"))
    DIAGNOSTIC_API_PORT = _to_int(os.getenv("DIAGNOSTIC_API_PORT", "8001"))
    DIAGNOSTIC_API_HOST = os.getenv("DIAGNOSTIC_API_HOST", "0.0.0.0")
    DIAGNOSTIC_API_TOKEN = os.getenv("DIAGNOSTIC_API_TOKEN", "your-api-token")
    
    # Diagnostic Dashboard
    DIAGNOSTIC_DASHBOARD_ENABLED = _to_bool(os.getenv("DIAGNOSTIC_DASHBOARD_ENABLED", "True"))
    DIAGNOSTIC_DASHBOARD_PORT = _to_int(os.getenv("DIAGNOSTIC_DASHBOARD_PORT", "8002"))
    DIAGNOSTIC_DASHBOARD_HOST = os.getenv("DIAGNOSTIC_DASHBOARD_HOST", "0.0.0.0")
    DASHBOARD_UPDATE_INTERVAL = _to_int(os.getenv("DASHBOARD_UPDATE_INTERVAL", "10"))  # seconds
    
    # Execution tracing
    EXECUTION_TRACING_ENABLED = _to_bool(os.getenv("EXECUTION_TRACING_ENABLED", "True"))
    TRACE_ALL_OPERATIONS = _to_bool(os.getenv("TRACE_ALL_OPERATIONS", "False"))
    TRACE_RETENTION_DAYS = _to_int(os.getenv("TRACE_RETENTION_DAYS", "7"))
    TRACE_PERFORMANCE_THRESHOLD_MS = _to_int(os.getenv("TRACE_PERFORMANCE_THRESHOLD_MS", "1000"))
    
    # System resource monitoring
    SYSTEM_MONITORING_ENABLED = _to_bool(os.getenv("SYSTEM_MONITORING_ENABLED", "True"))
    SYSTEM_METRICS_INTERVAL = _to_int(os.getenv("SYSTEM_METRICS_INTERVAL", "30"))  # seconds
    SYSTEM_METRICS_RETENTION_HOURS = _to_int(os.getenv("SYSTEM_METRICS_RETENTION_HOURS", "168"))  # 7 days
    
    # Database monitoring
    DB_MONITORING_ENABLED = _to_bool(os.getenv("DB_MONITORING_ENABLED", "True"))
    DB_SLOW_QUERY_THRESHOLD_MS = _to_int(os.getenv("DB_SLOW_QUERY_THRESHOLD_MS", "1000"))
    DB_CONNECTION_POOL_WARNING = _to_int(os.getenv("DB_CONNECTION_POOL_WARNING", "8"))
    DB_CONNECTION_POOL_CRITICAL = _to_int(os.getenv("DB_CONNECTION_POOL_CRITICAL", "9"))
    
    # Redis monitoring (if used)
    REDIS_MONITORING_ENABLED = _to_bool(os.getenv("REDIS_MONITORING_ENABLED", "False"))
    REDIS_MEMORY_WARNING_MB = _to_int(os.getenv("REDIS_MEMORY_WARNING_MB", "100"))
    REDIS_MEMORY_CRITICAL_MB = _to_int(os.getenv("REDIS_MEMORY_CRITICAL_MB", "200"))
    
    # Notification settings for diagnostics
    DIAGNOSTIC_DISCORD_WEBHOOK = os.getenv("DIAGNOSTIC_DISCORD_WEBHOOK", "").strip()
    DIAGNOSTIC_NOTIFICATIONS_ENABLED = _to_bool(os.getenv("DIAGNOSTIC_NOTIFICATIONS_ENABLED", "True"))
    DIAGNOSTIC_NOTIFICATION_COOLDOWN = _to_int(os.getenv("DIAGNOSTIC_NOTIFICATION_COOLDOWN", "300"))  # 5 minutes
    
    # Auto-recovery settings
    AUTO_RECOVERY_ENABLED = _to_bool(os.getenv("AUTO_RECOVERY_ENABLED", "False"))
    AUTO_RESTART_ON_CRITICAL = _to_bool(os.getenv("AUTO_RESTART_ON_CRITICAL", "False"))
    MAX_AUTO_RECOVERY_ATTEMPTS = _to_int(os.getenv("MAX_AUTO_RECOVERY_ATTEMPTS", "3"))
    AUTO_RECOVERY_COOLDOWN_MINUTES = _to_int(os.getenv("AUTO_RECOVERY_COOLDOWN_MINUTES", "30"))

    # ==== HELPER METHODS ====
    @classmethod
    def get_api_key(cls) -> str:
        """Get appropriate API key based on environment"""
        if cls.IS_TESTNET:
            return cls.BINANCE_TESTNET_API_KEY
        else:
            return cls.BINANCE_API_KEY

    @classmethod
    def get_api_secret(cls) -> str:
        """Get appropriate API secret based on environment"""
        if cls.IS_TESTNET:
            return cls.BINANCE_TESTNET_API_SECRET
        else:
            return cls.BINANCE_API_SECRET

    @classmethod
    def get_base_url(cls) -> str:
        """Get appropriate base URL based on environment"""
        if cls.IS_TESTNET:
            return cls.BINANCE_TESTNET_BASE_URL
        else:
            return cls.BINANCE_BASE_URL

    @classmethod
    def get_ws_url(cls) -> str:
        """Get appropriate WebSocket URL based on environment"""
        if cls.IS_TESTNET:
            return cls.BINANCE_TESTNET_WS_URL
        else:
            return cls.BINANCE_WS_URL

    @classmethod
    def get_max_leverage_for_symbol(cls, symbol: str) -> int:
        """Get maximum allowed leverage for a symbol"""
        # First check symbol-specific limits
        if symbol in cls.SYMBOL_MAX_LEVERAGE:
            symbol_max = cls.SYMBOL_MAX_LEVERAGE[symbol]
        else:
            symbol_max = cls.MAX_LEVERAGE

        # Return minimum of symbol limit and global limit
        return min(symbol_max, cls.MAX_LEVERAGE)

    @classmethod
    def get_tier_config(cls, tier: str) -> dict:
        """Get configuration for a specific tier"""
        return {
            "max_leverage": cls.TIER_LEVERAGE_LIMITS.get(tier, cls.DEFAULT_LEVERAGE),
            "risk_multiplier": cls.TIER_RISK_MULTIPLIERS.get(tier, 1.0),
        }

    @classmethod
    def validate_tier(cls, tier: str) -> bool:
        """Check if tier meets minimum requirements"""
        if tier not in cls.TIER_HIERARCHY:
            return False

        tier_index = cls.TIER_HIERARCHY.index(tier)
        min_tier_index = cls.TIER_HIERARCHY.index(cls.TIER_MINIMUM)

        # Lower index = higher tier (Emergency=0, Quick=4)
        return tier_index <= min_tier_index

    @classmethod
    def get_mode_config(cls, mode: str = None) -> dict:
        """Get configuration for a specific mode"""
        if mode is None:
            mode = cls.DEFAULT_MODE
        return cls.MODE_CONFIGS.get(mode, cls.MODE_CONFIGS["normal"])

    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        errors = []

        # Check API credentials
        if not cls.get_api_key() or not cls.get_api_secret():
            errors.append("Missing Binance API credentials")

        # Check Discord token
        if not cls.DISCORD_BOT_TOKEN:
            errors.append("Missing Discord token")

        # Check TP percentages sum to 100
        tp_sum = cls.TP1_SIZE_PERCENT + cls.TP2_SIZE_PERCENT + cls.TP3_SIZE_PERCENT
        if abs(tp_sum - 100) > 0.01:
            errors.append(f"TP percentages sum to {tp_sum}, should be 100")

        # Check tier minimum is valid
        if cls.TIER_MINIMUM not in cls.TIER_HIERARCHY:
            errors.append(f"Invalid TIER_MINIMUM: {cls.TIER_MINIMUM}")

        # Check default mode is valid
        if cls.DEFAULT_MODE not in cls.AVAILABLE_MODES:
            errors.append(f"Invalid DEFAULT_MODE: {cls.DEFAULT_MODE}")

        # Log errors if any
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            return False

        return True

    @classmethod
    def get_position_size_limits(cls, symbol: str) -> dict:
        """Get position size limits for a symbol"""
        return {
            "min_qty": cls.SYMBOL_MIN_QTY.get(symbol, 0.001),
            "max_exposure_pct": cls.MAX_EXPOSURE_PER_SYMBOL_PCT,
            "max_total_exposure_pct": cls.MAX_TOTAL_EXPOSURE_PCT,
        }

    @classmethod
    def should_use_indicator_leverage(cls) -> bool:
        """Check if indicator leverage should be used"""
        return cls.USE_INDICATOR_LEVERAGE

    @classmethod
    def should_use_size_multiplier(cls) -> bool:
        """Check if indicator size multiplier should be used"""
        return cls.USE_INDICATOR_SIZE_MULTIPLIER

    @classmethod
    def get_webhook_url(cls) -> str:
        """Get full webhook URL for TradingView"""
        if cls.WEBHOOK_SECRET:
            return f"/webhook/{cls.WEBHOOK_SECRET}"
        return "/webhook"

    @classmethod
    def get_diagnostic_config(cls) -> dict:
        """Get diagnostic system configuration"""
        return {
            "enabled": cls.DIAGNOSTICS_ENABLED,
            "health_check_interval": cls.HEALTH_CHECK_INTERVAL,
            "performance_check_interval": cls.PERFORMANCE_CHECK_INTERVAL,
            "alert_retention_days": cls.ALERT_RETENTION_DAYS,
            "decision_logging": cls.DECISION_LOGGING_ENABLED,
            "pattern_detection": cls.PATTERN_DETECTION_ENABLED,
            "api_enabled": cls.DIAGNOSTIC_API_ENABLED,
            "dashboard_enabled": cls.DIAGNOSTIC_DASHBOARD_ENABLED
        }
    
    @classmethod
    def get_health_thresholds(cls) -> dict:
        """Get health monitoring thresholds"""
        return {
            "cpu": {
                "warning": cls.CPU_WARNING_THRESHOLD,
                "critical": cls.CPU_CRITICAL_THRESHOLD
            },
            "memory": {
                "warning": cls.MEMORY_WARNING_THRESHOLD,
                "critical": cls.MEMORY_CRITICAL_THRESHOLD
            },
            "disk": {
                "warning": cls.DISK_WARNING_THRESHOLD,
                "critical": cls.DISK_CRITICAL_THRESHOLD
            },
            "api_latency": {
                "warning": cls.API_LATENCY_WARNING_MS,
                "critical": cls.API_LATENCY_CRITICAL_MS
            },
            "api_error_rate": {
                "warning": cls.API_ERROR_RATE_WARNING,
                "critical": cls.API_ERROR_RATE_CRITICAL
            }
        }
    
    @classmethod
    def get_performance_thresholds(cls) -> dict:
        """Get performance monitoring thresholds"""
        return {
            "trade_execution": {
                "warning": cls.TRADE_EXECUTION_WARNING_MS,
                "critical": cls.TRADE_EXECUTION_CRITICAL_MS
            },
            "order_fill_rate": {
                "warning": cls.ORDER_FILL_RATE_WARNING,
                "critical": cls.ORDER_FILL_RATE_CRITICAL
            },
            "trace_threshold": cls.TRACE_PERFORMANCE_THRESHOLD_MS
        }
    
    @classmethod
    def get_alert_config(cls) -> dict:
        """Get alert management configuration"""
        return {
            "enabled": cls.ALERT_MANAGEMENT_ENABLED,
            "retention_days": cls.ALERT_RETENTION_DAYS,
            "rate_limit_per_hour": cls.ALERT_RATE_LIMIT_PER_HOUR,
            "duplicate_window_minutes": cls.ALERT_DUPLICATE_WINDOW_MINUTES,
            "escalation_enabled": cls.ALERT_ESCALATION_ENABLED,
            "escalation_threshold_minutes": cls.ALERT_ESCALATION_THRESHOLD_MINUTES,
            "critical_immediate_notify": cls.CRITICAL_ALERT_IMMEDIATE_NOTIFY
        }
    
    @classmethod
    def get_pattern_detection_config(cls) -> dict:
        """Get pattern detection configuration"""
        return {
            "enabled": cls.PATTERN_DETECTION_ENABLED,
            "analysis_interval": cls.PATTERN_ANALYSIS_INTERVAL,
            "lookback_days": cls.PATTERN_LOOKBACK_DAYS,
            "anomaly_sensitivity": cls.ANOMALY_DETECTION_SENSITIVITY
        }
    
    @classmethod
    def get_shap_config(cls) -> dict:
        """Get SHAP explainability configuration"""
        return {
            "enabled": cls.SHAP_EXPLANATIONS_ENABLED,
            "sample_size": cls.SHAP_SAMPLE_SIZE,
            "cache_explanations": cls.SHAP_CACHE_EXPLANATIONS,
            "explanation_threshold": cls.SHAP_EXPLANATION_THRESHOLD
        }
    
    @classmethod
    def get_pine_monitoring_config(cls) -> dict:
        """Get Pine Script monitoring configuration"""
        return {
            "enabled": cls.PINE_HEALTH_MONITORING_ENABLED,
            "signal_timeout_minutes": cls.PINE_SIGNAL_TIMEOUT_MINUTES,
            "heartbeat_interval_minutes": cls.PINE_HEARTBEAT_INTERVAL_MINUTES,
            "alert_validation_enabled": cls.PINE_ALERT_VALIDATION_ENABLED
        }
    
    @classmethod
    def should_trace_operation(cls, operation_type: str = None) -> bool:
        """Check if operation should be traced"""
        if not cls.EXECUTION_TRACING_ENABLED:
            return False
        
        if cls.TRACE_ALL_OPERATIONS:
            return True
        
        # Trace only important operations by default
        important_operations = ["trade_entry", "trade_exit", "order_placement", "leverage_change"]
        return operation_type in important_operations if operation_type else True
    
    @classmethod
    def get_diagnostic_ports(cls) -> dict:
        """Get diagnostic service ports"""
        return {
            "api_port": cls.DIAGNOSTIC_API_PORT,
            "dashboard_port": cls.DIAGNOSTIC_DASHBOARD_PORT,
            "api_host": cls.DIAGNOSTIC_API_HOST,
            "dashboard_host": cls.DIAGNOSTIC_DASHBOARD_HOST
        }

    @classmethod
    def to_dict(cls) -> dict:
        """Export configuration as dictionary (for debugging/status)"""
        base_config = {
            "bot_version": cls.BOT_VERSION,
            "environment": cls.ENVIRONMENT,
            "is_testnet": cls.IS_TESTNET,
            "dry_run": cls.DRY_RUN,
            "mode": cls.DEFAULT_MODE,
            "tier_minimum": cls.TIER_MINIMUM,
            "emergency_enabled": cls.EMERGENCY_ENABLED,
            "use_indicator_leverage": cls.USE_INDICATOR_LEVERAGE,
            "use_size_multiplier": cls.USE_INDICATOR_SIZE_MULTIPLIER,
            "move_sl_to_be": cls.MOVE_SL_TO_BE_AT_TP1,
            "max_positions": cls.MAX_POSITIONS,
            "default_risk": cls.DEFAULT_RISK_PERCENT,
            "default_leverage": cls.DEFAULT_LEVERAGE,
            "position_guards": {
                "max_age": cls.ALERT_MAX_AGE_SEC,
                "max_drift": cls.MAX_PRICE_DRIFT_PCT,
            },
        }
        
        # Add diagnostic configuration
        if cls.DIAGNOSTICS_ENABLED:
            base_config["diagnostics"] = cls.get_diagnostic_config()
            base_config["health_thresholds"] = cls.get_health_thresholds()
            base_config["performance_thresholds"] = cls.get_performance_thresholds()
            base_config["alert_config"] = cls.get_alert_config()
            base_config["pattern_detection"] = cls.get_pattern_detection_config()
            base_config["diagnostic_ports"] = cls.get_diagnostic_ports()
        
        return base_config


# Validate configuration on import
if __name__ != "__main__":
    if not Config.validate_config():
        logger.warning("Configuration validation failed - check logs for details")
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
    INDICATOR_VERSION_REQUIRED = "9.1"
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    IS_TESTNET = _to_bool(os.getenv("IS_TESTNET", "False"))
    DRY_RUN = _to_bool(os.getenv("DRY_RUN", "False"))
    DEBUG = _to_bool(os.getenv("DEBUG", "False"))
    
    # ==== API CREDENTIALS ====
    # Binance API
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()
    BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "").strip()
    BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "").strip()
    
    # Discord
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
    DISCORD_CHANNEL_ID = _to_int(os.getenv("DISCORD_CHANNEL_ID", "0"))
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
    WEBHOOK_RATE_LIMIT_PER_MINUTE = _to_int(os.getenv("WEBHOOK_RATE_LIMIT_PER_MINUTE", "100"))
    WEBHOOK_RATE_LIMIT_PER_SECOND = _to_int(os.getenv("WEBHOOK_RATE_LIMIT_PER_SECOND", "10"))
    
    # ==== V9.1 FEATURE FLAGS ====
    # Leverage management
    USE_INDICATOR_LEVERAGE = _to_bool(os.getenv("USE_INDICATOR_LEVERAGE", "False"))
    USE_INDICATOR_SIZE_MULTIPLIER = _to_bool(os.getenv("USE_INDICATOR_SIZE_MULTIPLIER", "True"))
    
    # Position management
    MOVE_SL_TO_BE_AT_TP1 = _to_bool(os.getenv("MOVE_SL_TO_BE_AT_TP1", "True"))
    USE_TRAILING_AFTER_TP2 = _to_bool(os.getenv("USE_TRAILING_AFTER_TP2", "True"))
    TRAILING_STOP_PERCENTAGE = _to_float(os.getenv("TRAILING_STOP_PERCENTAGE", "0.5"))
    
    # Emergency mode
    EMERGENCY_ENABLED = _to_bool(os.getenv("EMERGENCY_ENABLED", "True"))
    EMERGENCY_CLOSE_AND_REVERSE = _to_bool(os.getenv("EMERGENCY_CLOSE_AND_REVERSE", "False"))
    
    # ==== POSITION GUARDS ====
    # Alert validation
    ALERT_MAX_AGE_SEC = _to_int(os.getenv("ALERT_MAX_AGE_SEC", "30"))
    MAX_PRICE_DRIFT_PCT = _to_float(os.getenv("MAX_PRICE_DRIFT_PCT", "0.6"))
    
    # Position limits
    SINGLE_POSITION_PER_SYMBOL = _to_bool(os.getenv("SINGLE_POSITION_PER_SYMBOL", "True"))
    MAX_POSITIONS = _to_int(os.getenv("MAX_POSITIONS", "10"))
    MAX_CONCURRENT_SLOTS = _to_int(os.getenv("MAX_CONCURRENT_SLOTS", "10"))  # Maximum concurrent positions
    
    # ==== PORTFOLIO LIMITS ====
    MAX_EXPOSURE_PER_SYMBOL_PCT = _to_float(os.getenv("MAX_EXPOSURE_PER_SYMBOL_PCT", "10.0"))
    MAX_TOTAL_EXPOSURE_PCT = _to_float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "30.0"))
    MIN_BALANCE_USDT = _to_float(os.getenv("MIN_BALANCE_USDT", "100.0"))
    
    # ==== TIER SYSTEM ====
    # Tier configuration
    TIER_MINIMUM = os.getenv("TIER_MINIMUM", "Quick")  # Platinum/Premium/Standard/Quick/Emergency
    TIER_HIERARCHY = ["Emergency", "Platinum", "Premium", "Standard", "Quick"]
    
    # Tier-specific leverage limits
    TIER_LEVERAGE_LIMITS = {
    "Platinum": _to_int(os.getenv("TIER_PLATINUM_MAX_LEVERAGE", "50")),
    "Premium": _to_int(os.getenv("TIER_PREMIUM_MAX_LEVERAGE", "40")),
    "Standard": _to_int(os.getenv("TIER_STANDARD_MAX_LEVERAGE", "30")),
    "Quick": _to_int(os.getenv("TIER_QUICK_MAX_LEVERAGE", "20")),
    "Emergency": _to_int(os.getenv("TIER_EMERGENCY_MAX_LEVERAGE", "50"))
    }
    
    # Tier-specific risk multipliers
    TIER_RISK_MULTIPLIERS = {
    "Platinum": _to_float(os.getenv("TIER_PLATINUM_RISK_MULT", "1.5")),
    "Premium": _to_float(os.getenv("TIER_PREMIUM_RISK_MULT", "1.2")),
    "Standard": _to_float(os.getenv("TIER_STANDARD_RISK_MULT", "1.0")),
    "Quick": _to_float(os.getenv("TIER_QUICK_RISK_MULT", "0.8")),
    "Emergency": _to_float(os.getenv("TIER_EMERGENCY_RISK_MULT", "2.0"))
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
    RISK_PER_TRADE = _to_float(os.getenv("RISK_PER_TRADE", "1.0"))  # Default risk per trade
    
    # Leverage
    DEFAULT_LEVERAGE = _to_int(os.getenv("DEFAULT_LEVERAGE", "10"))
    MAX_LEVERAGE = _to_int(os.getenv("MAX_LEVERAGE", "50"))
    MIN_LEVERAGE = _to_int(os.getenv("MIN_LEVERAGE", "1"))
    DEFAULT_MARGIN_TYPE = os.getenv("DEFAULT_MARGIN_TYPE", "ISOLATED")  # ISOLATED or CROSSED
    
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
    USE_ALERT_LEVELS = _to_bool(os.getenv("USE_ALERT_LEVELS", "True"))  # Use alert-based TP levels
    
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
    "require_confirmation": True
    },
    "normal": {
    "risk_multiplier": 1.0,
    "max_leverage": 20,
    "max_positions": 5,
    "tier_minimum": "Standard",
    "require_confirmation": False
    },
    "aggressive": {
    "risk_multiplier": 1.5,
    "max_leverage": 30,
    "max_positions": 8,
    "tier_minimum": "Quick",
    "require_confirmation": False
    },
    "scalping": {
    "risk_multiplier": 0.8,
    "max_leverage": 25,
    "max_positions": 10,
    "tier_minimum": "Quick",
    "require_confirmation": False
    },
    "emergency": {
    "risk_multiplier": 2.0,
    "max_leverage": 50,
    "max_positions": 1,
    "tier_minimum": "Emergency",
    "require_confirmation": False
    }
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
    TRACK_LEVERAGE_PERFORMANCE = _to_bool(os.getenv("TRACK_LEVERAGE_PERFORMANCE", "True"))
    
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
    ADAPTIVE_WIN_RATE_THRESHOLD = _to_float(os.getenv("ADAPTIVE_WIN_RATE_THRESHOLD", "0.6"))
    
    # ==== MACHINE LEARNING ====
    # ML-based decision making
    USE_ML_FOR_DECISION = _to_bool(os.getenv("USE_ML_FOR_DECISION", "False"))
    ML_MIN_WIN_PROB = _to_float(os.getenv("ML_MIN_WIN_PROB", "0.6"))  # Minimum win probability for ML trades
    
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
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    
    # ==== HELPER METHODS ====
    @classmethod
    def get_api_key(cls) -> str:
    """Get appropriate API key based on environment"""
    if cls.IS_TESTNET:
    return cls.BINANCE_TESTNET_API_KEY
    return cls.BINANCE_API_KEY
    
    @classmethod
    def get_api_secret(cls) -> str:
    """Get appropriate API secret based on environment"""
    if cls.IS_TESTNET:
    return cls.BINANCE_TESTNET_API_SECRET
    return cls.BINANCE_API_SECRET
    
    @classmethod
    def get_base_url(cls) -> str:
    """Get appropriate base URL based on environment"""
    if cls.IS_TESTNET:
    return cls.BINANCE_TESTNET_BASE_URL
    return cls.BINANCE_BASE_URL
    
    @classmethod
    def get_ws_url(cls) -> str:
    """Get appropriate WebSocket URL based on environment"""
    if cls.IS_TESTNET:
    return cls.BINANCE_TESTNET_WS_URL
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
    "risk_multiplier": cls.TIER_RISK_MULTIPLIERS.get(tier, 1.0)
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
    if not cls.DISCORD_TOKEN:
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
    "max_total_exposure_pct": cls.MAX_TOTAL_EXPOSURE_PCT
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
    def to_dict(cls) -> dict:
    """Export configuration as dictionary (for debugging/status)"""
    return {
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
    "max_drift": cls.MAX_PRICE_DRIFT_PCT
    }
    }


# Validate configuration on import
if __name__ != "__main__":
    if not Config.validate_config():
    logger.warning("Configuration validation failed - check logs for details")
    # Nowe flagi dla v9.1 i usprawnień
USE_INDICATOR_LEVERAGE = _to_bool(os.getenv("USE_INDICATOR_LEVERAGE", "False"), default=False)
USE_INDICATOR_SIZE_MULTIPLIER = _to_bool(os.getenv("USE_INDICATOR_SIZE_MULTIPLIER", "True"), default=True)
MOVE_SL_TO_BE_AT_TP1 = _to_bool(os.getenv("MOVE_SL_TO_BE_AT_TP1", "True"), default=True)
EMERGENCY_ENABLED = _to_bool(os.getenv("EMERGENCY_ENABLED", "True"), default=True)
EMERGENCY_CLOSE_AND_REVERSE = _to_bool(os.getenv("EMERGENCY_CLOSE_AND_REVERSE", "False"), default=False)

# Position Guards
ALERT_MAX_AGE_SEC = int(os.getenv("ALERT_MAX_AGE_SEC", "30"))
MAX_PRICE_DRIFT_PCT = float(os.getenv("MAX_PRICE_DRIFT_PCT", "0.6"))

# Portfolio limits
MAX_EXPOSURE_PER_SYMBOL_PCT = float(os.getenv("MAX_EXPOSURE_PER_SYMBOL_PCT", "10.0"))
MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "30.0"))

# Tier system
TIER_MINIMUM = os.getenv("TIER_MINIMUM", "Quick")  # Platinum/Premium/Standard/Quick/Emergency
TIER_HIERARCHY = ["Emergency", "Platinum", "Premium", "Standard", "Quick"]

# Webhook security
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
REQUIRE_HMAC_SIGNATURE = _to_bool(os.getenv("REQUIRE_HMAC_SIGNATURE", "False"), default=False)
"""
Database models and operations for Binance Trading Bot
Version: 9.1
"""

import json
import logging
from sqlalchemy import text
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
    func,
    Index,
    UniqueConstraint,
    and_,
    or_,
    inspect,
)
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

from config import Config

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

# Create engine with connection pooling
engine = create_engine(
    Config.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=Config.DB_POOL_SIZE,
    max_overflow=Config.DB_MAX_OVERFLOW,
    pool_timeout=Config.DB_POOL_TIMEOUT,
    pool_pre_ping=True,  # Verify connections before using
    echo=False,  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Session = scoped_session(SessionLocal)


class Trade(Base):
    """Main trade model with v9.1 enhancements"""

    __tablename__ = "trades"

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY/SELL
    status = Column(String(20), nullable=False, index=True)  # open/closed/cancelled
    # Dry run flag for ML training
    is_dry_run = Column(Boolean, default=False, index=True)
    # Idempotency and tracking - v9.1 CORE FEATURE
    idempotency_key = Column(String(64), unique=True, index=True)  # Prevents duplicates
    client_tags = Column(JSON)  # {"sl": "BOT_123_SL", "tp1": "BOT_123_TP1", etc}
    exchange_order_id = Column(String(50))

    # Entry information
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    entry_quantity = Column(Float, nullable=False)
    entry_commission = Column(Float, default=0.0)

    # Exit information
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    exit_quantity = Column(Float)
    exit_reason = Column(String(100))  # sl/tp1/tp2/tp3/manual/emergency/sync
    exit_commission = Column(Float, default=0.0)

    # Risk management levels - ENHANCED FOR v9.1
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)  # v9.1 Multi-TP
    take_profit_3 = Column(Float)  # v9.1 Multi-TP
    break_even_price = Column(Float)  # v9.1 BE tracking
    trailing_stop_price = Column(Float)  # v9.1 Trailing

    # Position management
    leverage_used = Column(Integer, default=1)
    leverage_hint = Column(Integer)  # v9.1 Suggested leverage from indicator
    margin_type = Column(String(20), default="ISOLATED")  # ISOLATED/CROSSED
    position_size_usdt = Column(Float)
    risk_amount_usdt = Column(Float)

    # Signal information - v9.1 ENHANCED
    signal_tier = Column(
        String(20), index=True
    )  # Platinum/Premium/Standard/Quick/Emergency
    signal_strength = Column(Float)
    signal_timeframe = Column(String(10))  # 1m/5m/15m/1h/4h/1d
    signal_session = Column(String(20))  # London/NewYork/Tokyo/Sydney
    indicator_version = Column(String(20))  # 8.0/9.1

    # v9.1 Enhanced fields - NEW INTELLIGENCE FEATURES
    institutional_flow = Column(Float, default=0.0)
    retest_confidence = Column(Float, default=0.0)
    fake_breakout_detected = Column(Boolean, default=False)
    fake_breakout_penalty = Column(Float, default=1.0)
    enhanced_regime = Column(String(50), default="NEUTRAL")
    regime_confidence = Column(Float, default=0.0)
    mtf_agreement_ratio = Column(Float, default=0.0)
    volume_context = Column(JSON)  # Store full volume context
    bar_close_time = Column(DateTime)

    # Performance metrics
    pnl_usdt = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    roe_percentage = Column(Float, default=0.0)  # Return on equity (with leverage)
    max_drawdown = Column(Float, default=0.0)
    time_in_position = Column(Integer)  # seconds

    # Partial fills tracking - v9.1 PRECISION
    filled_quantity = Column(Float, default=0.0)
    tp1_filled = Column(Boolean, default=False)
    tp2_filled = Column(Boolean, default=False)
    tp3_filled = Column(Boolean, default=False)
    sl_moved_to_be = Column(Boolean, default=False)
    trailing_activated = Column(Boolean, default=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
    alert_data = Column(JSON)  # Store full alert for debugging
    raw_signal_data = Column(JSON)  # Store raw signal data for ML training
    mode_used = Column(String(20))  # conservative/normal/aggressive/scalping/emergency

    # Indexes for performance - v9.1 OPTIMIZED
    __table_args__ = (
        Index("idx_trade_status_symbol", "status", "symbol"),
        Index("idx_trade_tier", "signal_tier"),
        Index("idx_trade_created", "created_at"),
        Index("idx_trade_idempotency", "idempotency_key"),
        Index("idx_trade_status_tier", "status", "signal_tier"),
        Index("idx_trade_indicator_version", "indicator_version"),
        Index("idx_trade_timeframe", "signal_timeframe"),
    )

    def to_dict(self) -> dict:
        """Convert trade to dictionary"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "pnl_usdt": self.pnl_usdt,
            "pnl_percentage": self.pnl_percentage,
            "roe_percentage": self.roe_percentage,
            "leverage_used": self.leverage_used,
            "leverage_hint": self.leverage_hint,
            "signal_tier": self.signal_tier,
            "signal_strength": self.signal_strength,
            "indicator_version": self.indicator_version,
            "sl_moved_to_be": self.sl_moved_to_be,
            "trailing_activated": self.trailing_activated,
            "institutional_flow": self.institutional_flow,
            "fake_breakout_detected": self.fake_breakout_detected,
            "enhanced_regime": self.enhanced_regime,
            "tp1_filled": self.tp1_filled,
            "tp2_filled": self.tp2_filled,
            "tp3_filled": self.tp3_filled,
            "mode_used": self.mode_used,
        }


class Signal(Base):
    """Signal history and analytics - v9.1 ENHANCED"""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Signal identification
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(20), nullable=False)  # buy/sell/emergency_close
    tier = Column(String(20), nullable=False, index=True)
    strength = Column(Float, nullable=False)

    # Signal decision
    accepted = Column(Boolean, nullable=False)
    rejection_reason = Column(String(100))

    # Signal metadata
    alert_price = Column(Float)
    market_price = Column(Float)
    price_drift_pct = Column(Float)
    alert_age_seconds = Column(Integer)

    # v9.1 fields - NEW INTELLIGENCE
    institutional_flow = Column(Float)
    fake_breakout_detected = Column(Boolean, default=False)
    regime = Column(String(50))
    retest_confidence = Column(Float)
    mtf_agreement_ratio = Column(Float)

    # Processing metrics
    processing_time_ms = Column(Integer)
    trade_id = Column(Integer)  # Link to trade if opened

    # Raw data
    raw_alert = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_signal_timestamp", "timestamp"),
        Index("idx_signal_symbol_tier", "symbol", "tier"),
        Index("idx_signal_accepted", "accepted"),
    )


class Position(Base):
    """Position tracking model - v9.1 ENHANCED"""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # LONG/SHORT
    status = Column(String(20), nullable=False, index=True)  # open/closed

    # Position details
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    quantity = Column(Float, nullable=False)
    leverage = Column(Integer, default=1)
    margin_type = Column(String(20), default="ISOLATED")

    # Risk management
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    take_profit_3 = Column(Float)
    break_even_price = Column(Float)
    trailing_stop_price = Column(Float)

    # PnL tracking
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    roe_percentage = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Link to trade
    trade_id = Column(Integer)

    def to_dict(self) -> dict:
        """Convert position to dictionary"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "unrealized_pnl": self.unrealized_pnl,
            "roe_percentage": self.roe_percentage,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BotSettings(Base):
    """Dynamic bot settings storage"""

    __tablename__ = "bot_settings"

    id = Column(Integer, primary_key=True)
    key = Column(String(50), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    value_type = Column(String(20), default="string")  # string/int/float/bool/json
    description = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50))  # Discord user who updated

    def get_typed_value(self):
        """Return value with correct type"""
        if self.value_type == "int":
            return int(self.value)
        elif self.value_type == "float":
            return float(self.value)
        elif self.value_type == "bool":
            return self.value.lower() in ("true", "1", "yes")
        elif self.value_type == "json":
            return json.loads(self.value)
        return self.value


class Performance(Base):
    """Performance metrics tracking - v9.1 ENHANCED"""

    __tablename__ = "performance"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # daily/tier/leverage/session
    metric_key = Column(String(50))  # e.g., "Platinum", "20x", "London"

    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)

    # PnL metrics
    gross_pnl = Column(Float, default=0.0)
    net_pnl = Column(Float, default=0.0)
    commission_paid = Column(Float, default=0.0)

    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)

    # Average metrics
    avg_win = Column(Float)
    avg_loss = Column(Float)
    avg_roe = Column(Float)
    avg_hold_time = Column(Integer)  # seconds

    # v9.1 specific metrics - NEW TRACKING
    be_triggered_count = Column(Integer, default=0)
    trailing_triggered_count = Column(Integer, default=0)
    emergency_trades = Column(Integer, default=0)
    fake_breakout_count = Column(Integer, default=0)
    institutional_flow_avg = Column(Float, default=0.0)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_performance_date_type", "date", "metric_type"),
        UniqueConstraint(
            "date", "metric_type", "metric_key", name="uq_performance_metric"
        ),
    )


class AlertHistory(Base):
    """Track all received alerts for debugging - v9.1 SECURITY"""

    __tablename__ = "alert_history"

    id = Column(Integer, primary_key=True)
    received_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Alert identification
    idempotency_key = Column(String(64), unique=True, index=True)
    symbol = Column(String(20), index=True)
    action = Column(String(20))
    tier = Column(String(20))

    # Processing result
    processed = Column(Boolean, default=False)
    duplicate = Column(Boolean, default=False)
    error = Column(Text)

    # Raw data
    raw_payload = Column(JSON)
    headers = Column(JSON)
    # v9.1 NEW: Latency tracking
    tv_ts = Column(Integer)  # TradingView timestamp (ms)
    api_latency_ms = Column(Integer)  # API processing latency
    end_to_end_latency_ms = Column(Integer)  # Total latency

    # Validation results - v9.1 SECURITY
    signature_valid = Column(Boolean)
    schema_valid = Column(Boolean)
    age_valid = Column(Boolean)
    price_drift_valid = Column(Boolean)

    success = Column(Boolean, default=False)
    processed_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_alert_received", "received_at"),
        Index("idx_alert_processed", "processed"),
    )

class DecisionTrace(Base):
    """Comprehensive decision tracing for explainable AI - Hybrid Ultra-Diagnostics"""
    
    __tablename__ = "decision_traces"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(64), unique=True, nullable=False, index=True)  # UUID for tracking
    
    # Alert context
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(20), nullable=False)  # BUY/SELL/emergency_close
    tier = Column(String(20), nullable=False, index=True)
    alert_timestamp = Column(DateTime, nullable=False, index=True)
    
    # Processing flow
    processing_stage = Column(String(50), nullable=False)  # received/validated/analyzed/decided/executed/completed
    final_decision = Column(String(20), nullable=False)  # ACCEPTED/REJECTED/ERROR
    rejection_reason = Column(String(200))
    
    # Decision parameters
    position_size_pct = Column(Float)
    position_size_usdt = Column(Float)
    leverage_used = Column(Integer)
    leverage_suggested = Column(Integer)  # From Pine Script
    stop_loss_pct = Column(Float)
    take_profit_1_pct = Column(Float)
    take_profit_2_pct = Column(Float)
    take_profit_3_pct = Column(Float)
    
    # ML & AI insights
    ml_confidence = Column(Float)  # 0.0-1.0
    risk_score = Column(Float)  # 0.0-1.0
    market_regime = Column(String(50))
    regime_confidence = Column(Float)
    
    # Pine Script diagnostics
    pine_health_score = Column(Float)  # 0.0-1.0
    atr_percentile = Column(Float)
    adx_strength = Column(Float)
    volume_profile_score = Column(Float)
    
    # Performance tracking
    processing_time_ms = Column(Integer)
    api_latency_ms = Column(Integer)
    decision_latency_ms = Column(Integer)
    
    # Links
    trade_id = Column(Integer)  # Link to Trade if executed
    alert_history_id = Column(Integer)  # Link to AlertHistory
    
    # Raw data storage
    raw_alert_data = Column(JSON)
    decision_context = Column(JSON)  # Full context used for decision
    alternative_scenarios = Column(JSON)  # What would happen with different params
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    
    __table_args__ = (
        Index("idx_trace_symbol_stage", "symbol", "processing_stage"),
        Index("idx_trace_decision", "final_decision"),
        Index("idx_trace_timestamp", "alert_timestamp"),
    )


class ParameterDecision(Base):
    """Detailed parameter decision explanations with SHAP values"""
    
    __tablename__ = "parameter_decisions"
    
    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(64), nullable=False, index=True)  # Link to DecisionTrace
    
    # Parameter identification
    parameter_name = Column(String(50), nullable=False)  # position_size/leverage/stop_loss/take_profit_1/etc
    parameter_type = Column(String(20), nullable=False)  # float/int/bool/string
    
    # Decision values
    original_value = Column(Float)  # From Pine Script or default
    final_value = Column(Float)  # After ML/risk management
    value_change_pct = Column(Float)  # Percentage change
    
    # Explainability - SHAP values
    shap_base_value = Column(Float)
    shap_prediction = Column(Float)
    feature_contributions = Column(JSON)  # {"atr": 0.15, "adx": -0.08, "regime": 0.12}
    
    # Decision reasoning
    primary_reason = Column(String(200))  # Human readable explanation
    confidence_score = Column(Float)  # 0.0-1.0
    risk_adjustment = Column(Float)  # How much risk management affected this
    
    # Alternative scenarios
    alternative_values = Column(JSON)  # {"if_atr_higher": 2.8, "if_regime_certain": 2.1}
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_param_trace_name", "trace_id", "parameter_name"),
    )


class PineHealthLog(Base):
    """Pine Script health monitoring and diagnostics"""
    
    __tablename__ = "pine_health_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(64), index=True)  # Optional link to DecisionTrace
    
    # Basic info
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Health scores (0.0-1.0)
    overall_health = Column(Float, nullable=False)
    parameter_stability = Column(Float)
    trend_clarity = Column(Float)
    volume_quality = Column(Float)
    
    # Technical indicators health
    atr_health = Column(Float)
    atr_value = Column(Float)
    atr_percentile = Column(Float)
    
    adx_health = Column(Float)
    adx_value = Column(Float)
    adx_trend_strength = Column(String(20))  # weak/moderate/strong/very_strong
    
    # Market regime analysis
    regime_detected = Column(String(50))
    regime_confidence = Column(Float)
    regime_stability = Column(Float)
    
    # Volume analysis
    volume_profile = Column(String(20))  # poor/fair/good/excellent
    volume_anomaly = Column(Boolean, default=False)
    institutional_flow = Column(Float)
    
    # Warnings and alerts
    warnings = Column(JSON)  # ["atr_too_high", "volume_declining", "regime_uncertain"]
    critical_issues = Column(JSON)  # ["data_gap", "calculation_error"]
    
    # Performance correlation
    correlation_with_performance = Column(Float)  # How this health correlates with trade success
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_pine_symbol_time", "symbol", "timestamp"),
        Index("idx_pine_health", "overall_health"),
    )


class ShapExplanation(Base):
    """SHAP explainability data for ML decisions"""
    
    __tablename__ = "shap_explanations"
    
    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(64), nullable=False, index=True)
    
    # Model information
    model_name = Column(String(50), nullable=False)  # position_sizer/leverage_optimizer/risk_manager
    model_version = Column(String(20))
    prediction_type = Column(String(30))  # regression/classification/ranking
    
    # SHAP values
    base_value = Column(Float, nullable=False)  # Model's base prediction
    prediction_value = Column(Float, nullable=False)  # Final prediction
    
    # Feature contributions (JSON format)
    feature_values = Column(JSON, nullable=False)  # {"atr": 0.023, "adx": 45.2, "regime": "BULLISH"}
    shap_values = Column(JSON, nullable=False)  # {"atr": 0.15, "adx": -0.08, "regime": 0.12}
    
    # Top contributors
    top_positive_features = Column(JSON)  # [{"feature": "atr", "contribution": 0.15, "description": "High ATR increases position size"}]
    top_negative_features = Column(JSON)  # [{"feature": "adx", "contribution": -0.08, "description": "Low ADX reduces confidence"}]
    
    # Model confidence
    prediction_confidence = Column(Float)  # 0.0-1.0
    explanation_quality = Column(Float)  # How well SHAP explains this prediction
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_shap_trace_model", "trace_id", "model_name"),
    )


class PatternAlert(Base):
    """Proactive pattern detection and alerts"""
    
    __tablename__ = "pattern_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Pattern identification
    pattern_type = Column(String(100), nullable=False, index=True)  # rejection_spike/parameter_drift/performance_decline
    pattern_severity = Column(String(20), nullable=False)  # low/medium/high/critical
    
    # Detection details
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    pattern_description = Column(Text, nullable=False)
    
    # Affected scope
    symbols_affected = Column(JSON)  # ["BTCUSDT", "ETHUSDT"] or ["ALL"]
    tiers_affected = Column(JSON)  # ["Platinum", "Premium"] or ["ALL"]
    timeframes_affected = Column(JSON)  # ["1h", "4h"] or ["ALL"]
    
    # Pattern metrics
    frequency_count = Column(Integer)  # How many times this pattern occurred
    impact_score = Column(Float)  # 0.0-1.0 - estimated impact on performance
    confidence_score = Column(Float)  # 0.0-1.0 - confidence in pattern detection
    
    # Recommendations
    recommended_action = Column(Text)  # Human readable recommendation
    auto_fix_available = Column(Boolean, default=False)
    auto_fix_applied = Column(Boolean, default=False)
    
    # Status tracking
    status = Column(String(20), default="active")  # active/investigating/resolved/ignored
    acknowledged_by = Column(String(50))  # Discord user who acknowledged
    acknowledged_at = Column(DateTime)
    
    # Resolution
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_pattern_type_severity", "pattern_type", "pattern_severity"),
        Index("idx_pattern_status", "status"),
    )


class SystemHealth(Base):
    """Overall system health monitoring"""
    
    __tablename__ = "system_health"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Component health scores (0.0-1.0)
    overall_health = Column(Float, nullable=False)
    pine_script_health = Column(Float)
    ml_model_health = Column(Float)
    binance_api_health = Column(Float)
    database_health = Column(Float)
    discord_health = Column(Float)
    
    # Performance metrics
    avg_processing_time_ms = Column(Integer)
    avg_api_latency_ms = Column(Integer)
    error_rate_pct = Column(Float)
    
    # Trading metrics (last 24h)
    signals_received = Column(Integer)
    signals_accepted = Column(Integer)
    trades_opened = Column(Integer)
    trades_closed = Column(Integer)
    
    # System resources
    cpu_usage_pct = Column(Float)
    memory_usage_pct = Column(Float)
    disk_usage_pct = Column(Float)
    
    # Alerts and warnings
    active_warnings = Column(JSON)
    critical_issues = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_system_health_time", "timestamp"),
    )

# Database operations
def init_db():
    """Initialize database and create tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")

        # Run migrations
        _migrate_database()

        # Set default settings
        _init_default_settings()

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def _migrate_database():
    """Run database migrations for v9.1 + Diagnostics"""
    with Session() as session:
        try:
            # Check if migrations are needed
            inspector = inspect(session.bind)

            # Migrate trades table
            if "trades" in inspector.get_table_names():
                _migrate_trades_table(session)

            # Migrate other tables as needed
            if "alert_history" in inspector.get_table_names():
                _migrate_alert_history_table(session)

            # Create diagnostic tables if they don't exist
            _create_diagnostic_tables()

            session.commit()
            logger.info("Database migrations completed (including diagnostics)")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            session.rollback()


def _create_diagnostic_tables():
    """Create diagnostic tables if they don't exist"""
    try:
        # This will create all tables defined in Base.metadata that don't exist yet
        Base.metadata.create_all(bind=engine)
        logger.info("Diagnostic tables created/verified")
    except Exception as e:
        logger.error(f"Failed to create diagnostic tables: {e}")


def _migrate_trades_table(session):
    """Add new columns to trades table if they don't exist"""
    try:
        # Get existing columns
        inspector = inspect(session.bind)
        existing_columns = [col["name"] for col in inspector.get_columns("trades")]

        # List of new columns for v9.1
        migrations = [
            (
                "is_dry_run",
                "ALTER TABLE trades ADD COLUMN is_dry_run BOOLEAN DEFAULT FALSE",
            ),
            (
                "idempotency_key",
                "ALTER TABLE trades ADD COLUMN idempotency_key VARCHAR(64)",
            ),
            ("client_tags", "ALTER TABLE trades ADD COLUMN client_tags TEXT"),
            ("leverage_hint", "ALTER TABLE trades ADD COLUMN leverage_hint INTEGER"),
            (
                "indicator_version",
                "ALTER TABLE trades ADD COLUMN indicator_version VARCHAR(20)",
            ),
            (
                "institutional_flow",
                "ALTER TABLE trades ADD COLUMN institutional_flow FLOAT DEFAULT 0.0",
            ),
            (
                "retest_confidence",
                "ALTER TABLE trades ADD COLUMN retest_confidence FLOAT DEFAULT 0.0",
            ),
            (
                "fake_breakout_detected",
                "ALTER TABLE trades ADD COLUMN fake_breakout_detected BOOLEAN DEFAULT 0",
            ),
            (
                "fake_breakout_penalty",
                "ALTER TABLE trades ADD COLUMN fake_breakout_penalty FLOAT DEFAULT 1.0",
            ),
            (
                "enhanced_regime",
                "ALTER TABLE trades ADD COLUMN enhanced_regime VARCHAR(50) DEFAULT 'NEUTRAL'",
            ),
            (
                "regime_confidence",
                "ALTER TABLE trades ADD COLUMN regime_confidence FLOAT DEFAULT 0.0",
            ),
            (
                "mtf_agreement_ratio",
                "ALTER TABLE trades ADD COLUMN mtf_agreement_ratio FLOAT DEFAULT 0.0",
            ),
            ("volume_context", "ALTER TABLE trades ADD COLUMN volume_context TEXT"),
            ("bar_close_time", "ALTER TABLE trades ADD COLUMN bar_close_time TIMESTAMP"),
            ("signal_tier", "ALTER TABLE trades ADD COLUMN signal_tier VARCHAR(20)"),
            (
                "signal_timeframe",
                "ALTER TABLE trades ADD COLUMN signal_timeframe VARCHAR(10)",
            ),
            (
                "signal_session",
                "ALTER TABLE trades ADD COLUMN signal_session VARCHAR(20)",
            ),
            ("take_profit_2", "ALTER TABLE trades ADD COLUMN take_profit_2 FLOAT"),
            ("take_profit_3", "ALTER TABLE trades ADD COLUMN take_profit_3 FLOAT"),
            (
                "break_even_price",
                "ALTER TABLE trades ADD COLUMN break_even_price FLOAT",
            ),
            (
                "trailing_stop_price",
                "ALTER TABLE trades ADD COLUMN trailing_stop_price FLOAT",
            ),
            (
                "tp1_filled",
                "ALTER TABLE trades ADD COLUMN tp1_filled BOOLEAN DEFAULT 0",
            ),
            (
                "tp2_filled",
                "ALTER TABLE trades ADD COLUMN tp2_filled BOOLEAN DEFAULT 0",
            ),
            (
                "tp3_filled",
                "ALTER TABLE trades ADD COLUMN tp3_filled BOOLEAN DEFAULT 0",
            ),
            (
                "sl_moved_to_be",
                "ALTER TABLE trades ADD COLUMN sl_moved_to_be BOOLEAN DEFAULT 0",
            ),
            (
                "trailing_activated",
                "ALTER TABLE trades ADD COLUMN trailing_activated BOOLEAN DEFAULT 0",
            ),
            ("mode_used", "ALTER TABLE trades ADD COLUMN mode_used VARCHAR(20)"),
            (
                "roe_percentage",
                "ALTER TABLE trades ADD COLUMN roe_percentage FLOAT DEFAULT 0.0",
            ),
            (
                "max_drawdown",
                "ALTER TABLE trades ADD COLUMN max_drawdown FLOAT DEFAULT 0.0",
            ),
            (
                "time_in_position",
                "ALTER TABLE trades ADD COLUMN time_in_position INTEGER",
            ),
            (
                "filled_quantity",
                "ALTER TABLE trades ADD COLUMN filled_quantity FLOAT DEFAULT 0.0",
            ),
            ("alert_data", "ALTER TABLE trades ADD COLUMN alert_data TEXT"),
            ("raw_signal_data", "ALTER TABLE trades ADD COLUMN raw_signal_data TEXT"),
        ]

        # Execute migrations for missing columns
        for column_name, sql in migrations:
            if column_name not in existing_columns:
                try:
                    session.execute(sql)
                    logger.info(f"Added column: {column_name}")
                except Exception as e:
                    # Column might already exist or other DB-specific issue
                    logger.debug(f"Could not add column {column_name}: {e}")

        # Create indexes if they don't exist
        existing_indexes = [idx["name"] for idx in inspector.get_indexes("trades")]

        index_sqls = [
            (
                "idx_trade_idempotency",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_idempotency ON trades(idempotency_key)",
            ),
            (
                "idx_trade_tier",
                "CREATE INDEX IF NOT EXISTS idx_trade_tier ON trades(signal_tier)",
            ),
            (
                "idx_trade_status_tier",
                "CREATE INDEX IF NOT EXISTS idx_trade_status_tier ON trades(status, signal_tier)",
            ),
            (
                "idx_trade_indicator_version",
                "CREATE INDEX IF NOT EXISTS idx_trade_indicator_version ON trades(indicator_version)",
            ),
            (
                "idx_trade_timeframe",
                "CREATE INDEX IF NOT EXISTS idx_trade_timeframe ON trades(signal_timeframe)",
            ),
        ]

        for index_name, sql in index_sqls:
            if index_name not in existing_indexes:
                try:
                    session.execute(sql)
                    logger.info(f"Created index: {index_name}")
                except Exception as e:
                    logger.debug(f"Could not create index {index_name}: {e}")

    except Exception as e:
        logger.error(f"Trade table migration failed: {e}")

def _migrate_alert_history_table(session):
    """Add new columns to alert_history table if they don't exist"""
    try:
        inspector = inspect(session.bind)
        existing_columns = [col["name"] for col in inspector.get_columns("alert_history")]

        migrations = []
        if "success" not in existing_columns:
            migrations.append("ALTER TABLE alert_history ADD COLUMN success BOOLEAN DEFAULT FALSE")
        if "processed_at" not in existing_columns:
            # TIMESTAMP dla Postgresa, DATETIME też działa w SQLite
            migrations.append("ALTER TABLE alert_history ADD COLUMN processed_at TIMESTAMP NULL")

        for sql in migrations:
            try:
                session.execute(text(sql))
                session.commit()
                logger.info(f"AlertHistory migration executed: {sql}")
            except Exception as e:
                session.rollback()
                logger.debug(f"Could not apply AlertHistory migration ({sql}): {e}")

        session.commit()
    except Exception as e:
        logger.error(f"AlertHistory table migration failed: {e}")
        session.rollback()

def _init_default_settings():
    """Initialize default bot settings"""
    with Session() as session:
        try:
            default_settings = [
                ("mode", Config.DEFAULT_MODE, "string", "Current trading mode"),
                ("is_paused", "False", "bool", "Bot pause state"),
                (
                    "emergency_enabled",
                    str(Config.EMERGENCY_ENABLED),
                    "bool",
                    "Emergency mode state",
                ),
                (
                    "use_indicator_leverage",
                    str(Config.USE_INDICATOR_LEVERAGE),
                    "bool",
                    "Use leverage from indicator",
                ),
                (
                    "use_size_multiplier",
                    str(Config.USE_INDICATOR_SIZE_MULTIPLIER),
                    "bool",
                    "Use size multiplier from indicator",
                ),
                (
                    "move_sl_to_be",
                    str(Config.MOVE_SL_TO_BE_AT_TP1),
                    "bool",
                    "Move SL to BE after TP1",
                ),
                (
                    "tier_minimum",
                    Config.TIER_MINIMUM,
                    "string",
                    "Minimum tier to accept",
                ),
                (
                    "last_restart",
                    datetime.utcnow().isoformat(),
                    "string",
                    "Last restart time",
                ),
            ]

            for key, value, value_type, description in default_settings:
                existing = session.query(BotSettings).filter_by(key=key).first()
                if not existing:
                    setting = BotSettings(
                        key=key,
                        value=value,
                        value_type=value_type,
                        description=description,
                    )
                    session.add(setting)

            session.commit()

        except Exception as e:
            logger.error(f"Failed to init default settings: {e}")
            session.rollback()



# Helper functions
def get_setting(session: Session, key: str, default: Any = None) -> Any:
    """Get a bot setting value"""
    setting = session.query(BotSettings).filter_by(key=key).first()
    if setting:
        return setting.get_typed_value()
    return default


def set_setting(
    session: Session,
    key: str,
    value: Any,
    value_type: str = "string",
    updated_by: str = "system",
):
    """Set a bot setting value"""
    setting = session.query(BotSettings).filter_by(key=key).first()

    if setting:
        setting.value = str(value)
        setting.value_type = value_type
        setting.updated_by = updated_by
    else:
        setting = BotSettings(
            key=key, value=str(value), value_type=value_type, updated_by=updated_by
        )
        session.add(setting)

    session.commit()


def check_idempotency(session: Session, idempotency_key: str) -> bool:
    """Check if an alert has already been processed - v9.1 CORE FEATURE"""
    # Check in trades
    existing_trade = (
        session.query(Trade).filter_by(idempotency_key=idempotency_key).first()
    )
    if existing_trade:
        return True

    # Check in alert history
    existing_alert = (
        session.query(AlertHistory).filter_by(idempotency_key=idempotency_key).first()
    )
    if existing_alert and existing_alert.processed:
        return True

    return False


def record_alert(
    session: Session,
    payload: dict,
    headers: dict,
    idempotency_key: str,
    processed: bool = False,
    error: str = None,
    signature_valid: bool = False,
    latency_metrics: dict = None,
):
    """Record an incoming alert with v9.1 enhancements"""
    alert = AlertHistory(
        idempotency_key=idempotency_key,
        symbol=payload.get("symbol"),
        action=payload.get("action"),
        tier=payload.get("tier"),
        processed=processed,
        duplicate=check_idempotency(session, idempotency_key),
        error=error,
        raw_payload=payload,
        headers=dict(headers) if headers else None,
        signature_valid=signature_valid,
        schema_valid=payload.get("indicator_version") == getattr(Config, "INDICATOR_VERSION_REQUIRED", "9.1"),
        age_valid=True,  # Will be calculated elsewhere
        price_drift_valid=True,  # Will be calculated elsewhere
        # v9.1 NEW: Latency fields
        tv_ts=latency_metrics.get("tv_ts") if latency_metrics else None,
        api_latency_ms=latency_metrics.get("tradingview_to_webhook_latency_ms") if latency_metrics else None,
        end_to_end_latency_ms=latency_metrics.get("end_to_end_latency_ms") if latency_metrics else None,
    )
    session.add(alert)
    session.commit()
    return alert

def finalize_alert_processing(session: Session, idempotency_key: str, success: bool, error: str | None = None):
    """Mark alert as processed and store final result."""
    alert = session.query(AlertHistory).filter_by(idempotency_key=idempotency_key).first()
    if not alert:
        return
    alert.processed = True
    # jeśli dodałeś nowe pola do modelu:
    try:
        setattr(alert, "success", bool(success))
        setattr(alert, "processed_at", datetime.utcnow())
    except Exception:
        # Pole istnieje w modelu runtime; jeśli nie, po prostu pomiń
        pass
    if error:
        alert.error = error
    session.commit()

def get_open_positions(session: Session) -> List[Trade]:
    """Get all open positions"""
    return session.query(Trade).filter_by(status="open").all()


def get_position_by_symbol(session: Session, symbol: str) -> Optional[Trade]:
    """Get open position for a symbol"""
    return session.query(Trade).filter_by(symbol=symbol, status="open").first()


def has_open_position(session: Session, symbol: str) -> bool:
    """Check if there's an open position for a symbol"""
    return session.query(Trade).filter_by(symbol=symbol, status="open").count() > 0


def create_trade(session: Session, trade_data: dict) -> Trade:
    """Create a new trade record"""
    trade = Trade(**trade_data)
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def update_trade(session: Session, trade_id: int, updates: dict):
    """Update a trade record"""
    trade = session.query(Trade).filter_by(id=trade_id).first()
    if trade:
        for key, value in updates.items():
            if hasattr(trade, key):
                setattr(trade, key, value)
        trade.updated_at = datetime.utcnow()
        session.commit()
        session.refresh(trade)
        return trade


def close_trade(session: Session, trade_id: int, exit_data: dict):
    """Close a trade with exit information"""
    trade = session.query(Trade).filter_by(id=trade_id).first()
    if trade and trade.status == "open":
        # Calculate time in position
        time_in_position = (datetime.utcnow() - trade.entry_time).total_seconds()

        # Calculate PnL
        if trade.side == "BUY":
            pnl_percentage = (
                (exit_data["exit_price"] - trade.entry_price) / trade.entry_price
            ) * 100
        else:
            pnl_percentage = (
                (trade.entry_price - exit_data["exit_price"]) / trade.entry_price
            ) * 100

        pnl_usdt = (pnl_percentage / 100) * trade.position_size_usdt
        roe_percentage = pnl_percentage * trade.leverage_used

        # Update trade
        trade.status = "closed"
        trade.exit_price = exit_data["exit_price"]
        trade.exit_time = exit_data.get("exit_time", datetime.utcnow())
        trade.exit_quantity = exit_data.get("exit_quantity", trade.entry_quantity)
        trade.exit_reason = exit_data.get("exit_reason", "unknown")
        trade.exit_commission = exit_data.get("exit_commission", 0.0)
        trade.pnl_usdt = pnl_usdt
        trade.pnl_percentage = pnl_percentage
        trade.roe_percentage = roe_percentage
        trade.time_in_position = int(time_in_position)
        trade.updated_at = datetime.utcnow()

        session.commit()
        session.refresh(trade)

        # Update performance metrics
        _update_performance_metrics(session, trade)

        return trade


def _update_performance_metrics(session: Session, trade: Trade):
    """Update performance metrics after trade closes - v9.1 ENHANCED"""
    try:
        today = datetime.utcnow().date()

        # Update daily performance
        daily_perf = (
            session.query(Performance)
            .filter_by(date=today, metric_type="daily", metric_key="overall")
            .first()
        )

        if not daily_perf:
            daily_perf = Performance(
                date=today, metric_type="daily", metric_key="overall"
            )
            session.add(daily_perf)

        daily_perf.total_trades += 1
        if trade.pnl_usdt > 0:
            daily_perf.winning_trades += 1
        else:
            daily_perf.losing_trades += 1

        daily_perf.net_pnl = (daily_perf.net_pnl or 0) + trade.pnl_usdt
        daily_perf.commission_paid = (
            (daily_perf.commission_paid or 0)
            + trade.exit_commission
            + trade.entry_commission
        )

        # Update tier performance
        if trade.signal_tier:
            tier_perf = (
                session.query(Performance)
                .filter_by(date=today, metric_type="tier", metric_key=trade.signal_tier)
                .first()
            )

            if not tier_perf:
                tier_perf = Performance(
                    date=today, metric_type="tier", metric_key=trade.signal_tier
                )
                session.add(tier_perf)

            tier_perf.total_trades += 1
            if trade.pnl_usdt > 0:
                tier_perf.winning_trades += 1
            else:
                tier_perf.losing_trades += 1

            tier_perf.net_pnl = (tier_perf.net_pnl or 0) + trade.pnl_usdt

            # v9.1 specific metrics
            if trade.sl_moved_to_be:
                tier_perf.be_triggered_count += 1
            if trade.trailing_activated:
                tier_perf.trailing_triggered_count += 1
            if trade.signal_tier == "Emergency":
                tier_perf.emergency_trades += 1
            if trade.fake_breakout_detected:
                tier_perf.fake_breakout_count += 1

            # Update institutional flow average
            if trade.institutional_flow:
                current_avg = tier_perf.institutional_flow_avg or 0
                tier_perf.institutional_flow_avg = (
                    current_avg * (tier_perf.total_trades - 1)
                    + trade.institutional_flow
                ) / tier_perf.total_trades

        # Update leverage performance
        leverage_key = f"{trade.leverage_used}x"
        leverage_perf = (
            session.query(Performance)
            .filter_by(date=today, metric_type="leverage", metric_key=leverage_key)
            .first()
        )

        if not leverage_perf:
            leverage_perf = Performance(
                date=today, metric_type="leverage", metric_key=leverage_key
            )
            session.add(leverage_perf)

        leverage_perf.total_trades += 1
        if trade.pnl_usdt > 0:
            leverage_perf.winning_trades += 1
        else:
            leverage_perf.losing_trades += 1

        leverage_perf.net_pnl = (leverage_perf.net_pnl or 0) + trade.pnl_usdt
        leverage_perf.avg_roe = (
            (leverage_perf.avg_roe or 0) * (leverage_perf.total_trades - 1)
            + trade.roe_percentage
        ) / leverage_perf.total_trades

        session.commit()

    except Exception as e:
        logger.error(f"Failed to update performance metrics: {e}")
        session.rollback()


def get_recent_trades(session: Session, limit: int = 10) -> List[Trade]:
    """Get recent trades"""
    return session.query(Trade).order_by(Trade.created_at.desc()).limit(limit).all()


def get_trades_by_date_range(
    session: Session, start_date: datetime, end_date: datetime
) -> List[Trade]:
    """Get trades within date range"""
    return (
        session.query(Trade)
        .filter(Trade.created_at >= start_date, Trade.created_at <= end_date)
        .all()
    )


def get_performance_stats(session: Session, period: str = "daily") -> dict:
    """Get performance statistics"""
    if period == "daily":
        date = datetime.utcnow().date()
    elif period == "weekly":
        date = datetime.utcnow().date() - timedelta(days=7)
    elif period == "monthly":
        date = datetime.utcnow().date() - timedelta(days=30)
    else:
        date = datetime.utcnow().date()

    stats = (
        session.query(Performance)
        .filter(Performance.date >= date, Performance.metric_type == "daily")
        .all()
    )

    total_trades = sum(s.total_trades for s in stats)
    winning_trades = sum(s.winning_trades for s in stats)
    net_pnl = sum(s.net_pnl for s in stats)

    return {
        "period": period,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": total_trades - winning_trades,
        "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        "net_pnl": net_pnl,
        "avg_pnl_per_trade": net_pnl / total_trades if total_trades > 0 else 0,
    }


def get_tier_performance(session: Session, date_from: datetime = None) -> dict:
    """Get performance breakdown by tier - v9.1 ENHANCED"""
    if not date_from:
        date_from = datetime.utcnow().date() - timedelta(days=7)

    tier_stats = (
        session.query(Performance)
        .filter(Performance.date >= date_from, Performance.metric_type == "tier")
        .all()
    )

    result = {}
    for tier in Config.TIER_HIERARCHY:
        tier_data = [s for s in tier_stats if s.metric_key == tier]
        if tier_data:
            total_trades = sum(s.total_trades for s in tier_data)
            winning_trades = sum(s.winning_trades for s in tier_data)
            net_pnl = sum(s.net_pnl for s in tier_data)
            be_count = sum(s.be_triggered_count for s in tier_data)
            trailing_count = sum(s.trailing_triggered_count for s in tier_data)
            fake_breakout_count = sum(s.fake_breakout_count for s in tier_data)

            result[tier] = {
                "total_trades": total_trades,
                "win_rate": (
                    (winning_trades / total_trades * 100) if total_trades > 0 else 0
                ),
                "net_pnl": net_pnl,
                "avg_pnl": net_pnl / total_trades if total_trades > 0 else 0,
                "be_triggered": be_count,
                "trailing_triggered": trailing_count,
                "fake_breakouts": fake_breakout_count,
            }

    return result


def cleanup_old_data(session: Session, days_to_keep: int = 30):
    """Clean up old data from database"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        # Clean old alert history
        deleted_alerts = (
            session.query(AlertHistory)
            .filter(AlertHistory.created_at < cutoff_date)
            .delete()
        )

        # Clean old performance data (keep monthly summaries)
        deleted_perf = (
            session.query(Performance)
            .filter(
                Performance.date < cutoff_date, Performance.metric_type != "monthly"
            )
            .delete()
        )

        session.commit()
        logger.info(
            f"Cleaned up {deleted_alerts} alerts and {deleted_perf} performance records"
        )

    except Exception as e:
        logger.error(f"Failed to cleanup old data: {e}")
        session.rollback()


# v9.1 NEW FUNCTIONS
def record_signal_decision(
    session: Session,
    signal_data: dict,
    accepted: bool,
    rejection_reason: str = None,
    processing_time_ms: int = 0,
):
    """Record signal processing decision for analytics"""
    signal = Signal(
        symbol=signal_data.get("symbol"),
        action=signal_data.get("action"),
        tier=signal_data.get("tier"),
        strength=signal_data.get("strength", 0.0),
        accepted=accepted,
        rejection_reason=rejection_reason,
        alert_price=signal_data.get("price"),
        market_price=signal_data.get("current_price"),
        price_drift_pct=signal_data.get("price_drift_pct"),
        alert_age_seconds=signal_data.get("alert_age_seconds"),
        institutional_flow=signal_data.get("institutional_flow"),
        fake_breakout_detected=signal_data.get("fake_breakout_detected", False),
        regime=signal_data.get("enhanced_regime"),
        retest_confidence=signal_data.get("retest_confidence"),
        mtf_agreement_ratio=signal_data.get("mtf_agreement_ratio"),
        processing_time_ms=processing_time_ms,
        raw_alert=signal_data,
    )
    session.add(signal)
    session.commit()
    return signal


def get_signal_stats(session: Session, hours: int = 24) -> dict:
    """Get signal processing statistics"""
    since = datetime.utcnow() - timedelta(hours=hours)

    signals = session.query(Signal).filter(Signal.timestamp >= since).all()

    total_signals = len(signals)
    accepted_signals = len([s for s in signals if s.accepted])

    rejection_reasons = {}
    for signal in signals:
        if not signal.accepted and signal.rejection_reason:
            rejection_reasons[signal.rejection_reason] = (
                rejection_reasons.get(signal.rejection_reason, 0) + 1
            )

    tier_stats = {}
    for tier in Config.TIER_HIERARCHY:
        tier_signals = [s for s in signals if s.tier == tier]
        if tier_signals:
            tier_accepted = len([s for s in tier_signals if s.accepted])
            tier_stats[tier] = {
                "total": len(tier_signals),
                "accepted": tier_accepted,
                "acceptance_rate": (
                    (tier_accepted / len(tier_signals) * 100) if tier_signals else 0
                ),
            }

    return {
        "period_hours": hours,
        "total_signals": total_signals,
        "accepted_signals": accepted_signals,
        "acceptance_rate": (
            (accepted_signals / total_signals * 100) if total_signals > 0 else 0
        ),
        "rejection_reasons": rejection_reasons,
        "tier_breakdown": tier_stats,
    }

# ===== HYBRID ULTRA-DIAGNOSTICS FUNCTIONS =====

def create_decision_trace(session: Session, trace_data: dict) -> DecisionTrace:
    """Create a new decision trace for explainable AI"""
    trace = DecisionTrace(**trace_data)
    session.add(trace)
    session.commit()
    session.refresh(trace)
    return trace


def update_decision_trace(session: Session, trace_id: str, updates: dict):
    """Update decision trace with new information"""
    trace = session.query(DecisionTrace).filter_by(trace_id=trace_id).first()
    if trace:
        for key, value in updates.items():
            if hasattr(trace, key):
                setattr(trace, key, value)
        session.commit()
        session.refresh(trace)
        return trace


def complete_decision_trace(session: Session, trace_id: str, final_data: dict):
    """Mark decision trace as completed with final results"""
    trace = session.query(DecisionTrace).filter_by(trace_id=trace_id).first()
    if trace:
        trace.processing_stage = "completed"
        trace.completed_at = datetime.utcnow()
        
        for key, value in final_data.items():
            if hasattr(trace, key):
                setattr(trace, key, value)
        
        session.commit()
        session.refresh(trace)
        return trace


def add_parameter_decision(session: Session, param_data: dict) -> ParameterDecision:
    """Add detailed parameter decision explanation"""
    param_decision = ParameterDecision(**param_data)
    session.add(param_decision)
    session.commit()
    session.refresh(param_decision)
    return param_decision


def log_pine_health(session: Session, health_data: dict) -> PineHealthLog:
    """Log Pine Script health metrics"""
    health_log = PineHealthLog(**health_data)
    session.add(health_log)
    session.commit()
    session.refresh(health_log)
    return health_log


def add_shap_explanation(session: Session, shap_data: dict) -> ShapExplanation:
    """Add SHAP explainability data"""
    shap_explanation = ShapExplanation(**shap_data)
    session.add(shap_explanation)
    session.commit()
    session.refresh(shap_explanation)
    return shap_explanation


def create_pattern_alert(session: Session, pattern_data: dict) -> PatternAlert:
    """Create a new pattern alert"""
    alert = PatternAlert(**pattern_data)
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return alert


def get_decision_trace(session: Session, trace_id: str) -> Optional[DecisionTrace]:
    """Get decision trace by trace_id"""
    return session.query(DecisionTrace).filter_by(trace_id=trace_id).first()


def get_recent_decision_traces(session: Session, limit: int = 20) -> List[DecisionTrace]:
    """Get recent decision traces"""
    return session.query(DecisionTrace).order_by(DecisionTrace.created_at.desc()).limit(limit).all()


def get_parameter_decisions(session: Session, trace_id: str) -> List[ParameterDecision]:
    """Get all parameter decisions for a trace"""
    return session.query(ParameterDecision).filter_by(trace_id=trace_id).all()


def get_pine_health_latest(session: Session, symbol: str = None) -> List[PineHealthLog]:
    """Get latest Pine Script health data"""
    query = session.query(PineHealthLog)
    if symbol:
        query = query.filter_by(symbol=symbol)
    return query.order_by(PineHealthLog.timestamp.desc()).limit(10).all()


def get_active_pattern_alerts(session: Session) -> List[PatternAlert]:
    """Get active pattern alerts"""
    return session.query(PatternAlert).filter_by(status="active").order_by(PatternAlert.detected_at.desc()).all()


def acknowledge_pattern_alert(session: Session, alert_id: int, acknowledged_by: str):
    """Acknowledge a pattern alert"""
    alert = session.query(PatternAlert).filter_by(id=alert_id).first()
    if alert:
        alert.status = "investigating"
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()
        session.commit()
        return alert


def resolve_pattern_alert(session: Session, alert_id: int, resolution_notes: str):
    """Resolve a pattern alert"""
    alert = session.query(PatternAlert).filter_by(id=alert_id).first()
    if alert:
        alert.status = "resolved"
        alert.resolved_at = datetime.utcnow()
        alert.resolution_notes = resolution_notes
        session.commit()
        return alert


def log_system_health(session: Session, health_data: dict) -> SystemHealth:
    """Log system health metrics"""
    health = SystemHealth(**health_data)
    session.add(health)
    session.commit()
    session.refresh(health)
    return health


def get_diagnostic_summary(session: Session, hours: int = 24) -> dict:
    """Get comprehensive diagnostic summary"""
    since = datetime.utcnow() - timedelta(hours=hours)
    
    # Decision traces stats
    traces = session.query(DecisionTrace).filter(DecisionTrace.created_at >= since).all()
    total_traces = len(traces)
    accepted_traces = len([t for t in traces if t.final_decision == "ACCEPTED"])
    rejected_traces = len([t for t in traces if t.final_decision == "REJECTED"])
    error_traces = len([t for t in traces if t.final_decision == "ERROR"])
    
    # Average processing times
    processing_times = [t.processing_time_ms for t in traces if t.processing_time_ms]
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # Pine health
    pine_health = session.query(PineHealthLog).filter(PineHealthLog.timestamp >= since).all()
    avg_pine_health = sum(p.overall_health for p in pine_health) / len(pine_health) if pine_health else 0
    
    # Active alerts
    active_alerts = get_active_pattern_alerts(session)
    critical_alerts = [a for a in active_alerts if a.pattern_severity == "critical"]
    
    # Recent system health
    latest_system_health = session.query(SystemHealth).order_by(SystemHealth.timestamp.desc()).first()
    
    return {
        "period_hours": hours,
        "decision_traces": {
            "total": total_traces,
            "accepted": accepted_traces,
            "rejected": rejected_traces,
            "errors": error_traces,
            "acceptance_rate": (accepted_traces / total_traces * 100) if total_traces > 0 else 0,
            "avg_processing_time_ms": avg_processing_time
        },
        "pine_script_health": {
            "avg_health_score": avg_pine_health,
            "total_logs": len(pine_health)
        },
        "pattern_alerts": {
            "active_total": len(active_alerts),
            "critical": len(critical_alerts),
            "high_priority": len([a for a in active_alerts if a.pattern_severity == "high"])
        },
        "system_health": {
            "overall_health": latest_system_health.overall_health if latest_system_health else 0,
            "last_check": latest_system_health.timestamp.isoformat() if latest_system_health else None
        }
    }


def cleanup_diagnostic_data(session: Session, days_to_keep: int = 7):
    """Clean up old diagnostic data"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Clean old decision traces (keep important ones longer)
        deleted_traces = session.query(DecisionTrace).filter(
            DecisionTrace.created_at < cutoff_date,
            DecisionTrace.final_decision != "ERROR"  # Keep error traces longer
        ).delete()
        
        # Clean old parameter decisions
        deleted_params = session.query(ParameterDecision).filter(
            ParameterDecision.created_at < cutoff_date
        ).delete()
        
        # Clean old Pine health logs
        deleted_pine = session.query(PineHealthLog).filter(
            PineHealthLog.timestamp < cutoff_date
        ).delete()
        
        # Clean old SHAP explanations
        deleted_shap = session.query(ShapExplanation).filter(
            ShapExplanation.created_at < cutoff_date
        ).delete()
        
        # Clean resolved pattern alerts older than 30 days
        old_cutoff = datetime.utcnow() - timedelta(days=30)
        deleted_patterns = session.query(PatternAlert).filter(
            PatternAlert.resolved_at < old_cutoff,
            PatternAlert.status == "resolved"
        ).delete()
        
        # Clean old system health (keep daily summaries)
        deleted_health = session.query(SystemHealth).filter(
            SystemHealth.timestamp < cutoff_date
        ).delete()
        
        session.commit()
        
        logger.info(f"Diagnostic cleanup: {deleted_traces} traces, {deleted_params} params, "
                   f"{deleted_pine} pine logs, {deleted_shap} SHAP, {deleted_patterns} patterns, "
                   f"{deleted_health} health records")
        
    except Exception as e:
        logger.error(f"Failed to cleanup diagnostic data: {e}")
        session.rollback()

def get_execution_diagnostics(session: Session, hours: int = 24) -> dict:
    """Get execution diagnostics summary"""
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get execution traces
        traces = session.query(ExecutionTrace).filter(ExecutionTrace.timestamp >= since).all()
        
        # Get health checks
        health_checks = session.query(DiagnosticHealthCheck).filter(
            DiagnosticHealthCheck.timestamp >= since
        ).all()
        
        # Calculate metrics
        total_traces = len(traces)
        successful_traces = len([t for t in traces if t.status == 'success'])
        error_traces = len([t for t in traces if t.status == 'error'])
        
        avg_processing_time = 0
        if traces:
            processing_times = [t.processing_time_ms for t in traces if t.processing_time_ms]
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Component health summary
        component_health = get_component_health_summary(session, hours)
        
        return {
            "period_hours": hours,
            "execution_traces": {
                "total": total_traces,
                "successful": successful_traces,
                "errors": error_traces,
                "success_rate": (successful_traces / total_traces * 100) if total_traces > 0 else 0,
                "avg_processing_time_ms": avg_processing_time
            },
            "health_checks": {
                "total": len(health_checks),
                "components": component_health
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get execution diagnostics: {e}")
        return {
            "period_hours": hours,
            "execution_traces": {"total": 0, "successful": 0, "errors": 0, "success_rate": 0},
            "health_checks": {"total": 0, "components": {}},
            "error": str(e)
        }

# Initialize database on module import
if __name__ != "__main__":
    try:
        init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


def get_profile_performance(session: Session, days: int = 30) -> dict:
    """Get profile performance statistics"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get trades from the period
        trades = session.query(Trade).filter(
            Trade.created_at >= cutoff_date,
            Trade.status == "closed"
        ).all()
        
        if not trades:
            return {
                "period_days": days,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl_per_trade": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "avg_hold_time_hours": 0.0,
                "total_commission": 0.0
            }
        
        # Calculate statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if (t.pnl_usdt or 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl_usdt or 0 for t in trades)
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        pnl_values = [t.pnl_usdt or 0 for t in trades]
        best_trade = max(pnl_values) if pnl_values else 0
        worst_trade = min(pnl_values) if pnl_values else 0
        
        # Calculate average hold time
        hold_times = []
        for trade in trades:
            if trade.exit_time and trade.entry_time:
                hold_time = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                hold_times.append(hold_time)
        
        avg_hold_time_hours = sum(hold_times) / len(hold_times) if hold_times else 0
        
        total_commission = sum((t.entry_commission or 0) + (t.exit_commission or 0) for t in trades)
        
        return {
            "period_days": days,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_hold_time_hours": avg_hold_time_hours,
            "total_commission": total_commission
        }
        
    except Exception as e:
        logger.error(f"Failed to get profile performance: {e}")
        return {
            "period_days": days,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl_per_trade": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_hold_time_hours": 0.0,
            "total_commission": 0.0,
            "error": str(e)
        }

def update_setting(session: Session, key: str, value: Any, value_type: str = "int") -> bool:
    """
    Aktualizuje ustawienie w bazie danych.
    Kompatybilna z signal_intelligence.py
    """
    try:
        # Użyj istniejącej funkcji set_setting
        set_setting(session, key, value, value_type, updated_by="adaptive_system")
        return True
    except Exception as e:
        logger.error(f"Błąd aktualizacji ustawienia {key}: {e}")
        return False

def log_command(user: str, command: str, details: str = ""):
    """Log Discord command usage"""
    try:
        with Session() as session:
            # Create a simple log entry - you can expand this as needed
            logger.info(f"Discord command: {user} executed {command} {details}")
            
            # You could also store this in a dedicated table if needed
            # For now, just log it
            
    except Exception as e:
        logger.error(f"Failed to log command: {e}")

class ExecutionTrace(Base):
    """Execution tracing for diagnostic purposes"""
    
    __tablename__ = "execution_traces"
    
    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(64), nullable=False, index=True)
    
    # Basic info
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(100), nullable=False)  # BUY/SELL
    stage = Column(String(100), nullable=False)  # received/validated/executed/completed
    
    # Timing
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processing_time_ms = Column(Integer)
    
    # Status
    status = Column(String(20), nullable=False)  # success/error/warning
    message = Column(Text)
    
    # Context data
    context_data = Column(JSON)
    
    # Links
    trade_id = Column(Integer)
    decision_trace_id = Column(String(64))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_execution_trace_symbol", "symbol", "timestamp"),
        Index("idx_execution_status", "status"),
    )


class DiagnosticHealthCheck(Base):
    """Health check results for diagnostics"""
    
    __tablename__ = "diagnostic_health_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Component being checked
    component = Column(String(50), nullable=False, index=True)  # binance_api/database/discord/ml_model
    
    # Health status
    status = Column(String(20), nullable=False)  # healthy/warning/critical/error
    health_score = Column(Float)  # 0.0-1.0
    
    # Metrics
    response_time_ms = Column(Integer)
    error_count = Column(Integer, default=0)
    success_rate = Column(Float)
    
    # Details
    details = Column(JSON)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_health_component_time", "component", "timestamp"),
        Index("idx_health_status", "status"),
    )


class DiagnosticPerformanceMetric(Base):
    """Performance metrics for diagnostics"""
    
    __tablename__ = "diagnostic_performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_category = Column(String(50), nullable=False)  # trading/system/api/ml
    
    # Values
    value = Column(Float, nullable=False)
    unit = Column(String(20))  # ms/pct/count/usdt
    
    # Context
    symbol = Column(String(20))
    timeframe = Column(String(10))
    additional_context = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_perf_metric_time", "metric_name", "timestamp"),
        Index("idx_perf_category", "metric_category"),
    )
# ===== EXECUTION TRACING FUNCTIONS =====

class DiagnosticAlert(Base):
    """Diagnostic alerts and notifications"""
    
    __tablename__ = "diagnostic_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(64), nullable=False, unique=True, index=True)
    
    # Alert info
    alert_type = Column(String(50), nullable=False, index=True)  # performance/error/warning/critical
    severity = Column(String(20), nullable=False)  # low/medium/high/critical
    component = Column(String(50), nullable=False)  # binance/discord/ml/system
    
    # Content
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON)
    
    # Status
    status = Column(String(20), nullable=False, default='active')  # active/acknowledged/resolved
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Timing
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_alert_type_status", "alert_type", "status"),
        Index("idx_alert_severity", "severity"),
    )

# ===== DIAGNOSTIC ALERT FUNCTIONS =====

def create_diagnostic_alert(session: Session, alert_data: dict) -> DiagnosticAlert:
    """Create diagnostic alert"""
    alert = DiagnosticAlert(**alert_data)
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return alert


def get_active_alerts(session: Session, component: str = None) -> List[DiagnosticAlert]:
    """Get active diagnostic alerts"""
    query = session.query(DiagnosticAlert).filter(DiagnosticAlert.status == 'active')
    if component:
        query = query.filter(DiagnosticAlert.component == component)
    return query.order_by(DiagnosticAlert.created_at.desc()).all()


def acknowledge_alert(session: Session, alert_id: str) -> bool:
    """Acknowledge diagnostic alert"""
    alert = session.query(DiagnosticAlert).filter(DiagnosticAlert.alert_id == alert_id).first()
    if alert:
        alert.status = 'acknowledged'
        alert.acknowledged_at = datetime.utcnow()
        session.commit()
        return True
    return False

def resolve_alert(session: Session, alert_id: str, resolution_notes: str = None) -> bool:
    """Resolve diagnostic alert"""
    alert = session.query(DiagnosticAlert).filter(DiagnosticAlert.alert_id == alert_id).first()
    if alert:
        alert.status = 'resolved'
        alert.resolved_at = datetime.utcnow()
        if resolution_notes:
            alert.details = alert.details or {}
            alert.details['resolution_notes'] = resolution_notes
        session.commit()
        return True
    return False

class DiagnosticSession(Base):
    """Diagnostic session tracking"""
    
    __tablename__ = "diagnostic_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, unique=True, index=True)
    
    # Session info
    session_type = Column(String(50), nullable=False)  # manual/scheduled/triggered
    trigger_reason = Column(String(100))
    
    # Timing
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Results
    status = Column(String(20), nullable=False, default='running')  # running/completed/failed
    overall_health_score = Column(Float)
    issues_found = Column(Integer, default=0)
    
    # Summary
    summary = Column(JSON)
    recommendations = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_session_type_time", "session_type", "started_at"),
        Index("idx_session_status", "status"),
    )

def create_execution_trace(session: Session, trace_data: dict) -> ExecutionTrace:
    """Create execution trace"""
    trace = ExecutionTrace(**trace_data)
    session.add(trace)
    session.commit()
    session.refresh(trace)
    return trace


def log_health_check(session: Session, health_data: dict) -> DiagnosticHealthCheck:
    """Log health check result"""
    health_check = DiagnosticHealthCheck(**health_data)
    session.add(health_check)
    session.commit()
    session.refresh(health_check)
    return health_check


def log_performance_metric(session: Session, metric_data: dict) -> DiagnosticPerformanceMetric:
    """Log performance metric"""
    metric = DiagnosticPerformanceMetric(**metric_data)
    session.add(metric)
    session.commit()
    session.refresh(metric)
    return metric

# ===== DIAGNOSTIC SESSION FUNCTIONS =====

def create_diagnostic_session(session: Session, session_data: dict) -> DiagnosticSession:
    """Create diagnostic session"""
    diag_session = DiagnosticSession(**session_data)
    session.add(diag_session)
    session.commit()
    session.refresh(diag_session)
    return diag_session


def complete_diagnostic_session(session: Session, session_id: str, results: dict) -> bool:
    """Complete diagnostic session"""
    diag_session = session.query(DiagnosticSession).filter(DiagnosticSession.session_id == session_id).first()
    if diag_session:
        diag_session.completed_at = datetime.utcnow()
        diag_session.duration_seconds = int((diag_session.completed_at - diag_session.started_at).total_seconds())
        diag_session.status = results.get('status', 'completed')
        diag_session.overall_health_score = results.get('health_score')
        diag_session.issues_found = results.get('issues_found', 0)
        diag_session.summary = results.get('summary')
        diag_session.recommendations = results.get('recommendations')
        session.commit()
        return True
    return False


def get_diagnostic_sessions(session: Session, limit: int = 10) -> List[DiagnosticSession]:
    """Get recent diagnostic sessions"""
    return session.query(DiagnosticSession).order_by(DiagnosticSession.started_at.desc()).limit(limit).all()


def get_diagnostic_session(session: Session, session_id: str) -> Optional[DiagnosticSession]:
    """Get diagnostic session by ID"""
    return session.query(DiagnosticSession).filter(DiagnosticSession.session_id == session_id).first()

# ===== ENHANCED DIAGNOSTIC QUERIES =====

def get_execution_traces(session: Session, symbol: str = None, limit: int = 100) -> List[ExecutionTrace]:
    """Get execution traces with optional filtering"""
    query = session.query(ExecutionTrace)
    if symbol:
        query = query.filter(ExecutionTrace.symbol == symbol)
    return query.order_by(ExecutionTrace.timestamp.desc()).limit(limit).all()


def get_health_checks(session: Session, component: str = None, hours: int = 24) -> List[DiagnosticHealthCheck]:
    """Get health checks for a component"""
    since = datetime.utcnow() - timedelta(hours=hours)
    query = session.query(DiagnosticHealthCheck).filter(DiagnosticHealthCheck.timestamp >= since)
    if component:
        query = query.filter(DiagnosticHealthCheck.component == component)
    return query.order_by(DiagnosticHealthCheck.timestamp.desc()).all()


def get_performance_metrics(session: Session, metric_name: str = None, hours: int = 24) -> List[DiagnosticPerformanceMetric]:
    """Get performance metrics"""
    since = datetime.utcnow() - timedelta(hours=hours)
    query = session.query(DiagnosticPerformanceMetric).filter(DiagnosticPerformanceMetric.timestamp >= since)
    if metric_name:
        query = query.filter(DiagnosticPerformanceMetric.metric_name == metric_name)
    return query.order_by(DiagnosticPerformanceMetric.timestamp.desc()).all()


def get_component_health_summary(session: Session, hours: int = 24) -> dict:
    """Get health summary for all components"""
    since = datetime.utcnow() - timedelta(hours=hours)
    
    health_checks = session.query(DiagnosticHealthCheck).filter(
        DiagnosticHealthCheck.timestamp >= since
    ).all()
    
    components = {}
    for check in health_checks:
        if check.component not in components:
            components[check.component] = {
                'latest_status': check.status,
                'latest_score': check.health_score,
                'checks_count': 0,
                'avg_response_time': 0,
                'error_count': 0
            }
        
        comp = components[check.component]
        comp['checks_count'] += 1
        comp['error_count'] += check.error_count or 0
        
        if check.response_time_ms:
            comp['avg_response_time'] = (
                (comp['avg_response_time'] * (comp['checks_count'] - 1) + check.response_time_ms) 
                / comp['checks_count']
            )
    
    return components

# ===== EXECUTION TRACING FUNCTIONS FOR DIAGNOSTICS =====

def log_execution_trace(operation: str, context: dict = None, success: bool = None, status: str = None, message: str = None, duration_ms: int = None, details: str = None) -> str:
    """Log execution trace and return trace_id"""
    import uuid
    trace_id = str(uuid.uuid4())
    final_context = context or {}
    final_message = message or details or f"Started {operation}"

    if duration_ms is not None:
        final_context['duration_ms'] = duration_ms

    try:
        with Session() as session:
            trace = ExecutionTrace(
                trace_id=trace_id,
                symbol=final_context.get('symbol', 'SYSTEM'),
                action=final_context.get('action', operation),
                stage='completed',  # Log as a single completed event
                status=status if status else ('success' if success else 'running'),
                message=final_message,
                context_data=final_context,
                processing_time_ms=duration_ms
            )
            session.add(trace)
            session.commit()

    except Exception as e:
        logger.error(f"Failed to log execution trace: {e}")

    return trace_id


def complete_execution_trace(trace_id: str, success: bool, processing_time_ms: int, message: str = None):
    """Complete execution trace with results"""
    try:
        with Session() as session:
            trace = session.query(ExecutionTrace).filter_by(trace_id=trace_id).first()
            if trace:
                trace.stage = 'completed'
                trace.status = 'success' if success else 'error'
                trace.processing_time_ms = processing_time_ms
                if message:
                    trace.message = message
                session.commit()
                
    except Exception as e:
        logger.error(f"Failed to complete execution trace: {e}")


# ===== ML PREDICTION CLASSES FOR DIAGNOSTICS =====

class MLPrediction(Base):
    """ML prediction results for diagnostics"""
    
    __tablename__ = "ml_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String(64), nullable=False, unique=True, index=True)
    
    # Basic info
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Prediction details
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(20))
    prediction_type = Column(String(30))  # win_probability/pnl_prediction/risk_score
    
    # Input features
    input_features = Column(JSON, nullable=False)
    
    # Prediction results
    prediction_value = Column(Float, nullable=False)
    confidence_score = Column(Float)
    
    # Actual outcome (filled later)
    actual_value = Column(Float)
    prediction_accuracy = Column(Float)
    
    # Performance tracking
    processing_time_ms = Column(Integer)
    
    # Links
    trade_id = Column(Integer)
    decision_trace_id = Column(String(64))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_ml_symbol_time", "symbol", "timestamp"),
        Index("idx_ml_model", "model_name"),
    )


class MLModelMetrics(Base):
    """ML model performance metrics"""
    
    __tablename__ = "ml_model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model identification
    model_name = Column(String(50), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    metric_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Regression metrics
    mae = Column(Float)  # Mean Absolute Error
    mse = Column(Float)  # Mean Squared Error
    rmse = Column(Float)  # Root Mean Squared Error
    r2_score = Column(Float)  # R-squared
    
    # Model statistics
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    training_samples = Column(Integer)
    
    # Feature importance
    feature_importance = Column(JSON)
    
    # Model health
    drift_score = Column(Float)  # Data drift detection
    model_health_score = Column(Float)  # Overall model health 0-1
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_model_metrics_name_date", "model_name", "metric_date"),
        UniqueConstraint("model_name", "model_version", "metric_date", name="uq_model_metrics"),
    )

def log_ml_prediction(session: Session, prediction_data: Dict[str, Any]) -> MLPrediction:
    """
    Log ML prediction to database
    
    Args:
        session: Database session
        prediction_data: Dictionary containing prediction data
        
    Returns:
        MLPrediction: Created prediction record
    """
    try:
        # Create MLPrediction instance
        ml_prediction = MLPrediction(
            prediction_id=prediction_data.get('prediction_id'),
            symbol=prediction_data.get('symbol'),
            model_name=prediction_data.get('model_name'),
            prediction_type=prediction_data.get('prediction_type'),
            input_features=prediction_data.get('input_features', {}),
            prediction_value=prediction_data.get('prediction_value'),
            confidence_score=prediction_data.get('confidence_score'),
            processing_time_ms=prediction_data.get('processing_time_ms'),
            timestamp=datetime.utcnow()
        )
        
        session.add(ml_prediction)
        session.commit()
        
        return ml_prediction
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to log ML prediction: {e}")
        raise
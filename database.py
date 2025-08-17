"""
Database models and operations for Binance Trading Bot
Version: 9.1
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, JSON, String, Text,
    create_engine, func, Index, UniqueConstraint, and_, or_
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
    echo=False  # Set to True for SQL debugging
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
    
    # Idempotency and tracking
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
    exit_reason = Column(String(50))  # sl/tp1/tp2/tp3/manual/emergency/sync
    exit_commission = Column(Float, default=0.0)
    
    # Risk management levels
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    take_profit_3 = Column(Float)
    break_even_price = Column(Float)
    trailing_stop_price = Column(Float)
    
    # Position management
    leverage_used = Column(Integer, default=1)
    leverage_hint = Column(Integer)  # Suggested leverage from indicator
    margin_type = Column(String(20), default="ISOLATED")  # ISOLATED/CROSSED
    position_size_usdt = Column(Float)
    risk_amount_usdt = Column(Float)
    
    # Signal information
    signal_tier = Column(String(20), index=True)  # Platinum/Premium/Standard/Quick/Emergency
    signal_strength = Column(Float)
    signal_timeframe = Column(String(10))  # 1m/5m/15m/1h/4h/1d
    signal_session = Column(String(20))  # London/NewYork/Tokyo/Sydney
    indicator_version = Column(String(20))  # 8.0/9.1
    
    # v9.1 Enhanced fields
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
    
    # Partial fills tracking
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
    mode_used = Column(String(20))  # conservative/normal/aggressive/scalping/emergency
    # Nowe kolumny dla v9.1 i usprawnień
    idempotency_key = Column(String, unique=True, index=True)
    client_tags = Column(JSON)  # {"sl": "clientOrderId", "tp1": "...", etc}
    leverage_hint = Column(Integer)
    indicator_version = Column(String(20))
    
    # v9.1 Enhanced fields
    institutional_flow = Column(Float, default=0.0)
    retest_confidence = Column(Float, default=0.0)
    fake_breakout_detected = Column(Boolean, default=False)
    fake_breakout_penalty = Column(Float, default=1.0)
    enhanced_regime = Column(String(50), default="NEUTRAL")
    regime_confidence = Column(Float, default=0.0)
    mtf_agreement_ratio = Column(Float, default=0.0)
    bar_close_time = Column(DateTime)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_trade_status_symbol', 'status', 'symbol'),
        Index('idx_trade_tier', 'signal_tier'),
        Index('idx_trade_created', 'created_at'),
        Index('idx_trade_idempotency', 'idempotency_key'),
        Index('idx_trade_status_tier', 'status', 'signal_tier'),
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
            "leverage_used": self.leverage_used,
            "signal_tier": self.signal_tier,
            "signal_strength": self.signal_strength,
            "indicator_version": self.indicator_version,
            "sl_moved_to_be": self.sl_moved_to_be,
            "trailing_activated": self.trailing_activated
        }


class Signal(Base):
    """Signal history and analytics"""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Signal identification
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # buy/sell
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
    
    # v9.1 fields
    institutional_flow = Column(Float)
    fake_breakout_detected = Column(Boolean, default=False)
    regime = Column(String(50))
    
    # Processing metrics
    processing_time_ms = Column(Integer)
    trade_id = Column(Integer)  # Link to trade if opened
    
    # Raw data
    raw_alert = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_signal_timestamp', 'timestamp'),
        Index('idx_signal_symbol_tier', 'symbol', 'tier'),
    )


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
    """Performance metrics tracking"""
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
    
    # v9.1 specific metrics
    be_triggered_count = Column(Integer, default=0)
    trailing_triggered_count = Column(Integer, default=0)
    emergency_trades = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_performance_date_type', 'date', 'metric_type'),
        UniqueConstraint('date', 'metric_type', 'metric_key', name='uq_performance_metric'),
    )


class AlertHistory(Base):
    """Track all received alerts for debugging"""
    __tablename__ = "alert_history"
    
    id = Column(Integer, primary_key=True)
    received_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Alert identification
    idempotency_key = Column(String(64), unique=True, index=True)
    symbol = Column(String(20), index=True)
    action = Column(String(10))
    tier = Column(String(20))
    
    # Processing result
    processed = Column(Boolean, default=False)
    duplicate = Column(Boolean, default=False)
    error = Column(Text)
    
    # Raw data
    raw_payload = Column(JSON)
    headers = Column(JSON)
    
    # Validation results
    signature_valid = Column(Boolean)
    schema_valid = Column(Boolean)
    age_valid = Column(Boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)


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
    """Run database migrations for v9.1"""
    with Session() as session:
        try:
            # Check if migrations are needed
            inspector = session.bind.dialect.get_inspector(session.bind)
            
            # Migrate trades table
            if 'trades' in inspector.get_table_names():
                _migrate_trades_table(session)
            
            # Migrate other tables as needed
            session.commit()
            logger.info("Database migrations completed")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            session.rollback()


def _migrate_trades_table(session):
    """Add new columns to trades table if they don't exist"""
    try:
        # Get existing columns
        inspector = session.bind.dialect.get_inspector(session.bind)
        existing_columns = [col['name'] for col in inspector.get_columns('trades')]
        
        # List of new columns for v9.1
        migrations = [
            ("idempotency_key", "ALTER TABLE trades ADD COLUMN idempotency_key VARCHAR(64)"),
            ("client_tags", "ALTER TABLE trades ADD COLUMN client_tags TEXT"),
            ("leverage_hint", "ALTER TABLE trades ADD COLUMN leverage_hint INTEGER"),
            ("indicator_version", "ALTER TABLE trades ADD COLUMN indicator_version VARCHAR(20)"),
            ("institutional_flow", "ALTER TABLE trades ADD COLUMN institutional_flow FLOAT DEFAULT 0.0"),
            ("retest_confidence", "ALTER TABLE trades ADD COLUMN retest_confidence FLOAT DEFAULT 0.0"),
            ("fake_breakout_detected", "ALTER TABLE trades ADD COLUMN fake_breakout_detected BOOLEAN DEFAULT 0"),
            ("fake_breakout_penalty", "ALTER TABLE trades ADD COLUMN fake_breakout_penalty FLOAT DEFAULT 1.0"),
            ("enhanced_regime", "ALTER TABLE trades ADD COLUMN enhanced_regime VARCHAR(50) DEFAULT 'NEUTRAL'"),
            ("regime_confidence", "ALTER TABLE trades ADD COLUMN regime_confidence FLOAT DEFAULT 0.0"),
            ("mtf_agreement_ratio", "ALTER TABLE trades ADD COLUMN mtf_agreement_ratio FLOAT DEFAULT 0.0"),
            ("volume_context", "ALTER TABLE trades ADD COLUMN volume_context TEXT"),
            ("bar_close_time", "ALTER TABLE trades ADD COLUMN bar_close_time DATETIME"),
            ("signal_tier", "ALTER TABLE trades ADD COLUMN signal_tier VARCHAR(20)"),
            ("signal_timeframe", "ALTER TABLE trades ADD COLUMN signal_timeframe VARCHAR(10)"),
            ("signal_session", "ALTER TABLE trades ADD COLUMN signal_session VARCHAR(20)"),
            ("take_profit_2", "ALTER TABLE trades ADD COLUMN take_profit_2 FLOAT"),
            ("take_profit_3", "ALTER TABLE trades ADD COLUMN take_profit_3 FLOAT"),
            ("break_even_price", "ALTER TABLE trades ADD COLUMN break_even_price FLOAT"),
            ("trailing_stop_price", "ALTER TABLE trades ADD COLUMN trailing_stop_price FLOAT"),
            ("tp1_filled", "ALTER TABLE trades ADD COLUMN tp1_filled BOOLEAN DEFAULT 0"),
            ("tp2_filled", "ALTER TABLE trades ADD COLUMN tp2_filled BOOLEAN DEFAULT 0"),
            ("tp3_filled", "ALTER TABLE trades ADD COLUMN tp3_filled BOOLEAN DEFAULT 0"),
            ("sl_moved_to_be", "ALTER TABLE trades ADD COLUMN sl_moved_to_be BOOLEAN DEFAULT 0"),
            ("trailing_activated", "ALTER TABLE trades ADD COLUMN trailing_activated BOOLEAN DEFAULT 0"),
            ("mode_used", "ALTER TABLE trades ADD COLUMN mode_used VARCHAR(20)"),
            ("roe_percentage", "ALTER TABLE trades ADD COLUMN roe_percentage FLOAT DEFAULT 0.0"),
            ("max_drawdown", "ALTER TABLE trades ADD COLUMN max_drawdown FLOAT DEFAULT 0.0"),
            ("time_in_position", "ALTER TABLE trades ADD COLUMN time_in_position INTEGER"),
            ("filled_quantity", "ALTER TABLE trades ADD COLUMN filled_quantity FLOAT DEFAULT 0.0"),
            ("alert_data", "ALTER TABLE trades ADD COLUMN alert_data TEXT"),
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
        existing_indexes = [idx['name'] for idx in inspector.get_indexes('trades')]
        
        index_sqls = [
            ("idx_trade_idempotency", "CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_idempotency ON trades(idempotency_key)"),
            ("idx_trade_tier", "CREATE INDEX IF NOT EXISTS idx_trade_tier ON trades(signal_tier)"),
            ("idx_trade_status_tier", "CREATE INDEX IF NOT EXISTS idx_trade_status_tier ON trades(status, signal_tier)"),
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


def _init_default_settings():
    """Initialize default bot settings"""
    with Session() as session:
        try:
            default_settings = [
                ("mode", Config.DEFAULT_MODE, "string", "Current trading mode"),
                ("is_paused", "False", "bool", "Bot pause state"),
                ("emergency_enabled", str(Config.EMERGENCY_ENABLED), "bool", "Emergency mode state"),
                ("last_restart", datetime.utcnow().isoformat(), "string", "Last restart time"),
            ]
            
            for key, value, value_type, description in default_settings:
                existing = session.query(BotSettings).filter_by(key=key).first()
                if not existing:
                    setting = BotSettings(
                        key=key,
                        value=value,
                        value_type=value_type,
                        description=description
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


def set_setting(session: Session, key: str, value: Any, value_type: str = "string", updated_by: str = "system"):
    """Set a bot setting value"""
    setting = session.query(BotSettings).filter_by(key=key).first()
    
    if setting:
        setting.value = str(value)
        setting.value_type = value_type
        setting.updated_by = updated_by
    else:
        setting = BotSettings(
            key=key,
            value=str(value),
            value_type=value_type,
            updated_by=updated_by
        )
        session.add(setting)
    
    session.commit()


def check_idempotency(session: Session, idempotency_key: str) -> bool:
    """Check if an alert has already been processed"""
    # Check in trades
    existing_trade = session.query(Trade).filter_by(idempotency_key=idempotency_key).first()
    if existing_trade:
        return True
    
    # Check in alert history
    existing_alert = session.query(AlertHistory).filter_by(idempotency_key=idempotency_key).first()
    if existing_alert and existing_alert.processed:
        return True
    
    return False


def record_alert(session: Session, payload: dict, headers: dict, idempotency_key: str, processed: bool = False, error: str = None):
    """Record an incoming alert"""
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
        signature_valid=headers.get("signature_valid", False) if headers else False,
        schema_valid=payload.get("indicator_version") == Config.INDICATOR_VERSION_REQUIRED,
    )
    session.add(alert)
    session.commit()


def get_open_positions(session: Session) -> List[Trade]:
    """Get all open positions"""
    return session.query(Trade).filter_by(status="open").all()


def get_position_by_symbol(session: Session, symbol: str) -> Optional[Trade]:
    """Get open position for a symbol"""
    return session.query(Trade).filter_by(
        symbol=symbol,
        status="open"
    ).first()


def has_open_position(session: Session, symbol: str) -> bool:
    """Check if there's an open position for a symbol"""
    return session.query(Trade).filter_by(
        symbol=symbol,
        status="open"
    ).count() > 0


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
            pnl_percentage = ((exit_data["exit_price"] - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl_percentage = ((trade.entry_price - exit_data["exit_price"]) / trade.entry_price) * 100
        
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
    """Update performance metrics after trade closes"""
    try:
        today = datetime.utcnow().date()
        
        # Update daily performance
        daily_perf = session.query(Performance).filter_by(
            date=today,
            metric_type="daily",
            metric_key="overall"
        ).first()
        
        if not daily_perf:
            daily_perf = Performance(
                date=today,
                metric_type="daily",
                metric_key="overall"
            )
            session.add(daily_perf)
        
        daily_perf.total_trades += 1
        if trade.pnl_usdt > 0:
            daily_perf.winning_trades += 1
        else:
            daily_perf.losing_trades += 1
        
        daily_perf.net_pnl = (daily_perf.net_pnl or 0) + trade.pnl_usdt
        daily_perf.commission_paid = (daily_perf.commission_paid or 0) + trade.exit_commission + trade.entry_commission
        
        # Update tier performance
        if trade.signal_tier:
            tier_perf = session.query(Performance).filter_by(
                date=today,
                metric_type="tier",
                metric_key=trade.signal_tier
            ).first()
            
            if not tier_perf:
                tier_perf = Performance(
                    date=today,
                    metric_type="tier",
                    metric_key=trade.signal_tier
                )
                session.add(tier_perf)
            
            tier_perf.total_trades += 1
            if trade.pnl_usdt > 0:
                tier_perf.winning_trades += 1
            else:
                tier_perf.losing_trades += 1
            
            tier_perf.net_pnl = (tier_perf.net_pnl or 0) + trade.pnl_usdt
            
            if trade.sl_moved_to_be:
                tier_perf.be_triggered_count += 1
            if trade.trailing_activated:
                tier_perf.trailing_triggered_count += 1
            if trade.signal_tier == "Emergency":
                tier_perf.emergency_trades += 1
        
        # Update leverage performance
        leverage_key = f"{trade.leverage_used}x"
        leverage_perf = session.query(Performance).filter_by(
            date=today,
            metric_type="leverage",
            metric_key=leverage_key
        ).first()
        
        if not leverage_perf:
            leverage_perf = Performance(
                date=today,
                metric_type="leverage",
                metric_key=leverage_key
            )
            session.add(leverage_perf)
        
        leverage_perf.total_trades += 1
        if trade.pnl_usdt > 0:
            leverage_perf.winning_trades += 1
        else:
            leverage_perf.losing_trades += 1
        
        leverage_perf.net_pnl = (leverage_perf.net_pnl or 0) + trade.pnl_usdt
        leverage_perf.avg_roe = (
            ((leverage_perf.avg_roe or 0) * (leverage_perf.total_trades - 1) + trade.roe_percentage) 
            / leverage_perf.total_trades
        )
        
        session.commit()
        
    except Exception as e:
        logger.error(f"Failed to update performance metrics: {e}")
        session.rollback()


def get_recent_trades(session: Session, limit: int = 10) -> List[Trade]:
    """Get recent trades"""
    return session.query(Trade).order_by(Trade.created_at.desc()).limit(limit).all()


def get_trades_by_date_range(session: Session, start_date: datetime, end_date: datetime) -> List[Trade]:
    """Get trades within date range"""
    return session.query(Trade).filter(
        Trade.created_at >= start_date,
        Trade.created_at <= end_date
    ).all()


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
    
    stats = session.query(Performance).filter(
        Performance.date >= date,
        Performance.metric_type == "daily"
    ).all()
    
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
        "avg_pnl_per_trade": net_pnl / total_trades if total_trades > 0 else 0
    }


def get_tier_performance(session: Session, date_from: datetime = None) -> dict:
    """Get performance breakdown by tier"""
    if not date_from:
        date_from = datetime.utcnow().date() - timedelta(days=7)
    
    tier_stats = session.query(Performance).filter(
        Performance.date >= date_from,
        Performance.metric_type == "tier"
    ).all()
    
    result = {}
    for tier in Config.TIER_HIERARCHY:
        tier_data = [s for s in tier_stats if s.metric_key == tier]
        if tier_data:
            total_trades = sum(s.total_trades for s in tier_data)
            winning_trades = sum(s.winning_trades for s in tier_data)
            net_pnl = sum(s.net_pnl for s in tier_data)
            
            result[tier] = {
                "total_trades": total_trades,
                "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                "net_pnl": net_pnl,
                "avg_pnl": net_pnl / total_trades if total_trades > 0 else 0
            }
    
    return result


def cleanup_old_data(session: Session, days_to_keep: int = 30):
    """Clean up old data from database"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Clean old alert history
        deleted_alerts = session.query(AlertHistory).filter(
            AlertHistory.created_at < cutoff_date
        ).delete()
        
        # Clean old performance data (keep monthly summaries)
        deleted_perf = session.query(Performance).filter(
            Performance.date < cutoff_date,
            Performance.metric_type != "monthly"
        ).delete()
        
        session.commit()
        logger.info(f"Cleaned up {deleted_alerts} alerts and {deleted_perf} performance records")
        
    except Exception as e:
        logger.error(f"Failed to cleanup old data: {e}")
        session.rollback()


# Initialize database on module import
if __name__ != "__main__":
    try:
        init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
# database.py (final – schema + settings/commands + auto-migrations)
from datetime import datetime, timedelta

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import Config

Base = declarative_base()
engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True, future=True)
Session = sessionmaker(bind=engine, expire_on_commit=False)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    action = Column(String)  # buy/sell
    entry_price = Column(Float)
    stop_loss = Column(Float)
    tp1 = Column(Float)
    tp2 = Column(Float)
    tp3 = Column(Float)
    quantity = Column(Float)
    leverage = Column(Integer)

    status = Column(String, default="open")  # open/closed/partial
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)

    signal_strength = Column(Float)
    confluence_score = Column(Float, default=0.0)
    signal_profile = Column(String, default="C")
    pair_tier = Column(Integer)
    session = Column(String)
    mode = Column(String)

    mfi_value = Column(Float, default=50.0)
    adx_value = Column(Float, default=0.0)
    volume_spike = Column(Boolean, default=False)
    near_key_level = Column(Boolean, default=False)
    htf_trend = Column(String, default="neutral")
    btc_correlation = Column(Float, default=0.0)

    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)

    exit_price = Column(Float)
    exit_reason = Column(String)  # tp1/tp2/tp3/sl/manual

    max_profit = Column(Float, default=0.0)
    max_loss = Column(Float, default=0.0)
    duration_minutes = Column(Integer)

    tp1_hit = Column(Boolean, default=False)
    tp2_hit = Column(Boolean, default=False)
    tp3_hit = Column(Boolean, default=False)
    time_to_tp1_minutes = Column(Integer)
    position_multiplier = Column(Float, default=1.0)

    is_dry_run = Column(Boolean, default=False)

    tier = Column(String(20), default="Standard")
    market_regime = Column(String(20), default="NEUTRAL")
    market_condition = Column(String(50), default="NORMAL")
    confidence_penalty = Column(Float, default=0.0)
    liquidity_sweep = Column(Boolean, default=False)
    fresh_bos = Column(Boolean, default=False)
    raw_signal_data = Column(JSON)
    ml_predicted_win_probability = Column(Float)
    ml_predicted_pnl_pct = Column(Float)
    ml_predicted_ev = Column(Float)
    ml_confidence_score = Column(Float)
    ml_model_version = Column(String(20))
    is_emergency_signal = Column(Boolean, default=False)
    mitigation_active = Column(Boolean, default=False)
    p1_volume_impulse = Column(Boolean, default=False)
    p2_fresh_bos = Column(Boolean, default=False)
    p3_micro_trend = Column(Boolean, default=False)
    p4_adx_rising = Column(Boolean, default=False)

    # NOWE: metryki wykonania (finalne wartości)
    planned_size = Column(Float, default=0.0)
    requested_size = Column(Float, default=0.0)
    fill_ratio_planned = Column(Float, default=0.0)
    fill_ratio_requested = Column(Float, default=0.0)
    partial_reason = Column(String, default="")  # np. margin_adjustment/limit_bracket/...


class Performance(Base):
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    total_pnl = Column(Float)

    profile_a_trades = Column(Integer, default=0)
    profile_b_trades = Column(Integer, default=0)
    profile_c_trades = Column(Integer, default=0)
    profile_a_winrate = Column(Float, default=0.0)
    profile_b_winrate = Column(Float, default=0.0)
    profile_c_winrate = Column(Float, default=0.0)


class Analytics(Base):
    __tablename__ = "analytics"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String(100))
    metric_value = Column(Float)
    additional_data = Column(JSON)


class SignalHistory(Base):
    __tablename__ = "signal_history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String)
    action = Column(String)

    confluence_score = Column(Float)
    signal_strength = Column(Float)
    mfi = Column(Float)
    macd_signal = Column(String)
    adx = Column(Float)
    volume_spike = Column(Boolean)
    near_key_level = Column(Boolean)
    key_level_strength = Column(Float)
    htf_trend = Column(String)
    btc_correlation = Column(Float)
    oi_signal = Column(String)
    pair_tier = Column(Integer)
    session = Column(String)

    decision = Column(String)
    rejection_reason = Column(String)
    assigned_profile = Column(String)
    position_multiplier = Column(Float)

    trade_id = Column(Integer)
    was_profitable = Column(Boolean)
    final_pnl = Column(Float)
    max_profit_reached = Column(Float)
    time_to_close_minutes = Column(Integer)


Base.metadata.create_all(engine)


def _ensure_settings_table(session) -> None:
    session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            created_at TEXT
        )
        """
        )
    )
    # brakujące kolumny
    cols = session.execute(text("PRAGMA table_info(settings)")).fetchall()
    names = {row[1] for row in cols}
    if "created_at" not in names:
        session.execute(text("ALTER TABLE settings ADD COLUMN created_at TEXT"))
        session.commit()


def _ensure_command_log_table(session) -> None:
    session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS command_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            command TEXT,
            details TEXT,
            created_at TEXT
        )
        """
        )
    )
    session.commit()


def _migrate_trades_table(session) -> None:
    # Dodaj brakujące kolumny w 'trades' jeśli nie istnieją
    cols = session.execute(text("PRAGMA table_info(trades)")).fetchall()
    names = {row[1] for row in cols}
    alters = []
    if "planned_size" not in names:
        alters.append("ALTER TABLE trades ADD COLUMN planned_size REAL DEFAULT 0.0")
    if "requested_size" not in names:
        alters.append("ALTER TABLE trades ADD COLUMN requested_size REAL DEFAULT 0.0")
    if "fill_ratio_planned" not in names:
        alters.append("ALTER TABLE trades ADD COLUMN fill_ratio_planned REAL DEFAULT 0.0")
    if "fill_ratio_requested" not in names:
        alters.append("ALTER TABLE trades ADD COLUMN fill_ratio_requested REAL DEFAULT 0.0")
    if "partial_reason" not in names:
        alters.append("ALTER TABLE trades ADD COLUMN partial_reason TEXT DEFAULT ''")
    for stmt in alters:
        session.execute(text(stmt))
    if alters:
        session.commit()


# Automatyczne migracje (lekko)
with Session() as _s:
    _ensure_settings_table(_s)
    _ensure_command_log_table(_s)
    _migrate_trades_table(_s)


def get_setting(session, key: str, default: str | None = None) -> str | None:
    _ensure_settings_table(session)
    val = session.execute(text("SELECT value FROM settings WHERE key = :k"), {"k": key}).scalar()
    if val is None and default is not None:
        set_setting(session, key, str(default))
        return str(default)
    return val


def set_setting(session, key: str, value: str) -> None:
    _ensure_settings_table(session)
    session.execute(
        text(
            """
            INSERT INTO settings (key, value, created_at)
            VALUES (:k, :v, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        ),
        {"k": key, "v": str(value)},
    )
    session.commit()


def log_command(username: str, command: str, details: str = "") -> None:
    with Session() as session:
        _ensure_command_log_table(session)
        session.execute(
            text(
                """
                INSERT INTO command_log (username, command, details, created_at)
                VALUES (:u, :c, :d, CURRENT_TIMESTAMP)
                """
            ),
            {"u": username, "c": command, "d": details},
        )
        session.commit()


def get_profile_performance(session, profile: str, days: int = 30):
    start_date = datetime.utcnow() - timedelta(days=days)

    trades = (
        session.query(Trade)
        .filter(
            Trade.signal_profile == profile,
            Trade.entry_time >= start_date,
            Trade.status == "closed",
            Trade.is_dry_run.is_(False),
        )
        .all()
    )

    if not trades:
        return {
            "profile": profile,
            "total_trades": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "total_pnl": 0,
        }

    winning = sum(1 for t in trades if (t.pnl is not None and t.pnl > 0))
    total_pnl = sum(t.pnl for t in trades if t.pnl is not None)

    return {
        "profile": profile,
        "total_trades": len(trades),
        "win_rate": (winning / len(trades)) * 100 if trades else 0,
        "avg_pnl": (total_pnl / len(trades)) if trades else 0,
        "total_pnl": total_pnl,
    }

# config.py

import os

from dotenv import load_dotenv

load_dotenv()


def _to_bool(s: str, default=False) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _sanitize_db_url(url: str) -> str:
    if not url:
        return "sqlite:////app/data/trading_bot.db"
    return url.strip().strip('"').strip("'")


def _to_id_set(s: str):
    if not s:
        return set()
    out = set()
    for part in s.split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


def _to_float_list(s: str, default: str = ""):
    src = s if s is not None else default
    vals = []
    for p in str(src).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except Exception:
            pass
    return vals


def _to_tp_split(s: str, default="0.5,0.3,0.2"):
    vals = _to_float_list(s, default=default)
    if len(vals) != 3:
        vals = [0.5, 0.3, 0.2]
    vals = [max(0.0, v) for v in vals]
    total = sum(vals) or 1.0
    vals = [v / total for v in vals]
    return {"tp1": vals[0], "tp2": vals[1], "tp3": vals[2]}


class Config:
    # Binance
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    BINANCE_TESTNET = _to_bool(os.getenv("BINANCE_TESTNET", "True"), default=True)

    # Discord
    DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
    DISCORD_LOGS_WEBHOOK = os.getenv("DISCORD_LOGS_WEBHOOK")
    DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
    DISCORD_LOG_CHANNEL_ID = os.getenv("DISCORD_LOG_CHANNEL_ID")
    DISCORD_TRADE_ENTRIES_WEBHOOK = os.getenv("DISCORD_TRADE_ENTRIES_WEBHOOK")
    DISCORD_TRADE_EXITS_WEBHOOK = os.getenv("DISCORD_TRADE_EXITS_WEBHOOK")
    DISCORD_SIGNAL_DECISIONS_WEBHOOK = os.getenv("DISCORD_SIGNAL_DECISIONS_WEBHOOK")
    DISCORD_ALERTS_WEBHOOK = os.getenv("DISCORD_ALERTS_WEBHOOK")
    DISCORD_PERFORMANCE_WEBHOOK = os.getenv("DISCORD_PERFORMANCE_WEBHOOK")

    # Trading
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1%
    TP_DISTRIBUTION = {"tp1": 0.5, "tp2": 0.3, "tp3": 0.2}

    # Czy używać leveli z alertu (SL/TP z payloadu). Gdy False, używamy RR + split.
    USE_ALERT_LEVELS = _to_bool(os.getenv("USE_ALERT_LEVELS", "True"), default=True)

    # RR poziomy (dla USE_ALERT_LEVELS=False), np. "1.0,1.5,2.0"
    TP_RR_LEVELS = _to_float_list(os.getenv("TP_RR_LEVELS", "1.0,1.5,2.0"), default="1.0,1.5,2.0")

    # Split TP (dla USE_ALERT_LEVELS=False)
    TP_SPLIT = _to_tp_split(os.getenv("TP_SPLIT", "0.5,0.3,0.2"))

    # SL jako % użytej marży po otwarciu (dla USE_ALERT_LEVELS=False)
    SL_MARGIN_PCT = float(os.getenv("SL_MARGIN_PCT", "0.5"))  # 50% użytej marży

    # ML (domyślnie wyłączone)
    USE_ML_FOR_DECISION = _to_bool(os.getenv("USE_ML_FOR_DECISION", "False"), default=False)
    USE_ML_FOR_SIZING = _to_bool(os.getenv("USE_ML_FOR_SIZING", "False"), default=False)

    # Kontrola Bota
    BOT_OVERRIDES_LEVERAGE = _to_bool(os.getenv("BOT_OVERRIDES_LEVERAGE", "True"), default=True)
    MIN_SIGNAL_STRENGTH = float(os.getenv("MIN_SIGNAL_STRENGTH", "0.45"))
    DRY_RUN = _to_bool(os.getenv("DRY_RUN", "False"), default=False)

    # Webhook
    WEBHOOK_SECRET = (os.getenv("WEBHOOK_SECRET", "") or "").strip()

    # Database
    DATABASE_URL = _sanitize_db_url(os.getenv("DATABASE_URL", "sqlite:////app/data/trading_bot.db"))

    # ACL
    ALLOWED_USER_IDS = _to_id_set(os.getenv("ALLOWED_USER_IDS", ""))
    ALLOWED_ROLE_IDS = _to_id_set(os.getenv("ALLOWED_ROLE_IDS", ""))

    # Margin i sloty
    MARGIN_PER_TRADE_FRACTION = float(os.getenv("MARGIN_PER_TRADE_FRACTION", "0.0"))
    MARGIN_SLOTS = int(os.getenv("MARGIN_SLOTS", "10"))
    MARGIN_SAFETY_BUFFER = float(os.getenv("MARGIN_SAFETY_BUFFER", "0.95"))  # zostaw 5%
    MAX_CONCURRENT_SLOTS = int(os.getenv("MAX_CONCURRENT_SLOTS", "3"))
    DEFAULT_MARGIN_TYPE = (os.getenv("DEFAULT_MARGIN_TYPE", "ISOLATED") or "ISOLATED").upper()
    SINGLE_POSITION_PER_SYMBOL = _to_bool(
        os.getenv("SINGLE_POSITION_PER_SYMBOL", "True"), default=True
    )

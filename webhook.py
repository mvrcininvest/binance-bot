# webhook.py (FINAL – walidacja zależna od USE_ALERT_LEVELS)
import hashlib
import json
import logging
from datetime import datetime, timedelta

from flask import Flask, current_app, jsonify, request
from sqlalchemy import create_engine, text

from config import Config

app = Flask(__name__)
logger = logging.getLogger("webhook")

engine = create_engine(Config.DATABASE_URL, future=True)


def _ensure_webhook_table(conn):
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS webhook_events (
                id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now'))
            )
            """
        )
    )


def register_webhook_event(evt_id: str) -> bool:
    try:
        with engine.begin() as conn:
            _ensure_webhook_table(conn)
            one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%f")
            exists = conn.execute(
                text("SELECT 1 FROM webhook_events WHERE id = :id AND created_at >= :timestamp"),
                {"id": evt_id, "timestamp": one_hour_ago},
            ).fetchone()
            if exists:
                return False
            conn.execute(text("INSERT INTO webhook_events (id) VALUES (:id)"), {"id": evt_id})
            return True
    except Exception as e:
        logger.error(f"Błąd sprawdzania duplikatu: {e}", exc_info=True)
        return True


def _validate_and_normalize(data: dict):
    payload = data.copy()

    # Minimalnie wymagane
    required = ["symbol", "action"]
    for k in required:
        if k not in payload:
            return False, f"Brak pola: {k}", None

    # Gdy USE_ALERT_LEVELS=True, wymagamy price i sl
    if Config.USE_ALERT_LEVELS:
        for k in ("price", "sl"):
            if k not in payload:
                return False, f"Brak pola: {k}", None

    # Parsowanie liczb jeśli są
    for nk in ("price", "sl", "tp1", "tp2", "tp3"):
        if nk in payload and payload[nk] is not None:
            try:
                payload[nk] = float(payload[nk])
            except (ValueError, TypeError):
                return False, f"Nieprawidłowa wartość dla {nk}", None

    return True, "ok", payload


def _event_hash(payload: dict) -> str:
    # Hashujemy tylko najważniejsze pola, by nie blokować podobnych sygnałów
    keys = ["symbol", "action", "price", "sl", "tp1", "tp2", "tp3"]
    base = json.dumps(
        {k: payload.get(k) for k in keys if k in payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


@app.route("/webhook", methods=["POST"])
def webhook():
    bot_instance = current_app.config.get("BOT_INSTANCE")
    if not bot_instance:
        return jsonify({"error": "bot_not_initialized"}), 503

    data = request.get_json(silent=True)
    if data is None:
        try:
            data = json.loads(request.data.decode("utf-8"))
        except Exception:
            return jsonify({"error": "payload.invalid_or_empty"}), 400

    ok, reason, payload = _validate_and_normalize(data)
    if not ok:
        return jsonify({"error": f"payload.{reason}"}), 400

    evt_hash = _event_hash(payload)
    if not register_webhook_event(evt_hash):
        return (
            jsonify(
                {
                    "status": "duplicate",
                    "message": "Event już przetworzony w ciągu ostatniej godziny.",
                }
            ),
            200,
        )

    logger.info(f"Odebrano nowy, unikalny alert: {payload}")
    bot_instance.process_signal_with_risk_management(payload)
    return jsonify({"status": "received"}), 200


@app.route("/health", methods=["GET"])
def health_check():
    bot_instance = current_app.config.get("BOT_INSTANCE")
    if bot_instance and getattr(bot_instance, "is_running", False):
        return jsonify({"status": "healthy"}), 200
    return jsonify({"status": "unhealthy"}), 503

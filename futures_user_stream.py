# futures_user_stream.py (szkic)
import logging
from datetime import datetime

from binance import ThreadedWebsocketManager

from config import Config
from database import Session, Trade

logger = logging.getLogger(__name__)


class FuturesUserStream:
    def __init__(self, bot):
        self.bot = bot
        self.twm = ThreadedWebsocketManager(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_API_SECRET,
            testnet=Config.BINANCE_TESTNET,
        )

    def start(self):
        self.twm.start()
        self.twm.start_futures_user_socket(self._on_msg)

    def stop(self):
        self.twm.stop()

    def _on_msg(self, msg: dict):
        try:
            if msg.get("e") != "ORDER_TRADE_UPDATE":
                return
            o = msg.get("o", {})
            symbol = o.get("s")
            status = o.get("X")
            order_type = (o.get("o") or "").upper()
            close_position = str(o.get("cp") or o.get("closePosition") or "").lower() == "true"

            if status != "FILLED":
                return

            ap = o.get("ap") or o.get("sp") or o.get("stopPrice")
            exit_price = float(ap) if ap is not None else None

            reason = None
            if order_type in ("STOP", "STOP_MARKET") and close_position:
                reason = "sl"
            elif order_type in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
                try:
                    pos = self.bot.binance.client.futures_position_information(symbol=symbol)
                    amt = abs(float(pos[0].get("positionAmt", 0.0)))
                    if amt == 0.0:
                        reason = "tp"
                except Exception:
                    pass

            if not reason:
                return

            with Session() as s:
                t = (
                    s.query(Trade)
                    .filter_by(symbol=symbol, status="open")
                    .order_by(Trade.entry_time.desc())
                    .first()
                )
                if not t:
                    return
                if exit_price is None:
                    try:
                        mark = self.bot.binance.client.futures_mark_price(symbol=symbol).get(
                            "markPrice"
                        )
                        exit_price = float(mark)
                    except Exception:
                        exit_price = float(t.entry_price or 0.0)

                t.status = "closed"
                t.exit_time = datetime.utcnow()
                t.exit_price = exit_price
                t.exit_reason = reason

                direction = 1 if t.action == "buy" else -1
                qty = float(t.quantity or 0.0)
                entry = float(t.entry_price or 0.0)
                t.pnl = direction * (exit_price - entry) * qty
                t.pnl_percent = direction * ((exit_price - entry) / max(entry, 1e-9)) * 100.0

                s.commit()

            try:
                self.bot.notifications.send_trade_exit_notification(
                    {
                        "symbol": t.symbol,
                        "pnl": t.pnl,
                        "pnl_percent": t.pnl_percent,
                        "exit_reason": t.exit_reason,
                    }
                )
                self.bot.notifications.send_log(
                    "Pozycja zamknięta (event)",
                    {
                        "symbol": t.symbol,
                        "reason": t.exit_reason,
                        "pnl": f"${t.pnl:.2f}",
                        "pnl_percent": f"{t.pnl_percent:+.2f}%",
                    },
                    level=("warning" if t.exit_reason == "sl" else "info"),
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"UserData handler error: {e}", exc_info=True)

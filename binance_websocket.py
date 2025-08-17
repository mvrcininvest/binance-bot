# binance_websocket.py — BE po TP1, kasowanie TP po SL, sprzątanie kiedy pozycja=0

import json
import logging
from contextlib import suppress

from binance.client import Client
from binance.streams import BinanceSocketManager

from config import Config

logger = logging.getLogger(__name__)


class BinanceWebsocketClient:
    def __init__(self, bot_instance):
        self.main_bot = bot_instance
        self.client = Client(
            Config.BINANCE_API_KEY,
            Config.BINANCE_API_SECRET,
            testnet=Config.BINANCE_TESTNET,
        )
        self.bsm: BinanceSocketManager | None = None
        self.user_socket = None
        self._be_moved_for: set[str] = set()  # symbole z już wykonanym BE po TP1

    def start(self):
        logger.info("Uruchamianie klienta WebSocket Binance...")
        try:
            self.bsm = BinanceSocketManager(self.client)
            self.user_socket = self.bsm.futures_user_socket(callback=self.on_message)
            self.bsm.start()
            logger.info("WebSocket Binance uruchomiony pomyślnie")
        except Exception as e:
            logger.error("Błąd inicjalizacji WebSocket: %s", e)

    def stop(self):
        logger.info("Zatrzymywanie WebSocket Binance...")
        if self.bsm:
            self.bsm.close()

    def on_message(self, message):
        try:
            if isinstance(message, str):
                message = json.loads(message)

            if message.get("e") == "ORDER_TRADE_UPDATE":
                od = message.get("o") or {}
                self.handle_order_update(od)
        except Exception as e:
            logger.error("Błąd przetwarzania wiadomości WebSocket: %s", e)

    def handle_order_update(self, order_data: dict):
        if not order_data:
            return

        symbol = order_data.get("s")
        status = order_data.get("X")  # NEW, PARTIALLY_FILLED, FILLED, CANCELED, ...
        order_type = order_data.get("o")  # TAKE_PROFIT_MARKET, STOP_MARKET itd.
        reduce_only = bool(order_data.get("R"))  # R=true => reduceOnly

        logger.info("WS: %s %s %s reduceOnly=%s", symbol, order_type, status, reduce_only)

        # TP hit (pierwszy TP): przenieś SL do BE raz
        if (
            status == "FILLED"
            and order_type == "TAKE_PROFIT_MARKET"
            and reduce_only
            and symbol not in self._be_moved_for
        ):
            ok, msg = self.main_bot.binance.move_sl_to_break_even(symbol)
            if ok:
                self._be_moved_for.add(symbol)
                with suppress(Exception):
                    self.main_bot.notifications.send_tp_hit_notification(
                        {
                            "symbol": symbol,
                            "tp_level": "TP (first)",
                            "price": float(order_data.get("sp") or 0.0),
                            "sl_moved_to_be": True,
                        }
                    )
            else:
                logger.warning("WS: Próba BE po TP1 dla %s nieudana: %s", symbol, msg)

        # SL filled: skasuj pozostałe TP/reduceOnly i posprzątaj
        if status == "FILLED" and order_type == "STOP_MARKET":
            with suppress(Exception):
                self.main_bot.binance.client.futures_cancel_all_open_orders(symbol=symbol)
            # Sprawdź czy pozycja jest zamknięta – jak tak, sprzątnij
            try:
                positions = self.main_bot.binance.check_positions()
                still_open = any(
                    p.get("symbol") == symbol and abs(float(p.get("positionAmt", 0))) > 0
                    for p in positions
                )
                if not still_open:
                    self._be_moved_for.discard(symbol)
                    cancelled = self.main_bot.binance.cleanup_stale_orders()
                    logger.info(
                        "WS: SL dla %s – pozostałe TP anulowane, cleanup=%s",
                        symbol,
                        cancelled,
                    )
            except Exception as e:
                logger.warning("WS cleanup error: %s", e)

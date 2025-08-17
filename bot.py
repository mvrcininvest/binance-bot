import logging
import threading
import time
from contextlib import suppress
from collections import defaultdict
from datetime import datetime
from typing import Any

from analytics import AnalyticsEngine
from binance_handler import BinanceHandler
from config import Config
from database import Session, Trade, get_setting, set_setting
from discord_notifications import DiscordNotificationSystem
from mode_manager import ModeManager
from mode_switcher import IntelligentModeSwitcher
from predictive_analytics import AutoOptimizer, PredictiveAnalytics
from signal_intelligence import SignalIntelligence

logger = logging.getLogger("bot")


class TradingBot:
    def __init__(self):
        logger.info("Inicjalizacja TradingBot...")

        # Komponenty
        self.session = Session()
        self.binance = BinanceHandler()
        self.notifications = DiscordNotificationSystem(
            webhook_url=Config.DISCORD_WEBHOOK,
            logs_webhook=Config.DISCORD_LOGS_WEBHOOK,
            trade_entries_webhook=Config.DISCORD_TRADE_ENTRIES_WEBHOOK,
            trade_exits_webhook=Config.DISCORD_TRADE_EXITS_WEBHOOK,
            signal_decisions_webhook=Config.DISCORD_SIGNAL_DECISIONS_WEBHOOK,
            alerts_webhook=Config.DISCORD_ALERTS_WEBHOOK,
            performance_webhook=Config.DISCORD_PERFORMANCE_WEBHOOK,
        )
        self.analytics = AnalyticsEngine(self)
        self.mode_manager = ModeManager(self)
        self.signal_intelligence = SignalIntelligence(self)
        self.mode_switcher = IntelligentModeSwitcher(self)
        self.predictive_analytics = PredictiveAnalytics(self)
        self.auto_optimizer = AutoOptimizer(self)
        self.user_stream = None

        # Stany
        self.is_running = True
        self.is_paused = False
        self.start_time = datetime.utcnow()
        self.runtime_risk = Config.RISK_PER_TRADE
        self.is_dry_run = Config.DRY_RUN
        self.max_daily_loss = -500.0

        self.use_ml_for_decision = Config.USE_ML_FOR_DECISION
        self.use_ml_for_sizing = Config.USE_ML_FOR_SIZING
        self.intelligent_mode_switcher_enabled = False

        self.circuit_breaker_tripped = False
        self.circuit_breaker_override = False
        self.last_daily_pnl_check = datetime.utcnow()
        self._trailing_state: dict[str, Any] = {}
        self.current_mode = "Balanced"
        self.leverage_override_enabled = Config.BOT_OVERRIDES_LEVERAGE

        self.signals_processed = 0
        self.signals_accepted = 0

        # Blokady na symbol
        self._symbol_locks = defaultdict(threading.Lock)

        # Stream zdarzeń futures
        self.user_stream = None

        self._load_persistent_settings()
        logger.info("Bot gotowy do pracy!")

    def _load_persistent_settings(self):
        try:
            with Session() as session:
                dry_run_db = get_setting(session, "dry_run", str(Config.DRY_RUN))
                self.is_dry_run = str(dry_run_db).lower() in ("true", "1", "yes")

                risk_db = get_setting(session, "risk_per_trade", str(Config.RISK_PER_TRADE))
                self.runtime_risk = float(risk_db)

                max_loss_db = get_setting(session, "max_daily_loss", "-500.0")
                self.max_daily_loss = float(max_loss_db)

                ml_decision_db = get_setting(
                    session, "use_ml_for_decision", str(self.use_ml_for_decision)
                )
                self.use_ml_for_decision = str(ml_decision_db).lower() in (
                    "true",
                    "1",
                    "yes",
                )

                ml_sizing_db = get_setting(
                    session, "use_ml_for_sizing", str(self.use_ml_for_sizing)
                )
                self.use_ml_for_sizing = str(ml_sizing_db).lower() in (
                    "true",
                    "1",
                    "yes",
                )

                ims_enabled_db = get_setting(session, "intelligent_mode_switcher_enabled", "False")
                self.intelligent_mode_switcher_enabled = str(ims_enabled_db).lower() in (
                    "true",
                    "1",
                    "yes",
                )

                logger.info(
                    "Wczytano ustawienia z DB: Dry Run=%s, Ryzyko=%.2f%%, "
                    "Max Strata Dzienna=$%s, ML=%s, IMS=%s",
                    self.is_dry_run,
                    self.runtime_risk * 100.0,
                    self.max_daily_loss,
                    self.use_ml_for_decision,
                    self.intelligent_mode_switcher_enabled,
                )
        except Exception as e:
            logger.error("Nie udało się wczytać ustawień z bazy danych: %s", e, exc_info=True)

    def toggle_ml_sizing(self, enabled: bool) -> str:
        self.use_ml_for_sizing = enabled
        with Session() as session:
            set_setting(session, "use_ml_for_sizing", str(enabled))
        status = "WŁĄCZONE" if enabled else "WYŁĄCZONE"
        logger.warning("ML sizing: %s", status)
        return f"✅ ML sizing jest teraz {status}."

    def _check_slots_or_block(self) -> bool:
        try:
            with Session() as s:
                open_count = s.query(Trade).filter_by(status="open").count()
                max_slots = int(
                    get_setting(s, "max_concurrent_slots", str(Config.MAX_CONCURRENT_SLOTS))
                    or Config.MAX_CONCURRENT_SLOTS
                )
                if open_count >= max_slots:
                    with suppress(Exception):
                        self.notifications.send_log(
                            "Brak wolnych slotów – blokuję nowe pozycje",
                            {"open": open_count, "max_slots": max_slots},
                            level="warning",
                        )
                    logger.warning(
                        "Sloty zajęte: %s/%s. Odrzucam nową pozycję.",
                        open_count,
                        max_slots,
                    )
                    return False
                return True
        except Exception as e:
            logger.warning("Slot check failed: %s", e)
            return True

    def process_signal_with_risk_management(self, signal: dict) -> bool:
        self.signals_processed += 1
        self.last_signal_raw = signal

        if self.is_paused:
            logger.warning("Bot jest zapauzowany. Ignoruję sygnał dla %s.", signal.get("symbol"))
            return False

        if not self._check_slots_or_block():
            return False

        symbol_raw = signal.get("symbol")
        norm = self.binance._normalize_symbol(symbol_raw or "")

        # Guard: jedna pozycja na symbol
        if Config.SINGLE_POSITION_PER_SYMBOL and self.binance.has_open_position(norm):
            logger.info("Odrzucam sygnał dla %s: pozycja już otwarta.", norm)
            if hasattr(self.notifications, "send_signal_decision"):
                self.notifications.send_signal_decision(
                    "rejected",
                    {
                        "symbol": symbol_raw,
                        "action": signal.get("action"),
                        "strength": signal.get("strength", 0),
                        "tier": signal.get("tier", "?"),
                        "reason": "Pozycja dla symbolu już otwarta",
                    },
                )
            return False

        lock = self._symbol_locks[norm]
        with lock:
            # jeszcze raz sprawdź (wyścig)
            if Config.SINGLE_POSITION_PER_SYMBOL and self.binance.has_open_position(norm):
                logger.info("[race] %s już otwarty – przerywam.", norm)
                return False

            try:
                decision = self.signal_intelligence.analyze_signal(signal)
                if decision.get("action") == "execute":
                    self.signals_accepted += 1
                    if hasattr(self.notifications, "send_signal_decision"):
                        self.notifications.send_signal_decision(
                            "accepted",
                            {
                                "symbol": decision["symbol"],
                                "action": decision["side"],
                                "strength": decision["metadata"].get("strength", 0),
                                "tier": decision.get("tier", "?"),
                                "reason": "Sygnał spełnia kryteria",
                            },
                        )
                    self._execute_intelligent_trade(decision)
                    return True
                else:
                    if hasattr(self.notifications, "send_signal_decision"):
                        self.notifications.send_signal_decision(
                            "rejected",
                            {
                                "symbol": signal.get("symbol"),
                                "action": signal.get("action"),
                                "strength": signal.get("strength", 0),
                                "tier": signal.get("tier", "?"),
                                "reason": decision.get("reason", "Nieznany powód"),
                            },
                        )
                    logger.info(
                        "Sygnał dla %s odrzucony. Powód: %s",
                        signal.get("symbol"),
                        decision.get("reason"),
                    )
                    return False
            except Exception as e:
                logger.error("Błąd podczas przetwarzania sygnału: %s", e, exc_info=True)
                return False

    def _execute_intelligent_trade(self, decision: dict[str, Any]):
        try:
            success, _, details = self.binance.place_order_with_multi_tp(
                signal=decision, is_dry_run=self.is_dry_run
            )
            if not success or not details:
                logger.error("Nie udało się złożyć zlecenia w BinanceHandler.")
                return

            self._save_enhanced_trade_to_db(decision, details, self.is_dry_run)

            with Session() as session:
                active_positions_count = session.query(Trade).filter_by(status="open").count()
                daily_pnl = self.analytics.get_pnl_report_for_period("daily").get("total_pnl", 0)

            self.notifications.send_trade_entry_notification(
                {
                    "symbol": details.get("symbol"),
                    "side": decision["side"].upper(),
                    "entry_price": details.get("entry_price"),
                    "size": details.get("final_size"),
                    "planned_size": details.get("planned_size"),
                    "requested_size": details.get("requested_size"),
                    "fill_ratio_planned": details.get("fill_ratio_planned"),
                    "fill_ratio_requested": details.get("fill_ratio_requested"),
                    "partial_reason": details.get("partial_reason"),
                    "leverage": decision["leverage"],
                    "risk_percent": decision["risk_percent"],
                    "signal_strength": decision["metadata"].get("strength", 0),
                    "sl_price": details.get("final_sl"),
                    "tp_details": details.get("tp_prices", []),
                    "mode": self.mode_manager.current_mode,
                    "tier": decision.get("tier", "Quick"),
                    "session": decision["metadata"].get("session", "N/A"),
                    "active_positions": active_positions_count,
                    "daily_pnl": daily_pnl,
                }
            )
        except Exception as e:
            logger.error("Błąd wykonania transakcji: %s", e, exc_info=True)

    def _save_enhanced_trade_to_db(
        self, decision: dict[str, Any], order_details: dict[str, Any], is_dry_run: bool
    ):
        metadata = decision.get("metadata", {})
        try:
            with Session() as session:
                final_tps = order_details.get("tp_prices") or []
                tp1 = final_tps[0]["price"] if len(final_tps) > 0 else None
                tp2 = final_tps[1]["price"] if len(final_tps) > 1 else None
                tp3 = final_tps[2]["price"] if len(final_tps) > 2 else None

                trade = Trade(
                    symbol=order_details.get("symbol"),
                    action=decision["side"],
                    entry_price=float(order_details.get("entry_price")),
                    quantity=float(order_details.get("final_size")),
                    stop_loss=(
                        float(order_details.get("final_sl"))
                        if order_details.get("final_sl") is not None
                        else None
                    ),
                    tp1=(float(tp1) if tp1 is not None else None),
                    tp2=(float(tp2) if tp2 is not None else None),
                    tp3=(float(tp3) if tp3 is not None else None),
                    leverage=int(decision["leverage"]),
                    status="open",
                    is_dry_run=is_dry_run,
                    signal_strength=float(metadata.get("strength", 0)),
                    tier=metadata.get("tier", "Quick"),
                    market_regime=metadata.get("market_regime", "NEUTRAL"),
                    market_condition=metadata.get("market_condition", "NORMAL"),
                    confidence_penalty=float(metadata.get("confidence_penalty", 0.0)),
                    liquidity_sweep=metadata.get("liquidity_sweep", False),
                    fresh_bos=metadata.get("fresh_bos", False),
                    raw_signal_data=metadata,
                    session=metadata.get("session", "Unknown"),
                    mode=self.mode_manager.current_mode,
                    pair_tier=int(metadata.get("pair_tier", 3)),
                    htf_trend=metadata.get("htf_trend", "neutral"),
                    mfi_value=float(metadata.get("mfi", 50.0)),
                    adx_value=float(metadata.get("adx", 0.0)),
                    btc_correlation=float(metadata.get("btc_correlation", 0.0)),
                    near_key_level=metadata.get("near_key_level", False),
                    volume_spike=metadata.get("volume_spike", False),
                    planned_size=float(order_details.get("planned_size") or 0),
                    requested_size=float(order_details.get("requested_size") or 0),
                    fill_ratio_planned=float(order_details.get("fill_ratio_planned") or 0),
                    fill_ratio_requested=float(order_details.get("fill_ratio_requested") or 0),
                    partial_reason=str(order_details.get("partial_reason") or ""),
                )
                session.add(trade)
                session.commit()
                logger.info("Zapisano nową transakcję w DB dla %s", trade.symbol)
        except Exception as e:
            logger.error("Nie udało się zapisać transakcji do DB: %s", e, exc_info=True)

    def get_active_positions(self) -> list:
        with Session() as session:
            return session.query(Trade).filter_by(status="open").all()

    def run_periodic_tasks(self):
        try:
            # Cleanup orphan TP/SL
            cancelled = 0
            with suppress(Exception):
                cancelled = self.binance.cleanup_stale_orders()
            if cancelled:
                logger.info("Porządkowanie: anulowano %s starych TP/SL bez pozycji.", cancelled)
                with suppress(Exception):
                    self.notifications.send_log(
                        "Cleanup: usunięto zlecenia TP/SL bez pozycji",
                        {"cancelled": cancelled, "okno": "run_periodic_tasks"},
                        level="info",
                    )

            # Reconcile DB ↔ giełda
            try:
                exchange_positions = self.binance.check_positions()
                exchange_open_symbols = {
                    str(p.get("symbol"))
                    for p in exchange_positions
                    if abs(float(p.get("positionAmt", 0.0))) != 0.0
                }
            except Exception as e:
                logger.warning("Reconcile: nie udało się pobrać pozycji z giełdy: %s", e)
                exchange_open_symbols = set()

            closed_by_sync: dict[str, int] = {}
            closed_trades_for_notify = []

            try:
                with Session() as s:
                    db_open_trades = s.query(Trade).filter_by(status="open").all()
                    for t in db_open_trades:
                        sym_norm = self.binance._normalize_symbol(t.symbol)

                        if sym_norm not in exchange_open_symbols:
                            # 1) Cena wyjścia – domyślnie markPrice
                            try:
                                mark = self.binance.client.futures_mark_price(symbol=sym_norm).get(
                                    "markPrice"
                                )
                                exit_price = float(mark)
                            except Exception:
                                exit_price = float(t.entry_price or 0.0)

                            # 2) Ustal reason (sl/tp) na podstawie ostatnich zleceń
                            exit_reason = "tp_or_sl"
                            with suppress(Exception):
                                orders = self.binance.client.futures_get_all_orders(
                                    symbol=sym_norm, limit=50
                                )
                                since_ms = (
                                    int((t.entry_time or datetime.utcnow()).timestamp() * 1000)
                                    - 60_000
                                )
                                filled = [
                                    o
                                    for o in orders
                                    if o.get("status") == "FILLED"
                                    and o.get("updateTime", 0) >= since_ms
                                ]
                                filled.sort(key=lambda o: o.get("updateTime", 0))
                                if filled:
                                    last = filled[-1]
                                    otype = (last.get("type") or "").upper()
                                    if otype in ("STOP", "STOP_MARKET"):
                                        exit_reason = "sl"
                                    elif otype in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
                                        exit_reason = "tp"
                                    ap = (
                                        last.get("avgPrice")
                                        or last.get("price")
                                        or last.get("stopPrice")
                                    )
                                    with suppress(Exception):
                                        exit_price = float(ap)

                            # 3) Zapis w DB
                            t.status = "closed"
                            t.exit_time = datetime.utcnow()
                            t.exit_price = exit_price

                            direction = 1 if t.action == "buy" else -1
                            qty = float(t.quantity or 0.0)
                            entry = float(t.entry_price or 0.0)
                            t.pnl = direction * (exit_price - entry) * qty
                            t.pnl_percent = (
                                direction * ((exit_price - entry) / max(entry, 1e-9)) * 100.0
                            )
                            t.exit_reason = exit_reason

                            s.commit()

                            closed_by_sync[sym_norm] = closed_by_sync.get(sym_norm, 0) + 1
                            closed_trades_for_notify.append(t)
            except Exception as e:
                logger.warning("Reconcile DB↔giełda: błąd przy zamykaniu sierot w DB: %s", e)

            # Notyfikacje
            for t in closed_trades_for_notify:
                with suppress(Exception):
                    self.notifications.send_trade_exit_notification(
                        {
                            "symbol": t.symbol,
                            "pnl": t.pnl,
                            "pnl_percent": t.pnl_percent,
                            "exit_reason": t.exit_reason,
                        }
                    )
                    self.notifications.send_log(
                        "Pozycja zamknięta",
                        {
                            "symbol": t.symbol,
                            "reason": t.exit_reason,
                            "pnl": f"${t.pnl:.2f}",
                            "pnl_percent": f"{t.pnl_percent:+.2f}%",
                        },
                        level=("warning" if t.exit_reason == "sl" else "info"),
                    )

            if closed_by_sync:
                with suppress(Exception):
                    self.notifications.send_log(
                        "DB sync: zamknięto brakujące pozycje",
                        {"zamkniete": ", ".join([f"{k}x{v}" for k, v in closed_by_sync.items()])},
                        level="warning",
                    )

            # Fallback: BE po TP1 (jeśli nie złapaliśmy eventu z user stream)
            active_positions = self.get_active_positions()
            if not active_positions:
                return

            for position in active_positions:
                symbol = position.symbol
                norm_symbol = self.binance._normalize_symbol(symbol)
                if exchange_open_symbols and norm_symbol not in exchange_open_symbols:
                    continue

                current_price = self.binance.get_last_price(symbol)
                if not current_price:
                    logger.warning("Nie udało się pobrać ceny dla %s, pomijam iterację.", symbol)
                    continue

                if not getattr(position, "tp1_hit", False) and position.tp1:
                    is_buy = position.action == "buy"
                    hit = (is_buy and current_price >= position.tp1) or (
                        not is_buy and current_price <= position.tp1
                    )
                    if hit:
                        success, message = self.binance.move_sl_to_break_even(symbol)
                        if success:
                            logger.info(
                                "TP1 trafiony dla %s! Przesunięto SL na Break-Even (%s).",
                                symbol,
                                position.entry_price,
                            )
                            with Session() as s:
                                pos_to_update = s.get(Trade, position.id)
                                if pos_to_update:
                                    pos_to_update.tp1_hit = True
                                    s.commit()
                            with suppress(Exception):
                                self.notifications.send_tp_hit_notification(
                                    {
                                        "symbol": symbol,
                                        "tp_level": "TP1",
                                        "price": position.tp1,
                                        "sl_moved_to_be": True,
                                    }
                                )
                            self._trailing_state[symbol] = {"last_sl_price": position.entry_price}
                        else:
                            logger.warning(
                                "TP1 dla %s, ale nie udało się przesunąć SL na BE: %s",
                                symbol,
                                message,
                            )
        except Exception as e:
            logger.error("Błąd w run_periodic_tasks: %s", e, exc_info=True)

    # ———— Public interface ————
    def pause_trading(self) -> str:
        self.is_paused = True
        logger.warning("Trading PAUSED")
        return "✅ Bot zapauzowany - nowe sygnały będą ignorowane."

    def resume_trading(self) -> str:
        self.is_paused = False
        logger.warning("Trading RESUMED")
        return "✅ Bot wznowiony - przyjmuje sygnały."

    def toggle_ml_decision(self, enabled: bool) -> str:
        self.use_ml_for_decision = enabled
        with Session() as session:
            set_setting(session, "use_ml_for_decision", str(enabled))
        status = "WŁĄCZONE" if enabled else "WYŁĄCZONE"
        logger.warning("ML decision making: %s", status)
        return f"✅ ML decision making jest teraz {status}."

    def toggle_intelligent_mode_switcher(self, enabled: bool) -> str:
        self.intelligent_mode_switcher_enabled = enabled
        with Session() as session:
            set_setting(session, "intelligent_mode_switcher_enabled", str(enabled))
        status = "WŁĄCZONE" if enabled else "WYŁĄCZONE"
        logger.warning("Intelligent Mode Switcher: %s", status)
        return f"✅ Automatyczne przełączanie trybów jest teraz {status}."

    def get_last_signal(self) -> dict[str, Any]:
        if hasattr(self, "last_signal_raw") and self.last_signal_raw:
            return self.last_signal_raw
        return {"message": "Brak ostatniego sygnału w pamięci."}

    def set_dry_run(self, enabled: bool) -> str:
        self.is_dry_run = enabled
        with Session() as session:
            set_setting(session, "dry_run", str(enabled))
        status = "WŁĄCZONY" if enabled else "WYŁĄCZONY"
        logger.warning("Dry Run: %s", status)
        return f"✅ Tryb symulacji jest teraz {status}."

    def toggle_dryrun(self) -> str:
        return self.set_dry_run(not self.is_dry_run)

    def set_risk(self, percent: float) -> str:
        if not 0.1 <= percent <= 5.0:
            return "Błąd: Ryzyko musi być w zakresie od 0.1% do 5.0%."
        self.runtime_risk = percent / 100.0
        with Session() as session:
            set_setting(session, "risk_per_trade", str(self.runtime_risk))
        return f"✅ Ryzyko na transakcję ustawione na {self.runtime_risk:.2%}"

    def set_max_daily_loss(self, amount: float) -> str:
        if amount > 0:
            return "Błąd: Maksymalna strata musi być wartością ujemną (np. -500)."
        self.max_daily_loss = amount
        with Session() as session:
            set_setting(session, "max_daily_loss", str(self.max_daily_loss))
        return f"✅ Maksymalna dzienna strata ustawiona na ${self.max_daily_loss}"

    def set_mode(self, mode: str) -> str:
        capitalized_mode = mode.capitalize()
        if capitalized_mode not in self.mode_manager.mode_configs:
            return "Błąd: Nieznany tryb '{mode}'. " "Dostępne: Conservative, Balanced, Aggressive."
        self.current_mode = capitalized_mode
        self.mode_manager.current_mode = capitalized_mode
        return f"✅ Tryb pracy zmieniony na: {self.current_mode}"

    def toggle_leverage_override(self) -> str:
        self.leverage_override_enabled = not self.leverage_override_enabled
        with Session() as session:
            set_setting(session, "leverage_override", str(self.leverage_override_enabled))
        status = "WŁĄCZONE" if self.leverage_override_enabled else "WYŁĄCZONE"
        return f"✅ Nadpisywanie dźwigni przez bota jest teraz {status}."

    def close_position(self, symbol: str) -> str:
        normalized_symbol = self.binance._normalize_symbol(symbol)
        result = self.binance.close_all_positions(specific_symbol=normalized_symbol)
        if result.get(normalized_symbol):
            return f"✅ Zamknięto pozycję dla {normalized_symbol}."
        else:
            return f"❌ Nie udało się zamknąć pozycji dla {normalized_symbol}."

    def get_circuit_status(self) -> dict:
        daily_pnl = self.analytics.get_pnl_report_for_period("daily").get("total_pnl", 0.0)
        return {
            "override": self.circuit_breaker_override,
            "tripped": self.circuit_breaker_tripped,
            "is_paused": self.is_paused or self.circuit_breaker_tripped,
            "max_daily_loss": self.max_daily_loss,
            "daily_pnl": daily_pnl,
        }

    def set_circuit_override(self, enabled: bool) -> str:
        self.circuit_breaker_override = enabled
        status = "WŁĄCZONY" if enabled else "WYŁĄCZONY"
        return f"✅ Override bezpiecznika jest teraz {status}."

    def reset_circuit_breaker(self) -> str:
        self.circuit_breaker_tripped = False
        self.is_paused = False
        return "✅ Bezpiecznik dziennej straty został zresetowany."

    def run(self):
        if not getattr(self.binance, "client", None):
            logger.critical("Brak połączenia z Binance - bot nie może wystartować.")
            return

        # Start Futures User Data Stream raz
        if self.user_stream is None:
            try:
                from futures_user_stream import FuturesUserStream

                self.user_stream = FuturesUserStream(self)
                self.user_stream.start()
                logger.info("Futures User Data Stream uruchomiony.")
            except Exception as e:
                logger.warning("Nie udało się uruchomić Futures User Data Stream: %s", e)

        while self.is_running:
            try:
                self.run_periodic_tasks()
                time.sleep(5)
            except KeyboardInterrupt:
                self.is_running = False
                logger.info("Zatrzymywanie bota...")
                break
            except Exception as e:
                logger.error("Błąd w głównej pętli bota: %s", e, exc_info=True)
                time.sleep(20)

        # Porządek przy wyjściu
        with suppress(Exception):
            if self.user_stream:
                self.user_stream.stop()

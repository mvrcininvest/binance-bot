"""
Binance Futures handler — wersja v9.1
- Dedykowana metoda do ustawiania dźwigni i typu marży
- Precyzyjne tagowanie zleceń SL/TP za pomocą newClientOrderId
- Ulepszone logowanie i obsługa błędów
"""

import logging
import math
import time
from contextlib import suppress
from decimal import Decimal
from typing import Any, Dict, List

from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import Config
from database import Session, get_setting

logger = logging.getLogger("binance_handler")


def _parse_float_list(s: str | None, default: str) -> list[float]:
    if s is None:
        s = default
    out: list[float] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        with suppress(Exception):
            out.append(float(p))
    return out


def _parse_split_dict(s: str | None, default: str = "0.5,0.3,0.2") -> dict[str, float]:
    vals = _parse_float_list(s, default)
    if len(vals) != 3:
        vals = [0.5, 0.3, 0.2]
    vals = [max(0.0, v) for v in vals]
    total = sum(vals) or 1.0
    vals = [v / total for v in vals]
    return {"tp1": vals[0], "tp2": vals[1], "tp3": vals[2]}


class BinanceHandler:
    def __init__(self) -> None:
        self.client: Client | None = None
        self.exchange_info: dict[str, Any] | None = None
        self.last_exchange_info_update: float = 0.0
        self.hedge_mode: bool = False

        try:
            if not Config.BINANCE_API_KEY or not Config.BINANCE_API_SECRET:
                raise ValueError("Brak kluczy API Binance w pliku .env")

            self.client = Client(
                api_key=(Config.BINANCE_API_KEY or "").strip(),
                api_secret=(Config.BINANCE_API_SECRET or "").strip(),
                testnet=Config.BINANCE_TESTNET,
            )
            self.client.futures_account()
            logger.info("Pomyślnie połączono z Binance Futures (testnet=%s)", Config.BINANCE_TESTNET)
            self._fetch_and_cache_exchange_info()
            self._check_position_mode()
        except BinanceAPIException as e:
            logger.error("BŁĄD Binance API podczas inicjalizacji: %s", e.message)
            self.client = None
        except Exception as e:
            logger.critical("KRYTYCZNY BŁĄD połączenia z Binance: %s", e, exc_info=True)
            self.client = None

    # ==== Exchange info / utils ====
    def _fetch_and_cache_exchange_info(self) -> None:
    if not self.client:
    return
    try:
    info = self.client.futures_exchange_info()
    self.exchange_info = {item["symbol"]: item for item in info.get("symbols", [])}
    self.last_exchange_info_update = time.time()
    logger.info(
    "Pobrano Exchange Info dla %d symboli.",
    len(self.exchange_info or {}),
    )
    except Exception as e:
    logger.error("Nie udało się pobrać Exchange Info: %s", e)

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
    if not self.exchange_info or (time.time() - self.last_exchange_info_update > 3600):
    self._fetch_and_cache_exchange_info()
    s = self._normalize_symbol(symbol)
    return (self.exchange_info or {}).get(s)

    def _normalize_symbol(self, symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if s.endswith(".P"):
    s = s[:-2]
    return s.replace("-", "").replace("/", "")

    def _get_filter_value(self, symbol: str, filter_type: str, param: str) -> Any | None:
    info = self.get_symbol_info(symbol)
    if info and "filters" in info:
    for f in info["filters"]:
    if f.get("filterType") == filter_type:
    return f.get(param)
    return None

    def _get_precision(self, symbol: str) -> tuple[int, int]:
    info = self.get_symbol_info(symbol)
    if not info:
    return 2, 3
    return int(info.get("pricePrecision", 2)), int(info.get("quantityPrecision", 3))

    def _round_value(self, value: float, precision: int) -> float:
    if precision <= 0:
    return math.floor(value)
    factor = 10**precision
    return math.floor(value * factor) / factor

    def _adjust_price_to_tick_size(self, symbol: str, price: float) -> float:
    tick_size_str = self._get_filter_value(symbol, "PRICE_FILTER", "tickSize")
    if tick_size_str:
    tick_size = Decimal(str(tick_size_str))
    price_decimal = Decimal(str(price))
    rounded_price = (price_decimal // tick_size) * tick_size
    return float(rounded_price)
    price_precision, _ = self._get_precision(symbol)
    return self._round_value(price, price_precision)

    def _check_position_mode(self) -> None:
    if not self.client:
    return
    try:
    mode = self.client.futures_get_position_mode()
    self.hedge_mode = bool(mode.get("dualSidePosition", False))
    logger.info("Tryb pozycji: %s", "Hedge" if self.hedge_mode else "One-way")
    except Exception:
    self.hedge_mode = False
    logger.warning("Nie udało się pobrać trybu pozycji; ustawiam One-way.")

    # ==== Dynamic settings getters ====
    def _get_bool_setting(self, key: str, default_bool: bool) -> bool:
    try:
    with Session() as s:
    v = get_setting(s, key, "1" if default_bool else "0")
    return str(v).lower() in ("1", "true", "on", "yes", "y", "t")
    except Exception:
    return default_bool

    def _get_float_setting(self, key: str, default_value: float) -> float:
    try:
    with Session() as s:
    v = get_setting(s, key, str(default_value))
    return float(v)
    except Exception:
    return default_value

    def _get_str_setting(self, key: str, default_value: str) -> str:
    try:
    with Session() as s:
    v = get_setting(s, key, default_value)
    return str(v)
    except Exception:
    return default_value

    # ==== Account helpers ====
    def get_balance(self) -> dict[str, float]:
    if not self.client:
    return {"total": 0.0, "available": 0.0}
    try:
    account = self.client.futures_account()
    for asset in account.get("assets", []):
    if asset.get("asset") == "USDT":
    return {
    "total": float(asset.get("walletBalance", 0.0)),
    "available": float(asset.get("availableBalance", 0.0)),
    }
    return {"total": 0.0, "available": 0.0}
    except Exception as e:
    logger.error("Błąd podczas pobierania salda: %s", e)
    return {"total": 0.0, "available": 0.0}

    def get_last_price(self, symbol: str) -> float | None:
    if not self.client:
    return None
    try:
    s = self._normalize_symbol(symbol)
    ticker = self.client.futures_symbol_ticker(symbol=s)
    return float(ticker["price"])
    except Exception as e:
    logger.error("Błąd pobierania ceny dla %s: %s", symbol, e)
    return None

    def has_open_position(self, symbol: str) -> bool:
    if not self.client:
    return False
    try:
    s = self._normalize_symbol(symbol)
    pos = self.client.futures_position_information(symbol=s)
    if not pos:
    return False
    amt = float(pos[0].get("positionAmt", 0.0))
    return abs(amt) > 0.0
    except Exception:
    return False

    # ==== Leverage brackets / notional cap ====
    def _get_max_notional_for_leverage(self, symbol: str, leverage: int) -> float | None:
    if not self.client:
    return None
    try:
    s = self._normalize_symbol(symbol)
    # Some SDK versions require symbol=, some return all symbols list
    try:
    data = self.client.futures_leverage_bracket(symbol=s)
    except TypeError:
    data = self.client.futures_leverage_bracket()

    # Normalize payload to brackets list
    if isinstance(data, dict) and "brackets" in data:
    entries = [data]
    elif isinstance(data, list):
    entries = data
    else:
    raise TypeError(f"Unexpected bracket payload type: {type(data)}")

    # Extract brackets for symbol
    if entries and isinstance(entries[0], dict) and "symbol" in entries[0]:
    entry = next(
    (e for e in entries if str(e.get("symbol", "")).upper() == s),
    entries[0],
    )
    brackets = entry.get("brackets") or []
    else:
    # some implementations return list of brackets directly
    brackets = entries  # type: ignore[assignment]

    if not brackets:
    return None

    # Select bracket that supports the requested leverage
    chosen: dict[str, Any] | None = None
    for b in brackets:
    try:
    maxLev = int(b.get("initialLeverage", b.get("maxLeverage", 0)) or 0)
    except Exception:
    maxLev = 0
    if int(leverage) <= maxLev:
    chosen = b
    break
    if not chosen:
    chosen = brackets[-1]

    cap_raw = chosen.get("notionalCap", chosen.get("maxNotional", 0))
    cap = float(cap_raw or 0)
    return cap if cap > 0 else None
    except Exception as e:
    logger.warning("Nie udało się pobrać leverage brackets dla %s: %s", symbol, e)
    return None

    def _get_current_position_notional(self, symbol: str) -> float:
    if not self.client:
    return 0.0
    try:
    s = self._normalize_symbol(symbol)
    pos = self.client.futures_position_information(symbol=s)
    if not pos:
    return 0.0
    qty = abs(float(pos[0].get("positionAmt", 0.0)))
    if qty == 0:
    return 0.0
    mark = float(self.client.futures_mark_price(symbol=s).get("markPrice", 0.0))
    return qty * mark
    except Exception:
    return 0.0

    def _compute_qty_cap(self, symbol: str, leverage: int, last_price: float) -> float:
    _, qty_precision = self._get_precision(symbol)
    available = max(0.0, self.get_balance().get("available", 0.0))

    # Limit from available margin
    allowed_by_margin = (available * leverage) / max(1e-9, last_price)
    allowed_by_margin = self._round_value(allowed_by_margin, qty_precision)

    # Limit from leverage bracket notional
    cap_notional = self._get_max_notional_for_leverage(symbol, leverage) or float("inf")
    try:
    pos_info = self.client.futures_position_information(
    symbol=self._normalize_symbol(symbol)
    )
    pos_amt = abs(float(pos_info[0].get("positionAmt", 0)))
    except Exception:
    pos_amt = 0.0

    allowed_by_notional = (
    float("inf")
    if cap_notional == float("inf")
    else max(0.0, (cap_notional / last_price) - pos_amt)
    )
    allowed_by_notional = self._round_value(allowed_by_notional, qty_precision)

    return max(0.0, min(allowed_by_margin, allowed_by_notional))

    # ==== Signal normalization ====
    def _canonicalize_signal(self, raw: dict[str, Any]) -> dict[str, Any]:
    def first_of(d: dict[str, Any], keys: list[str]) -> Any | None:
    for k in keys:
    if k in d and d[k] is not None:
    return d[k]
    return None

    symbol = self._normalize_symbol(str(raw.get("symbol", "")))

    side_raw = (raw.get("side") or raw.get("action") or "").lower()
    if side_raw.startswith("emergency_"):
    side_raw = side_raw.replace("emergency_", "")
    side = "buy" if side_raw == "buy" else "sell"

    price = first_of(raw, ["entry_price", "price"])
    stop_loss = first_of(raw, ["stop_loss", "sl", "sl_price"])
    leverage = int(first_of(raw, ["leverage"]) or 10)
    risk_percent = float(first_of(raw, ["risk_percent"]) or Config.RISK_PER_TRADE)

    # TPs
    tp_block = raw.get("take_profits") if isinstance(raw.get("take_profits"), dict) else {}
    tp1 = tp_block.get("tp1")
    tp2 = tp_block.get("tp2")
    tp3 = tp_block.get("tp3")
    if tp1 is None:
    tp1 = first_of(raw, ["tp1"])
    if tp2 is None:
    tp2 = first_of(raw, ["tp2"])
    if tp3 is None:
    tp3 = first_of(raw, ["tp3"])

    tp_distribution = raw.get("tp_distribution") or Config.TP_DISTRIBUTION

    return {
    "symbol": symbol,
    "side": side,
    "price": float(price) if price is not None else None,
    "stop_loss": float(stop_loss) if stop_loss is not None else None,
    "tp1": float(tp1) if tp1 is not None else None,
    "tp2": float(tp2) if tp2 is not None else None,
    "tp3": float(tp3) if tp3 is not None else None,
    "risk_percent": float(risk_percent),
    "leverage": leverage,
    "tp_distribution": tp_distribution,
    }

    def ensure_symbol_leverage_and_margin(
    self, symbol: str, leverage: int, margin_type: str = "ISOLATED"
    ):
    """Ustawia dźwignię i typ marginesu z wyraźnym logowaniem."""
    if not self.client:
    return

    try:
    if margin_type.upper() in ("ISOLATED", "CROSSED"):
    self.client.futures_change_margin_type(
    symbol=symbol, marginType=margin_type.upper()
    )
    except BinanceAPIException as e:
    if "No need to change margin type" not in e.message:
    logger.warning(
    f"Nie udało się zmienić typu marży na {margin_type} dla {symbol}: {e.message}"
    )

    try:
    self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
    logger.info(
    "Leverage set to %sx for %s (%s) przed zleceniem",
    leverage,
    symbol,
    margin_type,
    )
    except BinanceAPIException as e:
    logger.warning(
    "Nie udało się zmienić dźwigni na %s dla %s: %s",
    leverage,
    symbol,
    e.message,
    )

    def place_order_with_multi_tp(
    self, signal: dict[str, Any], is_dry_run: bool = False
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
    # ... (logika do pobierania symbol, side, etc. bez zmian) ...
    canon = self._canonicalize_signal(signal)
    symbol = canon["symbol"]
    side = "BUY" if canon["side"] == "buy" else "SELL"
    default_status = (False, None, None)

    if is_dry_run:
    # ... (logika dry run bez zmian) ...
    return True, {"dry_run": True}, {}

    if not self.client:
    return default_status

    leverage = int(signal.get("leverage", 10))
    margin_type = self._get_str_setting("default_margin_type", Config.DEFAULT_MARGIN_TYPE)
    self.ensure_symbol_leverage_and_margin(symbol, leverage, margin_type)

    # Confirm actual leverage as used by exchange
    try:
    pos = self.client.futures_position_information(symbol=symbol)
    cur_lev = int(float(pos[0].get("leverage", leverage)))
    if cur_lev != leverage:
    logger.warning(
    "Żądano dźwigni %sx, ale aktywna to %sx. Dostosowuję obliczenia.",
    leverage,
    cur_lev,
    )
    leverage = cur_lev
    except Exception:
    pass

    entry_price_signal = canon.get("price")
    use_alert_lvls = self._get_bool_setting("use_alert_levels", Config.USE_ALERT_LEVELS)
    risk_percent = canon.get("risk_percent", Config.RISK_PER_TRADE)
    trade_id = int(time.time() * 1000)

    # Price for sizing fallback
    last_px_now = self.get_last_price(symbol) or entry_price_signal
    if last_px_now is None or last_px_now <= 0:
    logger.error("Brak ceny dla %s – nie mogę policzyć rozmiaru.", symbol)
    return default_status

    # Ideal position sizing
    ideal_pos_size: float | None = None
    if (
    use_alert_lvls
    and (canon.get("stop_loss") is not None)
    and (entry_price_signal is not None)
    ):
    # risk-based sizing
    sl_distance = abs(float(entry_price_signal) - float(canon["stop_loss"]))
    if sl_distance > 0:
    available = self.get_balance()["available"] or 10_000.0
    risk_amount = available * float(risk_percent)
    ideal_pos_size = risk_amount / sl_distance

    if ideal_pos_size is None:
    # margin-budget based sizing (per trade fraction)
    available = self.get_balance()["available"] or 10_000.0
    per_trade_fraction = self._get_float_setting(
    "margin_per_trade_fraction",
    (
    Config.MARGIN_PER_TRADE_FRACTION
    if Config.MARGIN_PER_TRADE_FRACTION > 0
    else (1.0 / max(1, Config.MARGIN_SLOTS))
    ),
    )
    safety = self._get_float_setting(
    "margin_safety_buffer", Config.MARGIN_SAFETY_BUFFER
    )
    alloc_margin = available * per_trade_fraction * safety
    ideal_pos_size = (alloc_margin * leverage) / last_px_now

    # Adjust by balance and bracket caps
    final_pos_size = float(ideal_pos_size)
    last_px = last_px_now
    cap_qty = self._compute_qty_cap(symbol, leverage, last_px)
    if cap_qty <= 0:
    logger.error("Brak środków na minimalną pozycję.")
    return default_status
    if final_pos_size > cap_qty:
    logger.warning(
    "Przycinam wielkość pozycji: %.6f -> %.6f", final_pos_size, cap_qty
    )
    final_pos_size = cap_qty

    # Balance safety: ensure margin requirement fits (with safety buffer)
    available_balance = self.get_balance()["available"] or 0.0
    margin_required = (final_pos_size * last_px) / max(leverage, 1)
    if margin_required > available_balance:
    max_possible_margin = available_balance * self._get_float_setting(
    "margin_safety_buffer", Config.MARGIN_SAFETY_BUFFER
    )
    max_pos_by_balance = (max_possible_margin * leverage) / last_px
    if max_pos_by_balance <= 0:
    logger.error("Brak środków na minimalną pozycję.")
    return default_status
    logger.warning(
    "Dopasowuję wielkość pozycji pod saldo: %.6f -> %.6f",
    final_pos_size,
    max_pos_by_balance,
    )
    final_pos_size = max_pos_by_balance

    # Bracket cap against current used notional
    max_notional = self._get_max_notional_for_leverage(symbol, leverage)
    if max_notional:
    used_notional = self._get_current_position_notional(symbol)
    allowed_left = max(0.0, max_notional - used_notional)
    max_size_by_bracket = allowed_left / max(last_px, 1e-9)
    if max_size_by_bracket <= 0:
    logger.error(
    "Przekroczony limit notional dla %s przy %sx (pozost: $%.2f). "
    "Odrzucam.",
    symbol,
    leverage,
    allowed_left,
    )
    return default_status
    if final_pos_size > max_size_by_bracket:
    logger.info(
    "Przycinam wg bracketu %s: %.6f -> %.6f (left=$%.2f)",
    symbol,
    final_pos_size,
    max_size_by_bracket,
    allowed_left,
    )
    final_pos_size = max_size_by_bracket

    # Diagnostics before placement
    avail = available_balance
    logger.info(
    "[DIAG] avail=%.2f last=%.2f lev=%s ideal=%.6f cap=%.6f final=%.6f",
    avail,
    last_px,
    leverage,
    ideal_pos_size,
    cap_qty,
    final_pos_size,
    )

    # LOT_SIZE / MARKET_LOT_SIZE constraints
    max_order_qty = float(
    self._get_filter_value(symbol, "MARKET_LOT_SIZE", "maxQty") or 0
    ) or float(self._get_filter_value(symbol, "LOT_SIZE", "maxQty") or 10000.0)
    _, qty_precision = self._get_precision(symbol)
    cap_order_qty = max_order_qty  # will shrink on failures

    requested_qty = max(0.0, float(final_pos_size))
    total_executed_qty = 0.0
    last_order_response: dict[str, Any] | None = None
    logger.info(
    "Rozpoczynam otwieranie pozycji %.6f %s (%s).",
    requested_qty,
    symbol,
    side,
    )

    # Single meaningful MARKET with shrink-and-retry (no spamming)
    max_attempts = 6
    order_qty = self._round_value(
    min(requested_qty, max_order_qty, cap_order_qty), qty_precision
    )
    attempt = 0
    while attempt < max_attempts and order_qty > 0:
    attempt += 1
    try:
    logger.info(
    "[Try %d/%d] MARKET %s qty=%s %s",
    attempt,
    max_attempts,
    side,
    order_qty,
    symbol,
    )
    last_order_response = self.client.futures_create_order(
    symbol=symbol,
    side=side,
    type="MARKET",
    quantity=order_qty,
    newOrderRespType="RESULT",
    )
    except BinanceAPIException as e:
    msg = (e.message or "").lower()
    # Margin or allowable position exceeded — shrink and retry
    if ("margin is insufficient" in msg) or ("maximum allowable position" in msg):
    shrink = self._round_value(order_qty * 0.75, qty_precision)
    if 0 < shrink < order_qty:
    logger.warning(
    "Limit marginesu/allowable dla qty=%.6f; ponawiam z %.6f",
    order_qty,
    shrink,
    )
    order_qty = shrink
    time.sleep(0.25)
    continue
    logger.error("Nie mogę dalej zmniejszać qty (%.6f).", order_qty)
    return default_status
    # Percent price protection — try LIMIT IOC once per attempt
    if e.code == -4131:
    logger.warning("MARKET odrzucony (PERCENT_PRICE). Próbuję LIMIT IOC.")
    try:
    last_px_live = float(
    self.client.futures_symbol_ticker(symbol=symbol)["price"]
    )
    limit_price = last_px_live * (1.001 if side == "BUY" else 0.999)
    limit_price = self._adjust_price_to_tick_size(
    symbol, limit_price
    )
    last_order_response = self.client.futures_create_order(
    symbol=symbol,
    side=side,
    type="LIMIT",
    quantity=order_qty,
    price=limit_price,
    timeInForce="IOC",
    newOrderRespType="RESULT",
    )
    except Exception as e2:
    logger.error("LIMIT IOC nieudane: %s", e2)
    return default_status
    else:
    logger.error("Zlecenie odrzucone przez Binance: %s. Przerywam.", e.message)
    return default_status

    # Read executed quantity (supports both executedQty and cumQty)
    executed_qty = 0.0
    with suppress(Exception):
    executed_qty = float(last_order_response.get("executedQty", 0) or 0)
    if executed_qty == 0.0:
    with suppress(Exception):
    executed_qty = float(last_order_response.get("cumQty", 0) or 0)

    total_executed_qty += executed_qty
    if executed_qty > 0:
    logger.info(
    "Wykonano wejście: executedQty=%.6f/%s", executed_qty, symbol
    )
    break

    # If not executed at all, shrink and try again
    shrink = self._round_value(order_qty * 0.75, qty_precision)
    if 0 < shrink < order_qty:
    logger.warning(
    "Brak wykonania. Zmniejszam qty %.6f -> %.6f i ponawiam.",
    order_qty,
    shrink,
    )
    order_qty = shrink
    time.sleep(0.25)
    continue
    else:
    logger.error(
    "Brak wykonania i nie można dalej zmniejszać qty (%.6f).",
    order_qty,
    )
    return default_status

    if total_executed_qty <= 0:
    logger.error("Nie udało się faktycznie otworzyć pozycji (executedQty==0).")
    return default_status

    # Read actual entry price from position
    time.sleep(1.0)
    try:
    pos_info = self.client.futures_position_information(symbol=symbol)
    actual_entry_price = float(pos_info[0].get("entryPrice", last_px))
    except Exception:
    actual_entry_price = float(last_px)

    # Set SL/TPs
    setup_status, tp_prices, final_sl, client_tags = self._setup_sl_tp_after_entry(
    symbol=symbol,
    side=side,
    total_quantity=total_executed_qty,
    canon_signal=canon,
    actual_entry_price=actual_entry_price,
    use_alert_lvls=self._get_bool_setting(
    "use_alert_levels", Config.USE_ALERT_LEVELS
    ),
    rr_levels=_parse_float_list(
    self._get_str_setting("tp_rr_levels", ""), default="1.0,1.5,2.0"
    ),
    tp_split=_parse_split_dict(
    self._get_str_setting("tp_split", ""), default="0.5,0.3,0.2"
    ),
    sl_margin_pct=self._get_float_setting(
    "sl_margin_pct", Config.SL_MARGIN_PCT
    ),
    trade_id=trade_id,
    )

    details = {
    "setup_status": setup_status,
    "final_size": total_executed_qty,
    "symbol": symbol,
    "entry_price": actual_entry_price,
    "tp_prices": tp_prices,
    "final_sl": final_sl,
    "client_tags": client_tags,
    "trade_id": trade_id,
    "metadata": signal,  # Przekazujemy oryginalny sygnał dalej
    }
    return True, last_order_response, details

    def _setup_sl_tp_after_entry(
    self,
    symbol: str,
    side: str,
    total_quantity: float,
    canon_signal: dict[str, Any],
    actual_entry_price: float,
    use_alert_lvls: bool,
    rr_levels: list[float],
    tp_split: dict[str, float],
    sl_margin_pct: float,
    trade_id: int,
    ) -> tuple[dict[str, bool], list[dict[str, float]], float | None, dict[str, str]]:
    if not self.client:
    return {}, [], None, {}

    sl_opposite_side = "SELL" if side == "BUY" else "BUY"
    _, qty_precision = self._get_precision(symbol)
    status_report = {
    "sl_set": False,
    "tp1_set": False,
    "tp2_set": False,
    "tp3_set": False,
    }
    tp_prices_list: list[dict[str, float]] = []
    client_tags = {}
    final_sl: float | None = None

    # minimal gap to avoid immediate triggers
    try:
    last_price = float(self.client.futures_symbol_ticker(symbol=symbol)["price"])
    except Exception:
    last_price = actual_entry_price
    tick_size_str = (
    self._get_filter_value(symbol, "PRICE_FILTER", "tickSize") or "0.01"
    )
    tick = float(tick_size_str)
    min_gap = max(tick * 2.0, actual_entry_price * 0.0005)

    # Final SL
    if use_alert_lvls and (canon_signal.get("stop_loss") is not None):
    sl_price = float(canon_signal["stop_loss"])
    sl_price = self._adjust_price_to_tick_size(symbol, sl_price)
    # avoid immediate trigger
    if side == "BUY":
    min_allowed = min(last_price - min_gap, actual_entry_price - tick)
    if sl_price >= min_allowed:
    sl_price = self._adjust_price_to_tick_size(symbol, min_allowed)
    else:
    max_allowed = max(last_price + min_gap, actual_entry_price + tick)
    if sl_price <= max_allowed:
    sl_price = self._adjust_price_to_tick_size(symbol, max_allowed)
    final_sl = sl_price
    else:
    # SL as % of used margin
    used_margin = (total_quantity * actual_entry_price) / max(
    int(canon_signal.get("leverage", 10)) or 1, 1
    )
    loss_target = max(0.0, float(sl_margin_pct)) * used_margin
    per_unit = loss_target / max(total_quantity, 1e-9)
    if side == "BUY":
    final_sl = self._adjust_price_to_tick_size(
    symbol, actual_entry_price - per_unit
    )
    final_sl = min(final_sl, actual_entry_price - tick)
    final_sl = min(final_sl, last_price - min_gap)
    else:
    final_sl = self._adjust_price_to_tick_size(
    symbol, actual_entry_price + per_unit
    )
    final_sl = max(final_sl, actual_entry_price + tick)
    final_sl = max(final_sl, last_price + min_gap)

    # Validate SL relative to entry
    sl_is_valid = (side == "BUY" and final_sl < actual_entry_price) or (
    side == "SELL" and final_sl > actual_entry_price
    )
    if sl_is_valid:
    try:
    sl_client_id = f"BOT_{trade_id}_SL"
    client_tags["sl"] = sl_client_id
    self.client.futures_create_order(
    symbol=symbol,
    side=sl_opposite_side,
    type="STOP_MARKET",
    stopPrice=final_sl,
    closePosition=True,
    newClientOrderId=sl_client_id,
    )
    logger.info(
    "Ustawiono SL dla %s na %s (tag: %s)",
    symbol,
    final_sl,
    sl_client_id,
    )
    status_report["sl_set"] = True
    except Exception as e:
    logger.error("Nie udało się ustawić SL dla %s: %s", symbol, e)

    # TP levels
    desired_tps: dict[str, float | None] = {"tp1": None, "tp2": None, "tp3": None}
    if use_alert_lvls and (
    canon_signal.get("tp1")
    or canon_signal.get("tp2")
    or canon_signal.get("tp3")
    ):
    desired_tps["tp1"] = canon_signal.get("tp1")
    desired_tps["tp2"] = canon_signal.get("tp2")
    desired_tps["tp3"] = canon_signal.get("tp3")
    else:
    # RR-based from final SL
    if final_sl is not None:
    risk_per_unit = (
    (actual_entry_price - final_sl)
    if side == "BUY"
    else (final_sl - actual_entry_price)
    )
    for i, rr in enumerate(rr_levels[:3], 1):
    if side == "BUY":
    desired_tps[f"tp{i}"] = (
    actual_entry_price + float(rr) * risk_per_unit
    )
    else:
    desired_tps[f"tp{i}"] = (
    actual_entry_price - float(rr) * risk_per_unit
    )

    # Place TP1..TP3 (reduceOnly)
    remaining_qty_for_tp = total_quantity
    ordered_levels = [
    ("tp1", tp_split.get("tp1", 0.5)),
    ("tp2", tp_split.get("tp2", 0.3)),
    ("tp3", tp_split.get("tp3", 0.2)),
    ]
    for level_name, qty_perc in ordered_levels:
    tp_raw = desired_tps.get(level_name)
    if tp_raw is None or remaining_qty_for_tp <= 0 or (qty_perc or 0) <= 0:
    continue

    tp_qty = self._round_value(total_quantity * float(qty_perc), qty_precision)
    if tp_qty <= 0:
    continue
    tp_qty = min(tp_qty, remaining_qty_for_tp)

    tp_price = self._adjust_price_to_tick_size(symbol, float(tp_raw))
    # avoid immediate trigger
    if side == "BUY":
    min_tp = last_price + min_gap
    if tp_price <= min_tp:
    tp_price = self._adjust_price_to_tick_size(symbol, min_tp)
    else:
    max_tp = last_price - min_gap
    if tp_price >= max_tp:
    tp_price = self._adjust_price_to_tick_size(symbol, max_tp)

    try:
    tp_client_id = f"BOT_{trade_id}_{level_name.upper()}"
    client_tags[level_name] = tp_client_id
    self.client.futures_create_order(
    symbol=symbol,
    side=sl_opposite_side,
    type="TAKE_PROFIT_MARKET",
    quantity=tp_qty,
    stopPrice=tp_price,
    reduceOnly=True,
    newClientOrderId=tp_client_id,
    )
    logger.info(
    "Ustawiono %s dla %s na %s (qty=%s, tag=%s).",
    level_name.upper(),
    symbol,
    tp_price,
    tp_qty,
    tp_client_id,
    )
    status_report[f"{level_name}_set"] = True
    remaining_qty_for_tp -= tp_qty
    tp_prices_list.append(
    {"level": level_name.upper(), "price": tp_price, "qty": tp_qty}
    )
    except Exception as e:
    logger.error(
    "Nie udało się ustawić %s dla %s: %s",
    level_name.upper(),
    symbol,
    e,
    )

    return status_report, tp_prices_list, final_sl, client_tags

    # ==== Housekeeping ====
    def check_positions(self) -> list[dict[str, Any]]:
    if not self.client:
    return []
    try:
    positions = self.client.futures_position_information()
    return [p for p in positions if abs(float(p.get("positionAmt", 0))) != 0]
    except Exception as e:
    logger.error("Błąd sprawdzania pozycji: %s", e)
    return []

    def cleanup_stale_orders(self) -> int:
    if not self.client:
    return 0
    try:
    open_orders = self.client.futures_get_open_orders()
    if not open_orders:
    return 0
    symbols_with_position = set()
    with suppress(Exception):
    for p in self.client.futures_position_information():
    if abs(float(p.get("positionAmt", 0))) != 0:
    symbols_with_position.add(p.get("symbol"))

    cancelled = 0
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for o in open_orders:
    s = o.get("symbol")
    by_symbol.setdefault(s, []).append(o)

    for s, orders in by_symbol.items():
    if s not in symbols_with_position:
    with suppress(Exception):
    self.client.futures_cancel_all_open_orders(symbol=s)
    cancelled += len(orders)
    logger.info(
    "Cleanup: anulowano %d otwartych zleceń dla %s (brak pozycji).",
    len(orders),
    s,
    )

    return cancelled
    except Exception as e:
    logger.warning("Cleanup: błąd pobierania otwartych zleceń: %s", e)
    return 0

    def close_all_positions(self, specific_symbol: str = None) -> dict[str, bool]:
    if not self.client:
    return {}

    open_positions = self.check_positions()
    if specific_symbol:
    normalized = self._normalize_symbol(specific_symbol)
    open_positions = [
    p for p in open_positions if p.get("symbol") == normalized
    ]

    if not open_positions:
    logger.info(
    "Brak otwartych pozycji do zamknięcia (filtr: %s).",
    specific_symbol or "wszystkie",
    )
    return {}

    results: dict[str, bool] = {}
    logger.warning("Otrzymano żądanie zamknięcia %d pozycji.", len(open_positions))
    for pos in open_positions:
    symbol = pos["symbol"]
    side = "SELL" if float(pos["positionAmt"]) > 0 else "BUY"
    quantity = abs(float(pos["positionAmt"]))
    try:
    with suppress(Exception):
    self.client.futures_cancel_all_open_orders(symbol=symbol)
    order_response: dict[str, Any] | None = None

    try:
    order_response = self.client.futures_create_order(
    symbol=symbol,
    side=side,
    type="MARKET",
    quantity=quantity,
    reduceOnly=True,
    newOrderRespType="RESULT",
    )
    except BinanceAPIException as e:
    if e.code == -4131:
    logger.warning(
    "MARKET dla %s odrzucony (PERCENT_PRICE). Próbuję LIMIT IOC.",
    symbol,
    )
    last_price = float(
    self.client.futures_symbol_ticker(symbol=symbol)["price"]
    )
    limit_price = self._adjust_price_to_tick_size(
    symbol, last_price * (1.005 if side == "BUY" else 0.995)
    )
    order_response = self.client.futures_create_order(
    symbol=symbol,
    side=side,
    type="LIMIT",
    quantity=quantity,
    price=limit_price,
    timeInForce="IOC",
    reduceOnly=True,
    newOrderRespType="RESULT",
    )
    else:
    raise

    executed_qty = float(
    order_response.get("executedQty", 0)
    or order_response.get("cumQty", 0)
    or 0
    )
    if order_response and executed_qty > 0:
    results[symbol] = True
    logger.info("Potwierdzono zamknięcie pozycji dla %s.", symbol)
    else:
    results[symbol] = False
    logger.error(
    "Zlecenie zamknięcia dla %s nie zostało wykonane (IOC).", symbol
    )

    time.sleep(0.2)
    except Exception as e:
    logger.error("Nie udało się zamknąć pozycji dla %s: %s", symbol, e)
    results[symbol] = False

    return results

    # ==== SL updates ====
    def move_sl_to(self, symbol: str, new_sl_price: float) -> tuple[bool, str]:
    symbol = self._normalize_symbol(symbol)
    if not self.client:
    return False, "Brak połączenia z Binance."

    try:
    positions = self.client.futures_position_information(symbol=symbol)
    if not positions or abs(float(positions[0].get("positionAmt", 0))) == 0:
    return False, f"Brak otwartej pozycji dla {symbol}."

    pos_amt = float(positions[0]["positionAmt"])

    # Remove old SLs
    open_orders = self.client.futures_get_open_orders(symbol=symbol)
    for o in open_orders:
    if o.get("type") in ("STOP", "STOP_MARKET") and (
    o.get("reduceOnly") is True
    or str(o.get("closePosition")).lower() == "true"
    ):
    with suppress(Exception):
    self.client.futures_cancel_order(
    symbol=symbol, orderId=o["orderId"]
    )

    side = "SELL" if pos_amt > 0 else "BUY"
    adj_price = self._adjust_price_to_tick_size(symbol, float(new_sl_price))
    self.client.futures_create_order(
    symbol=symbol,
    side=side,
    type="STOP_MARKET",
    stopPrice=adj_price,
    closePosition=True,
    )
    logger.info(
    "SL zaktualizowany dla %s na %s (side: %s)", symbol, adj_price, side
    )
    return True, f"SL ustawiony na {adj_price:.6f}"

    except BinanceAPIException as e:
    logger.error("Błąd API przy ustawianiu SL dla %s: %s", symbol, e.message)
    return False, f"Błąd API: {e.message}"
    except Exception as e:
    logger.error("Nieoczekiwany błąd przy ustawianiu SL dla %s: %s", symbol, e)
    return False, "Wystąpił nieoczekiwany błąd."

    def move_sl_to_break_even(self, symbol: str) -> tuple[bool, str]:
        symbol = self._normalize_symbol(symbol)
        if not self.client:
            return False, "Brak połączenia z Binance."

        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions or abs(float(positions[0].get("positionAmt", 0))) == 0:
                return False, f"Brak otwartej pozycji dla {symbol}."

            entry_price = float(positions[0].get("entryPrice", 0.0))
            return self.move_sl_to(symbol, entry_price)

        except BinanceAPIException as e:
            logger.error(
                "Błąd API podczas przesuwania SL na BE dla %s: %s", symbol, e.message
            )
            return False, f"Błąd API: {e.message}"
        except Exception as e:
            logger.error(
                "Nieoczekiwany błąd podczas przesuwania SL na BE dla %s: %s",
                symbol,
                e,
            )
            return False, "Wystąpił nieoczekiwany błąd."

    # ==== New async methods for main.py compatibility ====
    async def init_async(self) -> None:
        """Initialize async components"""
        logger.info("Async initialization completed")
        return

    async def check_connection(self) -> bool:
        """Check if connection to Binance is working"""
        try:
            self.client.futures_ping()
            return True
        except Exception:
            return False

    async def get_futures_balance(self) -> float:
        """Get futures account balance in USDT"""
        try:
            account = self.client.futures_account()
            for asset in account.get("assets", []):
                if asset.get("asset") == "USDT":
                    return float(asset.get("availableBalance", 0))
            return 0.0
        except Exception as e:
            logger.error(f"Error getting futures balance: {e}")
            return 0.0

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    async def place_futures_order(self, symbol: str, side: str, quantity: float, leverage: int) -> dict[str, Any]:
        """Place futures market order"""
        try:
            # Set leverage first
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            
            # Place market order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    async def close_futures_position(self, symbol: str) -> dict[str, Any]:
        """Close futures position for symbol"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if positions and float(positions[0]["positionAmt"]) != 0:
                qty = abs(float(positions[0]["positionAmt"]))
                side = "SELL" if float(positions[0]["positionAmt"]) > 0 else "BUY"
                
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=qty,
                    reduceOnly=True
                )
                return {"symbol": symbol, "status": "CLOSED", "order": order}
            return {"symbol": symbol, "status": "NO_POSITION"}
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise

    async def get_futures_positions(self) -> list[dict[str, Any]]:
        """Get all open futures positions"""
        try:
            positions = self.client.futures_position_information()
            return [p for p in positions if float(p.get("positionAmt", 0)) != 0]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []


# Singleton instance
binance_handler = BinanceHandler()
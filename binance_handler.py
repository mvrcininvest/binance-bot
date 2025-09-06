"""
Binance Futures Handler - Version 9.1
Enhanced with v9.1 features:
- Precise leverage setting before orders
- Unique order tagging with newClientOrderId
- Enhanced error handling and logging
- Multi-TP support with proper quantity distribution
- Break-even and trailing stop functionality
- Comprehensive position management
"""

import logging
import math
import time
import hashlib
from contextlib import suppress
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import asyncio
from typing import Union

from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import Config
import uuid
from diagnostics import DiagnosticsEngine as DiagnosticManager
from decision_engine import DecisionEngine
from pine_health_monitor import PineScriptHealthMonitor as PineHealthMonitor
from shap_explainer import SHAPExplainer
from pattern_detector import PatternDetector
from database import (Session,
    get_setting, 
    create_decision_trace,
    update_decision_trace,
    complete_decision_trace,
    add_parameter_decision,
    log_pine_health,
    add_shap_explanation,
    create_pattern_alert,
    get_decision_trace,
    log_system_health,
    get_diagnostic_summary,
    cleanup_diagnostic_data,
)

logger = logging.getLogger("binance_handler")


def _parse_float_list(s: str | None, default: str) -> list[float]:
    """Parse comma-separated float values"""
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


def _parse_split_dict(s: str | None, default: str = "0.4,0.3,0.3") -> dict[str, float]:
    """Parse TP distribution percentages"""
    vals = _parse_float_list(s, default)
    if len(vals) != 3:
        vals = [0.4, 0.3, 0.3]
    vals = [max(0.0, v) for v in vals]
    total = sum(vals) or 1.0
    vals = [v / total for v in vals]
    return {"tp1": vals[0], "tp2": vals[1], "tp3": vals[2]}


class BinanceHandler:
    """Enhanced Binance Futures handler with v9.1 features"""
    
    def __init__(self) -> None:
        self.client: Client | None = None
        self.exchange_info: dict[str, Any] | None = None
        self.last_exchange_info_update: float = 0.0
        self.hedge_mode: bool = False
        self.leverage_cache: dict[str, tuple[int, float]] = {}  # symbol -> (leverage, timestamp)
        self.price_cache: dict[str, tuple[float, float]] = {}  # symbol -> (price, timestamp)
        self.diagnostic_enabled = getattr(Config, "ENABLE_DIAGNOSTICS", True)
        self.trace_tracking = {}  # trace_id -> execution_data
        self.order_trace_mapping = {}  # order_id -> trace_id
        self.execution_metrics = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "avg_execution_time_ms": 0.0,
            "leverage_changes": 0,
            "sl_tp_setups": 0,
        }
        
        try:
            if not Config.get_api_key() or not Config.get_api_secret():
                raise ValueError("Missing Binance API credentials in .env file")

            self.client = Client(
                api_key=Config.get_api_key(),
                api_secret=Config.get_api_secret(),
                testnet=Config.IS_TESTNET,
            )
            
            # Test connection
            self.client.futures_account()
            logger.info("Successfully connected to Binance Futures (testnet=%s)", Config.IS_TESTNET)
            
            # Initialize exchange info and position mode
            self._fetch_and_cache_exchange_info()
            self._check_position_mode()
            
        except BinanceAPIException as e:
            logger.error("Binance API error during initialization: %s", e.message)
            self.client = None
        except Exception as e:
            logger.critical("Critical error connecting to Binance: %s", e, exc_info=True)
            self.client = None

    # ==== v9.1 CORE FEATURE: Precise Leverage Setting ====
    def ensure_symbol_leverage_and_margin(self, symbol: str, leverage: int, margin_type: str = "ISOLATED") -> bool:
        """Set leverage and margin type with explicit logging and diagnostics - v9.1 ENHANCED
        Returns True if successful, False otherwise"""
        if not self.client:
            logger.error("No Binance client available for leverage setting")
            return False
        
        try:
            symbol = self._normalize_symbol(symbol)
            
            # Check current leverage to avoid unnecessary API calls
            cache_key = f"{symbol}_{leverage}_{margin_type}"
            current_time = time.time()
            
            if cache_key in self.leverage_cache:
                cached_leverage, cached_time = self.leverage_cache[cache_key]
                if current_time - cached_time < Config.LEVERAGE_CACHE_TTL:
                    logger.debug("Using cached leverage setting for %s", symbol)
                    return True
            
            # Get current position info
            position_info = self.client.futures_position_information(symbol=symbol)
            if not position_info:
                logger.error("Could not get position info for %s", symbol)
                return False
            
            current_leverage = int(position_info[0].get('leverage', 0))
            current_margin_type = position_info[0].get('marginType', '').upper()
            
            # Track leverage changes for diagnostics
            leverage_changed = False
            
            # Set margin type if different
            if current_margin_type != margin_type.upper():
                try:
                    self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type.upper())
                    logger.info("Margin type set to %s for %s", margin_type, symbol)
                    leverage_changed = True
                except BinanceAPIException as e:
                    if "No need to change margin type" not in e.message:
                        logger.warning("Failed to change margin type to %s for %s: %s", 
                                     margin_type, symbol, e.message)
                        return False
            
            # Set leverage if different
            if current_leverage != leverage:
                try:
                    self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
                    logger.info("Leverage set to %sx for %s (%s) - READY FOR ORDER", 
                              leverage, symbol, margin_type)
                    leverage_changed = True
                    
                    # Log leverage change for diagnostics
                    self.log_leverage_change(symbol, current_leverage, leverage, True)
                    
                except BinanceAPIException as e:
                    if "No need to change leverage" not in e.message:
                        logger.warning("Failed to change leverage to %sx for %s: %s", 
                                     leverage, symbol, e.message)
                        # Log failed leverage change
                        self.log_leverage_change(symbol, current_leverage, leverage, False)
                        return False
            else:
                logger.debug("Leverage already set to %sx for %s", leverage, symbol)
            
            # Cache the successful setting
            self.leverage_cache[cache_key] = (leverage, current_time)
            return True
            
        except Exception as e:
            logger.error("Unexpected error setting leverage for %s: %s", symbol, e, exc_info=True)
            return False

    # ==== Exchange Info & Utils ====
    def _fetch_and_cache_exchange_info(self) -> None:
        """Fetch and cache exchange information"""
        if not self.client:
            return
        try:
            info = self.client.futures_exchange_info()
            self.exchange_info = {item["symbol"]: item for item in info.get("symbols", [])}
            self.last_exchange_info_update = time.time()
            logger.info("Fetched Exchange Info for %d symbols", len(self.exchange_info or {}))
        except Exception as e:
            logger.error("Failed to fetch Exchange Info: %s", e)

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """Get symbol information with caching"""
        if not self.exchange_info or (time.time() - self.last_exchange_info_update > 3600):
            self._fetch_and_cache_exchange_info()
        s = self._normalize_symbol(symbol)
        return (self.exchange_info or {}).get(s)

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format"""
        s = (symbol or "").upper().strip()
        if s.endswith(".P"):
            s = s[:-2]
        return s.replace("-", "").replace("/", "")

    def _get_filter_value(self, symbol: str, filter_type: str, param: str) -> Any | None:
        """Get filter value from symbol info"""
        info = self.get_symbol_info(symbol)
        if info and "filters" in info:
            for f in info["filters"]:
                if f.get("filterType") == filter_type:
                    return f.get(param)
        return None

    def _get_precision(self, symbol: str) -> tuple[int, int]:
        """Get price and quantity precision for symbol"""
        info = self.get_symbol_info(symbol)
        if not info:
            return 2, 3
        return int(info.get("pricePrecision", 2)), int(info.get("quantityPrecision", 3))

    def _round_value(self, value: float, precision: int) -> float:
        """Round value to specified precision"""
        if precision <= 0:
            return math.floor(value)
        factor = 10**precision
        return math.floor(value * factor) / factor

    def _adjust_price_to_tick_size(self, symbol: str, price: float) -> float:
        """Adjust price to symbol's tick size"""
        tick_size_str = self._get_filter_value(symbol, "PRICE_FILTER", "tickSize")
        if tick_size_str:
            tick_size = Decimal(str(tick_size_str))
            price_decimal = Decimal(str(price))
            rounded_price = (price_decimal // tick_size) * tick_size
            return float(rounded_price)
        price_precision, _ = self._get_precision(symbol)
        return self._round_value(price, price_precision)

    def _check_position_mode(self) -> None:
        """Check and log position mode"""
        if not self.client:
            return
        try:
            mode = self.client.futures_get_position_mode()
            self.hedge_mode = bool(mode.get("dualSidePosition", False))
            logger.info("Position mode: %s", "Hedge" if self.hedge_mode else "One-way")
        except Exception:
            self.hedge_mode = False
            logger.warning("Could not get position mode; assuming One-way")

    # ==== Dynamic Settings Getters ====
    def _get_bool_setting(self, key: str, default_bool: bool) -> bool:
        """Get boolean setting from database"""
        try:
            with Session() as s:
                v = get_setting(s, key, "1" if default_bool else "0")
                return str(v).lower() in ("1", "true", "on", "yes", "y", "t")
        except Exception:
            return default_bool

    def _get_float_setting(self, key: str, default_value: float) -> float:
        """Get float setting from database"""
        try:
            with Session() as s:
                v = get_setting(s, key, str(default_value))
                return float(v)
        except Exception:
            return default_value

    def _get_str_setting(self, key: str, default_value: str) -> str:
        """Get string setting from database"""
        try:
            with Session() as s:
                v = get_setting(s, key, default_value)
                return str(v)
        except Exception:
            return default_value

    # ==== Account Helpers ====
    def get_balance(self) -> dict[str, float]:
        """Get USDT balance with caching"""
        if not self.client:
            return {"total": 0.0, "available": 0.0}
        
        try:
            current_time = time.time()
            cache_key = "balance"
            
            if hasattr(self, '_balance_cache'):
                cached_balance, cached_time = self._balance_cache
                if current_time - cached_time < Config.BALANCE_CACHE_TTL:
                    return cached_balance
            
            account = self.client.futures_account()
            for asset in account.get("assets", []):
                if asset.get("asset") == "USDT":
                    balance = {
                        "total": float(asset.get("walletBalance", 0.0)),
                        "available": float(asset.get("availableBalance", 0.0)),
                    }
                    self._balance_cache = (balance, current_time)
                    return balance
            
            balance = {"total": 0.0, "available": 0.0}
            self._balance_cache = (balance, current_time)
            return balance
            
        except Exception as e:
            logger.error("Error getting balance: %s", e)
            return {"total": 0.0, "available": 0.0}

    def get_last_price(self, symbol: str) -> float | None:
        """Get last price with caching"""
        if not self.client:
            return None
        
        try:
            s = self._normalize_symbol(symbol)
            current_time = time.time()
            
            if s in self.price_cache:
                cached_price, cached_time = self.price_cache[s]
                if current_time - cached_time < Config.PRICE_CACHE_TTL:
                    return cached_price
            
            ticker = self.client.futures_symbol_ticker(symbol=s)
            price = float(ticker["price"])
            self.price_cache[s] = (price, current_time)
            return price
            
        except Exception as e:
            logger.error("Error getting price for %s: %s", symbol, e)
            return None

    def has_open_position(self, symbol: str) -> bool:
        """Check if symbol has open position"""
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

    # ==== Leverage Brackets & Notional Cap ====
    def _get_max_notional_for_leverage(self, symbol: str, leverage: int) -> float | None:
        """Get maximum notional value for given leverage"""
        if not self.client:
            return None
        try:
            s = self._normalize_symbol(symbol)
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
                brackets = entries

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
            logger.warning("Could not get leverage brackets for %s: %s", symbol, e)
            return None

    def _get_current_position_notional(self, symbol: str) -> float:
        """Get current position notional value"""
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
        """Compute maximum quantity based on available margin and leverage brackets"""
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

    # ==== Signal Normalization ====
    def _canonicalize_signal(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize signal data to standard format"""
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
        leverage = int(first_of(raw, ["leverage"]) or Config.DEFAULT_LEVERAGE)
        risk_percent = float(first_of(raw, ["risk_percent"]) or Config.DEFAULT_RISK_PERCENT)

        # Take profits
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

        tp_distribution = raw.get("tp_distribution", f"{Config.TP1_SIZE_PERCENT},{Config.TP2_SIZE_PERCENT},{Config.TP3_SIZE_PERCENT}")

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

    # ==== v9.1 CORE FEATURE: Enhanced Order Placement ====
    def place_order_with_multi_tp(
        self, signal: dict[str, Any], is_dry_run: bool = False, trace_id: str = None
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
        """
        Place order with multi-TP setup and diagnostic tracing - v9.1 ENHANCED
        Returns: (success, order_response, details)
        """
        canon = self._canonicalize_signal(signal)
        symbol = canon["symbol"]
        side = "BUY" if canon["side"] == "buy" else "SELL"
        default_status = (False, None, None)

        # Start execution trace
        if trace_id:
            self._log_execution_trace(trace_id, "order_placement_started", {
                "symbol": symbol,
                "side": side,
                "signal_data": canon
            })

        if is_dry_run:
            logger.info("DRY RUN: Would place %s order for %s", side, symbol)
            if trace_id:
                self._complete_execution_trace(trace_id, {"status": "success", "dry_run": True})
            return True, {"dry_run": True}, {"symbol": symbol, "side": side}

        if not self.client:
            logger.error("No Binance client available for order placement")
            if trace_id:
                self._complete_execution_trace(trace_id, {"status": "error", "error": "no_client"})
            return default_status

        # v9.1 CORE: Generate unique trade ID for order tagging
        trade_id = int(time.time() * 1000)
        entry_client_id = f"BOT_{trade_id}_ENTRY"
        
        # Link trace to order
        if trace_id:
            self.order_trace_mapping[entry_client_id] = trace_id
        
        logger.info("Starting order placement for %s %s (trade_id: %s, trace: %s)", 
                   side, symbol, trade_id, trace_id)

        # v9.1 CORE: Set leverage and margin type BEFORE placing order
        leverage = canon.get("leverage", Config.DEFAULT_LEVERAGE)
        margin_type = Config.DEFAULT_MARGIN_TYPE
        
        if trace_id:
            self._log_execution_trace(trace_id, "setting_leverage", {
                "leverage": leverage,
                "margin_type": margin_type
            })
        
        if not self.ensure_symbol_leverage_and_margin(symbol, leverage, margin_type):
            logger.error("Failed to set leverage/margin for %s - aborting order", symbol)
            if trace_id:
                self._complete_execution_trace(trace_id, {
                    "status": "error", 
                    "error": "leverage_setting_failed"
                })
            return default_status

        # Get current price and validate
        if trace_id:
            self._log_execution_trace(trace_id, "fetching_price", {"symbol": symbol})
            
        last_price = self.get_last_price(symbol)
        if not last_price:
            logger.error("Could not get current price for %s", symbol)
            if trace_id:
                self._complete_execution_trace(trace_id, {
                    "status": "error", 
                    "error": "price_fetch_failed"
                })
            return default_status

        # Calculate position size
        if trace_id:
            self._log_execution_trace(trace_id, "calculating_position_size", {
                "last_price": last_price,
                "risk_percent": canon["risk_percent"]
            })
            
        risk_amount = self.get_balance()["available"] * (canon["risk_percent"] / 100.0)
        if canon.get("stop_loss"):
            if side == "BUY":
                risk_per_unit = max(0.001, last_price - canon["stop_loss"])
            else:
                risk_per_unit = max(0.001, canon["stop_loss"] - last_price)
            base_qty = risk_amount / risk_per_unit
        else:
            # Default 2% SL
            risk_per_unit = last_price * 0.02
            base_qty = risk_amount / risk_per_unit

        # Apply leverage and constraints
        leveraged_qty = base_qty * leverage
        max_qty = self._compute_qty_cap(symbol, leverage, last_price)
        final_qty = min(leveraged_qty, max_qty)

        # Round to precision
        _, qty_precision = self._get_precision(symbol)
        final_qty = self._round_value(final_qty, qty_precision)

        if final_qty <= 0:
            logger.error("Calculated quantity is 0 for %s", symbol)
            if trace_id:
                self._complete_execution_trace(trace_id, {
                    "status": "error", 
                    "error": "zero_quantity"
                })
            return default_status

        logger.info("Placing %s order: %s %s @ market (qty: %s, leverage: %sx)", 
                   side, symbol, side, final_qty, leverage)

        # Place market order with unique client ID
        if trace_id:
            self._log_execution_trace(trace_id, "placing_market_order", {
                "quantity": final_qty,
                "client_order_id": entry_client_id
            })
            
        try:
            order_response = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=final_qty,
                newClientOrderId=entry_client_id,  # v9.1 CORE: Unique tagging
                newOrderRespType="RESULT",
            )
            
            executed_qty = float(order_response.get("executedQty", 0) or 0)
            if executed_qty <= 0:
                logger.error("Order placed but no quantity executed for %s", symbol)
                if trace_id:
                    self._complete_execution_trace(trace_id, {
                        "status": "error", 
                        "error": "no_execution"
                    })
                return default_status
            
            logger.info("Entry order executed: %s %s (qty: %s, client_id: %s)", 
                       side, symbol, executed_qty, entry_client_id)

        except BinanceAPIException as e:
            logger.error("Binance API error placing order for %s: %s", symbol, e.message)
            if trace_id:
                self._complete_execution_trace(trace_id, {
                    "status": "error", 
                    "error": f"binance_api_error: {e.message}"
                })
            return default_status
        except Exception as e:
            logger.error("Unexpected error placing order for %s: %s", symbol, e, exc_info=True)
            if trace_id:
                self._complete_execution_trace(trace_id, {
                    "status": "error", 
                    "error": f"unexpected_error: {str(e)}"
                })
            return default_status

        # Get actual entry price
        if trace_id:
            self._log_execution_trace(trace_id, "fetching_entry_price", {})
            
        time.sleep(1.0)  # Allow position to settle
        try:
            pos_info = self.client.futures_position_information(symbol=symbol)
            actual_entry_price = float(pos_info[0].get("entryPrice", last_price))
        except Exception:
            actual_entry_price = float(last_price)

        # v9.1 CORE: Setup SL/TP with unique tagging
        if trace_id:
            self._log_execution_trace(trace_id, "setting_sl_tp", {
                "actual_entry_price": actual_entry_price
            })
            
        setup_status, tp_prices, final_sl, client_tags = self._setup_sl_tp_after_entry(
            symbol=symbol,
            side=side,
            total_quantity=executed_qty,
            canon_signal=canon,
            actual_entry_price=actual_entry_price,
            use_alert_lvls=self._get_bool_setting("use_alert_levels", Config.USE_ALERT_LEVELS),
            rr_levels=_parse_float_list(
                self._get_str_setting("tp_rr_levels", ""), default="1.5,3.0,5.0"
            ),
            tp_split=_parse_split_dict(
                self._get_str_setting("tp_split", ""), default="0.4,0.3,0.3"
            ),
            sl_margin_pct=self._get_float_setting("sl_margin_pct", 2.0),
            trade_id=trade_id,  # v9.1 CORE: Pass trade_id for tagging
            trace_id=trace_id,  # Pass trace_id for diagnostics
        )

        # Log SL/TP setup for diagnostics
        self.log_sl_tp_setup(symbol, setup_status)

        details = {
            "setup_status": setup_status,
            "final_size": executed_qty,
            "symbol": symbol,
            "entry_price": actual_entry_price,
            "tp_prices": tp_prices,
            "final_sl": final_sl,
            "client_tags": client_tags,
            "trade_id": trade_id,
            "entry_client_id": entry_client_id,
            "metadata": signal,
            "trace_id": trace_id,
        }
        
        logger.info("Order placement completed for %s (trade_id: %s, trace: %s)", 
                   symbol, trade_id, trace_id)
        
        # Complete execution trace
        if trace_id:
            self._complete_execution_trace(trace_id, {
                "status": "success",
                "trade_id": trade_id,
                "entry_client_id": entry_client_id,
                "executed_quantity": executed_qty,
                "entry_price": actual_entry_price,
                "sl_tp_setup": setup_status
            })
        
        return True, order_response, details

    def _setup_sl_tp_after_entry(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        canon_signal: dict,
        actual_entry_price: float,
        use_alert_lvls: bool,
        rr_levels: list[float],
        tp_split: dict[str, float],
        sl_margin_pct: float,
        trade_id: int,  # v9.1 CORE: For unique tagging
        trace_id: str = None,  # NEW: For diagnostics
    ) -> tuple[dict[str, bool], list[dict[str, Any]], float | None, dict[str, str]]:
        """
        Setup SL and TP orders with unique client IDs and diagnostics - v9.1 ENHANCED
        Returns: (status_report, tp_prices_list, final_sl, client_tags)
        """
        
        if trace_id:
            self._log_execution_trace(trace_id, "sl_tp_setup_started", {
                "symbol": symbol,
                "side": side,
                "total_quantity": total_quantity
            })
        
        # v9.1 CORE: Generate unique client IDs for all orders
        sl_client_id = f"BOT_{trade_id}_SL"
        tp1_client_id = f"BOT_{trade_id}_TP1"
        tp2_client_id = f"BOT_{trade_id}_TP2"
        tp3_client_id = f"BOT_{trade_id}_TP3"
        
        client_tags = {
            "sl": sl_client_id,
            "tp1": tp1_client_id,
            "tp2": tp2_client_id,
            "tp3": tp3_client_id,
        }

        sl_opposite_side = "SELL" if side == "BUY" else "BUY"
        _, qty_precision = self._get_precision(symbol)
        
        status_report = {
            "sl_set": False,
            "tp1_set": False,
            "tp2_set": False,
            "tp3_set": False,
        }
        tp_prices_list: list[dict[str, Any]] = []
        final_sl: float | None = None

        # Get current market price for gap calculations
        try:
            last_price = float(self.client.futures_symbol_ticker(symbol=symbol)["price"])
        except Exception:
            last_price = actual_entry_price
        
        tick_size_str = self._get_filter_value(symbol, "PRICE_FILTER", "tickSize") or "0.01"
        tick = float(tick_size_str)
        min_gap = max(tick * 2.0, actual_entry_price * 0.0005)

        # Calculate and set Stop Loss
        if trace_id:
            self._log_execution_trace(trace_id, "calculating_stop_loss", {
                "use_alert_levels": use_alert_lvls,
                "sl_margin_pct": sl_margin_pct
            })
            
        if use_alert_lvls and (canon_signal.get("stop_loss") is not None):
            sl_price = float(canon_signal["stop_loss"])
            sl_price = self._adjust_price_to_tick_size(symbol, sl_price)
            
            # Prevent immediate trigger
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
            # Calculate SL based on margin percentage
            used_margin = (total_quantity * actual_entry_price) / max(
                int(canon_signal.get("leverage", 10)) or 1, 1
            )
            loss_target = max(0.0, float(sl_margin_pct)) * used_margin / 100.0
            per_unit = loss_target / max(total_quantity, 1e-9)
            
            if side == "BUY":
                final_sl = self._adjust_price_to_tick_size(symbol, actual_entry_price - per_unit)
                final_sl = min(final_sl, actual_entry_price - tick)
                final_sl = min(final_sl, last_price - min_gap)
            else:
                final_sl = self._adjust_price_to_tick_size(symbol, actual_entry_price + per_unit)
                final_sl = max(final_sl, actual_entry_price + tick)
                final_sl = max(final_sl, last_price + min_gap)

        # Place Stop Loss order
        sl_is_valid = (side == "BUY" and final_sl < actual_entry_price) or (
            side == "SELL" and final_sl > actual_entry_price
        )
        
        if sl_is_valid and final_sl:
            if trace_id:
                self._log_execution_trace(trace_id, "placing_stop_loss", {
                    "sl_price": final_sl,
                    "client_id": sl_client_id
                })
                
            try:
                self.client.futures_create_order(
                    symbol=symbol,
                    side=sl_opposite_side,
                    type="STOP_MARKET",
                    stopPrice=final_sl,
                    closePosition=True,
                    newClientOrderId=sl_client_id,  # v9.1 CORE: Unique tagging
                )
                logger.info("SL set for %s at %s (tag: %s)", symbol, final_sl, sl_client_id)
                status_report["sl_set"] = True
            except Exception as e:
                logger.error("Failed to set SL for %s: %s", symbol, e)

        # Calculate Take Profit levels
        if trace_id:
            self._log_execution_trace(trace_id, "calculating_take_profits", {
                "use_alert_levels": use_alert_lvls,
                "rr_levels": rr_levels
            })
            
        desired_tps: dict[str, float | None] = {"tp1": None, "tp2": None, "tp3": None}
        
        if use_alert_lvls and (
            canon_signal.get("tp1") or canon_signal.get("tp2") or canon_signal.get("tp3")
        ):
            desired_tps["tp1"] = canon_signal.get("tp1")
            desired_tps["tp2"] = canon_signal.get("tp2")
            desired_tps["tp3"] = canon_signal.get("tp3")
        else:
            # Calculate TP levels based on risk-reward ratios
            if final_sl is not None:
                risk_per_unit = (
                    (actual_entry_price - final_sl) if side == "BUY" 
                    else (final_sl - actual_entry_price)
                )
                for i, rr in enumerate(rr_levels[:3], 1):
                    if side == "BUY":
                        desired_tps[f"tp{i}"] = actual_entry_price + float(rr) * risk_per_unit
                    else:
                        desired_tps[f"tp{i}"] = actual_entry_price - float(rr) * risk_per_unit

        # Place Take Profit orders
        remaining_qty_for_tp = total_quantity
        ordered_levels = [
            ("tp1", tp_split.get("tp1", 0.4)),
            ("tp2", tp_split.get("tp2", 0.3)),
            ("tp3", tp_split.get("tp3", 0.3)),
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
            
            # Prevent immediate trigger
            if side == "BUY":
                min_tp = last_price + min_gap
                if tp_price <= min_tp:
                    tp_price = self._adjust_price_to_tick_size(symbol, min_tp)
            else:
                max_tp = last_price - min_gap
                if tp_price >= max_tp:
                    tp_price = self._adjust_price_to_tick_size(symbol, max_tp)

            if trace_id:
                self._log_execution_trace(trace_id, f"placing_{level_name}", {
                    "tp_price": tp_price,
                    "quantity": tp_qty,
                    "client_id": client_tags[level_name]
                })

            try:
                tp_client_id = client_tags[level_name]
                self.client.futures_create_order(
                    symbol=symbol,
                    side=sl_opposite_side,
                    type="TAKE_PROFIT_MARKET",
                    quantity=tp_qty,
                    stopPrice=tp_price,
                    reduceOnly=True,
                    newClientOrderId=tp_client_id,  # v9.1 CORE: Unique tagging
                )
                logger.info("Set %s for %s at %s (qty=%s, tag=%s)", 
                           level_name.upper(), symbol, tp_price, tp_qty, tp_client_id)
                status_report[f"{level_name}_set"] = True
                remaining_qty_for_tp -= tp_qty
                tp_prices_list.append({
                    "level": level_name.upper(),
                    "price": tp_price,
                    "qty": tp_qty,
                    "client_id": tp_client_id
                })
            except Exception as e:
                logger.error("Failed to set %s for %s: %s", level_name.upper(), symbol, e)

        if trace_id:
            self._log_execution_trace(trace_id, "sl_tp_setup_completed", {
                "status_report": status_report,
                "tp_count": len(tp_prices_list),
                "final_sl": final_sl
            })

        return status_report, tp_prices_list, final_sl, client_tags

    # ==== Position Management ====
    def check_positions(self) -> list[dict[str, Any]]:
        """Get all open positions"""
        if not self.client:
            return []
        try:
            positions = self.client.futures_position_information()
            return [p for p in positions if abs(float(p.get("positionAmt", 0))) != 0]
        except Exception as e:
            logger.error("Error checking positions: %s", e)
            return []

    def cleanup_stale_orders(self) -> int:
        """Clean up orders for symbols without positions"""
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
                        logger.info("Cleanup: cancelled %d open orders for %s (no position)", 
                                   len(orders), s)

            return cancelled
        except Exception as e:
            logger.warning("Cleanup: error getting open orders: %s", e)
            return 0

    def close_all_positions(self, specific_symbol: str = None) -> dict[str, bool]:
        """Close all positions or specific symbol position"""
        if not self.client:
            return {}

        open_positions = self.check_positions()
        if specific_symbol:
            normalized = self._normalize_symbol(specific_symbol)
            open_positions = [p for p in open_positions if p.get("symbol") == normalized]

        if not open_positions:
            logger.info("No open positions to close (filter: %s)", specific_symbol or "all")
            return {}

        results: dict[str, bool] = {}
        logger.warning("Received request to close %d positions", len(open_positions))
        
        for pos in open_positions:
            symbol = pos["symbol"]
            side = "SELL" if float(pos["positionAmt"]) > 0 else "BUY"
            quantity = abs(float(pos["positionAmt"]))
            
            try:
                # Cancel all open orders first
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
                    if e.code == -4131:  # PERCENT_PRICE error
                        logger.warning("MARKET order for %s rejected (PERCENT_PRICE). Trying LIMIT IOC", symbol)
                        last_price = float(self.client.futures_symbol_ticker(symbol=symbol)["price"])
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
                    order_response.get("executedQty", 0) or 
                    order_response.get("cumQty", 0) or 0
                )
                
                if order_response and executed_qty > 0:
                    results[symbol] = True
                    logger.info("Confirmed position closure for %s", symbol)
                else:
                    results[symbol] = False
                    logger.error("Position closure order for %s was not executed (IOC)", symbol)

                time.sleep(0.2)
                
            except Exception as e:
                logger.error("Failed to close position for %s: %s", symbol, e)
                results[symbol] = False

        return results

    # ==== Stop Loss Management ====
    def move_sl_to(self, symbol: str, new_sl_price: float) -> tuple[bool, str]:
        """Move stop loss to specific price"""
        symbol = self._normalize_symbol(symbol)
        if not self.client:
            return False, "No Binance connection"

        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions or abs(float(positions[0].get("positionAmt", 0))) == 0:
                return False, f"No open position for {symbol}"

            pos_amt = float(positions[0]["positionAmt"])

            # Remove old SL orders
            open_orders = self.client.futures_get_open_orders(symbol=symbol)
            for o in open_orders:
                if o.get("type") in ("STOP", "STOP_MARKET") and (
                    o.get("reduceOnly") is True or 
                    str(o.get("closePosition")).lower() == "true"
                ):
                    with suppress(Exception):
                        self.client.futures_cancel_order(symbol=symbol, orderId=o["orderId"])

            side = "SELL" if pos_amt > 0 else "BUY"
            adj_price = self._adjust_price_to_tick_size(symbol, float(new_sl_price))
            
            self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                stopPrice=adj_price,
                closePosition=True,
            )
            
            logger.info("SL updated for %s to %s (side: %s)", symbol, adj_price, side)
            return True, f"SL set to {adj_price:.6f}"

        except BinanceAPIException as e:
            logger.error("API error setting SL for %s: %s", symbol, e.message)
            return False, f"API error: {e.message}"
        except Exception as e:
            logger.error("Unexpected error setting SL for %s: %s", symbol, e)
            return False, "Unexpected error occurred"

    def move_sl_to_break_even(self, symbol: str) -> tuple[bool, str]:
        """Move stop loss to break even (entry price)"""
        symbol = self._normalize_symbol(symbol)
        if not self.client:
            return False, "No Binance connection"

        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions or abs(float(positions[0].get("positionAmt", 0))) == 0:
                return False, f"No open position for {symbol}"

            entry_price = float(positions[0].get("entryPrice", 0.0))
            return self.move_sl_to(symbol, entry_price)

        except BinanceAPIException as e:
            logger.error("API error moving SL to BE for %s: %s", symbol, e.message)
            return False, f"API error: {e.message}"
        except Exception as e:
            logger.error("Unexpected error moving SL to BE for %s: %s", symbol, e)
            return False, "Unexpected error occurred"

    # ==== v9.1 Enhanced Methods for main.py compatibility ====
    async def init_async(self) -> None:
        """Initialize async components"""
        logger.info("Async initialization completed")
        return

    async def check_connection(self) -> bool:
        """Check if connection to Binance is working"""
        if not self.client:
            return False
        try:
            self.client.futures_ping()
            return True
        except Exception:
            return False

    async def get_futures_balance(self) -> float:
        """Get futures account balance in USDT"""
        try:
            balance = self.get_balance()
            return balance.get("available", 0.0)
        except Exception as e:
            logger.error(f"Error getting futures balance: {e}")
            return 0.0

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            price = self.get_last_price(symbol)
            return price or 0.0
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    async def place_futures_order(self, symbol: str, side: str, quantity: float, 
                                leverage: int, client_order_id: str = None) -> dict[str, Any]:
        """Place futures market order"""
        try:
            # Set leverage first
            if not self.ensure_symbol_leverage_and_margin(symbol, leverage):
                raise Exception(f"Failed to set leverage for {symbol}")
            
            # Place market order
            order_params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity
            }
            
            if client_order_id:
                order_params["newClientOrderId"] = client_order_id
            
            order = self.client.futures_create_order(**order_params)
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
            return self.check_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

# ==== Missing Methods for main.py Compatibility ====
    def get_open_orders(self, symbol: str = None) -> list[dict[str, Any]]:
        """Get open orders for symbol or all symbols"""
        if not self.client:
            logger.error("No Binance client available")
            return []
        
        try:
            if symbol:
                symbol = self._normalize_symbol(symbol)
                orders = self.client.futures_get_open_orders(symbol=symbol)
            else:
                orders = self.client.futures_get_open_orders()
            
            logger.debug("Retrieved %d open orders%s", len(orders), f" for {symbol}" if symbol else "")
            return orders
            
        except BinanceAPIException as e:
            logger.error("API error getting open orders: %s", e.message)
            return []
        except Exception as e:
            logger.error("Error getting open orders: %s", e, exc_info=True)
            return []

    def cancel_order(self, symbol: str, order_id: int = None, client_order_id: str = None) -> bool:
        """Cancel specific order by order ID or client order ID"""
        if not self.client:
            logger.error("No Binance client available")
            return False
        
        if not order_id and not client_order_id:
            logger.error("Either order_id or client_order_id must be provided")
            return False
        
        try:
            symbol = self._normalize_symbol(symbol)
            
            if order_id:
                self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
                logger.info("Cancelled order %s for %s", order_id, symbol)
            else:
                self.client.futures_cancel_order(symbol=symbol, origClientOrderId=client_order_id)
                logger.info("Cancelled order %s for %s", client_order_id, symbol)
            
            return True
            
        except BinanceAPIException as e:
            logger.error("API error cancelling order: %s", e.message)
            return False
        except Exception as e:
            logger.error("Error cancelling order: %s", e, exc_info=True)
            return False

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol - wrapper for ensure_symbol_leverage_and_margin"""
        return self.ensure_symbol_leverage_and_margin(symbol, leverage)

    async def get_account_info(self) -> dict[str, Any]:
        """Get futures account information"""
        if not self.client:
            return {}
        
        try:
            account = self.client.futures_account()
            return account
        except Exception as e:
            logger.error("Error getting account info: %s", e)
            return {}

    async def get_position_info(self, symbol: str = None) -> list[dict[str, Any]]:
        """Get position information for symbol or all positions"""
        if not self.client:
            return []
        
        try:
            if symbol:
                symbol = self._normalize_symbol(symbol)
                positions = self.client.futures_position_information(symbol=symbol)
            else:
                positions = self.client.futures_position_information()
            
            return positions
        except Exception as e:
            logger.error("Error getting position info: %s", e)
            return []

    def calculate_position_size(self, symbol: str, risk_percent: float, entry_price: float, 
                              stop_loss: float, leverage: int) -> float:
        """Calculate position size based on risk management"""
        try:
            balance = self.get_balance()
            available_balance = balance.get("available", 0.0)
            
            if available_balance <= 0:
                logger.warning("No available balance for position sizing")
                return 0.0
            
            risk_amount = available_balance * (risk_percent / 100.0)
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit <= 0:
                logger.warning("Invalid risk per unit calculation")
                return 0.0
            
            base_quantity = risk_amount / risk_per_unit
            leveraged_quantity = base_quantity * leverage
            
            # Apply quantity constraints
            max_qty = self._compute_qty_cap(symbol, leverage, entry_price)
            final_qty = min(leveraged_quantity, max_qty)
            
            # Round to precision
            _, qty_precision = self._get_precision(symbol)
            final_qty = self._round_value(final_qty, qty_precision)
            
            logger.info("Position size calculated: %s for %s (risk: %.2f%%, leverage: %dx)", 
                       final_qty, symbol, risk_percent, leverage)
            
            return final_qty
            
        except Exception as e:
            logger.error("Error calculating position size: %s", e, exc_info=True)
            return 0.0

    def get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        """Get symbol trading filters"""
        try:
            info = self.get_symbol_info(symbol)
            if not info or "filters" not in info:
                return {}
            
            filters = {}
            for f in info["filters"]:
                filter_type = f.get("filterType")
                if filter_type:
                    filters[filter_type] = f
            
            return filters
            
        except Exception as e:
            logger.error("Error getting symbol filters for %s: %s", symbol, e)
            return {}

    def validate_order_params(self, symbol: str, side: str, quantity: float, 
                            price: float = None) -> tuple[bool, str]:
        """Validate order parameters against symbol filters"""
        try:
            filters = self.get_symbol_filters(symbol)
            
            # Check minimum quantity
            lot_size = filters.get("LOT_SIZE", {})
            min_qty = float(lot_size.get("minQty", 0))
            if quantity < min_qty:
                return False, f"Quantity {quantity} below minimum {min_qty}"
            
            # Check minimum notional
            min_notional = filters.get("MIN_NOTIONAL", {})
            if min_notional and price:
                min_notional_value = float(min_notional.get("notional", 0))
                order_notional = quantity * price
                if order_notional < min_notional_value:
                    return False, f"Order notional {order_notional} below minimum {min_notional_value}"
            
            return True, "Valid"
            
        except Exception as e:
            logger.error("Error validating order params: %s", e)
            return False, f"Validation error: {e}"

    # ==== v9.1 Enhanced Status Methods ====
    def get_detailed_status(self) -> dict[str, Any]:
        """Get detailed status for bot status command with diagnostics - v9.1 ENHANCED"""
        try:
            balance = self.get_balance()
            positions = self.check_positions()
            
            # Get open orders
            open_orders = []
            if self.client:
                try:
                    open_orders = self.client.futures_get_open_orders()
                except Exception:
                    pass
            
            # Categorize orders by type
            sl_orders = [o for o in open_orders if o.get("type") in ("STOP", "STOP_MARKET")]
            tp_orders = [o for o in open_orders if o.get("type") in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET")]
            
            # Get execution diagnostics
            execution_diagnostics = self.get_execution_diagnostics() if self.diagnostic_enabled else {}
            
            return {
                "connection_status": "connected" if self.client else "disconnected",
                "testnet": Config.IS_TESTNET,
                "balance": balance,
                "positions_count": len(positions),
                "positions": positions,
                "open_orders_count": len(open_orders),
                "sl_orders_count": len(sl_orders),
                "tp_orders_count": len(tp_orders),
                "leverage_cache_size": len(self.leverage_cache),
                "price_cache_size": len(self.price_cache),
                "exchange_info_age": time.time() - self.last_exchange_info_update if self.exchange_info else None,
                "position_mode": "hedge" if self.hedge_mode else "one_way",
                # NEW: Diagnostic data
                "diagnostics_enabled": self.diagnostic_enabled,
                "execution_metrics": self.execution_metrics,
                "active_traces": len(self.trace_tracking),
                "execution_diagnostics": execution_diagnostics,
            }
        except Exception as e:
            logger.error("Error getting detailed status: %s", e)
            return {"error": str(e)}

    def clear_caches(self) -> None:
        """Clear all caches - v9.1 FEATURE"""
        self.leverage_cache.clear()
        self.price_cache.clear()
        if hasattr(self, '_balance_cache'):
            delattr(self, '_balance_cache')
        logger.info("All caches cleared")

    def validate_api_credentials(self) -> bool:
        """Validate API credentials by testing connection"""
        if not self.client:
            logger.error("No Binance client available for validation")
            return False
        
        try:
            # Test basic connection
            self.client.futures_ping()
            
            # Test account access
            account = self.client.futures_account()
            if not account:
                logger.error("Could not retrieve account information")
                return False
            
            logger.info("API credentials validated successfully")
            return True
            
        except BinanceAPIException as e:
            logger.error("Binance API error during validation: %s", e.message)
            return False
        except Exception as e:
            logger.error("Unexpected error during validation: %s", e, exc_info=True)
            return False
    # ==== HYBRID ULTRA-DIAGNOSTICS METHODS ====
    def _log_execution_trace(self, trace_id: str, stage: str, data: Dict[str, Any]):
        """Log execution trace for diagnostics"""
        if not self.diagnostic_enabled or not trace_id:
            return
        
        try:
            if trace_id not in self.trace_tracking:
                self.trace_tracking[trace_id] = {
                    "stages": [],
                    "start_time": time.time(),
                    "symbol": data.get("symbol"),
                    "side": data.get("side"),
                }
            
            self.trace_tracking[trace_id]["stages"].append({
                "stage": stage,
                "timestamp": time.time(),
                "data": data,
                "latency_ms": int((time.time() - self.trace_tracking[trace_id]["start_time"]) * 1000)
            })
            
            logger.debug(f"🔍 Execution trace {trace_id}: {stage}")
            
        except Exception as e:
            logger.error(f"Error logging execution trace: {e}")

    def _complete_execution_trace(self, trace_id: str, final_result: Dict[str, Any]):
        """Complete execution trace with final results"""
        if not self.diagnostic_enabled or not trace_id:
            return
        
        try:
            if trace_id in self.trace_tracking:
                trace_data = self.trace_tracking[trace_id]
                total_time_ms = int((time.time() - trace_data["start_time"]) * 1000)
                
                # Update metrics
                self.execution_metrics["total_orders"] += 1
                if final_result.get("status") == "success":
                    self.execution_metrics["successful_orders"] += 1
                else:
                    self.execution_metrics["failed_orders"] += 1
                
                # Update average execution time
                current_avg = self.execution_metrics["avg_execution_time_ms"]
                total_orders = self.execution_metrics["total_orders"]
                self.execution_metrics["avg_execution_time_ms"] = (
                    (current_avg * (total_orders - 1) + total_time_ms) / total_orders
                )
                
                # Store final result
                trace_data["final_result"] = final_result
                trace_data["total_execution_time_ms"] = total_time_ms
                trace_data["completed_at"] = time.time()
                
                logger.info(f"✅ Execution trace {trace_id} completed in {total_time_ms}ms")
                
                # Clean up old traces (keep last 100)
                if len(self.trace_tracking) > 100:
                    oldest_trace = min(self.trace_tracking.keys(), 
                                     key=lambda k: self.trace_tracking[k]["start_time"])
                    del self.trace_tracking[oldest_trace]
                    
        except Exception as e:
            logger.error(f"Error completing execution trace: {e}")

    def get_execution_diagnostics(self) -> Dict[str, Any]:
        """Get execution diagnostics data"""
        try:
            # Get recent traces
            recent_traces = []
            current_time = time.time()
            
            for trace_id, trace_data in self.trace_tracking.items():
                if current_time - trace_data["start_time"] < 3600:  # Last hour
                    recent_traces.append({
                        "trace_id": trace_id,
                        "symbol": trace_data.get("symbol"),
                        "side": trace_data.get("side"),
                        "execution_time_ms": trace_data.get("total_execution_time_ms"),
                        "stages_count": len(trace_data.get("stages", [])),
                        "status": trace_data.get("final_result", {}).get("status", "pending")
                    })
            
            # Calculate success rate
            total_orders = self.execution_metrics["total_orders"]
            success_rate = (
                (self.execution_metrics["successful_orders"] / total_orders * 100)
                if total_orders > 0 else 0
            )
            
            return {
                "execution_metrics": self.execution_metrics,
                "success_rate": success_rate,
                "recent_traces": recent_traces[-20:],  # Last 20 traces
                "cache_status": {
                    "leverage_cache_size": len(self.leverage_cache),
                    "price_cache_size": len(self.price_cache),
                    "balance_cache_active": hasattr(self, '_balance_cache'),
                },
                "connection_health": {
                    "client_available": self.client is not None,
                    "exchange_info_age": time.time() - self.last_exchange_info_update if self.exchange_info else None,
                    "position_mode": "hedge" if self.hedge_mode else "one_way",
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting execution diagnostics: {e}")
            return {"error": str(e)}

    def log_leverage_change(self, symbol: str, old_leverage: int, new_leverage: int, success: bool):
        """Log leverage change for diagnostics"""
        if self.diagnostic_enabled:
            self.execution_metrics["leverage_changes"] += 1
            logger.info(f"📊 Leverage change: {symbol} {old_leverage}x -> {new_leverage}x (success: {success})")

    def log_sl_tp_setup(self, symbol: str, setup_result: Dict[str, bool]):
        """Log SL/TP setup for diagnostics"""
        if self.diagnostic_enabled:
            self.execution_metrics["sl_tp_setups"] += 1
            successful_setups = sum(1 for success in setup_result.values() if success)
            total_setups = len(setup_result)
            logger.info(f"🎯 SL/TP setup: {symbol} ({successful_setups}/{total_setups} successful)")

    def get_trace_details(self, trace_id: str) -> Dict[str, Any]:
        """Get detailed trace information for diagnostics"""
        try:
            if trace_id not in self.trace_tracking:
                return {"error": "Trace not found"}
            
            trace_data = self.trace_tracking[trace_id]
            
            return {
                "trace_id": trace_id,
                "symbol": trace_data.get("symbol"),
                "side": trace_data.get("side"),
                "start_time": trace_data.get("start_time"),
                "total_execution_time_ms": trace_data.get("total_execution_time_ms"),
                "completed_at": trace_data.get("completed_at"),
                "stages": trace_data.get("stages", []),
                "final_result": trace_data.get("final_result", {}),
                "status": "completed" if "completed_at" in trace_data else "pending"
            }
            
        except Exception as e:
            logger.error(f"Error getting trace details: {e}")
            return {"error": str(e)}

    def clear_diagnostic_data(self):
        """Clear diagnostic data for maintenance"""
        try:
            self.trace_tracking.clear()
            self.order_trace_mapping.clear()
            self.execution_metrics = {
                "total_orders": 0,
                "successful_orders": 0,
                "failed_orders": 0,
                "avg_execution_time_ms": 0.0,
                "leverage_changes": 0,
                "sl_tp_setups": 0,
            }
            logger.info("🧹 Diagnostic data cleared")
            
        except Exception as e:
            logger.error(f"Error clearing diagnostic data: {e}")

# Singleton instance
binance_handler = BinanceHandler()
"""
Futures User Data Stream Handler for Trading Bot v9.1
Enhanced with v9.1 features:
- Centralized WebSocket management
- Enhanced order tracking with client IDs
- Break-even and trailing stop detection
- Comprehensive position monitoring
- Thread-safe state management
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any, List
from threading import Lock

from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

from config import Config
from database import (
    Session,
    Trade,
    update_trade,
    get_position_by_symbol,
    record_signal_decision,
)
from discord_notifications import discord_notifier

logger = logging.getLogger(__name__)


class FuturesUserStream:
    """Enhanced Futures User Data Stream Handler - v9.1"""

    def __init__(self, client: AsyncClient):
        """Initialize user stream handler with v9.1 enhancements"""
        self.client = client
        self.bm: Optional[BinanceSocketManager] = None
        self.listen_key: Optional[str] = None
        self.socket = None
        self.running = False
        self.connected = False

        # v9.1 CORE: Enhanced callback system
        self.callbacks = {
            "ORDER_TRADE_UPDATE": [],
            "ACCOUNT_UPDATE": [],
            "MARGIN_CALL": [],
            "ACCOUNT_CONFIG_UPDATE": [],
            "STRATEGY_UPDATE": [],
            "POSITION_UPDATE": [],  # v9.1 NEW
            "BALANCE_UPDATE": [],  # v9.1 NEW
        }

        # v9.1 CORE: Connection management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = Config.WS_MAX_RECONNECT_ATTEMPTS
        self.last_heartbeat = time.time()
        self.connection_lock = Lock()

        # v9.1 CORE: State tracking
        self.positions_cache = {}  # symbol -> position_data
        self.orders_cache = {}  # client_order_id -> order_data
        self.balance_cache = {"USDT": 0.0}
        self.last_update_time = time.time()

        # v9.1 CORE: Order tracking for SL/TP management
        self.tracked_orders = {}  # client_order_id -> trade_info
        self.tp_fills = {}  # trade_id -> {tp1: bool, tp2: bool, tp3: bool}
        self.be_moves = set()  # trade_ids that moved to BE
        self.trailing_active = set()  # trade_ids with trailing active

        logger.info("Futures User Stream handler initialized (v9.1)")

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for specific event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")

    async def start(self):
        """Start user data stream with v9.1 enhancements"""
        try:
            with self.connection_lock:
                if self.running:
                    logger.warning("User stream already running")
                    return

                self.running = True
                self.reconnect_attempts = 0

            self.bm = BinanceSocketManager(self.client)

            # Get listen key
            self.listen_key = await self._get_listen_key()

            # Start background tasks
            asyncio.create_task(self._keepalive_listen_key())
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._periodic_cleanup())

            # Start stream
            await self._connect_stream()

            logger.info("User data stream started successfully (v9.1)")

        except Exception as e:
            logger.error(f"Failed to start user stream: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop user data stream"""
        with self.connection_lock:
            if not self.running:
                return
            self.running = False
            self.connected = False

        try:
            if self.socket:
                await self.socket.__aexit__(None, None, None)

            if self.listen_key:
                await self._close_listen_key()

            # Clear caches
            self.positions_cache.clear()
            self.orders_cache.clear()
            self.tracked_orders.clear()
            self.tp_fills.clear()
            self.be_moves.clear()
            self.trailing_active.clear()

            logger.info("User data stream stopped")

        except Exception as e:
            logger.error(f"Error stopping user stream: {e}")

    async def _get_listen_key(self) -> str:
        """Get new listen key"""
        try:
            response = await self.client.futures_stream_get_listen_key()
            logger.debug("New listen key obtained")
            return response["listenKey"]
        except Exception as e:
            logger.error(f"Failed to get listen key: {e}")
            raise

    async def _keepalive_listen_key(self):
        """Keep listen key alive - v9.1 enhanced"""
        while self.running:
            try:
                await asyncio.sleep(Config.WS_PING_INTERVAL)  # 30 minutes default

                if self.listen_key and self.running:
                    await self.client.futures_stream_keepalive(self.listen_key)
                    logger.debug("Listen key keepalive sent")

            except Exception as e:
                logger.error(f"Keepalive error: {e}")

                # Try to get new listen key
                try:
                    self.listen_key = await self._get_listen_key()
                    await self._reconnect_stream()
                except Exception as e2:
                    logger.error(f"Failed to refresh listen key: {e2}")

    async def _heartbeat_monitor(self):
        """Monitor connection heartbeat - v9.1 NEW"""
        while self.running:
            try:
                await asyncio.sleep(Config.WS_HEARTBEAT_INTERVAL)

                current_time = time.time()
                if (
                    current_time - self.last_heartbeat
                    > Config.WS_HEARTBEAT_INTERVAL * 2
                ):
                    logger.warning("Heartbeat timeout detected, reconnecting...")
                    await self._reconnect_stream()

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")

    async def _periodic_cleanup(self):
        """Periodic cleanup of old data - v9.1 NEW"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minutes

                current_time = time.time()
                cutoff_time = current_time - 3600  # 1 hour

                # Clean old orders cache
                old_orders = [
                    order_id
                    for order_id, data in self.orders_cache.items()
                    if data.get("timestamp", 0) < cutoff_time
                ]
                for order_id in old_orders:
                    del self.orders_cache[order_id]

                # Clean old tracked orders
                old_tracked = [
                    order_id
                    for order_id, data in self.tracked_orders.items()
                    if data.get("timestamp", 0) < cutoff_time
                ]
                for order_id in old_tracked:
                    del self.tracked_orders[order_id]

                if old_orders or old_tracked:
                    logger.debug(
                        f"Cleaned {len(old_orders)} old orders, {len(old_tracked)} old tracked orders"
                    )

            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")

    async def _close_listen_key(self):
        """Close listen key"""
        try:
            if self.listen_key:
                await self.client.futures_stream_close(self.listen_key)
                logger.debug("Listen key closed")
        except Exception as e:
            logger.error(f"Error closing listen key: {e}")

    async def _connect_stream(self):
        """Connect to user data stream with v9.1 enhancements"""
        try:
            self.socket = self.bm.futures_user_socket()

            async with self.socket as stream:
                self.reconnect_attempts = 0
                self.connected = True
                self.last_heartbeat = time.time()
                logger.info("Connected to user data stream")

                # Send connection notification
                await discord_notifier.send_system_notification(
                    "WebSocket Connected", "User data stream connected successfully"
                )

                while self.running:
                    try:
                        msg = await asyncio.wait_for(stream.recv(), timeout=30.0)

                        if msg:
                            self.last_heartbeat = time.time()
                            await self._process_message(msg)

                    except asyncio.TimeoutError:
                        logger.debug("Stream timeout, sending ping...")
                        continue
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Stream error: {e}")

                        if self.running:
                            await self._reconnect_stream()
                        break

        except Exception as e:
            logger.error(f"Failed to connect stream: {e}")

            if self.running and self.reconnect_attempts < self.max_reconnect_attempts:
                await self._reconnect_stream()

    async def _reconnect_stream(self):
        """Reconnect to stream with v9.1 enhancements"""
        with self.connection_lock:
            if not self.running:
                return

            self.connected = False
            self.reconnect_attempts += 1

        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            await discord_notifier.send_error_notification(
                "User Stream Disconnected",
                f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached. Manual intervention required.",
            )
            return

        wait_time = min(60, Config.WS_RECONNECT_DELAY * self.reconnect_attempts)
        logger.info(
            f"Reconnecting in {wait_time} seconds... (attempt {self.reconnect_attempts})"
        )

        await asyncio.sleep(wait_time)

        try:
            # Get new listen key
            self.listen_key = await self._get_listen_key()

            # Reconnect
            await self._connect_stream()

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

            if self.running:
                await self._reconnect_stream()

    async def _process_message(self, msg: Dict):
        """Process stream message with v9.1 enhancements"""
        try:
            event_type = msg.get("e")
            self.last_update_time = time.time()

            # v9.1 CORE: Enhanced message processing
            if event_type == "ORDER_TRADE_UPDATE":
                await self._handle_order_update(msg)

            elif event_type == "ACCOUNT_UPDATE":
                await self._handle_account_update(msg)

            elif event_type == "MARGIN_CALL":
                await self._handle_margin_call(msg)

            elif event_type == "ACCOUNT_CONFIG_UPDATE":
                await self._handle_config_update(msg)

            elif event_type == "STRATEGY_UPDATE":
                await self._handle_strategy_update(msg)

            elif event_type == "listenKeyExpired":
                logger.warning("Listen key expired, reconnecting...")
                await self._reconnect_stream()
                return

            # v9.1 CORE: Call registered callbacks
            for callback in self.callbacks.get(event_type, []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(msg)
                    else:
                        callback(msg)
                except Exception as e:
                    logger.error(f"Callback error for {event_type}: {e}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.debug(f"Message: {json.dumps(msg, indent=2)}")

    async def _handle_order_update(self, msg: Dict):
        """Handle order trade update with v9.1 enhancements"""
        try:
            order = msg.get("o", {})

            symbol = order.get("s")
            order_id = order.get("i")
            client_order_id = order.get("c", "")
            side = order.get("S")
            order_type = order.get("o")
            status = order.get("X")
            execution_type = order.get("x")
            price = float(order.get("p", 0))
            quantity = float(order.get("q", 0))
            filled_qty = float(order.get("z", 0))
            avg_price = float(order.get("ap", 0))
            commission = float(order.get("n", 0))
            realized_pnl = float(order.get("rp", 0))

            # v9.1 CORE: Cache order data
            self.orders_cache[client_order_id] = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "status": status,
                "price": price,
                "quantity": quantity,
                "filled_qty": filled_qty,
                "avg_price": avg_price,
                "timestamp": time.time(),
            }

            logger.info(
                f"Order Update - {symbol} {side} {order_type} | Status: {status} | Execution: {execution_type}"
            )

            # v9.1 CORE: Enhanced order status handling
            if status == "NEW":
                await self._handle_new_order(order)

            elif status == "FILLED":
                await self._handle_filled_order(order)

            elif status == "PARTIALLY_FILLED":
                await self._handle_partial_fill(order)

            elif status == "CANCELED":
                await self._handle_canceled_order(order)

            elif status == "REJECTED":
                await self._handle_rejected_order(order)

            elif status == "EXPIRED":
                await self._handle_expired_order(order)

            # v9.1 CORE: Update database trade status
            if client_order_id:
                await self._update_trade_from_order(client_order_id, status, order)

            # v9.1 CORE: Check for TP fills and BE moves
            await self._check_tp_management(client_order_id, order)

        except Exception as e:
            logger.error(f"Error handling order update: {e}")

    async def _handle_new_order(self, order: Dict):
        """Handle new order placement - v9.1"""
        symbol = order.get("s")
        side = order.get("S")
        order_type = order.get("o")
        client_order_id = order.get("c", "")

        logger.info(
            f"New order placed: {symbol} {side} {order_type} (ID: {client_order_id})"
        )

        # Track if this is a bot order
        if client_order_id.startswith("BOT_"):
            trade_id = self._extract_trade_id(client_order_id)
            if trade_id:
                self.tracked_orders[client_order_id] = {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "side": side,
                    "type": order_type,
                    "timestamp": time.time(),
                }

    async def _handle_filled_order(self, order: Dict):
        """Handle filled order with v9.1 enhancements"""
        try:
            symbol = order.get("s")
            side = order.get("S")
            order_type = order.get("o")
            client_order_id = order.get("c", "")
            avg_price = float(order.get("ap", 0))
            filled_qty = float(order.get("z", 0))
            commission = float(order.get("n", 0))
            realized_pnl = float(order.get("rp", 0))

            logger.info(
                f"Order FILLED - {symbol} {side} {order_type} @ {avg_price}, Qty: {filled_qty}, PnL: {realized_pnl}"
            )

            # v9.1 CORE: Determine notification type and color
            notification_data = {
                "symbol": symbol,
                "side": side,
                "price": avg_price,
                "quantity": filled_qty,
                "commission": commission,
                "client_order_id": client_order_id,
            }

            if order_type == "MARKET" and realized_pnl == 0:
                # Entry order
                notification_data["type"] = "ENTRY"
                notification_data["color"] = "blue"
                await discord_notifier.send_trade_notification(**notification_data)

            elif (
                order_type in ["STOP_MARKET", "TAKE_PROFIT_MARKET"] or realized_pnl != 0
            ):
                # Exit order (SL or TP)
                notification_data["type"] = "EXIT"
                notification_data["realized_pnl"] = realized_pnl
                notification_data["color"] = "green" if realized_pnl > 0 else "red"

                # v9.1 CORE: Determine exit reason
                if "SL" in client_order_id:
                    notification_data["exit_reason"] = "Stop Loss"
                elif "TP" in client_order_id:
                    tp_level = self._extract_tp_level(client_order_id)
                    notification_data["exit_reason"] = (
                        f"Take Profit {tp_level}" if tp_level else "Take Profit"
                    )
                else:
                    notification_data["exit_reason"] = "Manual"

                await discord_notifier.send_trade_notification(**notification_data)

            # v9.1 CORE: Handle TP fills for position management
            if "TP" in client_order_id:
                await self._handle_tp_fill(client_order_id, order)

        except Exception as e:
            logger.error(f"Error handling filled order: {e}")

    async def _handle_partial_fill(self, order: Dict):
        """Handle partial fill - v9.1"""
        symbol = order.get("s")
        filled_qty = float(order.get("z", 0))
        total_qty = float(order.get("q", 0))

        logger.info(f"Order partially filled: {symbol} ({filled_qty}/{total_qty})")

    async def _handle_canceled_order(self, order: Dict):
        """Handle canceled order - v9.1"""
        symbol = order.get("s")
        client_order_id = order.get("c", "")

        logger.info(f"Order canceled: {symbol} (ID: {client_order_id})")

    async def _handle_rejected_order(self, order: Dict):
        """Handle rejected order - v9.1"""
        symbol = order.get("s")
        client_order_id = order.get("c", "")

        logger.warning(f"Order rejected: {symbol} (ID: {client_order_id})")

        await discord_notifier.send_error_notification(
            f"Order Rejected: {symbol}",
            f"Order {client_order_id} was rejected. Check account status and symbol settings.",
        )

    async def _handle_expired_order(self, order: Dict):
        """Handle expired order - v9.1"""
        symbol = order.get("s")
        client_order_id = order.get("c", "")

        logger.info(f"Order expired: {symbol} (ID: {client_order_id})")

    async def _handle_tp_fill(self, client_order_id: str, order: Dict):
        """Handle TP fill for position management - v9.1 CORE FEATURE"""
        try:
            trade_id = self._extract_trade_id(client_order_id)
            tp_level = self._extract_tp_level(client_order_id)

            if not trade_id or not tp_level:
                return

            # Track TP fill
            if trade_id not in self.tp_fills:
                self.tp_fills[trade_id] = {"tp1": False, "tp2": False, "tp3": False}

            self.tp_fills[trade_id][tp_level.lower()] = True

            logger.info(f"TP{tp_level} filled for trade {trade_id}")

            # v9.1 CORE: Check if we should move SL to BE after TP1
            if tp_level == "1" and Config.MOVE_SL_TO_BE_AT_TP1:
                await self._move_sl_to_break_even(trade_id, order.get("s"))

            # v9.1 CORE: Check if we should activate trailing after TP2
            if tp_level == "2" and Config.USE_TRAILING_AFTER_TP2:
                await self._activate_trailing_stop(trade_id, order.get("s"))

        except Exception as e:
            logger.error(f"Error handling TP fill: {e}")

    async def _move_sl_to_break_even(self, trade_id: str, symbol: str):
        """Move SL to break even - v9.1 CORE FEATURE"""
        try:
            if trade_id in self.be_moves:
                return  # Already moved

            # Import here to avoid circular imports
            from binance_handler import binance_handler

            success, message = binance_handler.move_sl_to_break_even(symbol)

            if success:
                self.be_moves.add(trade_id)
                logger.info(f"Moved SL to BE for {symbol} (trade {trade_id})")

                # Update database
                with Session() as session:
                    trade = session.query(Trade).filter_by(id=int(trade_id)).first()
                    if trade:
                        trade.sl_moved_to_be = True
                        trade.break_even_price = trade.entry_price
                        session.commit()

                # Send notification
                if Config.NOTIFY_ON_BE:
                    await discord_notifier.send_system_notification(
                        f"Break Even Activated: {symbol}",
                        f"Stop loss moved to break even after TP1 fill",
                    )
            else:
                logger.warning(f"Failed to move SL to BE for {symbol}: {message}")

        except Exception as e:
            logger.error(f"Error moving SL to BE: {e}")

    async def _activate_trailing_stop(self, trade_id: str, symbol: str):
        """Activate trailing stop - v9.1 CORE FEATURE"""
        try:
            if trade_id in self.trailing_active:
                return  # Already active

            self.trailing_active.add(trade_id)
            logger.info(f"Trailing stop activated for {symbol} (trade {trade_id})")

            # Update database
            with Session() as session:
                trade = session.query(Trade).filter_by(id=int(trade_id)).first()
                if trade:
                    trade.trailing_activated = True
                    session.commit()

            # Send notification
            if Config.NOTIFY_ON_TRAILING:
                await discord_notifier.send_system_notification(
                    f"Trailing Stop Activated: {symbol}",
                    f"Trailing stop activated after TP2 fill",
                )

        except Exception as e:
            logger.error(f"Error activating trailing stop: {e}")

    async def _check_tp_management(self, client_order_id: str, order: Dict):
        """Check TP management logic - v9.1 CORE FEATURE"""
        try:
            if not client_order_id.startswith("BOT_"):
                return

            trade_id = self._extract_trade_id(client_order_id)
            if not trade_id:
                return

            # Update database with order information
            with Session() as session:
                trade = session.query(Trade).filter_by(id=int(trade_id)).first()
                if trade:
                    # Update TP fill status
                    if "TP1" in client_order_id and order.get("X") == "FILLED":
                        trade.tp1_filled = True
                    elif "TP2" in client_order_id and order.get("X") == "FILLED":
                        trade.tp2_filled = True
                    elif "TP3" in client_order_id and order.get("X") == "FILLED":
                        trade.tp3_filled = True

                    session.commit()

        except Exception as e:
            logger.error(f"Error checking TP management: {e}")

    def _extract_trade_id(self, client_order_id: str) -> Optional[str]:
        """Extract trade ID from client order ID - v9.1 UTILITY"""
        try:
            # Format: BOT_{trade_id}_SL or BOT_{trade_id}_TP1, etc.
            if client_order_id.startswith("BOT_"):
                parts = client_order_id.split("_")
                if len(parts) >= 3:
                    return parts[1]
        except Exception:
            pass
        return None

    def _extract_tp_level(self, client_order_id: str) -> Optional[str]:
        """Extract TP level from client order ID - v9.1 UTILITY"""
        try:
            if "TP1" in client_order_id:
                return "1"
            elif "TP2" in client_order_id:
                return "2"
            elif "TP3" in client_order_id:
                return "3"
        except Exception:
            pass
        return None

    async def _handle_account_update(self, msg: Dict):
        """Handle account update with v9.1 enhancements"""
        try:
            event_reason = msg.get("m", "UNKNOWN")
            event_time = msg.get("E", 0)

            logger.debug(f"Account update - Reason: {event_reason}")

            # v9.1 CORE: Update balances cache
            balances = msg.get("a", {}).get("B", [])
            for balance in balances:
                asset = balance.get("a")
                wallet_balance = float(balance.get("wb", 0))
                cross_wallet = float(balance.get("cw", 0))

                if asset == "USDT":
                    old_balance = self.balance_cache.get("USDT", 0)
                    self.balance_cache["USDT"] = wallet_balance

                    if abs(wallet_balance - old_balance) > 1.0:  # Significant change
                        logger.info(
                            f"Balance Update - Wallet: {wallet_balance:.2f} USDT (Δ: {wallet_balance - old_balance:+.2f})"
                        )

            # v9.1 CORE: Update positions cache
            positions = msg.get("a", {}).get("P", [])
            for position in positions:
                symbol = position.get("s")
                amount = float(position.get("pa", 0))
                entry_price = float(position.get("ep", 0))
                unrealized_pnl = float(position.get("up", 0))
                margin_type = position.get("mt", "isolated")

                # Update cache
                if amount != 0:
                    self.positions_cache[symbol] = {
                        "amount": amount,
                        "entry_price": entry_price,
                        "unrealized_pnl": unrealized_pnl,
                        "margin_type": margin_type,
                        "timestamp": time.time(),
                    }
                    logger.debug(
                        f"Position Update - {symbol}: {amount} @ {entry_price}, uPnL: {unrealized_pnl:.2f}"
                    )
                else:
                    # Position closed
                    if symbol in self.positions_cache:
                        del self.positions_cache[symbol]
                        logger.info(f"Position closed: {symbol}")

            # Call position update callbacks
            for callback in self.callbacks.get("POSITION_UPDATE", []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(positions)
                    else:
                        callback(positions)
                except Exception as e:
                    logger.error(f"Position update callback error: {e}")

            # Call balance update callbacks
            for callback in self.callbacks.get("BALANCE_UPDATE", []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(balances)
                    else:
                        callback(balances)
                except Exception as e:
                    logger.error(f"Balance update callback error: {e}")

        except Exception as e:
            logger.error(f"Error handling account update: {e}")

    async def _handle_margin_call(self, msg: Dict):
        """Handle margin call with v9.1 enhancements"""
        try:
            logger.critical("🚨 MARGIN CALL RECEIVED! 🚨")

            # Send urgent Discord notification
            await discord_notifier.send_error_notification(
                "🚨 MARGIN CALL 🚨",
                "**IMMEDIATE ACTION REQUIRED!**\nAccount is at risk of liquidation.\nCheck positions and add margin or close positions.",
            )

            # Log details
            positions = msg.get("p", [])
            at_risk_positions = []

            for pos in positions:
                symbol = pos.get("s")
                side = pos.get("ps")
                amount = pos.get("pa")
                margin_type = pos.get("mt")
                unrealized_pnl = float(pos.get("up", 0))

                position_info = f"{symbol} {side} {amount} ({margin_type}) - uPnL: {unrealized_pnl:.2f}"
                at_risk_positions.append(position_info)
                logger.critical(f"At risk: {position_info}")

            # Send detailed notification
            if at_risk_positions:
                positions_text = "\n".join(
                    at_risk_positions[:5]
                )  # Limit to 5 positions
                await discord_notifier.send_error_notification(
                    "Positions at Risk", f"```\n{positions_text}\n```"
                )

        except Exception as e:
            logger.error(f"Error handling margin call: {e}")

    async def _handle_config_update(self, msg: Dict):
        """Handle account configuration update - v9.1"""
        try:
            logger.info("Account configuration updated")

            # Handle leverage updates
            if "ac" in msg:
                symbol = msg["ac"].get("s")
                leverage = msg["ac"].get("l")
                if symbol and leverage:
                    logger.info(f"Leverage updated for {symbol}: {leverage}x")

                    # Clear leverage cache for this symbol
                    from binance_handler import binance_handler

                    if hasattr(binance_handler, "leverage_cache"):
                        keys_to_remove = [
                            k
                            for k in binance_handler.leverage_cache.keys()
                            if k.startswith(symbol)
                        ]
                        for key in keys_to_remove:
                            del binance_handler.leverage_cache[key]

        except Exception as e:
            logger.error(f"Error handling config update: {e}")

    async def _handle_strategy_update(self, msg: Dict):
        """Handle strategy update - v9.1"""
        try:
            logger.debug(f"Strategy update: {msg}")

            strategy_id = msg.get("si")
            strategy_type = msg.get("st")
            strategy_status = msg.get("ss")

            if strategy_status == "NEW":
                logger.info(
                    f"New strategy activated: {strategy_type} (ID: {strategy_id})"
                )
            elif strategy_status == "CANCELED":
                logger.info(f"Strategy canceled: {strategy_type} (ID: {strategy_id})")

        except Exception as e:
            logger.error(f"Error handling strategy update: {e}")

    async def _update_trade_from_order(
        self, client_order_id: str, status: str, order_data: Dict
    ):
        """Update trade status in database - v9.1 ENHANCED"""
        try:
            # Only update for bot orders
            if not client_order_id.startswith("BOT_"):
                return

            trade_id = self._extract_trade_id(client_order_id)
            if not trade_id:
                return

            with Session() as session:
                trade = session.query(Trade).filter_by(id=int(trade_id)).first()

                if not trade:
                    return

                # Update based on order type and status
                if "ENTRY" in client_order_id and status == "FILLED":
                    # Entry order filled
                    trade.entry_price = float(order_data.get("ap", 0))
                    trade.entry_quantity = float(order_data.get("z", 0))
                    trade.entry_commission = float(order_data.get("n", 0))
                    trade.status = "open"

                elif status == "FILLED" and float(order_data.get("rp", 0)) != 0:
                    # Exit order filled (has realized PnL)
                    trade.exit_price = float(order_data.get("ap", 0))
                    trade.exit_quantity = float(order_data.get("z", 0))
                    trade.exit_commission = float(order_data.get("n", 0))
                    trade.exit_time = datetime.utcnow()
                    trade.pnl_usdt = float(order_data.get("rp", 0))

                    # Determine exit reason
                    if "SL" in client_order_id:
                        trade.exit_reason = "sl"
                    elif "TP1" in client_order_id:
                        trade.exit_reason = "tp1"
                    elif "TP2" in client_order_id:
                        trade.exit_reason = "tp2"
                    elif "TP3" in client_order_id:
                        trade.exit_reason = "tp3"
                    else:
                        trade.exit_reason = "manual"

                    # Check if position is fully closed
                    remaining_qty = trade.entry_quantity - trade.exit_quantity
                    if remaining_qty <= 0.001:  # Account for precision
                        trade.status = "closed"

                elif status == "CANCELED":
                    # Order canceled - update notes
                    if not trade.notes:
                        trade.notes = ""
                    trade.notes += f"Order {client_order_id} canceled. "

                elif status == "REJECTED":
                    # Order rejected
                    if not trade.notes:
                        trade.notes = ""
                    trade.notes += f"Order {client_order_id} rejected. "

                trade.updated_at = datetime.utcnow()
                session.commit()

                logger.debug(
                    f"Updated trade {trade_id} from order {client_order_id} (status: {status})"
                )

        except Exception as e:
            logger.error(f"Error updating trade from order: {e}")

    # v9.1 CORE: Status and monitoring methods
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status - v9.1 FEATURE"""
        return {
            "connected": self.connected,
            "running": self.running,
            "reconnect_attempts": self.reconnect_attempts,
            "last_heartbeat": self.last_heartbeat,
            "last_update": self.last_update_time,
            "positions_cached": len(self.positions_cache),
            "orders_cached": len(self.orders_cache),
            "tracked_orders": len(self.tracked_orders),
            "tp_fills_tracked": len(self.tp_fills),
            "be_moves": len(self.be_moves),
            "trailing_active": len(self.trailing_active),
        }

    def get_cached_positions(self) -> Dict[str, Any]:
        """Get cached positions - v9.1 FEATURE"""
        return self.positions_cache.copy()

    def get_cached_balance(self) -> Dict[str, float]:
        """Get cached balance - v9.1 FEATURE"""
        return self.balance_cache.copy()

    def clear_caches(self):
        """Clear all caches - v9.1 FEATURE"""
        self.positions_cache.clear()
        self.orders_cache.clear()
        self.tracked_orders.clear()
        self.tp_fills.clear()
        self.be_moves.clear()
        self.trailing_active.clear()
        logger.info("User stream caches cleared")


# v9.1 CORE: Singleton instance with enhanced management
user_stream: Optional[FuturesUserStream] = None
_stream_lock = Lock()


async def initialize_user_stream(client: AsyncClient) -> Optional[FuturesUserStream]:
    """Initialize user stream with v9.1 enhancements"""
    global user_stream

    with _stream_lock:
        if user_stream and user_stream.running:
            logger.warning("User stream already initialized and running")
            return user_stream

    try:
        user_stream = FuturesUserStream(client)
        await user_stream.start()
        logger.info("User stream initialized successfully (v9.1)")
        return user_stream

    except Exception as e:
        logger.error(f"Failed to initialize user stream: {e}")
        return None


async def stop_user_stream():
    """Stop user stream with v9.1 enhancements"""
    global user_stream

    with _stream_lock:
        if user_stream:
            await user_stream.stop()
            user_stream = None
            logger.info("User stream stopped")


def get_user_stream() -> Optional[FuturesUserStream]:
    """Get current user stream instance - v9.1 FEATURE"""
    return user_stream


def is_stream_connected() -> bool:
    """Check if stream is connected - v9.1 FEATURE"""
    return user_stream is not None and user_stream.connected

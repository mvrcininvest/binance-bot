"""
Main Trading Bot Class for v9.1 with Hybrid Ultra-Diagnostics
Enhanced with comprehensive features:
- Centralized state management
- Enhanced position management
- Multi-TP support with break-even and trailing
- Comprehensive error handling
- Thread-safe operations
- HYBRID ULTRA-DIAGNOSTICS integration
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from threading import Lock, Event
from dataclasses import dataclass, field
import signal
import sys
import psutil
import traceback
import uuid

from binance import AsyncClient
from binance.exceptions import BinanceAPIException

from config import Config
from database import (
    Session,
    Trade,
    BotSettings,
    AlertHistory,
    get_setting,
    set_setting,
    has_open_position,
    create_trade,
    update_trade,
    close_trade,
    check_idempotency,
    record_alert,
    record_signal_decision,
    get_open_positions,
    # HYBRID ULTRA-DIAGNOSTICS: New imports
    create_execution_trace,
    complete_execution_trace,
    log_performance_metric,
    log_system_health,
    create_decision_trace,
    get_diagnostic_summary,
    cleanup_diagnostic_data,
)
from binance_handler import binance_handler
from futures_user_stream import (
    initialize_user_stream,
    stop_user_stream,
    get_user_stream,
)
from discord_notifications import discord_notifier
from signal_intelligence import SignalIntelligence
from mode_manager import ModeManager
from analytics import Analytics
# HYBRID ULTRA-DIAGNOSTICS: New component imports
from diagnostics import DiagnosticsEngine
from decision_engine import DecisionEngine
from pine_health_monitor import pine_monitor
from pattern_detector import PatternDetector

logger = logging.getLogger(__name__)


@dataclass
class BotState:
    """Bot state container - v9.1 CORE with Hybrid Ultra-Diagnostics"""

    is_running: bool = False
    is_paused: bool = False
    emergency_mode: bool = False
    current_mode: str = "normal"
    start_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    open_positions: int = 0
    total_trades: int = 0
    session_pnl: float = 0.0
    balance_usdt: float = 0.0
    connection_status: Dict[str, Any] = None
    last_signal_time: Optional[datetime] = None
    processing_signal: bool = False
    
    # HYBRID ULTRA-DIAGNOSTICS: Enhanced state tracking
    system_health_score: float = 100.0
    last_diagnostic_check: Optional[datetime] = None
    diagnostic_alerts_count: int = 0
    performance_degradation_detected: bool = False
    anomaly_detection_active: bool = True
    pattern_recognition_accuracy: float = 0.0
    decision_confidence_avg: float = 0.0
    execution_efficiency: float = 100.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    diagnostic_mode: str = "NORMAL"  # NORMAL, ENHANCED, FULL
    last_pattern_analysis: Optional[datetime] = None
    ml_model_drift_score: float = 0.0


class TradingBot:
    """Enhanced Trading Bot v9.1 - Main orchestrator"""

    def __init__(self):
        """Initialize trading bot with v9.1 enhancements and Hybrid Ultra-Diagnostics"""
        self.state = BotState()
        self.client: Optional[AsyncClient] = None

        # v9.1 CORE: Thread safety
        self.state_lock = Lock()
        self.signal_lock = Lock()
        self.shutdown_event = Event()

        # v9.1 CORE: Component managers
        self.signal_intelligence = SignalIntelligence()
        self.mode_manager = ModeManager()
        self.analytics = Analytics()

        # HYBRID ULTRA-DIAGNOSTICS: New diagnostic components
        self.diagnostics_engine = DiagnosticsEngine()
        self.decision_engine = DecisionEngine()
        self.pine_monitor = pine_monitor()
        self.pattern_detector = PatternDetector()

        # v9.1 CORE: Enhanced tracking
        self.active_trades = {}  # trade_id -> Trade object
        self.pending_orders = {}  # client_order_id -> order_info
        self.signal_cooldowns = {}  # symbol -> last_signal_time

        # HYBRID ULTRA-DIAGNOSTICS: Enhanced tracking
        self.execution_traces = {}  # trace_id -> execution_data
        self.performance_metrics = {}  # metric_name -> metric_data
        self.diagnostic_alerts = []  # List of active diagnostic alerts
        self.pattern_cache = {}  # symbol -> pattern_data
        self.decision_history = []  # Recent decision traces

        # v9.1 CORE: Performance tracking
        self.session_stats = {
            "signals_received": 0,
            "signals_accepted": 0,
            "trades_opened": 0,
            "trades_closed": 0,
            "tp1_hits": 0,
            "tp2_hits": 0,
            "tp3_hits": 0,
            "sl_hits": 0,
            "be_moves": 0,
            "trailing_activations": 0,
            # HYBRID ULTRA-DIAGNOSTICS: Enhanced stats
            "diagnostic_checks": 0,
            "anomalies_detected": 0,
            "patterns_recognized": 0,
            "decision_overrides": 0,
            "performance_alerts": 0,
        }

        logger.info("Trading Bot v9.1 with Hybrid Ultra-Diagnostics initialized")

    async def start(self):
        """Start the trading bot with v9.1 enhancements"""
        try:
            with self.state_lock:
                if self.state.is_running:
                    logger.warning("Bot is already running")
                    return

                self.state.is_running = True
                self.state.start_time = datetime.utcnow()

            logger.info("🚀 Starting Trading Bot v9.1...")

            # Initialize Binance client
            await self._initialize_client()

            # Load bot settings from database
            await self._load_bot_settings()

            # Initialize components
            await self._initialize_components()

            # Start background tasks
            await self._start_background_tasks()

            # Send startup notification
            await discord_notifier.send_system_notification(
                "🚀 Bot Started",
                f"Trading Bot v{Config.BOT_VERSION} started successfully\n"
                f"Mode: {self.state.current_mode}\n"
                f"Emergency: {'Enabled' if self.state.emergency_mode else 'Disabled'}\n"
                f"Testnet: {'Yes' if Config.IS_TESTNET else 'No'}",
            )

            logger.info("✅ Trading Bot v9.1 started successfully")

        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the trading bot with v9.1 enhancements"""
        try:
            with self.state_lock:
                if not self.state.is_running:
                    return

                self.state.is_running = False

            logger.info("🛑 Stopping Trading Bot v9.1...")

            # Signal shutdown to all tasks
            self.shutdown_event.set()

            # Stop user stream
            await stop_user_stream()

            # Close Binance client
            if self.client:
                await self.client.close_connection()

            # Send shutdown notification
            uptime = (
                datetime.utcnow() - self.state.start_time
                if self.state.start_time
                else timedelta(0)
            )

            await discord_notifier.send_system_notification(
                "🛑 Bot Stopped",
                f"Trading Bot stopped after {uptime}\n"
                f"Session Stats:\n"
                f"• Signals: {self.session_stats['signals_received']} received, {self.session_stats['signals_accepted']} accepted\n"
                f"• Trades: {self.session_stats['trades_opened']} opened, {self.session_stats['trades_closed']} closed\n"
                f"• Session PnL: {self.state.session_pnl:.2f} USDT",
            )

            logger.info("✅ Trading Bot v9.1 stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping bot: {e}")

    async def _initialize_client(self):
        """Initialize Binance client"""
        try:
            self.client = await AsyncClient.create(
                api_key=Config.get_api_key(),
                api_secret=Config.get_api_secret(),
                testnet=Config.IS_TESTNET,
            )

            # Test connection
            account_info = await self.client.futures_account()
            self.state.balance_usdt = float(account_info["totalWalletBalance"])

            logger.info(
                f"✅ Binance client initialized - Balance: {self.state.balance_usdt:.2f} USDT"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise

    async def _load_bot_settings(self):
        """Load bot settings from database - v9.1 ENHANCED"""
        try:
            with Session() as session:
                # Load current mode
                mode = get_setting(session, "mode", Config.DEFAULT_MODE)
                self.state.current_mode = mode
                self.mode_manager.set_mode(mode)

                # Load pause state
                self.state.is_paused = get_setting(session, "is_paused", False)

                # Load emergency mode
                self.state.emergency_mode = get_setting(
                    session, "emergency_enabled", Config.EMERGENCY_ENABLED
                )

                # Update last restart time
                set_setting(
                    session,
                    "last_restart",
                    datetime.utcnow().isoformat(),
                    "string",
                    "system",
                )

                logger.info(
                    f"Settings loaded - Mode: {mode}, Paused: {self.state.is_paused}, Emergency: {self.state.emergency_mode}"
                )

        except Exception as e:
            logger.error(f"Failed to load bot settings: {e}")
            # Use defaults
            self.state.current_mode = Config.DEFAULT_MODE
            self.mode_manager.set_mode(Config.DEFAULT_MODE)

    async def _initialize_components(self):
        """Initialize bot components - v9.1 ENHANCED with Hybrid Ultra-Diagnostics"""
        try:
            # Initialize binance handler
            binance_handler.initialize(self.client)

            # Initialize user stream
            user_stream = await initialize_user_stream(self.client)
            if not user_stream:
                raise Exception("Failed to initialize user stream")

            # Register user stream callbacks
            user_stream.register_callback("ORDER_TRADE_UPDATE", self._on_order_update)
            user_stream.register_callback("ACCOUNT_UPDATE", self._on_account_update)
            user_stream.register_callback("POSITION_UPDATE", self._on_position_update)
            user_stream.register_callback("BALANCE_UPDATE", self._on_balance_update)

            # Initialize signal intelligence
            await self.signal_intelligence.initialize()

            # Initialize analytics
            await self.analytics.initialize()

            # HYBRID ULTRA-DIAGNOSTICS: Initialize diagnostic components
            await self.diagnostics_engine.initialize()
            await self.decision_engine.initialize()
            await self.pine_health_monitor.initialize()
            await self.pattern_detector.initialize()

            # HYBRID ULTRA-DIAGNOSTICS: Log initialization
            with Session() as session:
                create_execution_trace(
                    session,
                    "system_initialization",
                    "COMPLETED",
                    {"components": "all", "diagnostics_enabled": True}
                )

            logger.info("✅ All components with Hybrid Ultra-Diagnostics initialized")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            # HYBRID ULTRA-DIAGNOSTICS: Log initialization failure
            with Session() as session:
                create_execution_trace(
                    session,
                    "system_initialization",
                    "FAILED",
                    {"error": str(e), "traceback": traceback.format_exc()}
                )
            raise

    async def _start_background_tasks(self):
        """Start background tasks - v9.1 ENHANCED with Hybrid Ultra-Diagnostics"""
        try:
            # Heartbeat task
            asyncio.create_task(self._heartbeat_task())

            # Position monitoring task
            asyncio.create_task(self._position_monitor_task())

            # Balance monitoring task
            asyncio.create_task(self._balance_monitor_task())

            # Analytics task
            asyncio.create_task(self._analytics_task())

            # Cleanup task
            asyncio.create_task(self._cleanup_task())

            # HYBRID ULTRA-DIAGNOSTICS: New diagnostic tasks
            asyncio.create_task(self._diagnostic_monitor_task())
            asyncio.create_task(self._system_health_task())
            asyncio.create_task(self._pattern_analysis_task())
            asyncio.create_task(self._performance_monitor_task())
            asyncio.create_task(self._decision_analysis_task())

            logger.info("✅ Background tasks with Hybrid Ultra-Diagnostics started")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            raise

    async def _heartbeat_task(self):
        """Heartbeat task - v9.1 CORE"""
        while self.state.is_running and not self.shutdown_event.is_set():
            try:
                with self.state_lock:
                    self.state.last_heartbeat = datetime.utcnow()

                # Update connection status
                user_stream = get_user_stream()
                if user_stream:
                    self.state.connection_status = user_stream.get_connection_status()

                await asyncio.sleep(30)  # 30 second heartbeat

            except Exception as e:
                logger.error(f"Heartbeat task error: {e}")
                await asyncio.sleep(30)

    async def _position_monitor_task(self):
        """Position monitoring task - v9.1 CORE"""
        while self.state.is_running and not self.shutdown_event.is_set():
            try:
                # Update active trades from database
                with Session() as session:
                    open_trades = get_open_positions(session)

                    with self.state_lock:
                        self.state.open_positions = len(open_trades)
                        self.active_trades = {
                            str(trade.id): trade for trade in open_trades
                        }

                # Check for position management opportunities
                await self._check_position_management()

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Position monitor task error: {e}")
                await asyncio.sleep(10)

    async def _balance_monitor_task(self):
        """Balance monitoring task - v9.1 CORE"""
        while self.state.is_running and not self.shutdown_event.is_set():
            try:
                # Get current balance
                user_stream = get_user_stream()
                if user_stream:
                    balance_cache = user_stream.get_cached_balance()
                    if "USDT" in balance_cache:
                        with self.state_lock:
                            old_balance = self.state.balance_usdt
                            self.state.balance_usdt = balance_cache["USDT"]

                            # Update session PnL
                            if old_balance > 0:
                                self.state.session_pnl = (
                                    self.state.balance_usdt - old_balance
                                )

                # Check for low balance alert
                if (
                    Config.ALERT_ON_LOW_BALANCE
                    and self.state.balance_usdt < Config.MIN_BALANCE_USDT
                ):
                    await discord_notifier.send_error_notification(
                        "⚠️ Low Balance Alert",
                        f"Current balance: {self.state.balance_usdt:.2f} USDT\n"
                        f"Minimum required: {Config.MIN_BALANCE_USDT:.2f} USDT",
                    )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Balance monitor task error: {e}")
                await asyncio.sleep(60)

    async def _analytics_task(self):
        """Analytics task - v9.1 CORE"""
        while self.state.is_running and not self.shutdown_event.is_set():
            try:
                if Config.ENABLE_METRICS:
                    await self.analytics.update_metrics()

                await asyncio.sleep(Config.METRICS_INTERVAL_SECONDS)

            except Exception as e:
                logger.error(f"Analytics task error: {e}")
                await asyncio.sleep(Config.METRICS_INTERVAL_SECONDS)

    async def _cleanup_task(self):
        """Cleanup task - v9.1 CORE"""
        while self.state.is_running and not self.shutdown_event.is_set():
            try:
                # Clean old signal cooldowns
                current_time = time.time()
                expired_symbols = [
                    symbol
                    for symbol, last_time in self.signal_cooldowns.items()
                    if current_time - last_time > Config.SIGNAL_COOLDOWN_SECONDS
                ]
                for symbol in expired_symbols:
                    del self.signal_cooldowns[symbol]

                # Clean old pending orders
                expired_orders = [
                    order_id
                    for order_id, order_info in self.pending_orders.items()
                    if current_time - order_info.get("timestamp", 0) > 3600  # 1 hour
                ]
                for order_id in expired_orders:
                    del self.pending_orders[order_id]

                await asyncio.sleep(300)  # Clean every 5 minutes

            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(300)

    async def _check_position_management(self):
        """Check position management opportunities - v9.1 CORE"""
        try:
            for trade_id, trade in self.active_trades.items():
                # Check for break-even opportunities
                if (
                    not trade.sl_moved_to_be
                    and trade.tp1_filled
                    and Config.MOVE_SL_TO_BE_AT_TP1
                ):
                    success, message = binance_handler.move_sl_to_break_even(
                        trade.symbol
                    )
                    if success:
                        with Session() as session:
                            update_trade(
                                session,
                                int(trade_id),
                                {
                                    "sl_moved_to_be": True,
                                    "break_even_price": trade.entry_price,
                                },
                            )
                        self.session_stats["be_moves"] += 1
                        logger.info(f"Moved SL to BE for {trade.symbol}")

                # Check for trailing stop opportunities
                if (
                    not trade.trailing_activated
                    and trade.tp2_filled
                    and Config.USE_TRAILING_AFTER_TP2
                ):
                    # Activate trailing stop logic here
                    with Session() as session:
                        update_trade(
                            session, int(trade_id), {"trailing_activated": True}
                        )
                    self.session_stats["trailing_activations"] += 1
                    logger.info(f"Activated trailing stop for {trade.symbol}")

        except Exception as e:
            logger.error(f"Error checking position management: {e}")

    # v9.1 CORE: User stream callbacks
    async def _on_order_update(self, msg: Dict):
        """Handle order update callback - v9.1"""
        try:
            order = msg.get("o", {})
            client_order_id = order.get("c", "")
            status = order.get("X")

            if client_order_id.startswith("BOT_"):
                # Update pending orders
                if status in ["FILLED", "CANCELED", "REJECTED"]:
                    self.pending_orders.pop(client_order_id, None)

                # Update session stats
                if status == "FILLED":
                    if "TP1" in client_order_id:
                        self.session_stats["tp1_hits"] += 1
                    elif "TP2" in client_order_id:
                        self.session_stats["tp2_hits"] += 1
                    elif "TP3" in client_order_id:
                        self.session_stats["tp3_hits"] += 1
                    elif "SL" in client_order_id:
                        self.session_stats["sl_hits"] += 1

        except Exception as e:
            logger.error(f"Error in order update callback: {e}")

    async def _on_account_update(self, msg: Dict):
        """Handle account update callback - v9.1"""
        try:
            # Update balance from account update
            balances = msg.get("a", {}).get("B", [])
            for balance in balances:
                if balance.get("a") == "USDT":
                    with self.state_lock:
                        self.state.balance_usdt = float(balance.get("wb", 0))

        except Exception as e:
            logger.error(f"Error in account update callback: {e}")

    async def _on_position_update(self, positions: List[Dict]):
        """Handle position update callback - v9.1"""
        try:
            open_positions = sum(1 for pos in positions if float(pos.get("pa", 0)) != 0)
            with self.state_lock:
                self.state.open_positions = open_positions

        except Exception as e:
            logger.error(f"Error in position update callback: {e}")

    async def _on_balance_update(self, balances: List[Dict]):
        """Handle balance update callback - v9.1"""
        try:
            for balance in balances:
                if balance.get("a") == "USDT":
                    with self.state_lock:
                        self.state.balance_usdt = float(balance.get("wb", 0))

        except Exception as e:
            logger.error(f"Error in balance update callback: {e}")

    # v9.1 CORE: Signal processing
    async def process_signal(self, signal_data: Dict) -> Tuple[bool, str]:
        """Process trading signal with v9.1 enhancements"""
        start_time = time.time()

        try:
            with self.signal_lock:
                if self.state.processing_signal:
                    return False, "Another signal is being processed"
                self.state.processing_signal = True

            try:
                self.session_stats["signals_received"] += 1
                self.state.last_signal_time = datetime.utcnow()

                logger.info(
                    f"🔄 Processing signal: {signal_data.get('symbol')} {signal_data.get('action')} {signal_data.get('tier')}"
                )

                # v9.1 CORE: Enhanced signal validation
                is_valid, validation_message = await self._validate_signal(signal_data)
                if not is_valid:
                    await self._record_signal_rejection(
                        signal_data, validation_message, start_time
                    )
                    return False, validation_message

                # v9.1 CORE: Signal intelligence analysis
                intelligence_result = await self.signal_intelligence.analyze_signal(
                    signal_data
                )
                if not intelligence_result["accept"]:
                    await self._record_signal_rejection(
                        signal_data, intelligence_result["reason"], start_time
                    )
                    return False, intelligence_result["reason"]

                # v9.1 CORE: Mode-based filtering
                mode_result = self.mode_manager.should_accept_signal(signal_data)
                if not mode_result["accept"]:
                    await self._record_signal_rejection(
                        signal_data, mode_result["reason"], start_time
                    )
                    return False, mode_result["reason"]
                logger.info("✅ Przeszedłem wszystkie testy. Przystępuję do egzekucji zlecenia.")

                # v9.1 CORE: Execute trade
                success, message = await self._execute_trade(
                    signal_data, intelligence_result
                )

                if success:
                    self.session_stats["signals_accepted"] += 1
                    self.session_stats["trades_opened"] += 1

                    # Record successful signal
                    processing_time = int((time.time() - start_time) * 1000)
                    with Session() as session:
                        record_signal_decision(
                            session, signal_data, True, None, processing_time
                        )

                return success, message

            finally:
                with self.signal_lock:
                    self.state.processing_signal = False

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            with self.signal_lock:
                self.state.processing_signal = False
            return False, f"Processing error: {str(e)}"

    async def _validate_signal(self, signal_data: Dict) -> Tuple[bool, str]:
        """Validate signal with v9.1 enhancements"""
        try:
            # Check if bot is paused
            if self.state.is_paused:
                return False, "Bot is paused"

            # Check required fields
            required_fields = ["symbol", "action", "tier", "price", "timestamp"]
            for field in required_fields:
                if field not in signal_data:
                    return False, f"Missing required field: {field}"

            symbol = signal_data["symbol"]
            action = signal_data["action"].upper()
            tier = signal_data["tier"]
            price = float(signal_data["price"])

            # v9.1 CORE: Check indicator version
            indicator_version = signal_data.get("indicator_version")
            if indicator_version != Config.INDICATOR_VERSION_REQUIRED:
                return (
                    False,
                    f"Indicator version mismatch: {indicator_version} != {Config.INDICATOR_VERSION_REQUIRED}",
                )

            # Check tier minimum
            if not Config.validate_tier(tier):
                return False, f"Tier {tier} below minimum {Config.TIER_MINIMUM}"

            # Check emergency mode
            if tier == "Emergency" and not self.state.emergency_mode:
                return False, "Emergency signals disabled"

            # Check signal cooldown
            current_time = time.time()
            if symbol in self.signal_cooldowns:
                time_since_last = current_time - self.signal_cooldowns[symbol]
                if time_since_last < Config.SIGNAL_COOLDOWN_SECONDS:
                    return (
                        False,
                        f"Signal cooldown active for {symbol} ({Config.SIGNAL_COOLDOWN_SECONDS - time_since_last:.1f}s remaining)",
                    )

            # Check existing position
            if Config.SINGLE_POSITION_PER_SYMBOL:
                with Session() as session:
                    if has_open_position(session, symbol):
                        return False, f"Already have open position for {symbol}"

            # Check maximum positions
            if self.state.open_positions >= Config.MAX_POSITIONS:
                return False, f"Maximum positions reached ({Config.MAX_POSITIONS})"

            # v9.1 CORE: Check price drift
            try:
                current_price = await binance_handler.get_current_price(symbol)
                price_drift = abs(price - current_price) / current_price * 100

                if price_drift > Config.MAX_PRICE_DRIFT_PCT:
                    return (
                        False,
                        f"Price drift too high: {price_drift:.2f}% > {Config.MAX_PRICE_DRIFT_PCT}%",
                    )

                # Add current price to signal data
                signal_data["current_price"] = current_price
                signal_data["price_drift_pct"] = price_drift

            except Exception as e:
                logger.warning(f"Could not validate price drift: {e}")

            # v9.1 CORE: Check alert age
            try:
                alert_timestamp = signal_data.get("timestamp")
                if alert_timestamp:
                    if isinstance(alert_timestamp, str):
                        alert_time = datetime.fromisoformat(
                            alert_timestamp.replace("Z", "+00:00")
                        )
                    else:
                        alert_time = datetime.fromtimestamp(alert_timestamp)

                    age_seconds = (
                        datetime.utcnow() - alert_time.replace(tzinfo=None)
                    ).total_seconds()
                    signal_data["alert_age_seconds"] = age_seconds

                    if age_seconds > Config.ALERT_MAX_AGE_SEC:
                        return (
                            False,
                            f"Alert too old: {age_seconds:.1f}s > {Config.ALERT_MAX_AGE_SEC}s",
                        )

            except Exception as e:
                logger.warning(f"Could not validate alert age: {e}")

            return True, "Signal validation passed"

        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False, f"Validation error: {str(e)}"

    async def _execute_trade(
        self, signal_data: Dict, intelligence_result: Dict
    ) -> Tuple[bool, str]:
        """Execute trade with v9.1 enhancements"""
        try:
            symbol = signal_data["symbol"]
            action = signal_data["action"].upper()
            tier = signal_data["tier"]

            logger.info(f"🎯 Executing trade: {symbol} {action} {tier}")

            # v9.1 CORE: Calculate position parameters
            position_params = await self._calculate_position_params(
                signal_data, intelligence_result
            )

            # v9.1 CORE: Set leverage
            leverage_success, leverage_msg = binance_handler.set_leverage(
                symbol, position_params["leverage"]
            )
            if not leverage_success:
                return False, f"Failed to set leverage: {leverage_msg}"

            # v9.1 CORE: Create trade record with idempotency
            idempotency_key = self._generate_idempotency_key(signal_data)

            # Check for duplicate
            with Session() as session:
                if check_idempotency(session, idempotency_key):
                    return False, "Duplicate signal detected"

            # Create trade record
            trade_data = {
                "symbol": symbol,
                "side": action,
                "status": "pending",
                "idempotency_key": idempotency_key,
                "entry_price": signal_data["price"],
                "entry_quantity": position_params["quantity"],
                "leverage_used": position_params["leverage"],
                "leverage_hint": signal_data.get("leverage"),
                "position_size_usdt": position_params["position_size"],
                "risk_amount_usdt": position_params["risk_amount"],
                "stop_loss": position_params["stop_loss"],
                "take_profit_1": position_params["take_profit_1"],
                "take_profit_2": position_params.get("take_profit_2"),
                "take_profit_3": position_params.get("take_profit_3"),
                "signal_tier": tier,
                "signal_strength": signal_data.get("strength", 0.0),
                "signal_timeframe": signal_data.get("timeframe"),
                "signal_session": signal_data.get("session"),
                "indicator_version": signal_data.get("indicator_version"),
                "institutional_flow": intelligence_result.get(
                    "institutional_flow", 0.0
                ),
                "retest_confidence": intelligence_result.get("retest_confidence", 0.0),
                "fake_breakout_detected": intelligence_result.get(
                    "fake_breakout_detected", False
                ),
                "enhanced_regime": intelligence_result.get(
                    "enhanced_regime", "NEUTRAL"
                ),
                "regime_confidence": intelligence_result.get("regime_confidence", 0.0),
                "mtf_agreement_ratio": intelligence_result.get(
                    "mtf_agreement_ratio", 0.0
                ),
                "volume_context": intelligence_result.get("volume_context"),
                "mode_used": self.state.current_mode,
                "alert_data": signal_data,
            }

            with Session() as session:
                trade = create_trade(session, trade_data)
                trade_id = trade.id

            # v9.1 CORE: Generate client order IDs with trade ID
            client_tags = {
                "entry": f"BOT_{trade_id}_ENTRY",
                "sl": f"BOT_{trade_id}_SL",
                "tp1": f"BOT_{trade_id}_TP1",
                "tp2": f"BOT_{trade_id}_TP2",
                "tp3": f"BOT_{trade_id}_TP3",
            }

            # Update trade with client tags
            with Session() as session:
                update_trade(session, trade_id, {"client_tags": client_tags})

            # v9.1 CORE: Place orders
            order_results = binance_handler.place_trade_orders(
                symbol=symbol,
                side=action,
                quantity=position_params["quantity"],
                stop_loss=position_params["stop_loss"],
                take_profits=[
                    position_params["take_profit_1"],
                    position_params.get("take_profit_2"),
                    position_params.get("take_profit_3"),
                ],
                client_tags=client_tags,
                leverage=position_params["leverage"],
            )

            if not order_results["success"]:
                # Update trade status to failed
                with Session() as session:
                    update_trade(
                        session,
                        trade_id,
                        {"status": "failed", "notes": order_results["message"]},
                    )
                return False, order_results["message"]

            # Update trade with order information
            with Session() as session:
                update_trade(
                    session,
                    trade_id,
                    {
                        "status": "open",
                        "exchange_order_id": order_results.get("entry_order_id"),
                        "entry_time": datetime.utcnow(),
                    },
                )

            # Update signal cooldown
            self.signal_cooldowns[symbol] = time.time()

            # Send success notification
            await discord_notifier.send_trade_notification(
                symbol=symbol,
                side=action,
                price=signal_data["price"],
                quantity=position_params["quantity"],
                leverage=position_params["leverage"],
                stop_loss=position_params["stop_loss"],
                take_profit_1=position_params["take_profit_1"],
                tier=tier,
                type="ENTRY",
                color="blue",
            )

            logger.info(
                f"✅ Trade executed successfully: {symbol} {action} (ID: {trade_id})"
            )
            return True, f"Trade opened successfully (ID: {trade_id})"

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False, f"Execution error: {str(e)}"

    async def _calculate_position_params(
        self, signal_data: Dict, intelligence_result: Dict
    ) -> Dict:
        """Calculate position parameters with v9.1 enhancements"""
        try:
            symbol = signal_data["symbol"]
            tier = signal_data["tier"]
            price = float(signal_data["price"])

            # v9.1 CORE: Get mode configuration
            mode_config = Config.get_mode_config(self.state.current_mode)

            # v9.1 CORE: Calculate leverage
            if Config.USE_INDICATOR_LEVERAGE and "leverage" in signal_data:
                leverage = min(
                    int(signal_data["leverage"]),
                    Config.get_max_leverage_for_symbol(symbol),
                    mode_config["max_leverage"],
                )
            else:
                tier_config = Config.get_tier_config(tier)
                leverage = min(
                    tier_config["max_leverage"],
                    Config.get_max_leverage_for_symbol(symbol),
                    mode_config["max_leverage"],
                )

            # v9.1 CORE: Calculate risk amount
            base_risk = Config.DEFAULT_RISK_PERCENT
            tier_multiplier = Config.TIER_RISK_MULTIPLIERS.get(tier, 1.0)
            mode_multiplier = mode_config["risk_multiplier"]

            # v9.1 CORE: Apply size multiplier from indicator
            size_multiplier = 1.0
            if (
                Config.USE_INDICATOR_SIZE_MULTIPLIER
                and "size_multiplier" in signal_data
            ):
                size_multiplier = float(signal_data["size_multiplier"])
                size_multiplier = max(
                    0.5, min(2.0, size_multiplier)
                )  # Clamp between 0.5x and 2.0x

            risk_percent = (
                base_risk * tier_multiplier * mode_multiplier * size_multiplier
            )
            risk_percent = max(
                Config.MIN_RISK_PERCENT, min(Config.MAX_RISK_PERCENT, risk_percent)
            )

            risk_amount = self.state.balance_usdt * (risk_percent / 100)

            # v9.1 CORE: Calculate stop loss
            if "stop_loss" in signal_data:
                stop_loss = float(signal_data["stop_loss"])
            else:
                sl_percent = Config.DEFAULT_SL_PERCENT
                if signal_data["action"].upper() == "BUY":
                    stop_loss = price * (1 - sl_percent / 100)
                else:
                    stop_loss = price * (1 + sl_percent / 100)

            # Calculate position size
            if signal_data["action"].upper() == "BUY":
                sl_distance = abs(price - stop_loss) / price
            else:
                sl_distance = abs(stop_loss - price) / price

            position_size = risk_amount / sl_distance
            quantity = position_size / price

            # v9.1 CORE: Calculate take profits
            take_profits = {}

            if "take_profit_1" in signal_data:
                take_profits["take_profit_1"] = float(signal_data["take_profit_1"])
            else:
                tp1_percent = Config.DEFAULT_TP1_PERCENT
                if signal_data["action"].upper() == "BUY":
                    take_profits["take_profit_1"] = price * (1 + tp1_percent / 100)
                else:
                    take_profits["take_profit_1"] = price * (1 - tp1_percent / 100)

            # v9.1 CORE: Multi-TP support
            if "take_profit_2" in signal_data:
                take_profits["take_profit_2"] = float(signal_data["take_profit_2"])
            elif Config.DEFAULT_TP2_PERCENT > 0:
                tp2_percent = Config.DEFAULT_TP2_PERCENT
                if signal_data["action"].upper() == "BUY":
                    take_profits["take_profit_2"] = price * (1 + tp2_percent / 100)
                else:
                    take_profits["take_profit_2"] = price * (1 - tp2_percent / 100)

            if "take_profit_3" in signal_data:
                take_profits["take_profit_3"] = float(signal_data["take_profit_3"])
            elif Config.DEFAULT_TP3_PERCENT > 0:
                tp3_percent = Config.DEFAULT_TP3_PERCENT
                if signal_data["action"].upper() == "BUY":
                    take_profits["take_profit_3"] = price * (1 + tp3_percent / 100)
                else:
                    take_profits["take_profit_3"] = price * (1 - tp3_percent / 100)

            return {
                "leverage": leverage,
                "quantity": quantity,
                "position_size": position_size,
                "risk_amount": risk_amount,
                "stop_loss": stop_loss,
                **take_profits,
            }

        except Exception as e:
            logger.error(f"Error calculating position parameters: {e}")
            raise

    def _generate_idempotency_key(self, signal_data: Dict) -> str:
        """Generate idempotency key for signal - v9.1 CORE"""
        import hashlib

        # Create unique key from signal data
        key_data = f"{signal_data['symbol']}_{signal_data['action']}_{signal_data['tier']}_{signal_data['timestamp']}_{signal_data.get('price', 0)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    async def _record_signal_rejection(
        self, signal_data: Dict, reason: str, start_time: float
    ):
        """Record signal rejection - v9.1 CORE"""
        try:
            processing_time = int((time.time() - start_time) * 1000)

            with Session() as session:
                record_signal_decision(
                    session, signal_data, False, reason, processing_time
                )

            logger.info(f"❌ Signal rejected: {signal_data.get('symbol')} - {reason}")

        except Exception as e:
            logger.error(f"Error recording signal rejection: {e}")

    # v9.1 CORE: Status and control methods
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status - v9.1 FEATURE"""
        with self.state_lock:
            uptime = (
                datetime.utcnow() - self.state.start_time
                if self.state.start_time
                else timedelta(0)
            )

            return {
                "version": Config.BOT_VERSION,
                "running": self.state.is_running,
                "paused": self.state.is_paused,
                "emergency_mode": self.state.emergency_mode,
                "current_mode": self.state.current_mode,
                "uptime_seconds": int(uptime.total_seconds()),
                "uptime_formatted": str(uptime).split(".")[0],
                "balance_usdt": self.state.balance_usdt,
                "session_pnl": self.state.session_pnl,
                "open_positions": self.state.open_positions,
                "last_heartbeat": (
                    self.state.last_heartbeat.isoformat()
                    if self.state.last_heartbeat
                    else None
                ),
                "last_signal_time": (
                    self.state.last_signal_time.isoformat()
                    if self.state.last_signal_time
                    else None
                ),
                "processing_signal": self.state.processing_signal,
                "connection_status": self.state.connection_status,
                "session_stats": self.session_stats.copy(),
                "active_trades_count": len(self.active_trades),
                "pending_orders_count": len(self.pending_orders),
                "signal_cooldowns_active": len(self.signal_cooldowns),
                "config": {
                    "testnet": Config.IS_TESTNET,
                    "dry_run": Config.DRY_RUN,
                    "tier_minimum": Config.TIER_MINIMUM,
                    "max_positions": Config.MAX_POSITIONS,
                    "use_indicator_leverage": Config.USE_INDICATOR_LEVERAGE,
                    "move_sl_to_be": Config.MOVE_SL_TO_BE_AT_TP1,
                    "use_trailing": Config.USE_TRAILING_AFTER_TP2,
                },
            }

    async def pause(self) -> bool:
        """Pause the bot - v9.1 FEATURE"""
        try:
            with self.state_lock:
                self.state.is_paused = True

            with Session() as session:
                set_setting(session, "is_paused", "True", "bool", "system")

            await discord_notifier.send_system_notification(
                "⏸️ Bot Paused",
                "Trading bot has been paused. No new signals will be processed.",
            )

            logger.info("Bot paused")
            return True

        except Exception as e:
            logger.error(f"Error pausing bot: {e}")
            return False

    async def resume(self) -> bool:
        """Resume the bot - v9.1 FEATURE"""
        try:
            with self.state_lock:
                self.state.is_paused = False

            with Session() as session:
                set_setting(session, "is_paused", "False", "bool", "system")

            await discord_notifier.send_system_notification(
                "▶️ Bot Resumed",
                "Trading bot has been resumed. Signal processing is active.",
            )

            logger.info("Bot resumed")
            return True

        except Exception as e:
            logger.error(f"Error resuming bot: {e}")
            return False

    async def set_mode(self, mode: str) -> Tuple[bool, str]:
        """Set trading mode - v9.1 FEATURE"""
        try:
            if mode not in Config.AVAILABLE_MODES:
                return False, f"Invalid mode. Available: {Config.AVAILABLE_MODES}"

            old_mode = self.state.current_mode

            with self.state_lock:
                self.state.current_mode = mode

            self.mode_manager.set_mode(mode)

            with Session() as session:
                set_setting(session, "mode", mode, "string", "system")

            await discord_notifier.send_system_notification(
                "🔄 Mode Changed", f"Trading mode changed from {old_mode} to {mode}"
            )

            logger.info(f"Mode changed from {old_mode} to {mode}")
            return True, f"Mode changed to {mode}"

        except Exception as e:
            logger.error(f"Error setting mode: {e}")
            return False, f"Error setting mode: {str(e)}"

    async def toggle_emergency(self) -> Tuple[bool, str]:
        """Toggle emergency mode - v9.1 FEATURE"""
        try:
            with self.state_lock:
                self.state.emergency_mode = not self.state.emergency_mode
                new_state = self.state.emergency_mode

            with Session() as session:
                set_setting(
                    session, "emergency_enabled", str(new_state), "bool", "system"
                )

            status = "enabled" if new_state else "disabled"
            await discord_notifier.send_system_notification(
                f"🚨 Emergency Mode {status.title()}",
                f"Emergency mode has been {status}",
            )

            logger.info(f"Emergency mode {status}")
            return True, f"Emergency mode {status}"

        except Exception as e:
            logger.error(f"Error toggling emergency mode: {e}")
            return False, f"Error toggling emergency mode: {str(e)}"

    async def _diagnostic_monitor_task(self):
        """Monitor system diagnostics - v9.1 ENHANCED"""
        while True:
            try:
                # Run comprehensive system diagnostics
                health_report = await self.system_health.get_system_health()
            
                if health_report['status'] != 'healthy':
                    await self.discord.send_diagnostic_alert(
                        "System Health Warning",
                        health_report
                    )
            
                # Log diagnostic metrics
                await self.system_health.log_system_metrics()
            
                await asyncio.sleep(300)  # Every 5 minutes
            
            except Exception as e:
                logger.error(f"Diagnostic monitor task error: {e}")
                await asyncio.sleep(60)

    async def _system_health_task(self):
        """Monitor system health indicators - v9.1 ENHANCED"""
        while True:
            try:
                # Check system resources
                health_status = await self.system_health.check_system_resources()
            
                # Check API connectivity
                api_status = await self.system_health.check_api_connectivity()
            
                # Check database health
                db_status = await self.system_health.check_database_health()
            
                # Aggregate health status
                overall_health = {
                    'system': health_status,
                    'api': api_status,
                    'database': db_status,
                    'timestamp': datetime.utcnow()
                }
            
                # Store health metrics
                await self.system_health.store_health_metrics(overall_health)
            
                await asyncio.sleep(180)  # Every 3 minutes
            
            except Exception as e:
                logger.error(f"System health task error: {e}")
                await asyncio.sleep(60)

    async def _pattern_analysis_task(self):
        """Analyze trading patterns and anomalies - v9.1 ENHANCED"""
        while True:
            try:
                # Detect trading patterns
                patterns = await self.pattern_detector.detect_patterns()
            
                # Analyze anomalies
                anomalies = await self.pattern_detector.detect_anomalies()
            
                # Generate pattern insights
                insights = await self.pattern_detector.generate_insights(patterns, anomalies)
            
                # Store pattern analysis
                await self.pattern_detector.store_pattern_analysis({
                    'patterns': patterns,
                    'anomalies': anomalies,
                    'insights': insights,
                    'timestamp': datetime.utcnow()
                })
            
                # Send alerts for critical patterns
                if anomalies.get('critical_anomalies'):
                    await self.discord.send_pattern_alert(anomalies)
            
                await asyncio.sleep(900)  # Every 15 minutes
            
            except Exception as e:
                logger.error(f"Pattern analysis task error: {e}")
                await asyncio.sleep(300)

    async def _performance_monitor_task(self):
        """Monitor performance metrics - v9.1 ENHANCED"""
        while True:
            try:
                # Analyze performance metrics
                performance_data = await self.performance_analyzer.analyze_performance()
            
                # Generate performance insights
                insights = await self.performance_analyzer.generate_insights(performance_data)
            
                # Check for performance degradation
                if performance_data.get('performance_score', 100) < 70:
                    await self.discord.send_performance_alert(performance_data)
            
                # Store performance metrics
                await self.performance_analyzer.store_performance_data(performance_data)
            
                await asyncio.sleep(600)  # Every 10 minutes
            
            except Exception as e:
                logger.error(f"Performance monitor task error: {e}")
                await asyncio.sleep(180)

    async def _decision_analysis_task(self):
        """Analyze decision-making processes - v9.1 ENHANCED"""
        while True:
            try:
                # Analyze recent decisions
                decision_analysis = await self.decision_engine.analyze_recent_decisions()
            
                # Generate decision insights
                insights = await self.decision_engine.generate_decision_insights()
            
                # Check decision quality
                quality_metrics = await self.decision_engine.evaluate_decision_quality()
            
                # Store decision analysis
                await self.decision_engine.store_decision_analysis({
                    'analysis': decision_analysis,
                    'insights': insights,
                    'quality_metrics': quality_metrics,
                    'timestamp': datetime.utcnow()
                })
            
                await asyncio.sleep(1800)  # Every 30 minutes
            
            except Exception as e:
                logger.error(f"Decision analysis task error: {e}")
                await asyncio.sleep(300)

    async def get_comprehensive_diagnostics(self):
        """Get comprehensive diagnostic report - v9.1 ENHANCED"""
        try:
            # Generate comprehensive diagnostic report
            diagnostic_report = await self.diagnostic_reporter.generate_comprehensive_report()
        
            # Add Pine Script health
            pine_health = await self.pine_health_monitor.get_health_status()
            diagnostic_report['pine_script_health'] = pine_health
        
            # Add decision engine status
            decision_status = await self.decision_engine.get_engine_status()
            diagnostic_report['decision_engine'] = decision_status
        
            return diagnostic_report
        
        except Exception as e:
            logger.error(f"Error generating comprehensive diagnostics: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow()}


# v9.1 CORE: Global bot instance
bot_instance: Optional[TradingBot] = None
_bot_lock = Lock()


def get_bot() -> Optional[TradingBot]:
    """Get bot instance - v9.1 FEATURE"""
    return bot_instance


async def initialize_bot() -> TradingBot:
    """Initialize bot instance - v9.1 FEATURE"""
    global bot_instance

    with _bot_lock:
        if bot_instance:
            return bot_instance

        bot_instance = TradingBot()

    await bot_instance.start()
    return bot_instance


async def shutdown_bot():
    """Shutdown bot instance - v9.1 FEATURE"""
    global bot_instance

    with _bot_lock:
        if bot_instance:
            await bot_instance.stop()
            bot_instance = None


# v9.1 CORE: Signal handlers
def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown - v9.1 FEATURE"""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(shutdown_bot())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # This file should not be run directly
    # Use main.py instead
    print("⚠️  Please use 'python main.py' to start the bot")
    sys.exit(1)

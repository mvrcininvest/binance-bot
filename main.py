"""
Main entry point for Binance Trading Bot v9.1
Orchestrates all components and manages the main loop
Enhanced with v9.1 features: Signal Intelligence, ML Integration, Advanced Analytics
"""

import asyncio
import logging
import signal
import sys
import time
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import json
import traceback
from pathlib import Path
from collections import defaultdict
import time
import uuid
from database import (
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
    get_active_pattern_alerts,
    acknowledge_pattern_alert,
    update_trade,
    create_trade,
    ParameterDecision,
    ShapExplanation,
    ExecutionTrace,
    DecisionTrace,
)
from diagnostics import DiagnosticsEngine
from decision_engine import DecisionEngine
from pine_health_monitor import PineScriptHealthMonitor as PineHealthMonitor
from shap_explainer import SHAPExplainer as ShapExplainer
from pattern_detector import PatternDetector
from threading import Thread
from webhook import app as webhook_app
from binance_handler import binance_handler

# Rate limiting dla IP
request_counts = defaultdict(list)
RATE_LIMIT = 10  # max 10 requestów na minutę na IP

def rate_limit_check(client_ip: str) -> bool:
    now = time.time()
    # Usuń stare requesty (starsze niż 1 minuta)
    request_counts[client_ip] = [req_time for req_time in request_counts[client_ip] if now - req_time < 60]
    
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return False
    
    request_counts[client_ip].append(now)
    return True

import uvicorn
from concurrent.futures import ThreadPoolExecutor

from config import Config
from database import (
    Session,
    init_db,
    cleanup_old_data,
    get_setting,
    set_setting,
    Trade,
    Position,
    close_trade
)
from binance_handler import binance_handler
from discord_notifications import DiscordNotifier
from discord_client import run_discord
from webhook import app as webhook_app
from mode_manager import ModeManager
from analytics import AnalyticsEngine
from signal_intelligence import SignalIntelligence
from ml_predictor import TradingMLPredictor
from futures_user_stream import FuturesUserStream


# Create logs directory with proper permissions
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
try:
    logs_dir.chmod(0o777)
except:
    pass  # Ignore permission errors in containers

# Configure enhanced logging for v9.1 with SAFE file handling
log_format = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)

# Setup logging handlers SAFELY
handlers = [logging.StreamHandler()]  # Always have console output

# Try to add file handlers if possible
try:
    handlers.append(logging.FileHandler("logs/bot.log"))
    handlers.append(logging.FileHandler("logs/trades.log"))
except (PermissionError, OSError) as e:
    print(f"Warning: Could not create log files: {e}")
    print("Continuing with console logging only...")

logging.basicConfig(
    level=getattr(logging, getattr(Config, "LOG_LEVEL", "INFO"), logging.INFO),
    format=log_format,
    handlers=handlers,
)

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator with v9.1 enhancements"""

    def __init__(self):
        """Initialize bot components with v9.1 features"""
        self.running = False
        self.paused = False
        self.emergency_mode = False
        self.shutdown_event = asyncio.Event()

        # Core components
        self.discord = DiscordNotifier()
        self.mode_manager = ModeManager(self)
        self.analytics = AnalyticsEngine()
        self.signal_intelligence = SignalIntelligence()
        self.ml_predictor = (
            TradingMLPredictor()
            if getattr(Config, "USE_ML_FOR_DECISION", False)
            else None
        )
        self.diagnostic_manager = DiagnosticsEngine()
        self.decision_engine = DecisionEngine()
        self.pine_health_monitor = PineHealthMonitor()
        self.shap_explainer = ShapExplainer() if getattr(Config, "USE_ML_FOR_DECISION", False) else None
        self.pattern_detector = PatternDetector()
        self.diagnostic_enabled = getattr(Config, "ENABLE_DIAGNOSTICS", True)
        self.explainability_level = getattr(Config, "EXPLAINABILITY_LEVEL", "auto")  # basic/full/expert/auto
        self.proactive_alerts = getattr(Config, "ENABLE_PROACTIVE_ALERTS", True)
        self.user_stream = None  # Will be initialized later with client
        self.executor = ThreadPoolExecutor(max_workers=6)

        # Runtime settings
        self.runtime_risk = getattr(Config, "RISK_PER_TRADE", 0.02)
        self.connection_check_needed = False
        self.pending_trades = {}  # Przechowuj dane trade'ów w pamięci
        self.dry_run = getattr(Config, "DRY_RUN", True)
        self.blacklisted_symbols = set()
        self.last_signal = None
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0

        # v9.1 Enhanced tracking
        self.position_guards = {}  # Symbol -> timestamp of last alert
        self.signal_history = []  # Last 100 signals for analysis
        self.performance_metrics = {
            "total_signals": 0,
            "signals_taken": 0,
            "signals_rejected": 0,
            "ml_rejections": 0,
            "fake_breakout_detections": 0,
        }

        # Task intervals (seconds)
        self.health_check_interval = 30
        self.position_check_interval = 5  # More frequent for v9.1
        self.cleanup_interval = 3600
        self.analytics_interval = 300
        self.performance_report_interval = 3600 # 60 minutes

        logger.info("🚀 Trading Bot v9.1 initialized with enhanced features")
        logger.info(f"ML Enabled: {getattr(Config, 'USE_ML_FOR_DECISION', False)}")
        logger.info(
            f"Signal Intelligence: {getattr(Config, 'ENABLE_FAKE_BREAKOUT_DETECTION', False)}"
        )
        logger.info(
            f"Institutional Flow: {getattr(Config, 'ENABLE_INSTITUTIONAL_FLOW', False)}"
        )

    async def start(self):
        """Start the trading bot with v9.1 enhancements"""
        try:
            logger.info("🔄 Starting Trading Bot v9.1...")

            # Initialize database with v9.1 schema
            try:
                init_db()
                logger.info("✅ Database initialized successfully")
            except Exception as e:
                logger.error(f"❌ Database initialization failed: {e}")
                return

            # Validate configuration
            if not self._validate_config():
                logger.error("❌ Configuration validation failed")
                return

            # Test Binance connection
            try:
                if not binance_handler.validate_api_credentials():
                    logger.error("❌ Binance API credentials validation failed")
                    return
                logger.info("✅ Binance API credentials validated")
            except Exception as e:
                logger.error(f"❌ Binance API error during initialization: {e}")
                return

            # Initialize async components
            try:
                await binance_handler.init_async()
                logger.info("✅ Binance async components initialized")
            except Exception as e:
                logger.error(f"❌ Binance async initialization failed: {e}")
                return

            # Start user stream for real-time position updates
            if self.user_stream:
                try:
                    await self.user_stream.start()
                    logger.info("✅ Futures User Stream started")
                except Exception as e:
                    logger.warning(f"⚠️ User stream failed to start: {e}")

            # Set bot as running
            self.running = True
            # Initialize adaptive components for v9.1
            try:
                await self.signal_intelligence.initialize()
                if self.ml_predictor:
                    await self.ml_predictor.initialize()
                    if self.diagnostic_enabled:
                        await self.diagnostic_manager.initialize()
                        await self.decision_engine.initialize()
                        await self.pine_health_monitor.initialize()
                        if self.shap_explainer:
                            await self.shap_explainer.initialize()
                    await self.pattern_detector.initialize()
                    logger.info("✅ Hybrid Ultra-Diagnostics initialized")
                logger.info("✅ Adaptive components initialized")
            except Exception as e:
                logger.warning(f"⚠️ Adaptive components initialization failed: {e}")
            try:
                with Session() as session:
                    set_setting(
                        session,
                        "last_restart",
                        datetime.utcnow().isoformat(),
                        "string",
                        "system",
                    )
                    set_setting(session, "bot_version", "9.1", "string", "system")
            except Exception as e:
                logger.warning(f"⚠️ Could not save settings to DB: {e}")

            # Send enhanced startup notification
            try:
                await self.discord.send_startup_notification()
            except Exception as e:
                logger.warning(f"⚠️ Discord notification failed: {e}")

            # Start webhook server in background
            webhook_task = asyncio.create_task(self._run_webhook_server())
            # Uruchom Discord Assistant
            discord_task = None
            if getattr(Config, 'DISCORD_BOT_TOKEN', None):
                discord_task = asyncio.create_task(run_discord(bot_instance, getattr(Config, 'DISCORD_BOT_TOKEN')))
                logger.info("✅ Discord Assistant started")
            else:
                logger.warning("⚠️ Discord bot token not found, Discord Assistant disabled")

            # Start enhanced task loops
            tasks = [
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._position_monitor_loop()),
                asyncio.create_task(self._cleanup_loop()),
                asyncio.create_task(self._analytics_loop()),
                asyncio.create_task(self._performance_report_loop()),
                asyncio.create_task(self._emergency_monitor_loop()),
                asyncio.create_task(self._diagnostic_loop()),
            ]

            logger.info("✅ Bot started successfully with all v9.1 features")

            # Wait for shutdown signal
            await self.shutdown_event.wait()

            # Graceful shutdown
            logger.info("🔄 Shutting down bot...")
            self.running = False

            # Stop user stream
            if self.user_stream:
                await self.user_stream.stop()

            # Cancel all tasks
            for task in tasks:
                task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Close webhook server
            webhook_task.cancel()
            await asyncio.gather(webhook_task, return_exceptions=True)

            # Close executor
            self.executor.shutdown(wait=True)

            # Send shutdown notification
            try:
                await self.discord.send_shutdown_notification()
            except:
                pass

            logger.info("✅ Bot shutdown complete")

        except Exception as e:
            logger.error(f"💥 Fatal error in bot start: {e}", exc_info=True)
            try:
                await self.discord.send_error_notification(f"Bot crashed: {e}")
            except:
                pass
            raise

    def _validate_config(self) -> bool:
        """Validate configuration with v9.1 requirements"""
        try:
            errors = []

            # Try to get validation from Config if it exists
            try:
                if hasattr(Config, "validate"):
                    errors = Config.validate()
            except:
                pass

            # Additional v9.1 validations with safe attribute access
            if (
                hasattr(Config, "INDICATOR_VERSION_REQUIRED")
                and getattr(Config, "INDICATOR_VERSION_REQUIRED", "9.1") != "9.1"
            ):
                errors.append("INDICATOR_VERSION_REQUIRED must be '9.1'")

            if (
                hasattr(Config, "ALERT_MAX_AGE_SEC")
                and getattr(Config, "ALERT_MAX_AGE_SEC", 300) < 60
            ):
                errors.append("ALERT_MAX_AGE_SEC should be at least 60 seconds")

            if (
                hasattr(Config, "EMERGENCY_CLOSE_THRESHOLD")
                and getattr(Config, "EMERGENCY_CLOSE_THRESHOLD", -100) >= 0
            ):
                errors.append("EMERGENCY_CLOSE_THRESHOLD must be negative")

            if errors:
                for error in errors:
                    logger.error(f"Config error: {error}")
                return False

            logger.info("✅ Configuration validated successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Configuration validation error: {e}")
            return False
    async def _create_diagnostic_trace(self, signal_data: Dict[str, Any]) -> str:
        """Create diagnostic trace for explainable AI"""
        if not self.diagnostic_enabled:
            return None
            
        try:
            trace_id = str(uuid.uuid4())
            
            trace_data = {
                "trace_id": trace_id,
                "symbol": signal_data.get("symbol", "").upper(),
                "action": signal_data.get("action", "").upper(),
                "tier": signal_data.get("tier", "Unknown"),
                "alert_timestamp": datetime.utcnow(),
                "processing_stage": "received",
                "final_decision": "PROCESSING",
                "raw_alert_data": signal_data,
                "decision_context": {
                    "bot_mode": getattr(self.mode_manager, "current_mode", "unknown"),
                    "paused": self.paused,
                    "emergency_mode": self.emergency_mode,
                    "active_positions": len(self.active_positions),
                    "daily_pnl": self.daily_pnl,
                    "daily_trades": self.daily_trades,
                }
            }
            
            with Session() as session:
                create_decision_trace(session, trace_data)
            
            logger.info(f"🔍 Created diagnostic trace: {trace_id}")
            return trace_id
            
        except Exception as e:
            logger.error(f"Error creating diagnostic trace: {e}")
            return None

    async def _update_diagnostic_trace(self, trace_id: str, stage: str, updates: Dict[str, Any]):
        """Update diagnostic trace with new information"""
        if not self.diagnostic_enabled or not trace_id:
            return
            
        try:
            updates["processing_stage"] = stage
            
            with Session() as session:
                update_decision_trace(session, trace_id, updates)
                
        except Exception as e:
            logger.error(f"Error updating diagnostic trace {trace_id}: {e}")

    async def _complete_diagnostic_trace(self, trace_id: str, final_data: Dict[str, Any]):
        """Complete diagnostic trace with final results"""
        if not self.diagnostic_enabled or not trace_id:
            return
            
        try:
            with Session() as session:
                complete_decision_trace(session, trace_id, final_data)
                
            logger.info(f"✅ Completed diagnostic trace: {trace_id}")
            
        except Exception as e:
            logger.error(f"Error completing diagnostic trace {trace_id}: {e}")

    async def _log_parameter_decisions(self, trace_id: str, decisions: Dict[str, Any]):
        """Log detailed parameter decisions with SHAP explanations"""
        if not self.diagnostic_enabled or not trace_id:
            return
            
        try:
            with Session() as session:
                for param_name, decision_data in decisions.items():
                    if not isinstance(decision_data, dict):
                        logger.warning(f"Skipping invalid parameter decision for {param_name}: not a dictionary.")
                        continue
                    param_data = {
                        "trace_id": trace_id,
                        "parameter_name": param_name,
                        "parameter_type": decision_data.get("type", "float"),
                        "original_value": decision_data.get("original_value"),
                        "final_value": decision_data.get("final_value"),
                        "value_change_pct": decision_data.get("change_pct", 0.0),
                        "primary_reason": decision_data.get("reason", ""),
                        "confidence_score": decision_data.get("confidence", 0.5),
                        "alternative_values": decision_data.get("alternatives", {}),
                    }
                    
                    # Add SHAP values if available
                    if self.shap_explainer and decision_data.get("shap_values"):
                        shap_data = decision_data["shap_values"]
                        param_data.update({
                            "shap_base_value": shap_data.get("base_value", 0.0),
                            "shap_prediction": shap_data.get("prediction", 0.0),
                            "feature_contributions": shap_data.get("contributions", {}),
                        })
                    
                    add_parameter_decision(session, param_data)
                    
        except Exception as e:
            logger.error(f"Error logging parameter decisions for {trace_id}: {e}")

    async def _log_pine_health_data(self, trace_id: str, signal_data: Dict[str, Any]):
        """Log Pine Script health diagnostics"""
        if not self.diagnostic_enabled:
            return
            
        try:
            # Extract Pine health data from signal
            health_data = {
                "trace_id": trace_id,
                "symbol": signal_data.get("symbol", "").upper(),
                "timeframe": signal_data.get("timeframe", "unknown"),
                "timestamp": datetime.utcnow(),
                "overall_health": signal_data.get("pine_health_score", 0.5),
                "atr_health": signal_data.get("atr_health", 0.5),
                "atr_value": signal_data.get("atr_value"),
                "atr_percentile": signal_data.get("atr_percentile"),
                "adx_health": signal_data.get("adx_health", 0.5),
                "adx_value": signal_data.get("adx_value"),
                "adx_trend_strength": signal_data.get("adx_strength", "unknown"),
                "regime_detected": signal_data.get("enhanced_regime", "NEUTRAL"),
                "regime_confidence": signal_data.get("regime_confidence", 0.5),
                "volume_profile": signal_data.get("volume_profile", "fair"),
                "institutional_flow": signal_data.get("institutional_flow", 0.0),
                "warnings": signal_data.get("pine_warnings", []),
                "critical_issues": signal_data.get("pine_critical_issues", []),
            }
            
            with Session() as session:
                log_pine_health(session, health_data)
                
        except Exception as e:
            logger.error(f"Error logging Pine health data: {e}")

    async def _detect_patterns_and_alert(self):
        """Detect patterns and create proactive alerts"""
        from dataclasses import asdict
        from pattern_detector import Severity
        if not self.proactive_alerts:
            return
            
        try:
            # Analyze patterns
            patterns_result = await self.pattern_detector.detect_patterns()
            
            # Create alerts for significant patterns
            # Iteruj po liście obiektów, a nie po słowniku
            for pattern in patterns_result.get("detected_patterns", []):
                # Użyj atrybutów dataclass
                if pattern.severity in [Severity.HIGH, Severity.CRITICAL]:
                    # Konwertuj dataclass na słownik dla bazy danych
                    pattern_data = asdict(pattern)
                    
                    # Dostosuj klucze do modelu PatternAlert
                    db_pattern_data = {
                        "pattern_type": pattern_data.get("pattern_type").value,
                        "pattern_severity": pattern_data.get("severity").value,
                        "pattern_description": pattern_data.get("description"),
                        "symbols_affected": pattern_data.get("affected_components", ["ALL"]),
                        "frequency_count": pattern_data.get("occurrence_count", 1),
                        "confidence_score": pattern_data.get("confidence", 0.5),
                        "recommended_action": ", ".join(pattern_data.get("recommendations", [])),
                    }
                    
                    with Session() as session:
                        alert = create_pattern_alert(session, db_pattern_data)
                        
                    # Wyślij powiadomienie na Discord
                    if pattern.severity == Severity.CRITICAL:
                        # Przekaż słownik do funkcji powiadomień
                        await self.discord.send_system_notification(
                            message=f"Critical Pattern Detected: {pattern.description}",
                            notification_type="emergency"
                        )
                        
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}", exc_info=True)

    async def handle_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming trading signal with Hybrid Ultra-Diagnostics"""
        trace_id = None
        
        try:
            signal_timestamp = datetime.utcnow()
            self.performance_metrics["total_signals"] += 1
            
            # === STEP 1: CREATE DIAGNOSTIC TRACE FOR ALL SIGNALS ===
            trace_id = await self._create_diagnostic_trace(signal_data)
            tier = signal_data.get("tier", "Unknown")
            logger.debug(f"🔍 Created diagnostic trace: {trace_id}")
            
            # v9.1 NEW: Handle close/emergency_close signals immediately
            action = signal_data.get("action", "").lower()
            if action in ["close", "emergency_close"]:
                symbol = signal_data.get("symbol", "").upper()
                if symbol in self.active_positions:
                    close_reason = "emergency_close" if action == "emergency_close" else "manual_close"
                    result = await self.close_position(symbol, close_reason)
                    
                    # Complete diagnostic trace
                    if trace_id:
                        await self._complete_diagnostic_trace(trace_id, {
                            "final_decision": "EXECUTED",
                            "processing_stage": "completed",
                            "execution_result": result
                        })
                    
                    return {
                        "status": "success" if result.get("status") == "success" else "error",
                        "action": "position_closed",
                        "symbol": symbol,
                        "reason": close_reason,
                        "result": result,
                        "trace_id": trace_id
                    }
                else:
                    if trace_id:
                        await self._complete_diagnostic_trace(trace_id, {
                            "final_decision": "IGNORED",
                            "rejection_reason": "no_position_to_close"
                        })
                    
                    return {
                        "status": "ignored",
                        "reason": "no_position_to_close",
                        "symbol": symbol,
                        "trace_id": trace_id
                    }

            # Store last signal
            self.last_signal = signal_data.copy()
            self.last_signal["received_at"] = signal_timestamp.isoformat()
            self.last_signal["trace_id"] = trace_id
            
            # === STEP 2: LOG PINE HEALTH DATA ===
            await self._log_pine_health_data(trace_id, signal_data)
            
            # === STEP 3: LATENCY ANALYSIS ===
            if "tv_ts" in signal_data and signal_data["tv_ts"]:
                current_ms = int(time.time() * 1000)
                end_to_end_latency = current_ms - signal_data["tv_ts"]
                
                self.last_signal["end_to_end_latency_ms"] = end_to_end_latency
                
                # Update diagnostic trace with latency
                if trace_id:
                    await self._update_diagnostic_trace(trace_id, "latency_analyzed", {
                        "api_latency_ms": end_to_end_latency,
                        "decision_latency_ms": 0  # Will be updated later
                    })
                
                # Log latency warnings
                if end_to_end_latency > 800:
                    logger.warning(f"⚠️ M5 Latency warning: {end_to_end_latency}ms for {signal_data.get('symbol')}")
                
                if end_to_end_latency > 1500:
                    logger.error(f"🚨 M5 CRITICAL latency: {end_to_end_latency}ms for {signal_data.get('symbol')}")
                    
                    try:
                        await self.discord.send_error_notification(
                            f"🚨 CRITICAL M5 Latency Alert",
                            f"Signal for {signal_data.get('symbol')} took {end_to_end_latency}ms (>1500ms threshold)"
                        )
                    except:
                        pass

            # Add to signal history
            self.signal_history.append(self.last_signal)
            if len(self.signal_history) > 100:
                self.signal_history.pop(0)

            logger.info(f"📡 Received signal for {signal_data.get('symbol')} - {signal_data.get('action')} [Trace: {trace_id}]")

            # === STEP 4: BASIC VALIDATIONS ===
            if trace_id:
                await self._update_diagnostic_trace(trace_id, "validating", {})
            
            # Check alert age
            alert_timestamp = signal_data.get("timestamp")
            if alert_timestamp:
                try:
                    alert_time = datetime.fromisoformat(alert_timestamp.replace("Z", "+00:00"))
                    age_seconds = (signal_timestamp - alert_time.replace(tzinfo=None)).total_seconds()

                    alert_max_age = getattr(Config, "ALERT_MAX_AGE_SEC", 300)
                    if age_seconds > alert_max_age:
                        logger.warning(f"⏰ Alert too old: {age_seconds}s > {alert_max_age}s")
                        self.performance_metrics["signals_rejected"] += 1
                        
                        if trace_id:
                            await self._complete_diagnostic_trace(trace_id, {
                                "final_decision": "REJECTED",
                                "rejection_reason": "alert_too_old",
                                "processing_time_ms": int((datetime.utcnow() - signal_timestamp).total_seconds() * 1000)
                            })
                        
                        return {
                            "status": "rejected",
                            "reason": "alert_too_old",
                            "age_seconds": age_seconds,
                            "trace_id": trace_id
                        }
                
                except Exception as e:
                    logger.warning(f"Could not parse alert timestamp: {e}")

            # Check if paused or emergency mode
            if self.paused:
                logger.info("⏸️ Bot is paused, ignoring signal")
                if trace_id:
                    await self._complete_diagnostic_trace(trace_id, {
                        "final_decision": "IGNORED",
                        "rejection_reason": "bot_paused"
                    })
                return {"status": "ignored", "reason": "bot_paused", "trace_id": trace_id}

            if self.emergency_mode:
                logger.warning("🚨 Emergency mode active, ignoring signal")
                if trace_id:
                    await self._complete_diagnostic_trace(trace_id, {
                        "final_decision": "IGNORED",
                        "rejection_reason": "emergency_mode"
                    })
                return {"status": "ignored", "reason": "emergency_mode", "trace_id": trace_id}

            # Check blacklist
            symbol = signal_data.get("symbol", "").upper()
            if symbol in self.blacklisted_symbols:
                logger.info(f"🚫 Symbol {symbol} is blacklisted, ignoring signal")
                self.performance_metrics["signals_rejected"] += 1
                if trace_id:
                    await self._complete_diagnostic_trace(trace_id, {
                        "final_decision": "REJECTED",
                        "rejection_reason": "blacklisted_symbol"
                    })
                return {"status": "ignored", "reason": "blacklisted_symbol", "trace_id": trace_id}

            # Position guard check
            if symbol in self.position_guards:
                last_alert_time = self.position_guards[symbol]
                alert_max_age = getattr(Config, "ALERT_MAX_AGE_SEC", 300)
                if (signal_timestamp - last_alert_time).total_seconds() < alert_max_age:
                    logger.info(f"🛡️ Position guard active for {symbol}")
                    if trace_id:
                        await self._complete_diagnostic_trace(trace_id, {
                            "final_decision": "IGNORED",
                            "rejection_reason": "position_guard_active"
                        })
                    return {"status": "ignored", "reason": "position_guard_active", "trace_id": trace_id}

            # Tier validation
            tier = signal_data.get("tier", "Unknown")
            try:
                if hasattr(Config, 'validate_tier') and not Config.validate_tier(tier):
                    logger.warning(f"⚠️ Tier {tier} below minimum {getattr(Config, 'TIER_MINIMUM', 'Standard')}")
                    self.performance_metrics["signals_rejected"] += 1
                    if trace_id:
                        await self._complete_diagnostic_trace(trace_id, {
                            "final_decision": "REJECTED",
                            "rejection_reason": "tier_below_minimum"
                        })
                    return {"status": "rejected", "reason": "tier_below_minimum", "trace_id": trace_id}
            except:
                # Fallback validation
                allowed_tiers = ["Emergency", "Platinum", "Premium", "Standard", "Quick"]
                if tier not in allowed_tiers:
                    logger.warning(f"⚠️ Invalid tier: {tier}")
                    self.performance_metrics["signals_rejected"] += 1
                    if trace_id:
                        await self._complete_diagnostic_trace(trace_id, {
                            "final_decision": "REJECTED",
                            "rejection_reason": "invalid_tier"
                        })
                    return {"status": "rejected", "reason": "invalid_tier", "trace_id": trace_id}

            # === STEP 5: ENHANCED SIGNAL INTELLIGENCE ===
            if trace_id:
                await self._update_diagnostic_trace(trace_id, "analyzing", {})
            
            decision_start_time = datetime.utcnow()
            decision = await self.signal_intelligence.analyze_signal(signal_data)
            decision_time_ms = int((datetime.utcnow() - decision_start_time).total_seconds() * 1000)
            
            # Track fake breakout detections
            if decision.get("fake_breakout_detected"):
                self.performance_metrics["fake_breakout_detections"] += 1
                logger.info(f"🎯 Fake breakout detected for {symbol}")

            # === STEP 6: ML PREDICTION WITH SHAP EXPLAINABILITY ===
            if getattr(Config, "USE_ML_FOR_DECISION", False) and self.ml_predictor:
                if trace_id:
                    await self._update_diagnostic_trace(trace_id, "ml_predicting", {})
                
                ml_prediction = self.ml_predictor.predict(signal_data)
                should_take, reason = self.ml_predictor.should_take_trade(
                    ml_prediction, min_win_prob=getattr(Config, "ML_MIN_WIN_PROB", 0.55)
                )

                if not should_take:
                    logger.info(f"🤖 ML rejected trade: {reason}")
                    decision["should_trade"] = False
                    decision["ml_rejection"] = reason
                    self.performance_metrics["ml_rejections"] += 1

                decision["ml_prediction"] = ml_prediction
                
                # Generate SHAP explanations if available
                if self.shap_explainer and should_take:
                    try:
                        shap_explanation = await self.shap_explainer.explain_prediction(
                            signal_data, ml_prediction
                        )
                        decision["shap_explanation"] = shap_explanation
                        
                        # Log SHAP data to database
                        if trace_id:
                            shap_data = {
                                "trace_id": trace_id,
                                "model_name": "trading_predictor",
                                "model_version": getattr(Config, "ML_MODEL_VERSION", "1.0"),
                                "prediction_type": "classification",
                                "base_value": shap_explanation.get("base_value", 0.0),
                                "prediction_value": shap_explanation.get("prediction", 0.0),
                                "feature_values": shap_explanation.get("feature_values", {}),
                                "shap_values": shap_explanation.get("shap_values", {}),
                                "top_positive_features": shap_explanation.get("top_positive", []),
                                "top_negative_features": shap_explanation.get("top_negative", []),
                                "prediction_confidence": ml_prediction.get("confidence", 0.5),
                            }
                            
                            with Session() as session:
                                add_shap_explanation(session, shap_data)
                                
                    except Exception as e:
                        logger.error(f"Error generating SHAP explanation: {e}")

            # === STEP 7: DECISION ENGINE - PARAMETER OPTIMIZATION ===
            if decision.get("should_trade"):
                if trace_id:
                    await self._update_diagnostic_trace(trace_id, "optimizing_parameters", {})
                
                # Get base trade parameters
                trade_params = self.mode_manager.get_trade_parameters(decision)
                
                # Enhanced parameter optimization with explainability
                optimized_params = await self.decision_engine.optimize_parameters(
                    signal_data, decision, trade_params
                )
                
                decision.update(optimized_params)
                
                # Log parameter decisions with explanations
                if trace_id and optimized_params.get("parameter_decisions"):
                    await self._log_parameter_decisions(trace_id, optimized_params["parameter_decisions"])

                # === STEP 8: EXECUTE TRADE NATYCHMIAST ===
                result = await self._execute_trade(signal_data, decision)

                # Diagnostyka w tle TYLKO po udanym wykonaniu transakcji
                if (result.get("status") == "success" and trace_id):
                    # Uruchom diagnostyki w tle - nie blokuj głównego wątku
                    asyncio.create_task(self._run_diagnostics_background(signal_data, decision, result, trace_id))

                if result.get("status") == "success":
                    self.performance_metrics["signals_taken"] += 1
                    self.position_guards[symbol] = signal_timestamp
                    final_decision = "EXECUTED"
                else:
                    self.performance_metrics["signals_rejected"] += 1
                    final_decision = "ERROR"
                    
                # Complete diagnostic trace
                if trace_id:
                    trace_data = {
                        "final_decision": "EXECUTED",
                        "processing_stage": "completed",
                        "execution_result": result,
                    }
                    trade_id = result.get("trade_id")
                    if trade_id is not None:
                        safe_trade_id = trade_id
                        if trade_id > 2147483647 or trade_id < -2147483648:
                            safe_trade_id = hash(str(trade_id)) % 2147483647
                        trace_data["safe_trade_id"] = safe_trade_id
    
                    await self._complete_diagnostic_trace(trace_id, trace_data)
            else:
                self.performance_metrics["signals_rejected"] += 1
                
                # Complete diagnostic trace for rejection
                if trace_id:
                    await self._complete_diagnostic_trace(trace_id, {
                        "final_decision": "REJECTED",
                        "rejection_reason": decision.get("ml_rejection") or decision.get("reason", "analysis_rejection"),
                        "processing_time_ms": decision_time_ms,
                        "decision_latency_ms": decision_time_ms,
                    })

            # === STEP 9: ENHANCED NOTIFICATION ===
            decision["trace_id"] = trace_id
            await self.discord.send_signal_decision(signal_data, decision)

            return decision

        except BinanceAPIException as e:
            self.connection_check_needed = True  # Sprawdź połączenie przy następnym cyklu
            logger.error(f"💥 Binance API error handling signal: {e}", exc_info=True)
    
            # Complete diagnostic trace with error
            if trace_id:
                try:
                    await self._complete_diagnostic_trace(trace_id, {
                        "final_decision": "ERROR",
                        "rejection_reason": f"binance_api_error: {str(e)}"
                    })
                except:
                    pass
    
            await self.discord.send_error_notification("Binance API Error", f"Error: {e}")
            return {"status": "error", "error": str(e), "trace_id": trace_id}

        except Exception as e:
            logger.error(f"💥 Error handling signal: {e}", exc_info=True)
            
            # Complete diagnostic trace with error
            if trace_id:
                try:
                    await self._complete_diagnostic_trace(trace_id, {
                        "final_decision": "ERROR",
                        "rejection_reason": f"processing_error: {str(e)}"
                    })
                except:
                    pass
            
            await self.discord.send_error_notification("Signal Handling Error", f"Error: {e}")
            return {"status": "error", "error": str(e), "trace_id": trace_id}

    async def pause(self) -> bool:
        self.pause_trading()
        return True

    async def resume(self) -> bool:
        self.resume_trading()
        return True

    async def set_mode(self, mode: str) -> Tuple[bool, str]:
        ok = self.mode_manager.set_mode(mode, reason="API request")
        return ok, ("Mode changed" if ok else "Invalid mode")

    async def toggle_emergency(self) -> Tuple[bool, str]:
        if self.emergency_mode:
            self.disable_emergency_mode()
            return True, "Emergency mode disabled"
        else:
            self.enable_emergency_mode()
            return True, "Emergency mode enabled"

    async def process_signal(self, alert: Dict[str, Any]) -> Tuple[bool, str]:
        decision = await self.handle_signal(alert)

        # Wczesne zwroty z handle_signal (np. rejected/ignored/error)
        if isinstance(decision, dict) and "status" in decision:
            status = str(decision.get("status", ""))
            if status == "success":
                return True, decision.get("reason") or "Signal accepted"
            if status in ("rejected", "ignored", "error"):
                return False, decision.get("reason") or decision.get("error") or status

        # Decyzja zawiera wynik egzekucji
        exec_result = (decision or {}).get("execution_result") or {}
        if exec_result.get("status") == "success":
            return True, "Trade executed"
        if exec_result.get("status") == "skipped":
            return False, exec_result.get("reason", "Skipped")
        if exec_result.get("status") == "error":
            return False, exec_result.get("error", "Execution error")

        # Odmowy bez egzekucji (np. ML)
        if decision.get("ml_rejection"):
            return False, f"ML rejection: {decision['ml_rejection']}"
        if decision.get("should_trade") is False:
            return False, decision.get("reason", "Rejected by rules")

        return False, "Unknown decision"

    async def _execute_trade(
        self, signal_data: Dict[str, Any], decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute trade with v9.1 enhancements and corrected logic."""
        symbol = signal_data.get("symbol", "").upper()
        action = signal_data.get("action", "").lower()
        tier = signal_data.get("tier", "Standard")

        logger.info(f"🎯 Executing {action} trade for {symbol} (Tier: {tier})")

        try:
            # Określ parametry handlu na podstawie tier
            if tier == "Emergency":
                logger.info("🚨 Signal is an Emergency tier. Applying emergency trade parameters.")
                trade_params = self.mode_manager.get_mode_parameters("emergency")
            else:
                current_mode = self.mode_manager.get_current_mode()
                if current_mode == "normal":
                    current_mode = "balanced"
                    logger.warning("Mode 'normal' requested, using 'balanced' instead")
                logger.info(f"Signal is a standard tier. Applying '{current_mode}' trade parameters.")
                trade_params = self.mode_manager.get_mode_parameters(current_mode)

            leverage = trade_params.get("leverage", Config.DEFAULT_LEVERAGE)
            risk_percent = trade_params.get("risk_percent", self.runtime_risk)

            # Pobierz cenę i oblicz SL/TP
            current_price = await binance_handler.get_current_price(symbol)
            sl_price = self._calculate_stop_loss(current_price, action, signal_data)
            tp_levels = self._calculate_take_profits(current_price, action, signal_data, sl_price)

            if not sl_price or not tp_levels:
                raise ValueError("Failed to determine valid SL or TP levels.")

            logger.info(f"Determined trade parameters: SL={sl_price}, TPs={tp_levels}")

            # Ustaw leverage
            await binance_handler.set_leverage(symbol, leverage)
            logger.info(f"⚙️ Set leverage to {leverage}x for {symbol}")

            # Oblicz wielkość pozycji
            account_balance = await binance_handler.get_futures_balance()
            risk_amount = account_balance * risk_percent
            position_size = self._calculate_position_size(
                risk_amount, current_price, sl_price, leverage, tier
            )

            # Generuj unikalny tag dla zlecenia
            order_tag = f"{getattr(Config, 'ORDER_TAG_PREFIX', 'TBV91')}_{int(time.time())}"

            # Przygotuj dane trade'u w pamięci
            trade_data = self._prepare_trade_data(signal_data, decision, {
                "price": current_price, 
                "origQty": position_size, 
                "clientOrderId": order_tag
            }, sl_price, tp_levels)

            if not trade_data:
                raise ValueError("Failed to prepare trade data")

            # Zapisz w pamięci do późniejszego użycia
            self.pending_trades = getattr(self, 'pending_trades', {})
            self.pending_trades[symbol] = trade_data

            # Generuj unikalny trade_id
            trade_id = f"TEMP_{symbol}_{int(time.time())}"
            # === DODAJ ERROR HANDLING ===
            from binance.exceptions import BinanceAPIException
            # Wykonaj zlecenie
            if self.dry_run:
                logger.info(f"🧪 DRY RUN: Would place {action} order for {symbol}")
                order_result = {
                    "orderId": f"DRY_{trade_id if trade_id else 'UNKNOWN'}", 
                    "symbol": symbol, 
                    "side": action.upper(),
                    "price": current_price, 
                    "origQty": position_size, 
                    "status": "FILLED",
                    "clientOrderId": order_tag,
                }
            else:
                try:
                    order_result = await binance_handler.place_futures_order(
                        symbol=symbol, 
                        side=action.upper(), 
                        quantity=position_size,
                        leverage=leverage, 
                        client_order_id=order_tag,
                    )
                except BinanceAPIException as e:
                    logger.error(f"💥 Binance API error placing order: {e}")
                    # Usuń z pending_trades jeśli zlecenie się nie powiodło
                    self.pending_trades.pop(symbol, None)
                    raise e
                except Exception as e:
                    logger.error(f"💥 Unexpected error placing order: {e}")
                    # Usuń z pending_trades jeśli zlecenie się nie powiodło
                    self.pending_trades.pop(symbol, None)
                    raise e

            # Zapisz pozycję
            self.active_positions[symbol] = {
                "trade_id": trade_id, 
                "order_tag": order_tag, 
                "entry_price": current_price,
                "quantity": position_size, 
                "side": action, 
                "sl_price": sl_price,
                "tp_levels": tp_levels, 
                "entry_time": datetime.utcnow(), 
                "leverage": leverage,
                "tier": tier, 
                "signal_strength": signal_data.get("strength", 0.5),
            }

            self.daily_trades += 1
            await self.discord.send_entry_notification(order_result, signal_data)
            logger.info(f"✅ Trade executed successfully for {symbol}")

            return { 
                "status": "success", 
                "order": order_result, 
                "position": self.active_positions[symbol], 
                "trade_id": trade_id 
            }

        except Exception as e:
            logger.error(f"💥 Error executing trade: {e}", exc_info=True)
        
            # Wyczyść pending trade jeśli wystąpił błąd
            if symbol in getattr(self, 'pending_trades', {}):
                del self.pending_trades[symbol]
                logger.info(f"🧹 Cleaned up pending trade data for {symbol}")

            await self.discord.send_error_notification(
                f"Trade execution error for {symbol}", 
                f"Reason: {e}"
            )
            return {
                "status": "error", 
                "error": str(e), 
                "trade_id": trade_id if 'trade_id' in locals() else None
            }

    def _calculate_stop_loss(self, price: float, action: str, signal_data: Dict) -> float:
        """Oblicza cenę SL, dając priorytet wartości z alertu."""
        # Priorytet 1: Użyj SL z sygnału, jeśli jest poprawny
        if signal_data.get("sl") and isinstance(signal_data["sl"], (int, float)) and signal_data["sl"] > 0:
            sl_price = float(signal_data["sl"])
            # Prosta walidacja logiki
            if (action == "buy" and sl_price < price) or (action == "sell" and sl_price > price):
                logger.info(f"Using SL from alert: {sl_price}")
                return sl_price
            else:
                logger.warning(f"Invalid SL from alert for {action} action (price: {price}, sl: {sl_price}). Falling back to calculation.")

        # Fallback: Oblicz SL na podstawie ATR i konfiguracji
        logger.info("SL not in alert or invalid. Calculating fallback SL...")
        atr_value = float(signal_data.get("atr", 0))
        if atr_value <= 0:
            raise ValueError("ATR value is missing or invalid, cannot calculate fallback SL.")

        atr_multiplier = getattr(Config, "ATR_SL_MULTIPLIER", 1.0) # Pobierz mnożnik z config.py
        sl_distance = atr_value * atr_multiplier

        if action == "buy":
            return price - sl_distance
        else:
            return price + sl_distance

    def _calculate_take_profits(self, price: float, action: str, signal_data: Dict, sl_price: float) -> List[float]:
        """Oblicza ceny TP, dając priorytet wartościom z alertu."""
        tp_levels = []

        # Priorytet 1: Spróbuj użyć TP z sygnału
        for i in range(1, 4):
            tp_key = f"tp{i}"
            if signal_data.get(tp_key) and isinstance(signal_data[tp_key], (int, float)) and signal_data[tp_key] > 0:
                tp_price = float(signal_data[tp_key])
                # Prosta walidacja logiki
                if (action == "buy" and tp_price > price) or (action == "sell" and tp_price < price):
                    tp_levels.append(tp_price)
        
        if tp_levels:
            logger.info(f"Using {len(tp_levels)} TP level(s) from alert: {tp_levels}")
            return tp_levels

        # Fallback: Oblicz TP na podstawie RR i konfiguracji
        logger.info("TP levels not in alert or invalid. Calculating fallback TPs...")
        risk_per_unit = abs(price - sl_price)
        if risk_per_unit <= 0:
            raise ValueError("Risk per unit is zero, cannot calculate fallback TPs.")
            
        # Używamy teraz TP_RR_LEVELS z config.py, który powinien być listą, np. [0.5, 1.0, 1.5]
        tp_rr_levels = getattr(Config, "TP_RR_LEVELS", [0.5, 1.0, 1.5])

        for rr in tp_rr_levels:
            if action == "buy":
                tp_levels.append(price + (risk_per_unit * rr))
            else:
                tp_levels.append(price - (risk_per_unit * rr))
        
        logger.info(f"Calculated fallback TPs based on RR: {tp_levels}")
        return tp_levels

    def _calculate_position_size(
        self, risk_amount: float, entry_price: float, sl_price: float, leverage: int, tier: str = "Standard"
    ) -> float:
        """Calculate position size for scalping with dual risk management"""
    
        # Pobierz balans
        balance = binance_handler.get_balance()["available"]
    
        # PARAMETRY KONFIGURACYJNE (dodaj do .env)
        max_position_percent = float(os.getenv('MAX_POSITION_PERCENT', '15'))  # % kapitału na pozycję
        risk_percent = float(os.getenv('RISK_PER_TRADE_PERCENT', '2.0'))  # % maksymalnej straty
    
        # Oblicz maksymalną wartość pozycji (z leverage)
        max_position_value = balance * (max_position_percent / 100) * leverage
    
        # Oblicz wielkość pozycji na podstawie ryzyka
        price_diff = abs(entry_price - sl_price)
        risk_based_size = risk_amount / price_diff
        risk_based_value = risk_based_size * entry_price
    
        # Wybierz MNIEJSZĄ wartość (bezpieczeństwo)
        if risk_based_value > max_position_value:
            position_value = max_position_value
            position_size = position_value / entry_price
            logger.info(f"📉 Position limited by max size ({max_position_percent}% of balance)")
        else:
            position_value = risk_based_value
            position_size = risk_based_size
            logger.info(f"✅ Position sized by risk ({risk_percent}% max loss)")
    
        # Sprawdź czy mamy wystarczający margin
        required_margin = position_value / leverage
    
        logger.info(f"""
        📊 Position Calculation:
        ├── Balance: ${balance:.2f}
        ├── Max position ({max_position_percent}%): ${max_position_value:.2f}
        ├── Risk-based size: ${risk_based_value:.2f}
        ├── Selected: ${position_value:.2f}
        ├── Position size: {position_size:.3f} units
        ├── Required margin: ${required_margin:.2f}
        └── Leverage: {leverage}x
        """)
    
        # Ostateczne sprawdzenie
        if required_margin > balance * 0.95:
            logger.error(f"❌ Insufficient margin! Need ${required_margin:.2f}, have ${balance:.2f}")
            return 0
    
        # Apply multipliers
        multiplier = getattr(Config, "POSITION_SIZE_MULTIPLIER", 1.0)
        position_size *= multiplier
    
        # Tier adjustments
        tier_risk_mult = getattr(Config, "TIER_RISK_MULTIPLIERS", {}).get(tier, 1.0)
        if tier_risk_mult != 1.0:
            position_size *= tier_risk_mult
            logger.info(f"📊 Tier {tier} adjustment: {tier_risk_mult}x")
    
        return round(position_size, 3)

    def _prepare_trade_data(self, signal_data: Dict, decision: Dict, order_result: Dict, sl_price: float, tp_levels: List[float]):
        """Przygotuj dane trade'u w pamięci - nie zapisuj jeszcze do bazy"""
        try:
            price = float(order_result.get("price") or 0)
        
            if price <= 0:
                logger.error(f"Invalid entry_price: {price}")
                return None
            
            side = "BUY" if signal_data.get("action", "").lower() in ("buy", "long") else "SELL"
            qty = float(order_result.get("origQty") or 0)
            safe_tp_levels = tp_levels or []

            # Zwróć dane bez zapisywania do bazy
            return {
                "symbol": signal_data.get("symbol"),
                "side": side,
                "status": "open",
                "idempotency_key": signal_data.get("idempotency_key"),
                "client_tags": {"order_tag": order_result.get("clientOrderId")},
                "entry_price": price,
                "entry_time": datetime.utcnow(),
                "entry_quantity": qty,
                "position_size_usdt": price * qty if price and qty else None,
                "stop_loss": sl_price,
                "take_profit_1": safe_tp_levels[0] if len(safe_tp_levels) > 0 else None,
                "take_profit_2": safe_tp_levels[1] if len(safe_tp_levels) > 1 else None,
                "take_profit_3": safe_tp_levels[2] if len(safe_tp_levels) > 2 else None,
                "leverage_used": decision.get("leverage", 1),
                "leverage_hint": signal_data.get("leverage"),
                "signal_tier": signal_data.get("tier"),
                "signal_strength": signal_data.get("strength"),
                "signal_timeframe": signal_data.get("timeframe"),
                "signal_session": signal_data.get("session"),
                "indicator_version": signal_data.get("indicator_version"),
                "institutional_flow": signal_data.get("institutional_flow"),
                "retest_confidence": signal_data.get("retest_confidence"),
                "fake_breakout_detected": decision.get("fake_breakout_detected", False),
                "fake_breakout_penalty": signal_data.get("fake_breakout_penalty"),
                "enhanced_regime": signal_data.get("enhanced_regime"),
                "regime_confidence": signal_data.get("regime_confidence"),
                "mtf_agreement_ratio": signal_data.get("mtf_agreement_ratio"),
                "volume_context": {
                    "volume_spike": signal_data.get("volume_spike"),
                    "volume_ratio": signal_data.get("volume_ratio"),
                    "institutional_volume": signal_data.get("institutional_volume"),
                    "retail_volume": signal_data.get("retail_volume"),
                },
                "mode_used": getattr(self.mode_manager, "current_mode", None),
                "alert_data": signal_data,
            }
        except Exception as e:
            logger.error(f"💥 Error preparing trade data: {e}", exc_info=True)
            return None

    def _save_complete_trade_to_db(self, trade_data: Dict, exit_data: Dict = None):
        """Zapisz kompletny trade do bazy (z danymi wyjścia jeśli są)"""
        try:
            with Session() as session:
                if exit_data:
                    # Kompletny trade z wyjściem
                    trade_data.update({
                        "status": "closed",
                        "exit_price": exit_data.get("exit_price"),
                        "exit_time": exit_data.get("exit_time"),
                        "exit_quantity": exit_data.get("exit_quantity"),
                        "exit_reason": exit_data.get("exit_reason"),
                    })
            
                trade = Trade(**trade_data)
                session.add(trade)
                session.commit()
                logger.info(f"💾 Complete trade saved to DB: id={trade.id} {trade.symbol} {trade.side}")
                return trade
        except Exception as e:
            logger.error(f"💥 Error saving complete trade: {e}", exc_info=True)
            return None

    async def close_position(
        self, symbol: str, reason: str = "manual"
    ) -> Dict[str, Any]:
        """Close position with v9.1 enhancements"""
        try:
            if symbol not in self.active_positions:
                return {"status": "error", "error": "Position not found"}

            position = self.active_positions[symbol]

            logger.info(f"🔄 Closing position for {symbol} - Reason: {reason}")

            # Close on exchange
            if not self.dry_run:
                close_result = await binance_handler.close_futures_position(symbol)
            else:
                close_result = {
                    "symbol": symbol,
                    "status": "CLOSED",
                    "reason": reason,
                    "price": await binance_handler.get_current_price(symbol),
                }

            # Update database
            self.close_trade_in_db(symbol, close_result)

            # Remove from active positions
            del self.active_positions[symbol]

            # Remove position guard
            self.position_guards.pop(symbol, None)

            # Send enhanced notification
            await self.discord.send_exit_notification(close_result, reason)

            logger.info(f"✅ Position closed for {symbol}")

            return {"status": "success", "result": close_result}

        except Exception as e:
            logger.error(f"💥 Error closing position {symbol}: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def close_trade_in_db(self, symbol: str, close_result: Dict):
        """Zapisz kompletny trade po zamknięciu pozycji"""
        try:
            # Pobierz dane z pamięci
            if not hasattr(self, 'pending_trades') or symbol not in self.pending_trades:
                logger.warning(f"No pending trade data found for {symbol}")
                return
            
            trade_data = self.pending_trades[symbol]
        
            # Dodaj dane wyjścia
            exit_data = {
                "exit_price": float(close_result.get("price") or 0),
                "exit_time": datetime.utcnow(),
                "exit_quantity": trade_data.get("entry_quantity"),
                "exit_reason": close_result.get("reason", "manual"),
            }
        
            # Zapisz kompletny trade
            trade = self._save_complete_trade_to_db(trade_data, exit_data)
        
            # Usuń z pamięci
            del self.pending_trades[symbol]
        
            if trade:
                logger.info(f"💾 Complete trade {trade.id} saved to DB after close")
        
        except Exception as e:
            logger.error(f"💥 Error saving complete trade: {e}", exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Get enhanced bot status with v9.1 metrics"""
        return {
            "version": "9.1",
            "running": self.running,
            "paused": self.paused,
            "emergency_mode": self.emergency_mode,
            "mode": getattr(self.mode_manager, "current_mode", "unknown"),
            "dry_run": self.dry_run,
            "active_positions": len(self.active_positions),
            "positions": list(self.active_positions.keys()),
            "risk_per_trade": f"{self.runtime_risk:.2%}",
            "blacklisted_symbols": list(self.blacklisted_symbols),
            "ml_enabled": getattr(Config, "USE_ML_FOR_DECISION", False),
            "last_signal": (
                self.last_signal.get("timestamp") if self.last_signal else None
            ),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "performance_metrics": self.performance_metrics,
            "position_guards_active": len(self.position_guards),
            "signal_intelligence_enabled": getattr(
                Config, "ENABLE_FAKE_BREAKOUT_DETECTION", False
            ),
            "institutional_flow_enabled": getattr(
                Config, "ENABLE_INSTITUTIONAL_FLOW", False
            ),
            "diagnostics_enabled": self.diagnostic_enabled,
            "explainability_level": self.explainability_level,
            "proactive_alerts": self.proactive_alerts,
            "last_trace_id": self.last_signal.get("trace_id") if self.last_signal else None,
        }

    def pause_trading(self):
        """Pause trading"""
        self.paused = True
        logger.info("⏸️ Trading paused")

    def resume_trading(self):
        """Resume trading"""
        self.paused = False
        logger.info("▶️ Trading resumed")

    def enable_emergency_mode(self):
        """Enable emergency mode"""
        self.emergency_mode = True
        logger.warning("🚨 Emergency mode ENABLED")

    def disable_emergency_mode(self):
        """Disable emergency mode"""
        self.emergency_mode = False
        logger.info("✅ Emergency mode DISABLED")

    async def emergency_close_all(self):
        """Emergency close all positions"""
        logger.warning("🚨 EMERGENCY: Closing all positions")

        for symbol in list(self.active_positions.keys()):
            await self.close_position(symbol, "emergency_close")

        await self.discord.send_error_notification("🚨 EMERGENCY: All positions closed")

    def get_last_signal(self) -> Optional[Dict[str, Any]]:
        """Get last received signal"""
        return self.last_signal
    async def get_diagnostic_report(self, trace_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive diagnostic report"""
        try:
            if trace_id:
                # Get specific trace report
                with Session() as session:
                    trace = get_decision_trace(session, trace_id)
                    if not trace:
                        return {"error": "Trace not found"}
                    
                    # Get related data
                    from database import ParameterDecision, ShapExplanation
                    param_decisions = session.query(ParameterDecision).filter_by(trace_id=trace_id).all()
                    shap_explanations = session.query(ShapExplanation).filter_by(trace_id=trace_id).all()
                    
                    return {
                        "trace": trace.to_dict() if hasattr(trace, 'to_dict') else str(trace),
                        "parameter_decisions": [p.to_dict() if hasattr(p, 'to_dict') else str(p) for p in param_decisions],
                        "shap_explanations": [s.to_dict() if hasattr(s, 'to_dict') else str(s) for s in shap_explanations],
                    }
            else:
                # Get summary report
                with Session() as session:
                    summary = get_diagnostic_summary(session, hours)
                    return summary
                    
        except Exception as e:
            logger.error(f"Error getting diagnostic report: {e}")
            return {"error": str(e)}

    async def get_pattern_alerts(self) -> List[Dict[str, Any]]:
        """Get active pattern alerts"""
        try:
            with Session() as session:
                alerts = get_active_pattern_alerts(session)
                return [alert.to_dict() if hasattr(alert, 'to_dict') else str(alert) for alert in alerts]
        except Exception as e:
            logger.error(f"Error getting pattern alerts: {e}")
            return []

    async def acknowledge_alert(self, alert_id: int, user: str) -> bool:
        """Acknowledge a pattern alert"""
        try:
            with Session() as session:
                acknowledge_pattern_alert(session, alert_id, user)
                return True
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False

    def set_dry_run(self, enabled: bool):
        """Set dry run mode"""
        self.dry_run = enabled
        logger.info(f"🧪 Dry run mode: {enabled}")

    def set_risk(self, risk_percent: float):
        """Set risk per trade"""
        self.runtime_risk = max(0.001, min(0.1, risk_percent))
        logger.info(f"📊 Risk per trade set to: {self.runtime_risk:.2%}")

    def blacklist_symbol(self, symbol: str):
        """Add symbol to blacklist"""
        self.blacklisted_symbols.add(symbol.upper())
        logger.info(f"🚫 Symbol {symbol} blacklisted")

    def whitelist_symbol(self, symbol: str):
        """Remove symbol from blacklist"""
        self.blacklisted_symbols.discard(symbol.upper())
        logger.info(f"✅ Symbol {symbol} whitelisted")

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions with v9.1 enhancements"""
        positions = []

        # Get from exchange
        if not self.dry_run:
            try:
                exchange_positions = await binance_handler.get_futures_positions()
                for pos in exchange_positions:
                    if float(pos.get("positionAmt", 0)) != 0:
                        positions.append(pos)
            except Exception as e:
                logger.error(f"Error getting exchange positions: {e}")

        # Add local tracking with v9.1 data
        for symbol, pos_data in self.active_positions.items():
            positions.append(
                {
                    "symbol": symbol,
                    "side": pos_data["side"],
                    "quantity": pos_data["quantity"],
                    "entry_price": pos_data["entry_price"],
                    "entry_time": pos_data["entry_time"].isoformat(),
                    "leverage": pos_data.get("leverage", 1),
                    "tier": pos_data.get("tier", "Standard"),
                    "signal_strength": pos_data.get("signal_strength", 0.5),
                    "order_tag": pos_data.get("order_tag", ""),
                }
            )

        return positions

    async def _run_webhook_server(self):
        """Run webhook server"""
        try:
            # ===== DODAJ FILTR LOGÓW =====
            # Filtr dla health/metrics spam
            class HealthMetricsFilter(logging.Filter):
                def filter(self, record):
                    if hasattr(record, 'getMessage'):
                        message = record.getMessage()
                        # Filtruj health/metrics/status
                        if any(endpoint in message for endpoint in ['/health', '/metrics', '/status']):
                            return False
                    return True

            # Dodaj filtr do uvicorn access log
            uvicorn_access = logging.getLogger("uvicorn.access")
            uvicorn_access.addFilter(HealthMetricsFilter())

            webhook_port = getattr(Config, "WEBHOOK_PORT", 5000)
            config = uvicorn.Config(
                app=webhook_app,
                host="0.0.0.0",
                port=webhook_port,
                log_level="info",
                access_log=True,
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"💥 Webhook server error: {e}", exc_info=True)

    async def _health_check_loop(self):
        """Enhanced health check loop"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minut zamiast 30s

                # Sprawdzaj połączenie tylko gdy potrzeba
                if self.connection_check_needed:
                    if not await binance_handler.check_connection():
                        logger.error("❌ Binance connection lost")
                        await self.discord.send_error_notification("Binance connection lost")
                    self.connection_check_needed = False

                # Check database connection
                try:
                    with Session() as session:
                        session.execute("SELECT 1")
                except Exception as e:
                    logger.error(f"❌ Database connection lost: {e}")

                # Check user stream
                if self.user_stream and not self.user_stream.is_connected():
                    logger.warning("⚠️ User stream disconnected, reconnecting...")
                    await self.user_stream.reconnect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"💥 Health check error: {e}", exc_info=True)

    async def _position_monitor_loop(self):
        """Enhanced position monitoring"""
        while self.running:
            try:
                await asyncio.sleep(self.position_check_interval)

                if self.paused or self.dry_run:
                    continue

                # Check each position
                for symbol in list(self.active_positions.keys()):
                    position = self.active_positions[symbol]
                    current_price = await binance_handler.get_current_price(symbol)

                    # Check stop loss
                    if position["side"] == "buy":
                        if current_price <= position["sl_price"]:
                            await self.close_position(symbol, "stop_loss_hit")
                    else:
                        if current_price >= position["sl_price"]:
                            await self.close_position(symbol, "stop_loss_hit")

                    # Check take profits
                    for i, tp_price in enumerate(position["tp_levels"]):
                        if position["side"] == "buy":
                            if current_price >= tp_price:
                                await self.close_position(symbol, f"tp{i+1}_hit")
                                break
                        else:
                            if current_price <= tp_price:
                                await self.close_position(symbol, f"tp{i+1}_hit")
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"💥 Position monitor error: {e}", exc_info=True)

    async def _emergency_monitor_loop(self):
        """Monitor for emergency conditions"""
        while self.running:
            try:
                await asyncio.sleep(1800)  # Check every 30 minutes

                # Check daily loss threshold
                emergency_threshold = getattr(Config, "EMERGENCY_CLOSE_THRESHOLD", -100)
                if self.daily_pnl <= emergency_threshold:
                    logger.warning(f"🚨 Daily loss threshold reached: {self.daily_pnl}")
                    self.enable_emergency_mode()
                    await self.emergency_close_all()

                # Check max daily loss
                max_daily_loss = getattr(Config, "MAX_DAILY_LOSS", 500)
                if abs(self.daily_pnl) >= max_daily_loss:
                    logger.warning(f"🚨 Max daily loss reached: {abs(self.daily_pnl)}")
                    self.enable_emergency_mode()
                    await self.emergency_close_all()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"💥 Emergency monitor error: {e}", exc_info=True)

    async def _cleanup_loop(self):
        """Enhanced cleanup tasks"""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)

                # Cleanup old database records
                try:
                    with Session() as session:
                        cleanup_old_data(session)
                except Exception as e:
                    logger.error(f"Error cleaning up old data: {e}")

                # Train ML model if needed
                if self.ml_predictor:
                    try:
                        self.ml_predictor.retrain_if_needed()
                    except Exception as e:
                        logger.error(f"Error retraining ML model: {e}")

                # Clean old position guards
                current_time = datetime.utcnow()
                alert_max_age = getattr(Config, "ALERT_MAX_AGE_SEC", 300)
                expired_guards = [
                    symbol
                    for symbol, timestamp in self.position_guards.items()
                    if (current_time - timestamp).total_seconds() > alert_max_age * 2
                ]

                for symbol in expired_guards:
                    del self.position_guards[symbol]

                logger.info(
                    f"🧹 Cleanup completed. Removed {len(expired_guards)} expired position guards"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"💥 Cleanup error: {e}", exc_info=True)

    async def _analytics_loop(self):
        """Enhanced analytics loop"""
        while self.running:
            try:
                await asyncio.sleep(self.analytics_interval)

                # Generate performance report
                try:
                    report = self.analytics.generate_performance_report()

                    # Send to Discord if significant
                    if report.get("total_trades", 0) > 0:
                        await self.discord.send_performance_report(report)
                except Exception as e:
                    logger.error(f"Error generating analytics: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"💥 Analytics error: {e}", exc_info=True)

    async def _performance_report_loop(self):
        """Regular performance reporting"""
        while self.running:
            try:
                await asyncio.sleep(self.performance_report_interval)

                # Generate and send performance metrics
                metrics = {
                    "daily_pnl": self.daily_pnl,
                    "daily_trades": self.daily_trades,
                    "active_positions": len(self.active_positions),
                    "performance_metrics": self.performance_metrics,
                }

                try:
                    await self.discord.send_performance_update(metrics)
                except Exception as e:
                    logger.error(f"Error sending performance update: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"💥 Performance report error: {e}", exc_info=True)

    async def _diagnostic_loop(self):
        """Diagnostic and pattern detection loop"""
        while self.running:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                if not self.diagnostic_enabled:
                    continue
                # Uruchom pełną diagnostykę tylko co godzinę
                if datetime.utcnow().minute not in [0, 30]:  # Tylko o pełnych i półgodzinach
                    continue
                
                # Pattern detection and proactive alerts
                await self._detect_patterns_and_alert()
                
                # System health logging
                try:
                    # --- POCZĄTEK POPRAWKI ---
                    health_report = await self.diagnostic_manager.run_full_diagnostics()
                    # Reaguj na krytyczne problemy
                    await self._handle_critical_diagnostics(health_report)

                    health_data = {
                        "timestamp": datetime.utcnow(),
                        "overall_status": health_report.get("overall_health", {}).get("status", "UNKNOWN"),
                        "health_score": health_report.get("overall_health", {}).get("score", 0.0),
                        "component_results": health_report.get("component_results", []),
                        "system_metrics": health_report.get("system_metrics", {}),
                        "trading_metrics": health_report.get("trading_metrics", {}),
                        "recommendations": health_report.get("recommendations", []),
                        "execution_time_ms": health_report.get("execution_summary", {}).get("execution_time_ms", 0)
                    }
                    # --- KONIEC POPRAWKI ---

                    with Session() as session:
                        log_system_health(session, health_data)

                except Exception as e:
                    logger.error(f"Error logging system health: {e}")
                
                # Cleanup old diagnostic data
                try:
                    with Session() as session:
                        cleanup_diagnostic_data(session, days_to_keep=7)
                except Exception as e:
                    logger.error(f"Error cleaning diagnostic data: {e}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"💥 Diagnostic loop error: {e}", exc_info=True)

    async def check_health(self) -> Dict:
        """Sprawdza zdrowie systemu"""
        try:
            return {
                'overall_status': 'HEALTHY',
                'components': {
                    'binance': 'HEALTHY' if await binance_handler.check_connection() else 'UNHEALTHY',
                    'database': 'HEALTHY',
                    'discord': 'HEALTHY'
                },
                'recent_issues': []
            }
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {'overall_status': 'UNKNOWN', 'components': {}}


# Global bot instance
bot_instance: Optional[TradingBot] = None


def get_performance_metrics(self, hours: int = 24) -> Dict:
    """Zwraca metryki wydajności"""
    try:
        return {
            'avg_response_time': 0,
            'max_response_time': 0,
            'min_response_time': 0,
            'total_trades': 0,
            'success_rate': 0
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {}

async def main():
    """Enhanced main entry point"""
    global bot_instance

    try:
        logger.info("🚀 Starting Trading Bot v9.1...")

        # Create bot instance
        bot_instance = TradingBot()
        webhook_app.state.bot = bot_instance
        
        # Initialize diagnostics engine with bot instance
        from diagnostics import DiagnosticsEngine
        diagnostics_engine = DiagnosticsEngine(bot_instance)
        bot_instance.diagnostics_engine = diagnostics_engine
        
        # Set global diagnostics_engine for convenience functions
        import diagnostics
        diagnostics.diagnostics_engine = diagnostics_engine

        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info(f"📡 Received signal {sig}")
            if bot_instance:
                bot_instance.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start bot
        await bot_instance.start()

    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        if bot_instance:
            try:
                await bot_instance.discord.send_error_notification(
                    f"Bot fatal error: {e}"
                )
            except:
                pass
        sys.exit(1)

async def _run_diagnostics_background(self, signal_data: Dict, decision: Dict, result: Dict):
    """Uruchom diagnostykę w tle po wykonaniu trade'u"""
    try:
        trace_id = await self._create_diagnostic_trace(signal_data)
        await self._log_pine_health_data(trace_id, signal_data)
        
        if decision.get("parameter_decisions"):
            await self._log_parameter_decisions(trace_id, decision["parameter_decisions"])
            
        await self._complete_diagnostic_trace(trace_id, {
            "final_decision": "EXECUTED",
            "processing_stage": "completed_background",
            "execution_result": result
        })
        
        logger.info(f"🔍 Background diagnostics completed for trace: {trace_id}")
        
    except Exception as e:
        logger.error(f"Background diagnostics error: {e}")

async def _handle_critical_diagnostics(self, health_report: dict):
        """Reaguj na krytyczne problemy diagnostyczne"""
        try:
            overall_status = health_report.get("overall_health", {}).get("status", "UNKNOWN")
          
            if overall_status == "CRITICAL":
                critical_components = []
                for component in health_report.get("component_results", []):
                    if component.get("status") == "critical":
                        critical_components.append(component.get("component"))
              
                logger.error(f"🚨 CRITICAL system status detected: {critical_components}")
              
                # Automatyczne działania naprawcze
                if "database" in critical_components:
                    logger.warning("🔄 Attempting database reconnection...")
                    # Dodaj logikę reconnect do bazy
                  
                if "binance_api" in critical_components:
                    logger.warning("🔄 Marking connection check needed...")
                    self.connection_check_needed = True
                  
                if "bot_core" in critical_components:
                    logger.warning("🔄 Enabling emergency mode...")
                    self.enable_emergency_mode()
              
                # Wyślij alert na Discord
                await self.discord.send_error_notification(
                    "🚨 CRITICAL System Status",
                    f"Critical issues detected in: {', '.join(critical_components)}"
                )
              
                # Jeśli więcej niż 2 komponenty krytyczne - zatrzymaj trading
                if len(critical_components) >= 2:
                    logger.error("🚨 Multiple critical components - pausing trading")
                    self.pause_trading()
                  
        except Exception as e:
            logger.error(f"Error handling critical diagnostics: {e}")


if __name__ == "__main__":
    # Run the enhanced bot
    asyncio.run(main())

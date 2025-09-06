"""
Comprehensive Diagnostics System v9.1 - HYBRID ULTRA-DIAGNOSTICS
Centralny system diagnostyczny dla trading bota z pełną analizą i monitoringiem
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import traceback

from database import (
    Session, Trade, Position, AlertHistory, ExecutionTrace, MLPrediction,
    MLModelMetrics, SystemHealth, get_open_positions, log_execution_trace,
    complete_execution_trace
)
from config import Config

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """System component types"""
    DATABASE = "database"
    BINANCE_API = "binance_api"
    DISCORD = "discord"
    ML_PREDICTOR = "ml_predictor"
    WEBHOOK = "webhook"
    BOT_CORE = "bot_core"
    PINE_SCRIPT = "pine_script"


@dataclass
class DiagnosticResult:
    """Single diagnostic check result"""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    execution_time_ms: int
    recommendations: List[str]


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    database_response_time: float
    api_response_time: float
    active_connections: int
    error_rate: float


@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_signals: int
    signals_taken: int
    signals_rejected: int
    ml_rejections: int
    fake_breakout_detections: int
    active_positions: int
    daily_pnl: float
    daily_trades: int
    win_rate: float
    avg_trade_duration: float
    risk_exposure: float


class DiagnosticsEngine:
    """
    Centralny silnik diagnostyczny v9.1 HYBRID ULTRA-DIAGNOSTICS
    Zapewnia kompleksową analizę wszystkich komponentów systemu
    """
    
    def __init__(self, bot=None):
        self.logger = logging.getLogger(__name__)
        self.bot = bot
        self.last_full_check = None
        self.diagnostic_history = []
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0,
            "api_latency": 2000,  # ms
            "database_latency": 1000,  # ms
            "position_risk": 0.8  # 80% of max risk
        }
        self.component_weights = {
            ComponentType.DATABASE: 0.25,
            ComponentType.BINANCE_API: 0.20,
            ComponentType.BOT_CORE: 0.20,
            ComponentType.ML_PREDICTOR: 0.15,
            ComponentType.DISCORD: 0.10,
            ComponentType.WEBHOOK: 0.05,
            ComponentType.PINE_SCRIPT: 0.05
        }
    
    def update_pine_health(self, health_data: dict):
        """Update Pine Script health data"""
        self.pine_health = health_data
        self.last_update = datetime.utcnow()
        
        # Analiza trendów
        if health_data.get('health_score', 0) < 0.3:
            self.add_warning("Pine Script health critically low")
        
        # Zapisz do bazy
        with Session() as session:
            log_pine_health(session, health_data)
    
    # DODAJ też metodę add_warning jeśli jej nie masz:
    def add_warning(self, warning: str):
        """Add warning to diagnostic history"""
        if not hasattr(self, 'warnings'):
            self.warnings = []
        self.warnings.append({
            'timestamp': datetime.utcnow(),
            'message': warning
        })
        self.logger.warning(f"Diagnostic warning: {warning}")

    async def run_full_diagnostics(self) -> Dict[str, Any]:
        """
        Uruchom pełną diagnostykę systemu
        Zwraca kompletny raport diagnostyczny
        """
        start_time = time.time()
        trace_id = log_execution_trace("full_diagnostics", {"initiated_by": "diagnostics_engine"})
        
        try:
            self.logger.info("🔍 Starting full system diagnostics...")
            
            # Uruchom wszystkie testy diagnostyczne równolegle
            diagnostic_tasks = [
                self._check_database_health(),
                self._check_binance_api_health(),
                self._check_discord_health(),
                self._check_ml_predictor_health(),
                self._check_webhook_health(),
                self._check_bot_core_health(),
                self._check_pine_script_health()
            ]
            
            # Wykonaj testy
            diagnostic_results = await asyncio.gather(*diagnostic_tasks, return_exceptions=True)
            
            # Przetwórz wyniki
            results = []
            for i, result in enumerate(diagnostic_results):
                if isinstance(result, Exception):
                    component_name = list(ComponentType)[i].value
                    results.append(DiagnosticResult(
                        component=component_name,
                        status=HealthStatus.CRITICAL,
                        message=f"Diagnostic check failed: {str(result)}",
                        details={"error": str(result), "traceback": traceback.format_exc()},
                        timestamp=datetime.utcnow(),
                        execution_time_ms=0,
                        recommendations=[f"Investigate {component_name} component failure"]
                    ))
                else:
                    results.append(result)
            
            # Zbierz metryki systemowe
            system_metrics = await self._collect_system_metrics()
            trading_metrics = await self._collect_trading_metrics()
            
            # Oblicz ogólny stan zdrowia
            overall_health = self._calculate_overall_health(results)
            
            # Generuj rekomendacje
            recommendations = self._generate_system_recommendations(results, system_metrics, trading_metrics)
            
            # Przygotuj raport
            diagnostic_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "9.1",
                "overall_health": {
                    "status": overall_health.value,
                    "score": self._calculate_health_score(results),
                    "critical_issues": len([r for r in results if r.status == HealthStatus.CRITICAL]),
                    "warnings": len([r for r in results if r.status == HealthStatus.WARNING])
                },
                "component_results": [asdict(result) for result in results],
                "system_metrics": asdict(system_metrics) if system_metrics else {},
                "trading_metrics": asdict(trading_metrics) if trading_metrics else {},
                "recommendations": recommendations,
                "execution_summary": {
                    "total_checks": len(results),
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                    "checks_passed": len([r for r in results if r.status == HealthStatus.HEALTHY]),
                    "checks_failed": len([r for r in results if r.status == HealthStatus.CRITICAL])
                }
            }
            
            # Zapisz do historii
            self.diagnostic_history.append(diagnostic_report)
            if len(self.diagnostic_history) > 100:  # Zachowaj ostatnie 100 raportów
                self.diagnostic_history = self.diagnostic_history[-100:]
            
            self.last_full_check = datetime.utcnow()
            
            # Zapisz do bazy danych
            await self._save_diagnostic_report(diagnostic_report)
            
            # Zakończ trace
            execution_time = int((time.time() - start_time) * 1000)
            complete_execution_trace(trace_id, True, execution_time, f"Full diagnostics completed - {overall_health.value}")
            
            self.logger.info(f"✅ Full diagnostics completed in {execution_time}ms - Status: {overall_health.value}")
            
            return diagnostic_report
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            complete_execution_trace(trace_id, False, execution_time, f"Diagnostics failed: {str(e)}")
            self.logger.error(f"❌ Full diagnostics failed: {e}", exc_info=True)
            raise
    
    async def _check_database_health(self) -> DiagnosticResult:
        """Sprawdź stan zdrowia bazy danych"""
        start_time = time.time()
        
        try:
            details = {}
            recommendations = []
            
            with Session() as session:
                # Test podstawowej łączności
                session.execute("SELECT 1")
                details["connection"] = "healthy"
                
                # Sprawdź rozmiar bazy danych
                try:
                    result = session.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
                    db_size = result.scalar()
                    details["database_size"] = db_size
                except:
                    details["database_size"] = "unknown"
                
                # Sprawdź aktywne połączenia
                try:
                    result = session.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                    active_connections = result.scalar()
                    details["active_connections"] = active_connections
                    
                    if active_connections > 50:
                        recommendations.append("High number of active database connections detected")
                except:
                    details["active_connections"] = "unknown"
                
                # Sprawdź ostatnie transakcje
                try:
                    recent_trades = session.query(Trade).filter(
                        Trade.entry_time >= datetime.utcnow() - timedelta(hours=24)
                    ).count()
                    details["recent_trades_24h"] = recent_trades
                    
                    recent_alerts = session.query(AlertHistory).filter(
                        AlertHistory.received_at >= datetime.utcnow() - timedelta(hours=1)
                    ).count()
                    details["recent_alerts_1h"] = recent_alerts
                    
                except Exception as e:
                    details["query_error"] = str(e)
                    recommendations.append("Database query performance issues detected")
                
                # Sprawdź indeksy i wydajność
                try:
                    # Sprawdź czy są długo działające zapytania
                    result = session.execute("""
                        SELECT count(*) FROM pg_stat_activity 
                        WHERE state = 'active' AND now() - query_start > interval '30 seconds'
                    """)
                    long_queries = result.scalar()
                    details["long_running_queries"] = long_queries
                    
                    if long_queries > 0:
                        recommendations.append(f"Found {long_queries} long-running queries")
                        
                except:
                    pass
            
            execution_time = int((time.time() - start_time) * 1000)
            details["response_time_ms"] = execution_time
            
            # Określ status
            if execution_time > self.alert_thresholds["database_latency"]:
                status = HealthStatus.WARNING
                message = f"Database response time high: {execution_time}ms"
                recommendations.append("Database performance optimization needed")
            elif recommendations:
                status = HealthStatus.WARNING
                message = "Database operational with warnings"
            else:
                status = HealthStatus.HEALTHY
                message = "Database operating normally"
            
            return DiagnosticResult(
                component="database",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return DiagnosticResult(
                component="database",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e), "response_time_ms": execution_time},
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=["Check database connection and credentials", "Verify PostgreSQL service status"]
            )
    
    async def _check_binance_api_health(self) -> DiagnosticResult:
        """Sprawdź stan zdrowia Binance API"""
        start_time = time.time()
        
        try:
            # Import tutaj żeby uniknąć circular imports
            from binance_handler import BinanceHandler
            
            details = {}
            recommendations = []
            
            # Test łączności z API
            handler = BinanceHandler()
            
            # Sprawdź status serwera
            server_time = await handler.get_server_time()
            if server_time:
                details["server_connection"] = "healthy"
                details["server_time"] = server_time
                
                # Sprawdź różnicę czasu
                local_time = int(time.time() * 1000)
                time_diff = abs(local_time - server_time)
                details["time_sync_diff_ms"] = time_diff
                
                if time_diff > 5000:  # 5 sekund
                    recommendations.append(f"Time synchronization issue: {time_diff}ms difference")
            else:
                details["server_connection"] = "failed"
                recommendations.append("Cannot connect to Binance servers")
            
            # Sprawdź informacje o koncie
            try:
                account_info = await handler.get_account_info()
                if account_info:
                    details["account_status"] = "accessible"
                    details["can_trade"] = account_info.get("canTrade", False)
                    details["can_withdraw"] = account_info.get("canWithdraw", False)
                    details["can_deposit"] = account_info.get("canDeposit", False)
                    
                    # Sprawdź saldo
                    balances = account_info.get("balances", [])
                    usdt_balance = next((b for b in balances if b["asset"] == "USDT"), None)
                    if usdt_balance:
                        free_usdt = float(usdt_balance["free"])
                        details["usdt_balance"] = free_usdt
                        
                        if free_usdt < 100:  # Minimum 100 USDT
                            recommendations.append(f"Low USDT balance: ${free_usdt:.2f}")
                else:
                    details["account_status"] = "inaccessible"
                    recommendations.append("Cannot access account information")
                    
            except Exception as e:
                details["account_error"] = str(e)
                recommendations.append("Account access issues detected")
            
            # Sprawdź limity API
            try:
                # Symuluj sprawdzenie limitów przez próbę pobrania ceny
                price_data = await handler.get_symbol_price("BTCUSDT")
                if price_data:
                    details["market_data_access"] = "healthy"
                    details["btc_price"] = price_data
                else:
                    details["market_data_access"] = "failed"
                    recommendations.append("Market data access issues")
                    
            except Exception as e:
                details["market_data_error"] = str(e)
                recommendations.append("Market data API issues detected")
            
            execution_time = int((time.time() - start_time) * 1000)
            details["response_time_ms"] = execution_time
            
            # Określ status
            if execution_time > self.alert_thresholds["api_latency"]:
                status = HealthStatus.WARNING
                message = f"Binance API response time high: {execution_time}ms"
                recommendations.append("API performance issues detected")
            elif not details.get("server_connection") == "healthy":
                status = HealthStatus.CRITICAL
                message = "Cannot connect to Binance API"
            elif not details.get("account_status") == "accessible":
                status = HealthStatus.CRITICAL
                message = "Account access issues"
            elif recommendations:
                status = HealthStatus.WARNING
                message = "Binance API operational with warnings"
            else:
                status = HealthStatus.HEALTHY
                message = "Binance API operating normally"
            
            return DiagnosticResult(
                component="binance_api",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return DiagnosticResult(
                component="binance_api",
                status=HealthStatus.CRITICAL,
                message=f"Binance API check failed: {str(e)}",
                details={"error": str(e), "response_time_ms": execution_time},
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=["Check Binance API credentials", "Verify network connectivity", "Check API key permissions"]
            )
    
    async def _check_discord_health(self) -> DiagnosticResult:
        """Sprawdź stan zdrowia Discord notifications"""
        start_time = time.time()
        
        try:
            from discord_notifications import discord_notifier
            
            details = {}
            recommendations = []
            
            # Sprawdź konfigurację
            if not Config.DISCORD_WEBHOOK_URL:
                details["configuration"] = "missing"
                recommendations.append("Discord webhook URL not configured")
                status = HealthStatus.WARNING
                message = "Discord notifications not configured"
            else:
                details["configuration"] = "present"
                
                # Test połączenia (bez wysyłania wiadomości)
                try:
                    # Sprawdź czy webhook URL jest prawidłowy
                    import re
                    webhook_pattern = r'https://discord\.com/api/webhooks/\d+/[\w-]+'
                    if re.match(webhook_pattern, Config.DISCORD_WEBHOOK_URL):
                        details["webhook_url_format"] = "valid"
                    else:
                        details["webhook_url_format"] = "invalid"
                        recommendations.append("Discord webhook URL format appears invalid")
                    
                    # Sprawdź ostatnie powiadomienia
                    if hasattr(discord_notifier, 'last_notification_time'):
                        last_notification = discord_notifier.last_notification_time
                        if last_notification:
                            time_since_last = (datetime.utcnow() - last_notification).total_seconds()
                            details["last_notification_seconds_ago"] = int(time_since_last)
                            
                            if time_since_last > 3600:  # 1 godzina
                                recommendations.append("No Discord notifications sent in the last hour")
                    
                    # Sprawdź kolejkę powiadomień
                    if hasattr(discord_notifier, 'notification_queue'):
                        queue_size = len(discord_notifier.notification_queue)
                        details["notification_queue_size"] = queue_size
                        
                        if queue_size > 10:
                            recommendations.append(f"Large notification queue: {queue_size} pending")
                    
                    status = HealthStatus.HEALTHY if not recommendations else HealthStatus.WARNING
                    message = "Discord notifications operational" if not recommendations else "Discord operational with warnings"
                    
                except Exception as e:
                    details["test_error"] = str(e)
                    recommendations.append("Discord notification system test failed")
                    status = HealthStatus.WARNING
                    message = "Discord notifications may have issues"
            
            execution_time = int((time.time() - start_time) * 1000)
            details["response_time_ms"] = execution_time
            
            return DiagnosticResult(
                component="discord",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return DiagnosticResult(
                component="discord",
                status=HealthStatus.CRITICAL,
                message=f"Discord check failed: {str(e)}",
                details={"error": str(e), "response_time_ms": execution_time},
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=["Check Discord notification configuration", "Verify webhook URL"]
            )
    
    async def _check_ml_predictor_health(self) -> DiagnosticResult:
        """Sprawdź stan zdrowia ML Predictor"""
        start_time = time.time()
        
        try:
            from ml_predictor import MLPredictor
            
            details = {}
            recommendations = []
            
            # Sprawdź czy ML Predictor jest dostępny
            try:
                predictor = MLPredictor()
                details["predictor_available"] = True
                
                # Sprawdź załadowane modele
                details["win_prob_model_loaded"] = predictor.win_prob_model is not None
                details["pnl_model_loaded"] = predictor.pnl_model is not None
                
                if not details["win_prob_model_loaded"]:
                    recommendations.append("Win probability model not loaded")
                if not details["pnl_model_loaded"]:
                    recommendations.append("PnL prediction model not loaded")
                
                # Sprawdź metryki modelu
                if hasattr(predictor, 'model_metrics'):
                    metrics = predictor.model_metrics
                    details["model_accuracy"] = metrics.get("accuracy", 0.0)
                    details["last_model_update"] = metrics.get("last_updated")
                    
                    if metrics.get("accuracy", 0.0) < 0.6:
                        recommendations.append(f"Low model accuracy: {metrics.get('accuracy', 0.0):.3f}")
                    
                    # Sprawdź czy model wymaga aktualizacji
                    if metrics.get("last_updated"):
                        try:
                            last_update = datetime.fromisoformat(metrics["last_updated"])
                            days_old = (datetime.utcnow() - last_update).days
                            details["model_age_days"] = days_old
                            
                            if days_old > 30:
                                recommendations.append(f"Model is {days_old} days old - consider retraining")
                        except:
                            pass
                
                # Sprawdź dane treningowe
                with Session() as session:
                    training_data_count = session.query(Trade).filter(
                        Trade.status == "closed",
                        Trade.raw_signal_data.is_not(None),
                        Trade.pnl_usdt.is_not(None),
                        Trade.is_dry_run.is_(False)
                    ).count()
                    
                    details["training_data_count"] = training_data_count
                    
                    if training_data_count < predictor.min_trades_for_training:
                        recommendations.append(f"Insufficient training data: {training_data_count} < {predictor.min_trades_for_training}")
                
                # Sprawdź ostatnie predykcje
                if hasattr(predictor, 'prediction_count'):
                    details["total_predictions"] = predictor.prediction_count
                    details["successful_predictions"] = getattr(predictor, 'successful_predictions', 0)
                    
                    if predictor.prediction_count > 0:
                        success_rate = predictor.successful_predictions / predictor.prediction_count
                        details["prediction_success_rate"] = success_rate
                        
                        if success_rate < 0.8:
                            recommendations.append(f"Low prediction success rate: {success_rate:.3f}")
                
                # Test predykcji
                try:
                    test_signal = {
                        "symbol": "BTCUSDT",
                        "tier": "Standard",
                        "strength": 0.5,
                        "rsi": 50.0,
                        "mfi": 50.0,
                        "adx": 25.0
                    }
                    
                    prediction = await predictor.predict(test_signal)
                    if prediction and "win_probability" in prediction:
                        details["prediction_test"] = "passed"
                        details["test_prediction"] = prediction
                    else:
                        details["prediction_test"] = "failed"
                        recommendations.append("ML prediction test failed")
                        
                except Exception as e:
                    details["prediction_test_error"] = str(e)
                    recommendations.append("ML prediction test error")
                
            except Exception as e:
                details["predictor_available"] = False
                details["predictor_error"] = str(e)
                recommendations.append("ML Predictor initialization failed")
            
            execution_time = int((time.time() - start_time) * 1000)
            details["response_time_ms"] = execution_time
            
            # Określ status
            if not details.get("predictor_available"):
                status = HealthStatus.CRITICAL
                message = "ML Predictor not available"
            elif not details.get("win_prob_model_loaded") or not details.get("pnl_model_loaded"):
                status = HealthStatus.CRITICAL
                message = "ML models not loaded"
            elif len(recommendations) > 2:
                status = HealthStatus.WARNING
                message = "ML Predictor operational with multiple warnings"
            elif recommendations:
                status = HealthStatus.WARNING
                message = "ML Predictor operational with warnings"
            else:
                status = HealthStatus.HEALTHY
                message = "ML Predictor operating normally"
            
            return DiagnosticResult(
                component="ml_predictor",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return DiagnosticResult(
                component="ml_predictor",
                status=HealthStatus.CRITICAL,
                message=f"ML Predictor check failed: {str(e)}",
                details={"error": str(e), "response_time_ms": execution_time},
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=["Check ML Predictor configuration", "Verify model files", "Check training data availability"]
            )
    
    async def _check_webhook_health(self) -> DiagnosticResult:
        """Sprawdź stan zdrowia Webhook"""
        start_time = time.time()
        
        try:
            details = {}
            recommendations = []
            
            # Sprawdź konfigurację webhook
            details["webhook_port"] = Config.WEBHOOK_PORT
            details["webhook_secret_configured"] = bool(Config.WEBHOOK_SECRET)
            
            if not Config.WEBHOOK_SECRET:
                recommendations.append("Webhook secret not configured - security risk")
            
            # Sprawdź ostatnie alerty
            with Session() as session:
                # Ostatnie alerty w ciągu godziny
                recent_cutoff = datetime.utcnow() - timedelta(hours=1)
                recent_alerts = session.query(AlertHistory).filter(
                    AlertHistory.received_at >= recent_cutoff
                ).all()
                
                details["recent_alerts_count"] = len(recent_alerts)
                
                if recent_alerts:
                    processed_count = len([a for a in recent_alerts if a.processed])
                    successful_count = len([a for a in recent_alerts if a.processed and not a.error])
                    
                    details["processing_rate"] = processed_count / len(recent_alerts) if recent_alerts else 0
                    details["success_rate"] = successful_count / processed_count if processed_count else 0
                    
                    # Sprawdź błędy
                    error_alerts = [a for a in recent_alerts if a.error]
                    if error_alerts:
                        details["error_count"] = len(error_alerts)
                        details["common_errors"] = {}
                        
                        for alert in error_alerts:
                            error_key = alert.error[:50] if alert.error else "unknown"
                            details["common_errors"][error_key] = details["common_errors"].get(error_key, 0) + 1
                        
                        error_rate = len(error_alerts) / len(recent_alerts)
                        if error_rate > 0.1:  # 10% błędów
                            recommendations.append(f"High error rate: {error_rate:.1%}")
                    
                    # Sprawdź opóźnienia
                    processing_times = []
                    for alert in recent_alerts:
                        if alert.processed and alert.received_at:
                            # Szacunkowy czas przetwarzania (brak dokładnych timestampów)
                            processing_times.append(1000)  # Placeholder
                    
                    if processing_times:
                        avg_processing_time = statistics.mean(processing_times)
                        details["avg_processing_time_ms"] = avg_processing_time
                        
                        if avg_processing_time > 5000:  # 5 sekund
                            recommendations.append(f"Slow alert processing: {avg_processing_time:.0f}ms average")
                
                # Sprawdź tier distribution
                tier_counts = {}
                for alert in recent_alerts:
                    tier = alert.tier or "Unknown"
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                
                details["tier_distribution"] = tier_counts
            
            execution_time = int((time.time() - start_time) * 1000)
            details["response_time_ms"] = execution_time
            
            # Określ status
            if details.get("success_rate", 1.0) < 0.8:
                status = HealthStatus.WARNING
                message = f"Low webhook success rate: {details.get('success_rate', 0):.1%}"
            elif recommendations:
                status = HealthStatus.WARNING
                message = "Webhook operational with warnings"
            else:
                status = HealthStatus.HEALTHY
                message = "Webhook operating normally"
            
            return DiagnosticResult(
                component="webhook",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return DiagnosticResult(
                component="webhook",
                status=HealthStatus.CRITICAL,
                message=f"Webhook check failed: {str(e)}",
                details={"error": str(e), "response_time_ms": execution_time},
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=["Check webhook configuration", "Verify database connectivity"]
            )
    
    async def _check_bot_core_health(self) -> DiagnosticResult:
        """Sprawdź stan zdrowia głównego bota"""
        start_time = time.time()
        
        try:
            details = {}
            recommendations = []
            
            # Import bot instance
            try:
                from main import bot_instance
                bot = bot_instance
                
                if bot:
                    details["bot_available"] = True
                    details["bot_running"] = getattr(bot, "running", False)
                    details["bot_paused"] = getattr(bot, "paused", False)
                    details["emergency_mode"] = getattr(bot, "emergency_mode", False)
                    
                    # Sprawdź aktywne pozycje
                    if hasattr(bot, "active_positions"):
                        details["active_positions_count"] = len(bot.active_positions)
                        
                        # Sprawdź ryzyko
                        total_risk = 0.0
                        for pos in bot.active_positions.values():
                            if hasattr(pos, "risk_amount"):
                                total_risk += pos.risk_amount
                        
                        details["total_risk_exposure"] = total_risk
                        max_risk = getattr(Config, "MAX_TOTAL_RISK", 1000)  # Default
                        
                        if total_risk > max_risk * 0.8:
                            recommendations.append(f"High risk exposure: ${total_risk:.2f}")
                    
                    # Sprawdź performance metrics
                    if hasattr(bot, "performance_metrics"):
                        perf = bot.performance_metrics
                        details["performance_metrics"] = perf
                        
                        # Sprawdź win rate
                        if "win_rate" in perf and perf["win_rate"] < 0.5:
                            recommendations.append(f"Low win rate: {perf['win_rate']:.1%}")
                        
                        # Sprawdź rejection rate
                        total_signals = perf.get("total_signals", 0)
                        rejected_signals = perf.get("signals_rejected", 0)
                        if total_signals > 0:
                            rejection_rate = rejected_signals / total_signals
                            details["signal_rejection_rate"] = rejection_rate
                            
                            if rejection_rate > 0.7:  # 70% odrzuconych
                                recommendations.append(f"High signal rejection rate: {rejection_rate:.1%}")
                    
                    # Sprawdź ostatnią aktywność
                    if hasattr(bot, "last_signal_time"):
                        last_signal = bot.last_signal_time
                        if last_signal:
                            time_since_signal = (datetime.utcnow() - last_signal).total_seconds()
                            details["seconds_since_last_signal"] = int(time_since_signal)
                            
                            if time_since_signal > 3600:  # 1 godzina
                                recommendations.append("No signals received in the last hour")
                    
                    # Sprawdź mode manager
                    if hasattr(bot, "mode_manager"):
                        mode_manager = bot.mode_manager
                        details["current_mode"] = getattr(mode_manager, "current_mode", "unknown")
                        details["mode_changes_today"] = getattr(mode_manager, "mode_changes_today", 0)
                        
                        if details["mode_changes_today"] > 10:
                            recommendations.append("Frequent mode changes detected")
                    
                    # Status ogólny
                    if not details["bot_running"]:
                        status = HealthStatus.CRITICAL
                        message = "Bot is not running"
                    elif details["bot_paused"]:
                        status = HealthStatus.WARNING
                        message = "Bot is paused"
                    elif details["emergency_mode"]:
                        status = HealthStatus.WARNING
                        message = "Bot in emergency mode"
                    elif recommendations:
                        status = HealthStatus.WARNING
                        message = "Bot operational with warnings"
                    else:
                        status = HealthStatus.HEALTHY
                        message = "Bot operating normally"
                        
                else:
                    details["bot_available"] = False
                    status = HealthStatus.CRITICAL
                    message = "Bot instance not available"
                    recommendations.append("Bot instance not initialized")
                    
            except Exception as e:
                details["bot_check_error"] = str(e)
                status = HealthStatus.CRITICAL
                message = "Cannot access bot instance"
                recommendations.append("Bot initialization or import issues")
            
            execution_time = int((time.time() - start_time) * 1000)
            details["response_time_ms"] = execution_time
            
            return DiagnosticResult(
                component="bot_core",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return DiagnosticResult(
                component="bot_core",
                status=HealthStatus.CRITICAL,
                message=f"Bot core check failed: {str(e)}",
                details={"error": str(e), "response_time_ms": execution_time},
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=["Check bot initialization", "Verify main.py configuration"]
            )
    
    async def _check_pine_script_health(self) -> DiagnosticResult:
        """Sprawdź stan zdrowia Pine Script integration"""
        start_time = time.time()
        
        try:
            details = {}
            recommendations = []
            
            # Sprawdź ostatnie alerty z Pine Script
            with Session() as session:
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                recent_alerts = session.query(AlertHistory).filter(
                    AlertHistory.received_at >= recent_cutoff
                ).all()
                
                details["total_alerts_24h"] = len(recent_alerts)
                
                if recent_alerts:
                    # Sprawdź wersje wskaźnika
                    version_counts = {}
                    for alert in recent_alerts:
                        if alert.raw_payload and isinstance(alert.raw_payload, dict):
                            version = alert.raw_payload.get("indicator_version", "unknown")
                            version_counts[version] = version_counts.get(version, 0) + 1
                    
                    details["indicator_versions"] = version_counts
                    
                    # Sprawdź czy są alerty v9.1
                    v91_alerts = version_counts.get("9.1", 0)
                    if v91_alerts == 0 and len(recent_alerts) > 0:
                        recommendations.append("No v9.1 indicator alerts detected")
                    
                    # Sprawdź tier distribution
                    tier_counts = {}
                    for alert in recent_alerts:
                        tier = alert.tier or "Unknown"
                        tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    
                    details["tier_distribution"] = tier_counts
                    
                    # Sprawdź emergency alerts
                    emergency_count = 0
                    for alert in recent_alerts:
                        if alert.raw_payload and isinstance(alert.raw_payload, dict):
                            action = alert.raw_payload.get("action", "")
                            if "emergency" in action.lower():
                                emergency_count += 1
                    
                    details["emergency_alerts_24h"] = emergency_count
                    
                    if emergency_count > 10:
                        recommendations.append(f"High number of emergency alerts: {emergency_count}")
                    
                    # Sprawdź opóźnienia alertów
                    latency_issues = 0
                    for alert in recent_alerts:
                        if alert.raw_payload and isinstance(alert.raw_payload, dict):
                            tv_ts = alert.raw_payload.get("tv_ts")
                            if tv_ts and alert.received_at:
                                received_ts = int(alert.received_at.timestamp() * 1000)
                                latency = received_ts - tv_ts
                                
                                if latency > 2000:  # 2 sekundy
                                    latency_issues += 1
                    
                    details["high_latency_alerts"] = latency_issues
                    
                    if latency_issues > len(recent_alerts) * 0.1:  # 10% alertów
                        recommendations.append(f"High latency detected in {latency_issues} alerts")
                    
                    # Sprawdź fake breakout detections
                    fake_breakout_count = 0
                    for alert in recent_alerts:
                        if alert.raw_payload and isinstance(alert.raw_payload, dict):
                            v91_data = alert.raw_payload.get("v91_enhancements", {})
                            if isinstance(v91_data, dict):
                                fake_breakout = v91_data.get("fake_breakout", {})
                                if isinstance(fake_breakout, dict) and fake_breakout.get("detected"):
                                    fake_breakout_count += 1
                    
                    details["fake_breakout_detections_24h"] = fake_breakout_count
                    
                else:
                    recommendations.append("No Pine Script alerts received in 24 hours")
            
            execution_time = int((time.time() - start_time) * 1000)
            details["response_time_ms"] = execution_time
            
            # Określ status
            if details["total_alerts_24h"] == 0:
                status = HealthStatus.WARNING
                message = "No Pine Script alerts in 24 hours"
            elif len(recommendations) > 1:
                status = HealthStatus.WARNING
                message = "Pine Script integration has multiple issues"
            elif recommendations:
                status = HealthStatus.WARNING
                message = "Pine Script integration operational with warnings"
            else:
                status = HealthStatus.HEALTHY
                message = "Pine Script integration operating normally"
            
            return DiagnosticResult(
                component="pine_script",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return DiagnosticResult(
                component="pine_script",
                status=HealthStatus.CRITICAL,
                message=f"Pine Script check failed: {str(e)}",
                details={"error": str(e), "response_time_ms": execution_time},
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                recommendations=["Check Pine Script configuration", "Verify TradingView alerts"]
            )
    
    async def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Zbierz metryki systemowe"""
        try:
            import psutil
            
            # CPU i pamięć
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network (podstawowe)
            network_latency = 0.0  # Placeholder - można dodać ping test
            
            # Database response time
            db_start = time.time()
            with Session() as session:
                session.execute("SELECT 1")
            database_response_time = (time.time() - db_start) * 1000
            
            # API response time (placeholder)
            api_response_time = 0.0
            
            # Active connections (placeholder)
            active_connections = 0
            
            # Error rate (z ostatniej godziny)
            error_rate = 0.0
            with Session() as session:
                recent_cutoff = datetime.utcnow() - timedelta(hours=1)
                total_alerts = session.query(AlertHistory).filter(
                    AlertHistory.received_at >= recent_cutoff
                ).count()
                
                error_alerts = session.query(AlertHistory).filter(
                    AlertHistory.received_at >= recent_cutoff,
                    AlertHistory.error.is_not(None)
                ).count()
                
                if total_alerts > 0:
                    error_rate = (error_alerts / total_alerts) * 100
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_latency=network_latency,
                database_response_time=database_response_time,
                api_response_time=api_response_time,
                active_connections=active_connections,
                error_rate=error_rate
            )
            
        except ImportError:
            self.logger.warning("psutil not available - system metrics disabled")
            return None
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    async def _collect_trading_metrics(self) -> Optional[TradingMetrics]:
        """Zbierz metryki tradingowe"""
        try:
            with Session() as session:
                # Podstawowe statystyki
                total_signals = 0
                signals_taken = 0
                signals_rejected = 0
                ml_rejections = 0
                fake_breakout_detections = 0
                
                # Sprawdź czy bot ma performance metrics
                try:
                    from main import bot_instance
                    if bot_instance and hasattr(bot_instance, "performance_metrics"):
                        perf = bot_instance.performance_metrics
                        total_signals = perf.get("total_signals", 0)
                        signals_taken = perf.get("signals_taken", 0)
                        signals_rejected = perf.get("signals_rejected", 0)
                        ml_rejections = perf.get("ml_rejections", 0)
                        fake_breakout_detections = perf.get("fake_breakout_detections", 0)
                except:
                    pass
                
                # Aktywne pozycje
                active_positions = len(get_open_positions(session))
                
                # Daily metrics
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                
                daily_trades = session.query(Trade).filter(
                    Trade.entry_time >= today_start,
                    Trade.status == "closed"
                ).count()
                
                # Daily PnL
                daily_pnl_result = session.query(Trade.pnl_usdt).filter(
                    Trade.entry_time >= today_start,
                    Trade.status == "closed",
                    Trade.pnl_usdt.is_not(None)
                ).all()
                
                daily_pnl = sum(pnl[0] for pnl in daily_pnl_result if pnl[0] is not None)
                
                # Win rate (ostatnie 100 transakcji)
                recent_trades = session.query(Trade).filter(
                    Trade.status == "closed",
                    Trade.pnl_usdt.is_not(None)
                ).order_by(Trade.entry_time.desc()).limit(100).all()
                
                win_rate = 0.0
                avg_trade_duration = 0.0
                
                if recent_trades:
                    winning_trades = len([t for t in recent_trades if t.pnl_usdt > 0])
                    win_rate = winning_trades / len(recent_trades)
                    
                    # Średni czas trwania transakcji
                    durations = []
                    for trade in recent_trades:
                        if trade.entry_time and trade.exit_time:
                            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # godziny
                            durations.append(duration)
                    
                    if durations:
                        avg_trade_duration = statistics.mean(durations)
                
                # Risk exposure
                risk_exposure = 0.0
                open_positions = get_open_positions(session)
                for pos in open_positions:
                    if pos.entry_price and pos.stop_loss:
                        risk_per_position = abs(pos.entry_price - pos.stop_loss) * pos.quantity
                        risk_exposure += risk_per_position
                
                return TradingMetrics(
                    total_signals=total_signals,
                    signals_taken=signals_taken,
                    signals_rejected=signals_rejected,
                    ml_rejections=ml_rejections,
                    fake_breakout_detections=fake_breakout_detections,
                    active_positions=active_positions,
                    daily_pnl=daily_pnl,
                    daily_trades=daily_trades,
                    win_rate=win_rate,
                    avg_trade_duration=avg_trade_duration,
                    risk_exposure=risk_exposure
                )
                
        except Exception as e:
            self.logger.error(f"Failed to collect trading metrics: {e}")
            return None
    
    def _calculate_overall_health(self, results: List[DiagnosticResult]) -> HealthStatus:
        """Oblicz ogólny stan zdrowia systemu"""
        if not results:
            return HealthStatus.UNKNOWN
        
        # Sprawdź czy są komponenty krytyczne
        critical_components = [r for r in results if r.status == HealthStatus.CRITICAL]
        if critical_components:
            # Sprawdź czy to są krytyczne komponenty
            critical_names = [r.component for r in critical_components]
            essential_components = ["database", "binance_api", "bot_core"]
            
            if any(comp in critical_names for comp in essential_components):
                return HealthStatus.CRITICAL
        
        # Oblicz ważoną ocenę
        total_weight = 0.0
        weighted_score = 0.0
        
        for result in results:
            component_type = None
            for comp_type in ComponentType:
                if comp_type.value == result.component:
                    component_type = comp_type
                    break
            
            if component_type:
                weight = self.component_weights.get(component_type, 0.1)
                total_weight += weight
                
                if result.status == HealthStatus.HEALTHY:
                    weighted_score += weight * 1.0
                elif result.status == HealthStatus.WARNING:
                    weighted_score += weight * 0.5
                elif result.status == HealthStatus.CRITICAL:
                    weighted_score += weight * 0.0
        
        if total_weight == 0:
            return HealthStatus.UNKNOWN
        
        overall_score = weighted_score / total_weight
        
        if overall_score >= 0.8:
            return HealthStatus.HEALTHY
        elif overall_score >= 0.5:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL
    
    def _calculate_health_score(self, results: List[DiagnosticResult]) -> float:
        """Oblicz numeryczną ocenę zdrowia (0.0 - 1.0)"""
        if not results:
            return 0.0
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for result in results:
            component_type = None
            for comp_type in ComponentType:
                if comp_type.value == result.component:
                    component_type = comp_type
                    break
            
            if component_type:
                weight = self.component_weights.get(component_type, 0.1)
                total_weight += weight
                
                if result.status == HealthStatus.HEALTHY:
                    weighted_score += weight * 1.0
                elif result.status == HealthStatus.WARNING:
                    weighted_score += weight * 0.6
                elif result.status == HealthStatus.CRITICAL:
                    weighted_score += weight * 0.0
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_system_recommendations(
        self, 
        results: List[DiagnosticResult], 
        system_metrics: Optional[SystemMetrics], 
        trading_metrics: Optional[TradingMetrics]
    ) -> List[str]:
        """Generuj rekomendacje systemowe"""
        recommendations = []
        
        # Zbierz wszystkie rekomendacje z komponentów
        for result in results:
            recommendations.extend(result.recommendations)
        
        # Dodaj rekomendacje systemowe
        if system_metrics:
            if system_metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
                recommendations.append(f"High CPU usage: {system_metrics.cpu_usage:.1f}% - consider optimization")
            
            if system_metrics.memory_usage > self.alert_thresholds["memory_usage"]:
                recommendations.append(f"High memory usage: {system_metrics.memory_usage:.1f}% - check for memory leaks")
            
            if system_metrics.disk_usage > self.alert_thresholds["disk_usage"]:
                recommendations.append(f"High disk usage: {system_metrics.disk_usage:.1f}% - clean up old data")
            
            if system_metrics.error_rate > self.alert_thresholds["error_rate"]:
                recommendations.append(f"High error rate: {system_metrics.error_rate:.1f}% - investigate error causes")
        
        # Dodaj rekomendacje tradingowe
        if trading_metrics:
            if trading_metrics.win_rate < 0.5:
                recommendations.append(f"Low win rate: {trading_metrics.win_rate:.1%} - review trading strategy")
            
            if trading_metrics.active_positions > 10:
                recommendations.append(f"High number of active positions: {trading_metrics.active_positions} - monitor risk")
            
            if trading_metrics.daily_pnl < -100:
                recommendations.append(f"Negative daily PnL: ${trading_metrics.daily_pnl:.2f} - review recent trades")
        
        # Usuń duplikaty i ogranicz do top 10
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:10]
    
    async def _save_diagnostic_report(self, report: Dict[str, Any]) -> None:
        """Zapisz raport diagnostyczny do bazy danych"""
        try:
            with Session() as session:
                health_record = SystemHealth(
                    timestamp=datetime.utcnow(),
                    overall_status=report["overall_health"]["status"],
                    health_score=report["overall_health"]["score"],
                    component_results=report["component_results"],
                    system_metrics=report.get("system_metrics"),
                    trading_metrics=report.get("trading_metrics"),
                    recommendations=report["recommendations"],
                    execution_time_ms=report["execution_summary"]["execution_time_ms"]
                )
                
                session.add(health_record)
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save diagnostic report: {e}")
    
    async def get_diagnostic_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Pobierz historię diagnostyk"""
        try:
            with Session() as session:
                records = session.query(SystemHealth).order_by(
                    SystemHealth.timestamp.desc()
                ).limit(limit).all()
                
                return [
                    {
                        "timestamp": record.timestamp.isoformat(),
                        "overall_status": record.overall_status,
                        "health_score": record.health_score,
                        "execution_time_ms": record.execution_time_ms,
                        "recommendations_count": len(record.recommendations) if record.recommendations else 0
                    }
                    for record in records
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get diagnostic history: {e}")
            return []
    
    async def get_component_health_trend(self, component: str, hours: int = 24) -> Dict[str, Any]:
        """Pobierz trend zdrowia komponentu"""
        try:
            with Session() as session:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                records = session.query(SystemHealth).filter(
                    SystemHealth.timestamp >= cutoff_time
                ).order_by(SystemHealth.timestamp.asc()).all()
                
                component_data = []
                for record in records:
                    if record.component_results:
                        for comp_result in record.component_results:
                            if comp_result.get("component") == component:
                                component_data.append({
                                    "timestamp": record.timestamp.isoformat(),
                                    "status": comp_result.get("status"),
                                    "execution_time_ms": comp_result.get("execution_time_ms", 0)
                                })
                
                # Oblicz statystyki
                if component_data:
                    statuses = [d["status"] for d in component_data]
                    healthy_count = statuses.count("healthy")
                    warning_count = statuses.count("warning")
                    critical_count = statuses.count("critical")
                    
                    avg_response_time = statistics.mean([
                        d["execution_time_ms"] for d in component_data 
                        if d["execution_time_ms"] > 0
                    ]) if component_data else 0
                    
                    return {
                        "component": component,
                        "time_period_hours": hours,
                        "total_checks": len(component_data),
                        "healthy_count": healthy_count,
                        "warning_count": warning_count,
                        "critical_count": critical_count,
                        "health_percentage": (healthy_count / len(component_data)) * 100,
                        "avg_response_time_ms": avg_response_time,
                        "trend_data": component_data
                    }
                else:
                    return {
                        "component": component,
                        "time_period_hours": hours,
                        "total_checks": 0,
                        "message": "No data available for this component"
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get component health trend: {e}")
            return {"error": str(e)}
    
    async def run_quick_health_check(self) -> Dict[str, Any]:
        """Szybki check zdrowia - tylko podstawowe komponenty"""
        start_time = time.time()
        
        try:
            # Tylko najważniejsze testy
            quick_tasks = [
                self._check_database_health(),
                self._check_binance_api_health(),
                self._check_bot_core_health()
            ]
            
            results = await asyncio.gather(*quick_tasks, return_exceptions=True)
            
            # Przetwórz wyniki
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    component_name = ["database", "binance_api", "bot_core"][i]
                    processed_results.append(DiagnosticResult(
                        component=component_name,
                        status=HealthStatus.CRITICAL,
                        message=f"Quick check failed: {str(result)}",
                        details={"error": str(result)},
                        timestamp=datetime.utcnow(),
                        execution_time_ms=0,
                        recommendations=[f"Investigate {component_name} issues"]
                    ))
                else:
                    processed_results.append(result)
            
            overall_health = self._calculate_overall_health(processed_results)
            execution_time = int((time.time() - start_time) * 1000)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "check_type": "quick",
                "overall_health": overall_health.value,
                "health_score": self._calculate_health_score(processed_results),
                "execution_time_ms": execution_time,
                "components_checked": len(processed_results),
                "critical_issues": len([r for r in processed_results if r.status == HealthStatus.CRITICAL]),
                "warnings": len([r for r in processed_results if r.status == HealthStatus.WARNING]),
                "component_results": [asdict(result) for result in processed_results]
            }
            
        except Exception as e:
            self.logger.error(f"Quick health check failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "check_type": "quick",
                "overall_health": "critical",
                "error": str(e),
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }

    async def process_pine_diagnostics(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Przetwarzanie danych diagnostycznych z Pine Script v9.1"""
        try:
            diagnostics = alert_data.get('diagnostics', {})
            if not diagnostics:
                return {}
        
            # Analiza health score
            health_score = diagnostics.get('health_score', 0.5)
            pattern_count = diagnostics.get('pattern_count', 0)
            anomalies = diagnostics.get('anomalies', [])
        
            # Zapisz do bazy
            with Session() as session:
                # Zapisz Pine Health Log
                pine_health_data = {
                    'symbol': alert_data.get('symbol'),
                    'timeframe': alert_data.get('timeframe', '1h'),
                    'overall_health': health_score,
                    'parameter_stability': diagnostics.get('parameter_stability', 0.5),
                    'trend_clarity': diagnostics.get('trend_clarity', 0.5),
                    'volume_quality': diagnostics.get('volume_quality', 0.5),
                    'atr_health': diagnostics.get('atr_health', 0.5),
                    'atr_value': alert_data.get('atr'),
                    'atr_percentile': diagnostics.get('atr_percentile'),
                    'adx_health': diagnostics.get('adx_health', 0.5),
                    'adx_value': diagnostics.get('adx'),
                    'adx_trend_strength': diagnostics.get('adx_trend_strength', 'moderate'),
                    'regime_detected': alert_data.get('regime', 'NEUTRAL'),
                    'regime_confidence': alert_data.get('regime_confidence', 0.0),
                    'regime_stability': diagnostics.get('regime_stability', 0.5),
                    'volume_profile': diagnostics.get('volume_profile', 'fair'),
                    'volume_anomaly': diagnostics.get('volume_anomaly', False),
                    'institutional_flow': alert_data.get('institutional_flow', 0.0),
                    'warnings': diagnostics.get('warnings', []),
                    'critical_issues': diagnostics.get('critical_issues', []),
                    'correlation_with_performance': diagnostics.get('correlation', 0.0)
                }
                log_pine_health(session, pine_health_data)
            
                # Jeśli są anomalie, utwórz pattern alert
                if anomalies and len(anomalies) > 0:
                    pattern_alert_data = {
                        'pattern_type': 'pine_script_anomaly',
                        'pattern_severity': 'high' if health_score < 0.3 else 'medium',
                        'pattern_description': f"Pine Script detected {len(anomalies)} anomalies",
                        'symbols_affected': [alert_data.get('symbol')],
                        'tiers_affected': ['ALL'],
                        'timeframes_affected': [alert_data.get('timeframe', '1h')],
                        'frequency_count': len(anomalies),
                        'impact_score': 1.0 - health_score,
                        'confidence_score': 0.8,
                        'recommended_action': 'Review Pine Script parameters and market conditions',
                        'auto_fix_available': False
                    }
                    create_pattern_alert(session, pattern_alert_data)
        
            return {
                'health_score': health_score,
                'patterns_detected': pattern_count,
                'anomalies': anomalies,
                'requires_attention': health_score < 0.3
            }
        
        except Exception as e:
            self.logger.error(f"Error processing Pine diagnostics: {e}")
            return {}

    def get_summary(self) -> Dict[str, Any]:
        """Zwraca podsumowanie dla ulepszonej komendy /diagnostics"""
        try:
            # Pobierz ostatni raport
            if self.diagnostic_history:
                last_report = self.diagnostic_history[-1]
                overall_health = last_report.get('overall_health', {})
            
                return {
                    'status': overall_health.get('status', 'UNKNOWN'),
                    'regime': 'TRENDING',  # Możesz pobrać z bota
                    'confidence': 0.8,
                    'signals': {
                        'buy': 0.6,
                        'sell': 0.4
                    },
                    'zones': 3,
                    'alerts': []
                }
            return {
                'status': 'UNKNOWN',
                'regime': 'NEUTRAL',
                'confidence': 0.5,
                'signals': {'buy': 0.5, 'sell': 0.5},
                'zones': 0,
                'alerts': []
            }
        except Exception as e:
            self.logger.error(f"Error getting summary: {e}")
            return {'status': 'ERROR'}

# --- ALIASY I METODY DODANE DLA KOMPATYBILNOŚCI Z DISCORD_CLIENT ---

    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Alias dla run_full_diagnostics() dla komendy /diagnostics."""
        self.logger.info("Wywołano alias: generate_comprehensive_report -> run_full_diagnostics")
        return await self.run_full_diagnostics()

    async def quick_health_check(self) -> Dict[str, Any]:
        """Alias dla run_quick_health_check() dla komendy /health."""
        self.logger.info("Wywołano alias: quick_health_check -> run_quick_health_check")
        return await self.run_quick_health_check()

    async def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Zbiera i zwraca metryki wydajnościowe dla komendy /performance.
        Używa istniejących prywatnych metod do zbierania danych.
        """
        self.logger.info(f"Zbieranie metryk wydajnościowych z ostatnich {hours} godzin.")
        try:
            # Na razie ignorujemy parametr 'hours' i zbieramy aktualne metryki
            # W przyszłości można rozbudować logikę o dane historyczne
            trading_metrics = await self._collect_trading_metrics()
            system_metrics = await self._collect_system_metrics()
            
            return {
                "trading_metrics": asdict(trading_metrics) if trading_metrics else {},
                "system_metrics": asdict(system_metrics) if system_metrics else {},
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas zbierania metryk wydajności: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

# Global diagnostics instance
diagnostics_engine = None  # Będzie utworzony w main.py


# Convenience functions
async def run_full_diagnostics() -> Dict[str, Any]:
    """Uruchom pełną diagnostykę systemu"""
    return await diagnostics_engine.run_full_diagnostics()


async def run_quick_health_check() -> Dict[str, Any]:
    """Uruchom szybki check zdrowia"""
    return await diagnostics_engine.run_quick_health_check()


async def get_component_health(component: str, hours: int = 24) -> Dict[str, Any]:
    """Pobierz trend zdrowia komponentu"""
    return await diagnostics_engine.get_component_health_trend(component, hours)


async def get_diagnostic_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Pobierz historię diagnostyk"""
    return await diagnostics_engine.get_diagnostic_history(limit)


if __name__ == "__main__":
    # Test diagnostics
    async def test_diagnostics():
        print("🔍 Running test diagnostics...")
        
        # Quick check
        quick_result = await run_quick_health_check()
        print(f"Quick check: {quick_result['overall_health']} ({quick_result['execution_time_ms']}ms)")
        
        # Full diagnostics
        full_result = await run_full_diagnostics()
        print(f"Full diagnostics: {full_result['overall_health']['status']} (Score: {full_result['overall_health']['score']:.2f})")
        
        print("✅ Test diagnostics completed!")

    # Run the test only when script is executed directly
    asyncio.run(test_diagnostics())
        
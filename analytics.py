"""
Advanced Analytics Engine for Trading Bot v9.1
Comprehensive performance analysis, ML insights, and predictive analytics
Enhanced with v9.1 features: Signal Intelligence Analytics, ML Performance Tracking, Advanced Metrics
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import dataclass
from collections import defaultdict
import asyncio
import psutil
import time
from config import Config
from database import Session, Position, Trade, get_performance_stats
try:
    from diagnostics import DiagnosticsEngine
    from decision_engine import DecisionEngine
    from pattern_detector import PatternDetector
    from pine_health_monitor import PineScriptHealthMonitor
    DIAGNOSTICS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Diagnostics modules not available: {e}")
    DIAGNOSTICS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics structure for v9.1"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # v9.1 NEW: Enhanced metrics
    fake_breakout_accuracy: float = 0.0
    institutional_flow_correlation: float = 0.0
    ml_prediction_accuracy: float = 0.0
    regime_accuracy: float = 0.0
    mtf_agreement_effectiveness: float = 0.0
    tier_performance: Dict[str, float] = None
    mode_performance: Dict[str, float] = None
    timeframe_performance: Dict[str, float] = None

    # v9.1 NEW: Enhanced metrics
    fake_breakout_accuracy: float = 0.0
    institutional_flow_correlation: float = 0.0
    ml_prediction_accuracy: float = 0.0
    regime_accuracy: float = 0.0
    mtf_agreement_effectiveness: float = 0.0
    tier_performance: Dict[str, float] = None
    mode_performance: Dict[str, float] = None
    timeframe_performance: Dict[str, float] = None

    # HYBRID ULTRA-DIAGNOSTICS: New diagnostic metrics
    system_health_score: float = 0.0
    decision_confidence_avg: float = 0.0
    pattern_detection_accuracy: float = 0.0
    anomaly_detection_rate: float = 0.0
    execution_efficiency: float = 0.0
    resource_utilization: Dict[str, float] = None
    diagnostic_alerts_count: int = 0
    model_drift_score: float = 0.0

    def __post_init__(self):
        if self.tier_performance is None:
            self.tier_performance = {}
        if self.mode_performance is None:
            self.mode_performance = {}
        if self.timeframe_performance is None:
            self.timeframe_performance = {}
        if self.resource_utilization is None:
            self.resource_utilization = {}


class AnalyticsEngine:
    """Advanced analytics engine with v9.1 enhancements and diagnostics integration"""

    def __init__(self):
        """Initialize analytics engine with v9.1 features and diagnostics"""
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_analysis = None

        # v9.1 Enhanced tracking
        self.signal_intelligence_stats = defaultdict(list)
        self.ml_performance_history = []
        self.regime_tracking = defaultdict(list)
        self.institutional_flow_tracking = []

        # HYBRID ULTRA-DIAGNOSTICS: Initialize diagnostic components
        if DIAGNOSTICS_AVAILABLE and Config.DIAGNOSTICS_ENABLED:
            try:
                self.diagnostics_manager = DiagnosticsEngine()
                self.decision_engine = DecisionEngine()
                self.pattern_detector = PatternDetector()
                self.pine_health_monitor = PineScriptHealthMonitor()
                self.diagnostics_enabled = True
                logger.info("🔬 Analytics Engine v9.1 initialized with Hybrid Ultra-Diagnostics")
            except Exception as e:
                logger.error(f"Failed to initialize diagnostics: {e}")
                self.diagnostics_enabled = False
        else:
            self.diagnostics_enabled = False
            logger.info("🔬 Analytics Engine v9.1 initialized (diagnostics disabled)")

        # Performance tracking
        self.performance_start_time = time.time()
        self.analytics_call_count = 0
        self.last_system_check = time.time()
    def _calculate_diagnostic_metrics(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate diagnostic-specific metrics"""
        if not self.diagnostics_enabled:
            return {}

        try:
            # System health score
            system_health = self._calculate_system_health_score()
            
            # Decision confidence analysis
            decision_confidence = self._analyze_decision_confidence(positions)
            
            # Pattern detection accuracy
            pattern_accuracy = self._calculate_pattern_detection_accuracy(positions)
            
            # Anomaly detection rate
            anomaly_rate = self._calculate_anomaly_detection_rate(positions)
            
            # Execution efficiency
            execution_efficiency = self._calculate_execution_efficiency(positions)
            
            # Resource utilization
            resource_utilization = self._get_resource_utilization()
            
            # Model drift score
            model_drift = self._calculate_model_drift_score()

            return {
                "system_health_score": system_health,
                "decision_confidence_avg": decision_confidence,
                "pattern_detection_accuracy": pattern_accuracy,
                "anomaly_detection_rate": anomaly_rate,
                "execution_efficiency": execution_efficiency,
                "resource_utilization": resource_utilization,
                "model_drift_score": model_drift
            }

        except Exception as e:
            logger.error(f"Error calculating diagnostic metrics: {e}")
            return {}

    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        if not self.diagnostics_enabled:
            return 100.0

        try:
            health_factors = []
            
            # CPU usage (inverted - lower is better)
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_score = max(0, 100 - cpu_percent)
            health_factors.append(cpu_score)
            
            # Memory usage (inverted - lower is better)
            memory = psutil.virtual_memory()
            memory_score = max(0, 100 - memory.percent)
            health_factors.append(memory_score)
            
            # Disk usage (inverted - lower is better)
            disk = psutil.disk_usage('/')
            disk_score = max(0, 100 - (disk.used / disk.total * 100))
            health_factors.append(disk_score)
            
            # Database connection health
            db_health = self._check_database_health()
            health_factors.append(db_health)
            
            # API response times
            api_health = self._check_api_health()
            health_factors.append(api_health)

            return np.mean(health_factors)

        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return 50.0  # Default moderate health

    def _analyze_decision_confidence(self, positions: List[Position]) -> float:
        """Analyze average decision confidence from decision engine"""
        if not self.diagnostics_enabled:
            return 0.0

        try:
            confidences = []
            for position in positions:
                if hasattr(position, 'decision_confidence') and position.decision_confidence:
                    confidences.append(position.decision_confidence)
            
            return np.mean(confidences) if confidences else 0.0

        except Exception as e:
            logger.error(f"Error analyzing decision confidence: {e}")
            return 0.0

    def _calculate_pattern_detection_accuracy(self, positions: List[Position]) -> float:
        """Calculate pattern detection accuracy"""
        if not self.diagnostics_enabled:
            return 0.0

        try:
            pattern_positions = [p for p in positions if hasattr(p, 'pattern_detected') and p.pattern_detected]
            
            if not pattern_positions:
                return 0.0
            
            correct_patterns = len([p for p in pattern_positions if p.pnl and p.pnl > 0])
            return (correct_patterns / len(pattern_positions)) * 100

        except Exception as e:
            logger.error(f"Error calculating pattern detection accuracy: {e}")
            return 0.0

    def _calculate_anomaly_detection_rate(self, positions: List[Position]) -> float:
        """Calculate anomaly detection rate"""
        if not self.diagnostics_enabled:
            return 0.0

        try:
            total_positions = len(positions)
            anomaly_positions = len([p for p in positions if hasattr(p, 'anomaly_detected') and p.anomaly_detected])
            
            return (anomaly_positions / total_positions * 100) if total_positions > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating anomaly detection rate: {e}")
            return 0.0

    def _calculate_execution_efficiency(self, positions: List[Position]) -> float:
        """Calculate execution efficiency score"""
        if not self.diagnostics_enabled:
            return 100.0

        try:
            execution_times = []
            for position in positions:
                if hasattr(position, 'execution_time') and position.execution_time:
                    execution_times.append(position.execution_time)
            
            if not execution_times:
                return 100.0
            
            avg_execution_time = np.mean(execution_times)
            # Convert to efficiency score (lower time = higher efficiency)
            # Assume 1 second is perfect (100%), scale accordingly
            efficiency = max(0, 100 - (avg_execution_time - 1) * 10)
            return min(100, efficiency)

        except Exception as e:
            logger.error(f"Error calculating execution efficiency: {e}")
            return 100.0

    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent if psutil.disk_usage('/') else 0,
                "network_io": self._get_network_io_rate()
            }
        except Exception as e:
            logger.error(f"Error getting resource utilization: {e}")
            return {"cpu_percent": 0, "memory_percent": 0, "disk_percent": 0, "network_io": 0}

    def _get_network_io_rate(self) -> float:
        """Get network I/O rate"""
        try:
            net_io = psutil.net_io_counters()
            return (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
        except:
            return 0.0

    def _calculate_model_drift_score(self) -> float:
        """Calculate ML model drift score"""
        if not self.diagnostics_enabled:
            return 0.0

        try:
            # This would require historical model performance data
            # For now, return a placeholder
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating model drift: {e}")
            return 0.0

    def _check_database_health(self) -> float:
        """Check database connection health"""
        try:
            with Session() as session:
                start_time = time.time()
                session.execute("SELECT 1")
                response_time = time.time() - start_time
                
                # Convert response time to health score
                if response_time < 0.1:
                    return 100.0
                elif response_time < 0.5:
                    return 80.0
                elif response_time < 1.0:
                    return 60.0
                else:
                    return 30.0
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return 0.0

    def _check_api_health(self) -> float:
        """Check API health"""
        try:
            # This would require actual API health checks
            # For now, return a default good health score
            return 90.0
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return 50.0

    def generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        if not self.diagnostics_enabled:
            return {"diagnostics_enabled": False, "message": "Diagnostics not available"}

        try:
            # System diagnostics
            system_diagnostics = self.diagnostics_manager.run_system_diagnostics()
            
            # Performance diagnostics
            performance_diagnostics = self.diagnostics_manager.analyze_performance_metrics()
            
            # Pattern analysis
            pattern_analysis = self.pattern_detector.analyze_recent_patterns()
            
            # Pine Script health
            pine_health = self.pine_health_monitor.get_health_status()
            
            # Decision engine insights
            decision_insights = self.decision_engine.get_recent_decisions_analysis()

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "diagnostics_enabled": True,
                "system_diagnostics": system_diagnostics,
                "performance_diagnostics": performance_diagnostics,
                "pattern_analysis": pattern_analysis,
                "pine_health": pine_health,
                "decision_insights": decision_insights,
                "recommendations": self._generate_diagnostic_recommendations(
                    system_diagnostics, performance_diagnostics
                )
            }

        except Exception as e:
            logger.error(f"Error generating diagnostic report: {e}")
            return {"diagnostics_enabled": True, "error": str(e)}

    def _generate_diagnostic_recommendations(self, system_diag: Dict, perf_diag: Dict) -> List[str]:
        """Generate diagnostic-based recommendations"""
        recommendations = []

        try:
            # System health recommendations
            if system_diag.get("health_score", 100) < 70:
                recommendations.append("🔴 System health is degraded. Check resource usage and optimize.")

            # Performance recommendations
            if perf_diag.get("execution_efficiency", 100) < 80:
                recommendations.append("⚡ Execution efficiency is low. Review order processing pipeline.")

            # Pattern detection recommendations
            if perf_diag.get("pattern_accuracy", 0) < 60:
                recommendations.append("🎯 Pattern detection accuracy is low. Review pattern parameters.")

            # Resource recommendations
            resource_util = system_diag.get("resource_utilization", {})
            if resource_util.get("cpu_percent", 0) > 80:
                recommendations.append("🔥 High CPU usage detected. Consider scaling or optimization.")
            
            if resource_util.get("memory_percent", 0) > 85:
                recommendations.append("💾 High memory usage detected. Check for memory leaks.")

        except Exception as e:
            logger.error(f"Error generating diagnostic recommendations: {e}")

        return recommendations

    def generate_performance_report(self, period: str = "daily") -> Dict[str, Any]:
        """Generate comprehensive performance report with v9.1 enhancements and diagnostics"""
        try:
            cache_key = f"performance_report_{period}"

            # Check cache
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]["data"]

            with Session() as session:
                # Get base performance stats
                base_stats = get_performance_stats(session, period)

                # Get enhanced metrics
                positions = self._get_positions_for_period(session, period)

                if not positions:
                    return self._empty_report()

                # Calculate comprehensive metrics
                metrics = self._calculate_enhanced_metrics(positions)

                # HYBRID ULTRA-DIAGNOSTICS: Add diagnostic metrics
                diagnostic_metrics = self._calculate_diagnostic_metrics(positions)

                # Update metrics with diagnostic data
                if diagnostic_metrics:
                    metrics.system_health_score = diagnostic_metrics.get("system_health_score", 0)
                    metrics.decision_confidence_avg = diagnostic_metrics.get("decision_confidence_avg", 0)
                    metrics.pattern_detection_accuracy = diagnostic_metrics.get("pattern_detection_accuracy", 0)
                    metrics.anomaly_detection_rate = diagnostic_metrics.get("anomaly_detection_rate", 0)
                    metrics.execution_efficiency = diagnostic_metrics.get("execution_efficiency", 100)
                    metrics.resource_utilization = diagnostic_metrics.get("resource_utilization", {})
                    metrics.model_drift_score = diagnostic_metrics.get("model_drift_score", 0)

                # v9.1 CORE: Signal Intelligence Analysis
                signal_analysis = self._analyze_signal_intelligence(positions)

                # v9.1 CORE: ML Performance Analysis
                ml_analysis = self._analyze_ml_performance(positions)

                # v9.1 CORE: Regime Analysis
                regime_analysis = self._analyze_regime_performance(positions)

                # v9.1 CORE: Tier Performance Analysis
                tier_analysis = self._analyze_tier_performance(positions)

                # v9.1 CORE: Mode Performance Analysis
                mode_analysis = self._analyze_mode_performance(positions)

                # v9.1 CORE: Risk Analysis
                risk_analysis = self._analyze_risk_metrics(positions)

                # Compile comprehensive report with diagnostics
                report = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "period": period,
                    "summary": {
                        "total_trades": len(positions),
                        "win_rate": metrics.win_rate,
                        "total_pnl": metrics.total_pnl,
                        "profit_factor": metrics.profit_factor,
                        "max_drawdown": metrics.max_drawdown,
                        "sharpe_ratio": metrics.sharpe_ratio,
                    },
                    "enhanced_metrics": {
                        "fake_breakout_accuracy": metrics.fake_breakout_accuracy,
                        "institutional_flow_correlation": metrics.institutional_flow_correlation,
                        "ml_prediction_accuracy": metrics.ml_prediction_accuracy,
                        "regime_accuracy": metrics.regime_accuracy,
                        "mtf_agreement_effectiveness": metrics.mtf_agreement_effectiveness,
                    },
                    # HYBRID ULTRA-DIAGNOSTICS: Add diagnostic section
                    "diagnostic_metrics": {
                        "system_health_score": metrics.system_health_score,
                        "decision_confidence_avg": metrics.decision_confidence_avg,
                        "pattern_detection_accuracy": metrics.pattern_detection_accuracy,
                        "anomaly_detection_rate": metrics.anomaly_detection_rate,
                        "execution_efficiency": metrics.execution_efficiency,
                        "resource_utilization": metrics.resource_utilization,
                        "model_drift_score": metrics.model_drift_score
                    },
                    "signal_intelligence": signal_analysis,
                    "ml_performance": ml_analysis,
                    "regime_analysis": regime_analysis,
                    "tier_performance": tier_analysis,
                    "mode_performance": mode_analysis,
                    "risk_analysis": risk_analysis,
                    "recommendations": self._generate_recommendations(metrics, positions),
                }

                # Add full diagnostic report if enabled
                if self.diagnostics_enabled:
                    report["full_diagnostics"] = self.generate_diagnostic_report()

                # Cache the report
                self._cache_data(cache_key, report)

                logger.info(f"📊 Generated comprehensive {period} performance report with diagnostics")
                return report

        except Exception as e:
            logger.error(f"💥 Error generating performance report: {e}", exc_info=True)
            return self._empty_report()

    def _calculate_enhanced_metrics(
        self, positions: List[Position]
    ) -> PerformanceMetrics:
        """Calculate enhanced performance metrics with v9.1 features"""
        if not positions:
            return PerformanceMetrics()

        # Basic calculations
        total_trades = len(positions)
        winning_trades = len([p for p in positions if p.pnl and p.pnl > 0])
        losing_trades = len([p for p in positions if p.pnl and p.pnl < 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(p.pnl or 0 for p in positions)

        wins = [p.pnl for p in positions if p.pnl and p.pnl > 0]
        losses = [abs(p.pnl) for p in positions if p.pnl and p.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        profit_factor = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else 0

        # Calculate drawdown
        pnl_series = [p.pnl or 0 for p in positions]
        cumulative_pnl = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Calculate Sharpe ratio
        returns = np.array(pnl_series)
        sharpe_ratio = (
            (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
        )

        # Calculate Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = (np.mean(returns) / downside_std) if downside_std > 0 else 0

        # v9.1 CORE: Enhanced metrics calculations
        fake_breakout_accuracy = self._calculate_fake_breakout_accuracy(positions)
        institutional_flow_correlation = self._calculate_institutional_flow_correlation(
            positions
        )
        ml_prediction_accuracy = self._calculate_ml_prediction_accuracy(positions)
        regime_accuracy = self._calculate_regime_accuracy(positions)
        mtf_agreement_effectiveness = self._calculate_mtf_agreement_effectiveness(
            positions
        )

        # Performance by categories
        tier_performance = self._calculate_tier_performance(positions)
        mode_performance = self._calculate_mode_performance(positions)
        timeframe_performance = self._calculate_timeframe_performance(positions)

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            fake_breakout_accuracy=fake_breakout_accuracy,
            institutional_flow_correlation=institutional_flow_correlation,
            ml_prediction_accuracy=ml_prediction_accuracy,
            regime_accuracy=regime_accuracy,
            mtf_agreement_effectiveness=mtf_agreement_effectiveness,
            tier_performance=tier_performance,
            mode_performance=mode_performance,
            timeframe_performance=timeframe_performance,
        )

    def _calculate_fake_breakout_accuracy(self, positions: List[Position]) -> float:
        """Calculate accuracy of fake breakout detection"""
        fake_breakout_positions = [p for p in positions if p.fake_breakout_detected]

        if not fake_breakout_positions:
            return 0.0

        # Positions where fake breakout was detected and trade was profitable
        # (meaning the detection was correct and we avoided a loss)
        correct_detections = len(
            [p for p in fake_breakout_positions if p.pnl and p.pnl > 0]
        )

        return (correct_detections / len(fake_breakout_positions)) * 100

    def _calculate_institutional_flow_correlation(
        self, positions: List[Position]
    ) -> float:
        """Calculate correlation between institutional flow and trade success"""
        flow_data = [
            (p.institutional_flow or 0, p.pnl or 0)
            for p in positions
            if p.institutional_flow is not None and p.pnl is not None
        ]

        if len(flow_data) < 2:
            return 0.0

        flows, pnls = zip(*flow_data)
        correlation = np.corrcoef(flows, pnls)[0, 1]

        return correlation if not np.isnan(correlation) else 0.0

    def _calculate_ml_prediction_accuracy(self, positions: List[Position]) -> float:
        """Calculate ML prediction accuracy"""
        ml_positions = [p for p in positions if p.ml_predictions]

        if not ml_positions:
            return 0.0

        correct_predictions = 0
        total_predictions = 0

        for position in ml_positions:
            try:
                ml_data = (
                    json.loads(position.ml_predictions)
                    if isinstance(position.ml_predictions, str)
                    else position.ml_predictions
                )
                predicted_win_prob = ml_data.get("win_probability", 0.5)

                # Consider prediction correct if:
                # - Predicted win (>0.5) and actual win (pnl > 0)
                # - Predicted loss (<=0.5) and actual loss (pnl <= 0)
                actual_win = position.pnl and position.pnl > 0
                predicted_win = predicted_win_prob > 0.5

                if (predicted_win and actual_win) or (
                    not predicted_win and not actual_win
                ):
                    correct_predictions += 1

                total_predictions += 1

            except (json.JSONDecodeError, AttributeError):
                continue

        return (
            (correct_predictions / total_predictions * 100)
            if total_predictions > 0
            else 0.0
        )

    def _calculate_regime_accuracy(self, positions: List[Position]) -> float:
        """Calculate regime detection accuracy"""
        regime_positions = [p for p in positions if p.regime and p.regime != "NEUTRAL"]

        if not regime_positions:
            return 0.0

        correct_regime_trades = 0

        for position in regime_positions:
            # Consider regime correct if trade was profitable in trending regime
            if (
                position.regime in ["TRENDING_UP", "TRENDING_DOWN"]
                and position.pnl
                and position.pnl > 0
            ):
                correct_regime_trades += 1

        return (correct_regime_trades / len(regime_positions)) * 100

    def _calculate_mtf_agreement_effectiveness(
        self, positions: List[Position]
    ) -> float:
        """Calculate effectiveness of multi-timeframe agreement"""
        mtf_positions = [p for p in positions if p.mtf_agreement is not None]

        if not mtf_positions:
            return 0.0

        # Group by MTF agreement levels
        high_agreement = [p for p in mtf_positions if p.mtf_agreement > 0.7]
        low_agreement = [p for p in mtf_positions if p.mtf_agreement < 0.4]

        if not high_agreement or not low_agreement:
            return 0.0

        high_agreement_win_rate = len(
            [p for p in high_agreement if p.pnl and p.pnl > 0]
        ) / len(high_agreement)
        low_agreement_win_rate = len(
            [p for p in low_agreement if p.pnl and p.pnl > 0]
        ) / len(low_agreement)

        # Effectiveness is the difference in win rates
        return (high_agreement_win_rate - low_agreement_win_rate) * 100

    def _calculate_tier_performance(
        self, positions: List[Position]
    ) -> Dict[str, float]:
        """Calculate performance by signal tier"""
        tier_stats = defaultdict(list)

        for position in positions:
            if position.tier and position.pnl is not None:
                tier_stats[position.tier].append(position.pnl)

        tier_performance = {}
        for tier, pnls in tier_stats.items():
            win_rate = len([pnl for pnl in pnls if pnl > 0]) / len(pnls) * 100
            avg_pnl = np.mean(pnls)
            tier_performance[tier] = {
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "total_trades": len(pnls),
                "total_pnl": sum(pnls),
            }

        return tier_performance

    def _calculate_mode_performance(
        self, positions: List[Position]
    ) -> Dict[str, float]:
        """Calculate performance by trading mode"""
        # This would require mode tracking in positions table
        # For now, return empty dict
        return {}

    def _calculate_timeframe_performance(
        self, positions: List[Position]
    ) -> Dict[str, float]:
        """Calculate performance by timeframe"""
        # This would require timeframe tracking in positions table
        # For now, return empty dict
        return {}

    def _analyze_signal_intelligence(self, positions: List[Position]) -> Dict[str, Any]:
        """Analyze signal intelligence performance"""
        total_positions = len(positions)

        # Fake breakout analysis
        fake_breakout_detected = len([p for p in positions if p.fake_breakout_detected])
        fake_breakout_avoided_losses = len(
            [p for p in positions if p.fake_breakout_detected and p.pnl and p.pnl > 0]
        )

        # Institutional flow analysis
        high_flow_positions = [
            p for p in positions if p.institutional_flow and p.institutional_flow > 0.7
        ]
        high_flow_win_rate = (
            (
                len([p for p in high_flow_positions if p.pnl and p.pnl > 0])
                / len(high_flow_positions)
                * 100
            )
            if high_flow_positions
            else 0
        )

        # MTF agreement analysis
        high_mtf_positions = [
            p for p in positions if p.mtf_agreement and p.mtf_agreement > 0.8
        ]
        high_mtf_win_rate = (
            (
                len([p for p in high_mtf_positions if p.pnl and p.pnl > 0])
                / len(high_mtf_positions)
                * 100
            )
            if high_mtf_positions
            else 0
        )

        return {
            "fake_breakout_detection": {
                "total_detected": fake_breakout_detected,
                "detection_rate": (
                    (fake_breakout_detected / total_positions * 100)
                    if total_positions > 0
                    else 0
                ),
                "avoided_losses": fake_breakout_avoided_losses,
                "effectiveness": (
                    (fake_breakout_avoided_losses / fake_breakout_detected * 100)
                    if fake_breakout_detected > 0
                    else 0
                ),
            },
            "institutional_flow": {
                "high_flow_trades": len(high_flow_positions),
                "high_flow_win_rate": high_flow_win_rate,
                "correlation_with_success": self._calculate_institutional_flow_correlation(
                    positions
                ),
            },
            "mtf_agreement": {
                "high_agreement_trades": len(high_mtf_positions),
                "high_agreement_win_rate": high_mtf_win_rate,
                "effectiveness": self._calculate_mtf_agreement_effectiveness(positions),
            },
        }

    def _analyze_ml_performance(self, positions: List[Position]) -> Dict[str, Any]:
        """Analyze ML prediction performance"""
        ml_positions = [p for p in positions if p.ml_predictions]

        if not ml_positions:
            return {
                "enabled": False,
                "total_predictions": 0,
                "accuracy": 0.0,
                "confidence_analysis": {},
                "feature_importance": {},
            }

        accuracy = self._calculate_ml_prediction_accuracy(positions)

        # Confidence analysis
        confidence_buckets = {"high": [], "medium": [], "low": []}

        for position in ml_positions:
            try:
                ml_data = (
                    json.loads(position.ml_predictions)
                    if isinstance(position.ml_predictions, str)
                    else position.ml_predictions
                )
                confidence = ml_data.get("confidence", 0.5)

                if confidence > 0.8:
                    confidence_buckets["high"].append(position)
                elif confidence > 0.5:
                    confidence_buckets["medium"].append(position)
                else:
                    confidence_buckets["low"].append(position)
            except (json.JSONDecodeError, AttributeError):
                continue

        confidence_analysis = {}
        for bucket, positions_list in confidence_buckets.items():
            if positions_list:
                win_rate = (
                    len([p for p in positions_list if p.pnl and p.pnl > 0])
                    / len(positions_list)
                    * 100
                )
                avg_pnl = np.mean([p.pnl or 0 for p in positions_list])
                confidence_analysis[bucket] = {
                    "count": len(positions_list),
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                }

        return {
            "enabled": True,
            "total_predictions": len(ml_positions),
            "accuracy": accuracy,
            "confidence_analysis": confidence_analysis,
            "feature_importance": {},  # Would need to be calculated from ML model
        }

    def _analyze_regime_performance(self, positions: List[Position]) -> Dict[str, Any]:
        """Analyze market regime performance"""
        regime_stats = defaultdict(list)

        for position in positions:
            if position.regime and position.pnl is not None:
                regime_stats[position.regime].append(position.pnl)

        regime_analysis = {}
        for regime, pnls in regime_stats.items():
            win_rate = len([pnl for pnl in pnls if pnl > 0]) / len(pnls) * 100
            avg_pnl = np.mean(pnls)
            regime_analysis[regime] = {
                "total_trades": len(pnls),
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "total_pnl": sum(pnls),
            }

        return regime_analysis

    def _analyze_tier_performance(self, positions: List[Position]) -> Dict[str, Any]:
        """Analyze performance by signal tier"""
        return self._calculate_tier_performance(positions)

    def _analyze_mode_performance(self, positions: List[Position]) -> Dict[str, Any]:
        """Analyze performance by trading mode"""
        # This would require mode tracking in database
        return {}

    def _analyze_risk_metrics(self, positions: List[Position]) -> Dict[str, Any]:
        """Analyze risk metrics"""
        if not positions:
            return {}

        pnls = [p.pnl or 0 for p in positions]

        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(pnls, 5) if pnls else 0

        # Expected Shortfall (Conditional VaR)
        losses = [pnl for pnl in pnls if pnl < var_95]
        expected_shortfall = np.mean(losses) if losses else 0

        # Maximum consecutive losses
        max_consecutive_losses = 0
        current_streak = 0

        for pnl in pnls:
            if pnl < 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        # Risk-adjusted returns
        total_return = sum(pnls)
        volatility = np.std(pnls) if len(pnls) > 1 else 0

        return {
            "var_95": var_95,
            "expected_shortfall": expected_shortfall,
            "max_consecutive_losses": max_consecutive_losses,
            "volatility": volatility,
            "total_return": total_return,
            "risk_adjusted_return": (
                (total_return / volatility) if volatility > 0 else 0
            ),
        }

    def _generate_recommendations(
        self, metrics: PerformanceMetrics, positions: List[Position]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Win rate recommendations
        if metrics.win_rate < 40:
            recommendations.append(
                "🔴 Low win rate detected. Consider tightening signal filters or switching to conservative mode."
            )
        elif metrics.win_rate > 80:
            recommendations.append(
                "🟢 Excellent win rate. Consider increasing position sizes or switching to aggressive mode."
            )

        # Profit factor recommendations
        if metrics.profit_factor < 1.2:
            recommendations.append(
                "🔴 Low profit factor. Review stop-loss and take-profit levels."
            )
        elif metrics.profit_factor > 2.0:
            recommendations.append(
                "🟢 Strong profit factor. Current strategy is performing well."
            )

        # Fake breakout recommendations
        if metrics.fake_breakout_accuracy > 70:
            recommendations.append(
                "🎯 Fake breakout detection is highly effective. Continue using this feature."
            )
        elif metrics.fake_breakout_accuracy < 30:
            recommendations.append(
                "⚠️ Fake breakout detection needs improvement. Review detection parameters."
            )

        # ML recommendations
        if metrics.ml_prediction_accuracy > 60:
            recommendations.append(
                "🤖 ML predictions are performing well. Consider increasing ML influence on decisions."
            )
        elif metrics.ml_prediction_accuracy < 40:
            recommendations.append(
                "🤖 ML predictions underperforming. Review model training or disable ML mode."
            )

        # Institutional flow recommendations
        if metrics.institutional_flow_correlation > 0.3:
            recommendations.append(
                "📈 Strong correlation with institutional flow. Consider institutional mode."
            )
        elif metrics.institutional_flow_correlation < -0.3:
            recommendations.append(
                "📉 Negative correlation with institutional flow. Review flow analysis."
            )

        # Risk recommendations
        if metrics.max_drawdown > Config.MAX_DAILY_LOSS * 0.5:
            recommendations.append(
                "⚠️ High drawdown detected. Consider reducing position sizes or switching to conservative mode."
            )

        return recommendations

    def _get_positions_for_period(
        self, session: Session, period: str
    ) -> List[Position]:
        """Get positions for specified period"""
        now = datetime.utcnow()

        if period == "daily":
            start_time = now - timedelta(days=1)
        elif period == "weekly":
            start_time = now - timedelta(weeks=1)
        elif period == "monthly":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)

        return (
            session.query(Position)
            .filter(Position.created_at >= start_time, Position.status == "closed")
            .all()
        )

    def _empty_report(self) -> Dict[str, Any]:
        """Return empty report structure"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "period": "daily",
            "summary": {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            },
            "enhanced_metrics": {},
            "signal_intelligence": {},
            "ml_performance": {"enabled": False},
            "regime_analysis": {},
            "tier_performance": {},
            "mode_performance": {},
            "risk_analysis": {},
            "recommendations": [],
        }

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False

        cache_time = self.cache[key]["timestamp"]
        return (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl

    def _cache_data(self, key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[key] = {"data": data, "timestamp": datetime.utcnow()}

    async def generate_real_time_metrics(self) -> Dict[str, Any]:
        """Generate real-time performance metrics"""
        try:
            with Session() as session:
                # Get today's positions
                today_start = datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

                open_positions = (
                    session.query(Position).filter(Position.status == "open").all()
                )

                closed_positions_today = (
                    session.query(Position)
                    .filter(
                        Position.status == "closed", Position.exit_time >= today_start
                    )
                    .all()
                )

                # Calculate real-time metrics
                total_open_pnl = sum(
                    self._calculate_unrealized_pnl(p) for p in open_positions
                )
                total_closed_pnl = sum(p.pnl or 0 for p in closed_positions_today)

                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "open_positions": len(open_positions),
                    "closed_positions_today": len(closed_positions_today),
                    "unrealized_pnl": total_open_pnl,
                    "realized_pnl_today": total_closed_pnl,
                    "total_pnl_today": total_open_pnl + total_closed_pnl,
                    "positions_details": [
                        {
                            "symbol": p.symbol,
                            "side": p.side,
                            "entry_price": p.entry_price,
                            "current_pnl": self._calculate_unrealized_pnl(p),
                            "tier": p.tier,
                            "fake_breakout_detected": p.fake_breakout_detected,
                        }
                        for p in open_positions
                    ],
                }

        except Exception as e:
            logger.error(f"💥 Error generating real-time metrics: {e}", exc_info=True)
            return {}

    def _calculate_unrealized_pnl(self, position: Position) -> float:
        """Calculate unrealized PnL for open position"""
        # This would require current market price
        # For now, return 0 as placeholder
        return 0.0

    def export_performance_data(
        self, period: str = "monthly", format: str = "json"
    ) -> str:
        """Export performance data in specified format"""
        try:
            report = self.generate_performance_report(period)

            if format.lower() == "json":
                return json.dumps(report, indent=2, default=str)
            elif format.lower() == "csv":
                # Convert to CSV format
                return self._convert_to_csv(report)
            else:
                return json.dumps(report, indent=2, default=str)

        except Exception as e:
            logger.error(f"💥 Error exporting performance data: {e}", exc_info=True)
            return "{}"

    def _convert_to_csv(self, report: Dict) -> str:
        """Convert report to CSV format"""
        # Simplified CSV conversion
        lines = []
        lines.append("Metric,Value")

        summary = report.get("summary", {})
        for key, value in summary.items():
            lines.append(f"{key},{value}")

        return "\n".join(lines)

    def get_performance_insights(self) -> Dict[str, Any]:
        """Get AI-powered performance insights"""
        try:
            report = self.generate_performance_report("weekly")

            insights = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_health": self._assess_overall_health(report),
                "key_strengths": self._identify_strengths(report),
                "areas_for_improvement": self._identify_improvements(report),
                "risk_assessment": self._assess_risk_level(report),
                "recommended_actions": report.get("recommendations", []),
            }

            return insights

        except Exception as e:
            logger.error(
                f"💥 Error generating performance insights: {e}", exc_info=True
            )
            return {}

    def _assess_overall_health(self, report: Dict) -> str:
        """Assess overall trading performance health"""
        summary = report.get("summary", {})
        win_rate = summary.get("win_rate", 0)
        profit_factor = summary.get("profit_factor", 0)
        total_pnl = summary.get("total_pnl", 0)

        if win_rate > 60 and profit_factor > 1.5 and total_pnl > 0:
            return "Excellent"
        elif win_rate > 45 and profit_factor > 1.2 and total_pnl >= 0:
            return "Good"
        elif win_rate > 35 and profit_factor > 1.0:
            return "Fair"
        else:
            return "Poor"

    def _identify_strengths(self, report: Dict) -> List[str]:
        """Identify key performance strengths"""
        strengths = []

        summary = report.get("summary", {})
        enhanced = report.get("enhanced_metrics", {})

        if summary.get("win_rate", 0) > 60:
            strengths.append("High win rate consistency")

        if summary.get("profit_factor", 0) > 2.0:
            strengths.append("Excellent risk-reward ratio")

        if enhanced.get("fake_breakout_accuracy", 0) > 70:
            strengths.append("Effective fake breakout detection")

        if enhanced.get("ml_prediction_accuracy", 0) > 60:
            strengths.append("Strong ML prediction performance")

        return strengths

    def _identify_improvements(self, report: Dict) -> List[str]:
        """Identify areas for improvement"""
        improvements = []

        summary = report.get("summary", {})
        enhanced = report.get("enhanced_metrics", {})

        if summary.get("win_rate", 0) < 40:
            improvements.append("Improve signal quality and filtering")

        if summary.get("profit_factor", 0) < 1.2:
            improvements.append("Optimize stop-loss and take-profit levels")

        if enhanced.get("fake_breakout_accuracy", 0) < 30:
            improvements.append("Refine fake breakout detection parameters")

        if enhanced.get("ml_prediction_accuracy", 0) < 40:
            improvements.append("Retrain ML model with recent data")

        return improvements

    def _assess_risk_level(self, report: Dict) -> str:
        """Assess current risk level"""
        risk_analysis = report.get("risk_analysis", {})
        summary = report.get("summary", {})

        max_drawdown = summary.get("max_drawdown", 0)
        consecutive_losses = risk_analysis.get("max_consecutive_losses", 0)

        if max_drawdown > Config.MAX_DAILY_LOSS * 0.8 or consecutive_losses > 5:
            return "High"
        elif max_drawdown > Config.MAX_DAILY_LOSS * 0.5 or consecutive_losses > 3:
            return "Medium"
        else:
            return "Low"


# Singleton instance
Analytics = AnalyticsEngine()

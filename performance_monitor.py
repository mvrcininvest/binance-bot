"""
Performance Monitor - Advanced Performance Monitoring for Trading Bot v9.1
Real-time performance tracking, analysis, and optimization recommendations
"""

import asyncio
import logging
import time
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import json
import statistics
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    TRADING_PERFORMANCE = "trading_performance"
    API_PERFORMANCE = "api_performance"

@dataclass
class PerformanceMetric:
    name: str
    value: float
    unit: str
    timestamp: datetime
    metric_type: MetricType
    level: PerformanceLevel
    threshold_good: float
    threshold_poor: float
    metadata: Dict[str, Any]

@dataclass
class PerformanceSnapshot:
    timestamp: datetime
    overall_score: float
    level: PerformanceLevel
    metrics: List[PerformanceMetric]
    recommendations: List[str]
    alerts: List[str]

@dataclass
class TradingPerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_trade_duration: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

class PerformanceMonitor:
    """Advanced performance monitoring and analysis"""
    
    def __init__(self, db_pool, alert_manager, config):
        self.db_pool = db_pool
        self.alert_manager = alert_manager
        self.config = config
        
        # Performance data storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_snapshots: deque = deque(maxlen=100)
        
        # Real-time tracking
        self.active_operations: Dict[str, float] = {}
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Trading performance tracking
        self.trading_metrics: Dict[str, Any] = {}
        self.position_history: deque = deque(maxlen=1000)
        
        # System performance baselines
        self.performance_baselines = {
            'api_response_time': {'good': 500, 'poor': 2000},  # ms
            'database_query_time': {'good': 100, 'poor': 1000},  # ms
            'order_execution_time': {'good': 1000, 'poor': 5000},  # ms
            'memory_usage': {'good': 70, 'poor': 85},  # %
            'cpu_usage': {'good': 60, 'poor': 80},  # %
            'error_rate': {'good': 1, 'poor': 5},  # %
            'win_rate': {'good': 60, 'poor': 40},  # %
            'profit_factor': {'good': 1.5, 'poor': 1.0}
        }
        
        # Performance monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        
    async def start_monitoring(self):
        """Start performance monitoring tasks"""
        try:
            # System performance monitoring
            self.monitoring_tasks.append(
                asyncio.create_task(self._monitor_system_performance())
            )
            
            # Trading performance monitoring
            self.monitoring_tasks.append(
                asyncio.create_task(self._monitor_trading_performance())
            )
            
            # API performance monitoring
            self.monitoring_tasks.append(
                asyncio.create_task(self._monitor_api_performance())
            )
            
            # Database performance monitoring
            self.monitoring_tasks.append(
                asyncio.create_task(self._monitor_database_performance())
            )
            
            # Performance analysis task
            self.monitoring_tasks.append(
                asyncio.create_task(self._analyze_performance_trends())
            )
            
            logger.info("🚀 Performance monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting performance monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop performance monitoring tasks"""
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        logger.info("🛑 Performance monitoring stopped")
    
    @asynccontextmanager
    async def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to track operation performance"""
        start_time = time.time()
        self.active_operations[operation_name] = start_time
        
        try:
            yield
            # Operation completed successfully
            duration = (time.time() - start_time) * 1000  # Convert to ms
            self.operation_times[operation_name].append(duration)
            
            # Log slow operations
            threshold = self.performance_baselines.get(operation_name, {}).get('poor', 5000)
            if duration > threshold:
                logger.warning(f"⚠️ Slow operation: {operation_name} took {duration:.2f}ms")
                
                # Create alert for very slow operations
                if duration > threshold * 2:
                    await self.alert_manager.create_alert(
                        title=f"Slow Operation: {operation_name}",
                        message=f"Operation took {duration:.2f}ms (threshold: {threshold}ms)",
                        severity=self.alert_manager.AlertSeverity.HIGH,
                        category=self.alert_manager.AlertCategory.PERFORMANCE,
                        source="performance_monitor",
                        metadata={
                            "operation": operation_name,
                            "duration_ms": duration,
                            "threshold_ms": threshold,
                            **(metadata or {})
                        }
                    )
            
        except Exception as e:
            # Operation failed
            duration = (time.time() - start_time) * 1000
            self.error_counts[operation_name].append(time.time())
            
            logger.error(f"❌ Operation failed: {operation_name} after {duration:.2f}ms - {e}")
            
            # Create alert for operation failures
            await self.alert_manager.create_alert(
                title=f"Operation Failed: {operation_name}",
                message=f"Operation failed after {duration:.2f}ms: {str(e)}",
                severity=self.alert_manager.AlertSeverity.HIGH,
                category=self.alert_manager.AlertCategory.PERFORMANCE,
                source="performance_monitor",
                metadata={
                    "operation": operation_name,
                    "duration_ms": duration,
                    "error": str(e),
                    **(metadata or {})
                }
            )
            raise
        finally:
            self.active_operations.pop(operation_name, None)
    
    async def record_trading_performance(self, trade_data: Dict[str, Any]):
        """Record trading performance metrics"""
        try:
            self.position_history.append({
                'timestamp': datetime.utcnow(),
                'symbol': trade_data.get('symbol'),
                'side': trade_data.get('side'),
                'quantity': trade_data.get('quantity'),
                'entry_price': trade_data.get('entry_price'),
                'exit_price': trade_data.get('exit_price'),
                'pnl': trade_data.get('pnl', 0),
                'duration': trade_data.get('duration', 0),
                'fees': trade_data.get('fees', 0)
            })
            
            # Update trading metrics
            await self._update_trading_metrics()
            
        except Exception as e:
            logger.error(f"Error recording trading performance: {e}")
    
    async def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""
        try:
            timestamp = datetime.utcnow()
            metrics = []
            
            # System metrics
            system_metrics = await self._get_system_performance_metrics()
            metrics.extend(system_metrics)
            
            # Trading metrics
            trading_metrics = await self._get_trading_performance_metrics()
            metrics.extend(trading_metrics)
            
            # API metrics
            api_metrics = await self._get_api_performance_metrics()
            metrics.extend(api_metrics)
            
            # Database metrics
            db_metrics = await self._get_database_performance_metrics()
            metrics.extend(db_metrics)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            level = self._get_performance_level(overall_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)
            
            # Generate alerts
            alerts = self._generate_performance_alerts(metrics)
            
            snapshot = PerformanceSnapshot(
                timestamp=timestamp,
                overall_score=overall_score,
                level=level,
                metrics=metrics,
                recommendations=recommendations,
                alerts=alerts
            )
            
            # Store snapshot
            self.performance_snapshots.append(snapshot)
            await self._store_performance_snapshot(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error getting performance snapshot: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                overall_score=0,
                level=PerformanceLevel.CRITICAL,
                metrics=[],
                recommendations=["Error getting performance data"],
                alerts=["Performance monitoring error"]
            )
    
    async def _monitor_system_performance(self):
        """Monitor system performance metrics"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                await self._record_metric(
                    "cpu_usage", cpu_percent, "%", MetricType.RESOURCE_USAGE
                )
                
                # Memory usage
                memory = psutil.virtual_memory()
                await self._record_metric(
                    "memory_usage", memory.percent, "%", MetricType.RESOURCE_USAGE
                )
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    await self._record_metric(
                        "disk_read_speed", disk_io.read_bytes, "bytes/s", MetricType.THROUGHPUT
                    )
                    await self._record_metric(
                        "disk_write_speed", disk_io.write_bytes, "bytes/s", MetricType.THROUGHPUT
                    )
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    await self._record_metric(
                        "network_sent", net_io.bytes_sent, "bytes/s", MetricType.THROUGHPUT
                    )
                    await self._record_metric(
                        "network_recv", net_io.bytes_recv, "bytes/s", MetricType.THROUGHPUT
                    )
                
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system performance: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_trading_performance(self):
        """Monitor trading performance metrics"""
        while True:
            try:
                # Update trading metrics
                await self._update_trading_metrics()
                
                # Check for performance degradation
                if self.trading_metrics:
                    win_rate = self.trading_metrics.get('win_rate', 0)
                    if win_rate < self.performance_baselines['win_rate']['poor']:
                        await self.alert_manager.create_alert(
                            title="Poor Trading Performance",
                            message=f"Win rate has dropped to {win_rate:.1f}%",
                            severity=self.alert_manager.AlertSeverity.HIGH,
                            category=self.alert_manager.AlertCategory.TRADING,
                            source="performance_monitor"
                        )
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring trading performance: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_api_performance(self):
        """Monitor API performance metrics"""
        while True:
            try:
                # Calculate API response times
                for operation, times in self.operation_times.items():
                    if times and 'api' in operation.lower():
                        avg_time = statistics.mean(times)
                        await self._record_metric(
                            f"{operation}_response_time", avg_time, "ms", MetricType.LATENCY
                        )
                
                # Calculate error rates
                for operation, errors in self.error_counts.items():
                    if errors:
                        # Count errors in last hour
                        hour_ago = time.time() - 3600
                        recent_errors = [e for e in errors if e > hour_ago]
                        error_rate = len(recent_errors) / 60  # errors per minute
                        
                        await self._record_metric(
                            f"{operation}_error_rate", error_rate, "errors/min", MetricType.ERROR_RATE
                        )
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Error monitoring API performance: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_database_performance(self):
        """Monitor database performance metrics"""
        while True:
            try:
                async with self.db_pool.acquire() as conn:
                    # Query execution time test
                    start_time = time.time()
                    await conn.fetchval("SELECT 1")
                    query_time = (time.time() - start_time) * 1000
                    
                    await self._record_metric(
                        "database_query_time", query_time, "ms", MetricType.LATENCY
                    )
                    
                    # Database size
                    db_size = await conn.fetchval("""
                        SELECT pg_database_size(current_database())
                    """)
                    
                    await self._record_metric(
                        "database_size", db_size, "bytes", MetricType.RESOURCE_USAGE
                    )
                    
                    # Active connections
                    active_connections = await conn.fetchval("""
                        SELECT count(*) FROM pg_stat_activity WHERE state = 'active'
                    """)
                    
                    await self._record_metric(
                        "database_connections", active_connections, "count", MetricType.RESOURCE_USAGE
                    )
                
                await asyncio.sleep(120)  # Every 2 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring database performance: {e}")
                await asyncio.sleep(120)
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends and generate insights"""
        while True:
            try:
                # Analyze trends for each metric
                for metric_name, history in self.metrics_history.items():
                    if len(history) >= 10:  # Need at least 10 data points
                        values = [m.value for m in history]
                        
                        # Calculate trend
                        trend = self._calculate_trend(values)
                        
                        # Check for concerning trends
                        if abs(trend) > 0.1:  # 10% change
                            trend_direction = "increasing" if trend > 0 else "decreasing"
                            
                            # Determine if trend is concerning
                            concerning = False
                            if metric_name in ['cpu_usage', 'memory_usage', 'error_rate'] and trend > 0:
                                concerning = True
                            elif metric_name in ['win_rate', 'profit_factor'] and trend < 0:
                                concerning = True
                            
                            if concerning:
                                await self.alert_manager.create_alert(
                                    title=f"Performance Trend Alert: {metric_name}",
                                    message=f"{metric_name} is {trend_direction} by {abs(trend)*100:.1f}%",
                                    severity=self.alert_manager.AlertSeverity.MEDIUM,
                                    category=self.alert_manager.AlertCategory.PERFORMANCE,
                                    source="performance_monitor",
                                    metadata={
                                        "metric": metric_name,
                                        "trend": trend,
                                        "direction": trend_direction
                                    }
                                )
                
                await asyncio.sleep(900)  # Every 15 minutes
                
            except Exception as e:
                logger.error(f"Error analyzing performance trends: {e}")
                await asyncio.sleep(900)
    
    async def _record_metric(self, name: str, value: float, unit: str, metric_type: MetricType):
        """Record a performance metric"""
        try:
            # Get performance level
            baseline = self.performance_baselines.get(name, {'good': 0, 'poor': float('inf')})
            level = self._get_metric_level(value, baseline, metric_type)
            
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=datetime.utcnow(),
                metric_type=metric_type,
                level=level,
                threshold_good=baseline['good'],
                threshold_poor=baseline['poor'],
                metadata={}
            )
            
            self.metrics_history[name].append(metric)
            
        except Exception as e:
            logger.error(f"Error recording metric {name}: {e}")
    
    def _get_metric_level(self, value: float, baseline: Dict[str, float], metric_type: MetricType) -> PerformanceLevel:
        """Determine performance level for a metric"""
        good_threshold = baseline['good']
        poor_threshold = baseline['poor']
        
        # For metrics where lower is better (latency, error rate, resource usage)
        if metric_type in [MetricType.LATENCY, MetricType.ERROR_RATE, MetricType.RESOURCE_USAGE]:
            if value <= good_threshold:
                return PerformanceLevel.EXCELLENT
            elif value <= (good_threshold + poor_threshold) / 2:
                return PerformanceLevel.GOOD
            elif value <= poor_threshold:
                return PerformanceLevel.AVERAGE
            elif value <= poor_threshold * 1.5:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL
        
        # For metrics where higher is better (throughput, trading performance)
        else:
            if value >= good_threshold:
                return PerformanceLevel.EXCELLENT
            elif value >= (good_threshold + poor_threshold) / 2:
                return PerformanceLevel.GOOD
            elif value >= poor_threshold:
                return PerformanceLevel.AVERAGE
            elif value >= poor_threshold * 0.5:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL
    
    async def _get_system_performance_metrics(self) -> List[PerformanceMetric]:
        """Get current system performance metrics"""
        metrics = []
        
        # Get latest metrics from history
        for name in ['cpu_usage', 'memory_usage']:
            if name in self.metrics_history and self.metrics_history[name]:
                metrics.append(self.metrics_history[name][-1])
        
        return metrics
    
    async def _get_trading_performance_metrics(self) -> List[PerformanceMetric]:
        """Get current trading performance metrics"""
        metrics = []
        
        if self.trading_metrics:
            for key, value in self.trading_metrics.items():
                if isinstance(value, (int, float)):
                    baseline = self.performance_baselines.get(key, {'good': 0, 'poor': 0})
                    level = self._get_metric_level(value, baseline, MetricType.TRADING_PERFORMANCE)
                    
                    metric = PerformanceMetric(
                        name=key,
                        value=float(value),
                        unit=self._get_metric_unit(key),
                        timestamp=datetime.utcnow(),
                        metric_type=MetricType.TRADING_PERFORMANCE,
                        level=level,
                        threshold_good=baseline['good'],
                        threshold_poor=baseline['poor'],
                        metadata={}
                    )
                    metrics.append(metric)
        
        return metrics
    
    async def _get_api_performance_metrics(self) -> List[PerformanceMetric]:
        """Get current API performance metrics"""
        metrics = []
        
        for name in self.metrics_history:
            if 'api' in name.lower() and 'response_time' in name:
                if self.metrics_history[name]:
                    metrics.append(self.metrics_history[name][-1])
        
        return metrics
    
    async def _get_database_performance_metrics(self) -> List[PerformanceMetric]:
        """Get current database performance metrics"""
        metrics = []
        
        for name in ['database_query_time', 'database_connections']:
            if name in self.metrics_history and self.metrics_history[name]:
                metrics.append(self.metrics_history[name][-1])
        
        return metrics
    
    def _calculate_overall_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall performance score (0-100)"""
        if not metrics:
            return 0
        
        level_scores = {
            PerformanceLevel.EXCELLENT: 100,
            PerformanceLevel.GOOD: 80,
            PerformanceLevel.AVERAGE: 60,
            PerformanceLevel.POOR: 40,
            PerformanceLevel.CRITICAL: 20
        }
        
        total_score = sum(level_scores[metric.level] for metric in metrics)
        return total_score / len(metrics)
    
    def _get_performance_level(self, score: float) -> PerformanceLevel:
        """Get performance level from score"""
        if score >= 90:
            return PerformanceLevel.EXCELLENT
        elif score >= 75:
            return PerformanceLevel.GOOD
        elif score >= 60:
            return PerformanceLevel.AVERAGE
        elif score >= 40:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def _generate_recommendations(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for metric in metrics:
            if metric.level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
                if metric.name == 'cpu_usage':
                    recommendations.append("Consider optimizing CPU-intensive operations or scaling resources")
                elif metric.name == 'memory_usage':
                    recommendations.append("Review memory usage patterns and consider increasing available RAM")
                elif metric.name == 'database_query_time':
                    recommendations.append("Optimize database queries and consider adding indexes")
                elif 'response_time' in metric.name:
                    recommendations.append(f"Optimize {metric.name} - consider caching or connection pooling")
                elif metric.name == 'win_rate':
                    recommendations.append("Review trading strategy parameters and market conditions")
                elif metric.name == 'error_rate':
                    recommendations.append("Investigate and fix recurring errors in the system")
        
        # Remove duplicates
        return list(set(recommendations))
    
    def _generate_performance_alerts(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Generate performance alerts"""
        alerts = []
        
        critical_metrics = [m for m in metrics if m.level == PerformanceLevel.CRITICAL]
        poor_metrics = [m for m in metrics if m.level == PerformanceLevel.POOR]
        
        if critical_metrics:
            alerts.append(f"🚨 {len(critical_metrics)} metrics in CRITICAL state")
        
        if poor_metrics:
            alerts.append(f"⚠️ {len(poor_metrics)} metrics in POOR state")
        
        return alerts
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (percentage change) from values"""
        if len(values) < 2:
            return 0
        
        # Use linear regression to calculate trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Convert to percentage change
        if values[0] != 0:
            trend = (slope * len(values)) / values[0]
        else:
            trend = 0
        
        return trend
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric name"""
        unit_map = {
            'win_rate': '%',
            'profit_factor': 'ratio',
            'total_pnl': 'USDT',
            'avg_trade_duration': 'minutes',
            'max_drawdown': '%',
            'sharpe_ratio': 'ratio'
        }
        return unit_map.get(metric_name, '')
    
    async def _update_trading_metrics(self):
        """Update trading performance metrics"""
        try:
            if not self.position_history:
                return
            
            trades = list(self.position_history)
            
            # Basic metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
            
            # Win rate
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # PnL metrics
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
            losses = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]
            
            avg_win = statistics.mean(wins) if wins else 0
            avg_loss = statistics.mean(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0
            
            # Profit factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Average trade duration
            durations = [t.get('duration', 0) for t in trades if t.get('duration', 0) > 0]
            avg_trade_duration = statistics.mean(durations) if durations else 0
            
            # Max drawdown (simplified)
            pnls = [t.get('pnl', 0) for t in trades]
            cumulative_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - running_max) / running_max * 100
            max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
            
            # Sharpe ratio (simplified)
            if len(pnls) > 1:
                returns_std = np.std(pnls)
                avg_return = np.mean(pnls)
                sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            self.trading_metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_trade_duration': avg_trade_duration,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss
            }
            
        except Exception as e:
            logger.error(f"Error updating trading metrics: {e}")
    
    async def _store_performance_snapshot(self, snapshot: PerformanceSnapshot):
        """Store performance snapshot in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO performance_snapshots (
                        timestamp, overall_score, level, metrics_count,
                        recommendations, alerts
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                snapshot.timestamp, snapshot.overall_score, snapshot.level.value,
                len(snapshot.metrics), json.dumps(snapshot.recommendations),
                json.dumps(snapshot.alerts)
                )
                
                # Store individual metrics
                for metric in snapshot.metrics:
                    await conn.execute("""
                        INSERT INTO performance_metrics (
                            timestamp, name, value, unit, metric_type, level,
                            threshold_good, threshold_poor, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    metric.timestamp, metric.name, metric.value, metric.unit,
                    metric.metric_type.value, metric.level.value,
                    metric.threshold_good, metric.threshold_poor,
                    json.dumps(metric.metadata)
                    )
                
        except Exception as e:
            logger.error(f"Error storing performance snapshot: {e}")
    
    async def get_performance_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance history"""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT timestamp, overall_score, level, metrics_count
                    FROM performance_snapshots
                    WHERE timestamp >= $1
                    ORDER BY timestamp DESC
                """, since)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []
    
    async def cleanup_old_data(self, days: int = 7):
        """Clean up old performance data"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            async with self.db_pool.acquire() as conn:
                deleted_snapshots = await conn.fetchval("""
                    DELETE FROM performance_snapshots 
                    WHERE timestamp < $1
                    RETURNING count(*)
                """, cutoff)
                
                deleted_metrics = await conn.fetchval("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < $1
                    RETURNING count(*)
                """, cutoff)
                
                logger.info(f"🧹 Cleaned up {deleted_snapshots} performance snapshots and {deleted_metrics} metrics")
                
        except Exception as e:
            logger.error(f"Error cleaning up performance data: {e}")

# Utility functions
def format_performance_report(snapshot: PerformanceSnapshot) -> str:
    """Format performance snapshot into readable report"""
    report = f"""
📊 **Performance Report** - {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

**Overall Score:** {snapshot.overall_score:.1f}/100 ({snapshot.level.value.upper()})

**Key Metrics:**
"""
    
    for metric in snapshot.metrics[:10]:  # Show top 10 metrics
        report += f"• {metric.name}: {metric.value:.2f}{metric.unit} ({metric.level.value})\n"
    
    if snapshot.recommendations:
        report += f"\n**Recommendations:**\n"
        for rec in snapshot.recommendations:
            report += f"• {rec}\n"
    
    if snapshot.alerts:
        report += f"\n**Alerts:**\n"
        for alert in snapshot.alerts:
            report += f"• {alert}\n"
    
    return report
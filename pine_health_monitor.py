"""
Pine Script Health Monitor - Advanced monitoring and analysis of Pine Script performance
Part of the Hybrid Ultra-Diagnostics system for Trading Bot v9.1
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PineScriptMetrics:
    """Metrics for Pine Script performance analysis"""
    script_name: str
    timeframe: str
    symbol: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    alert_count: int
    error_count: int
    warning_count: int
    last_execution: datetime
    performance_score: float
    health_status: str

@dataclass
class AlertAnalysis:
    """Analysis of Pine Script alerts"""
    alert_type: str
    frequency: int
    accuracy_rate: float
    avg_execution_time: float
    success_rate: float
    error_patterns: List[str]
    performance_trend: str
    recommendations: List[str]

@dataclass
class PineHealthReport:
    """Comprehensive Pine Script health report"""
    overall_health: str
    health_score: float
    total_scripts: int
    active_scripts: int
    error_rate: float
    avg_performance: float
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    metrics_by_script: Dict[str, PineScriptMetrics]
    alert_analysis: Dict[str, AlertAnalysis]
    trend_analysis: Dict[str, Any]
    timestamp: datetime

class PineScriptHealthMonitor:
    """Advanced Pine Script health monitoring system"""
    
    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history = defaultdict(lambda: deque(maxlen=500))
        self.error_patterns = defaultdict(int)
        self.performance_baselines = {}
        self.monitoring_active = False
        
    async def start_monitoring(self):
        """Start continuous Pine Script monitoring"""
        self.monitoring_active = True
        logger.info("🌲 Pine Script Health Monitor started")
        
        # Start background monitoring tasks
        asyncio.create_task(self._continuous_monitoring())
        asyncio.create_task(self._performance_analysis())
        asyncio.create_task(self._error_pattern_detection())
        
    async def stop_monitoring(self):
        """Stop Pine Script monitoring"""
        self.monitoring_active = False
        logger.info("🛑 Pine Script Health Monitor stopped")
        
    async def record_script_execution(self, script_data: Dict[str, Any]):
        """Record Pine Script execution metrics"""
        try:
            script_name = script_data.get('script_name', 'unknown')
            
            metrics = PineScriptMetrics(
                script_name=script_name,
                timeframe=script_data.get('timeframe', '1m'),
                symbol=script_data.get('symbol', 'UNKNOWN'),
                execution_time_ms=script_data.get('execution_time', 0.0),
                memory_usage_mb=script_data.get('memory_usage', 0.0),
                cpu_usage_percent=script_data.get('cpu_usage', 0.0),
                alert_count=script_data.get('alert_count', 0),
                error_count=script_data.get('error_count', 0),
                warning_count=script_data.get('warning_count', 0),
                last_execution=datetime.now(),
                performance_score=self._calculate_performance_score(script_data),
                health_status=self._determine_health_status(script_data)
            )
            
            # Store metrics
            self.metrics_history[script_name].append(metrics)
            
            # Update baselines
            await self._update_performance_baselines(script_name, metrics)
            
            # Log execution trace for diagnostics
            from database import log_execution_trace
            await log_execution_trace(
                component="pine_script",
                operation="script_execution",
                input_data=script_data,
                output_data=asdict(metrics),
                execution_time=metrics.execution_time_ms,
                success=metrics.error_count == 0
            )
            
        except Exception as e:
            logger.error(f"Error recording script execution: {e}")
            
    async def record_alert(self, alert_data: Dict[str, Any]):
        """Record Pine Script alert for analysis"""
        try:
            alert_type = alert_data.get('alert_type', 'unknown')
            script_name = alert_data.get('script_name', 'unknown')
            
            alert_record = {
                'timestamp': datetime.now(),
                'alert_type': alert_type,
                'script_name': script_name,
                'symbol': alert_data.get('symbol', 'UNKNOWN'),
                'timeframe': alert_data.get('timeframe', '1m'),
                'execution_time': alert_data.get('execution_time', 0.0),
                'success': alert_data.get('success', True),
                'error_message': alert_data.get('error_message', ''),
                'alert_data': alert_data
            }
            
            self.alert_history[f"{script_name}_{alert_type}"].append(alert_record)
            
            # Track error patterns
            if not alert_record['success'] and alert_record['error_message']:
                self.error_patterns[alert_record['error_message']] += 1
                
        except Exception as e:
            logger.error(f"Error recording alert: {e}")
            
    async def analyze_script_performance(self, script_name: str) -> Dict[str, Any]:
        """Analyze performance of specific Pine Script"""
        try:
            if script_name not in self.metrics_history:
                return {"error": f"No data found for script: {script_name}"}
                
            metrics = list(self.metrics_history[script_name])
            if not metrics:
                return {"error": f"No metrics available for script: {script_name}"}
                
            # Calculate performance statistics
            execution_times = [m.execution_time_ms for m in metrics]
            performance_scores = [m.performance_score for m in metrics]
            error_counts = [m.error_count for m in metrics]
            
            analysis = {
                'script_name': script_name,
                'total_executions': len(metrics),
                'avg_execution_time': statistics.mean(execution_times),
                'median_execution_time': statistics.median(execution_times),
                'max_execution_time': max(execution_times),
                'min_execution_time': min(execution_times),
                'avg_performance_score': statistics.mean(performance_scores),
                'error_rate': sum(error_counts) / len(metrics) if metrics else 0,
                'health_trend': self._calculate_health_trend(metrics),
                'performance_trend': self._calculate_performance_trend(performance_scores),
                'recommendations': await self._generate_script_recommendations(script_name, metrics)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing script performance: {e}")
            return {"error": str(e)}
            
    async def analyze_alert_patterns(self) -> Dict[str, AlertAnalysis]:
        """Analyze Pine Script alert patterns"""
        try:
            alert_analyses = {}
            
            for alert_key, alerts in self.alert_history.items():
                if not alerts:
                    continue
                    
                alerts_list = list(alerts)
                script_name, alert_type = alert_key.split('_', 1)
                
                # Calculate metrics
                total_alerts = len(alerts_list)
                successful_alerts = sum(1 for a in alerts_list if a['success'])
                execution_times = [a['execution_time'] for a in alerts_list]
                
                # Detect error patterns
                error_patterns = []
                error_messages = [a['error_message'] for a in alerts_list if not a['success']]
                for error_msg in set(error_messages):
                    if error_messages.count(error_msg) > 1:
                        error_patterns.append(f"{error_msg} (occurred {error_messages.count(error_msg)} times)")
                
                # Generate recommendations
                recommendations = await self._generate_alert_recommendations(alert_key, alerts_list)
                
                alert_analyses[alert_key] = AlertAnalysis(
                    alert_type=alert_type,
                    frequency=total_alerts,
                    accuracy_rate=successful_alerts / total_alerts if total_alerts > 0 else 0,
                    avg_execution_time=statistics.mean(execution_times) if execution_times else 0,
                    success_rate=successful_alerts / total_alerts if total_alerts > 0 else 0,
                    error_patterns=error_patterns,
                    performance_trend=self._calculate_alert_trend(alerts_list),
                    recommendations=recommendations
                )
                
            return alert_analyses
            
        except Exception as e:
            logger.error(f"Error analyzing alert patterns: {e}")
            return {}
            
    async def generate_health_report(self) -> PineHealthReport:
        """Generate comprehensive Pine Script health report"""
        try:
            # Collect all current metrics
            all_metrics = {}
            total_scripts = 0
            active_scripts = 0
            total_errors = 0
            total_executions = 0
            performance_scores = []
            
            for script_name, metrics_deque in self.metrics_history.items():
                if not metrics_deque:
                    continue
                    
                latest_metric = metrics_deque[-1]
                all_metrics[script_name] = latest_metric
                total_scripts += 1
                
                # Check if script is active (executed in last 10 minutes)
                if (datetime.now() - latest_metric.last_execution).seconds < 600:
                    active_scripts += 1
                    
                # Collect statistics
                for metric in metrics_deque:
                    total_errors += metric.error_count
                    total_executions += 1
                    performance_scores.append(metric.performance_score)
                    
            # Calculate overall metrics
            error_rate = total_errors / total_executions if total_executions > 0 else 0
            avg_performance = statistics.mean(performance_scores) if performance_scores else 0
            
            # Determine overall health
            health_score = self._calculate_overall_health_score(error_rate, avg_performance, active_scripts, total_scripts)
            overall_health = self._determine_overall_health_status(health_score)
            
            # Identify critical issues and warnings
            critical_issues = await self._identify_critical_issues(all_metrics)
            warnings = await self._identify_warnings(all_metrics)
            recommendations = await self._generate_overall_recommendations(all_metrics, error_rate, avg_performance)
            
            # Analyze alerts
            alert_analysis = await self.analyze_alert_patterns()
            
            # Generate trend analysis
            trend_analysis = await self._generate_trend_analysis()
            
            report = PineHealthReport(
                overall_health=overall_health,
                health_score=health_score,
                total_scripts=total_scripts,
                active_scripts=active_scripts,
                error_rate=error_rate,
                avg_performance=avg_performance,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations,
                metrics_by_script=all_metrics,
                alert_analysis=alert_analysis,
                trend_analysis=trend_analysis,
                timestamp=datetime.now()
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return PineHealthReport(
                overall_health="ERROR",
                health_score=0.0,
                total_scripts=0,
                active_scripts=0,
                error_rate=1.0,
                avg_performance=0.0,
                critical_issues=[f"Error generating report: {str(e)}"],
                warnings=[],
                recommendations=["Check Pine Script Health Monitor logs"],
                metrics_by_script={},
                alert_analysis={},
                trend_analysis={},
                timestamp=datetime.now()
            )
            
    async def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time Pine Script status"""
        try:
            current_time = datetime.now()
            status = {
                'timestamp': current_time,
                'monitoring_active': self.monitoring_active,
                'scripts_monitored': len(self.metrics_history),
                'active_scripts': 0,
                'recent_errors': 0,
                'avg_performance_last_hour': 0.0,
                'alert_frequency_last_hour': 0,
                'top_performing_scripts': [],
                'problematic_scripts': []
            }
            
            # Analyze recent activity (last hour)
            one_hour_ago = current_time - timedelta(hours=1)
            recent_performance_scores = []
            recent_alerts = 0
            
            for script_name, metrics_deque in self.metrics_history.items():
                recent_metrics = [m for m in metrics_deque if m.last_execution > one_hour_ago]
                
                if recent_metrics:
                    status['active_scripts'] += 1
                    
                    # Count recent errors
                    status['recent_errors'] += sum(m.error_count for m in recent_metrics)
                    
                    # Collect performance scores
                    recent_performance_scores.extend([m.performance_score for m in recent_metrics])
                    
                    # Identify top performing and problematic scripts
                    avg_score = statistics.mean([m.performance_score for m in recent_metrics])
                    if avg_score > 0.8:
                        status['top_performing_scripts'].append({
                            'script': script_name,
                            'score': avg_score
                        })
                    elif avg_score < 0.5:
                        status['problematic_scripts'].append({
                            'script': script_name,
                            'score': avg_score
                        })
                        
            # Count recent alerts
            for alert_deque in self.alert_history.values():
                recent_alerts += len([a for a in alert_deque if a['timestamp'] > one_hour_ago])
                
            status['avg_performance_last_hour'] = statistics.mean(recent_performance_scores) if recent_performance_scores else 0.0
            status['alert_frequency_last_hour'] = recent_alerts
            
            # Sort top performers and problematic scripts
            status['top_performing_scripts'] = sorted(status['top_performing_scripts'], key=lambda x: x['score'], reverse=True)[:5]
            status['problematic_scripts'] = sorted(status['problematic_scripts'], key=lambda x: x['score'])[:5]
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting real-time status: {e}")
            return {'error': str(e)}
            
    # Private helper methods
    
    def _calculate_performance_score(self, script_data: Dict[str, Any]) -> float:
        """Calculate performance score for Pine Script execution"""
        try:
            # Base score
            score = 1.0
            
            # Penalize high execution time
            execution_time = script_data.get('execution_time', 0)
            if execution_time > 1000:  # > 1 second
                score -= 0.3
            elif execution_time > 500:  # > 0.5 second
                score -= 0.1
                
            # Penalize errors
            error_count = script_data.get('error_count', 0)
            score -= error_count * 0.2
            
            # Penalize warnings
            warning_count = script_data.get('warning_count', 0)
            score -= warning_count * 0.05
            
            # Penalize high resource usage
            cpu_usage = script_data.get('cpu_usage', 0)
            if cpu_usage > 80:
                score -= 0.2
            elif cpu_usage > 50:
                score -= 0.1
                
            memory_usage = script_data.get('memory_usage', 0)
            if memory_usage > 100:  # > 100MB
                score -= 0.1
                
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default score on error
            
    def _determine_health_status(self, script_data: Dict[str, Any]) -> str:
        """Determine health status based on script data"""
        score = self._calculate_performance_score(script_data)
        
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.6:
            return "GOOD"
        elif score >= 0.4:
            return "FAIR"
        elif score >= 0.2:
            return "POOR"
        else:
            return "CRITICAL"
            
    async def _update_performance_baselines(self, script_name: str, metrics: PineScriptMetrics):
        """Update performance baselines for script"""
        try:
            if script_name not in self.performance_baselines:
                self.performance_baselines[script_name] = {
                    'avg_execution_time': metrics.execution_time_ms,
                    'avg_performance_score': metrics.performance_score,
                    'baseline_established': datetime.now()
                }
            else:
                # Update rolling average
                baseline = self.performance_baselines[script_name]
                baseline['avg_execution_time'] = (baseline['avg_execution_time'] + metrics.execution_time_ms) / 2
                baseline['avg_performance_score'] = (baseline['avg_performance_score'] + metrics.performance_score) / 2
                
        except Exception as e:
            logger.error(f"Error updating baselines: {e}")
            
    def _calculate_health_trend(self, metrics: List[PineScriptMetrics]) -> str:
        """Calculate health trend for script"""
        try:
            if len(metrics) < 5:
                return "INSUFFICIENT_DATA"
                
            recent_scores = [m.performance_score for m in metrics[-5:]]
            older_scores = [m.performance_score for m in metrics[-10:-5]] if len(metrics) >= 10 else recent_scores
            
            recent_avg = statistics.mean(recent_scores)
            older_avg = statistics.mean(older_scores)
            
            if recent_avg > older_avg + 0.1:
                return "IMPROVING"
            elif recent_avg < older_avg - 0.1:
                return "DECLINING"
            else:
                return "STABLE"
                
        except Exception:
            return "UNKNOWN"
            
    def _calculate_performance_trend(self, scores: List[float]) -> str:
        """Calculate performance trend from scores"""
        try:
            if len(scores) < 3:
                return "INSUFFICIENT_DATA"
                
            # Simple linear trend calculation
            x = list(range(len(scores)))
            y = scores
            
            # Calculate slope
            n = len(scores)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            if slope > 0.01:
                return "IMPROVING"
            elif slope < -0.01:
                return "DECLINING"
            else:
                return "STABLE"
                
        except Exception:
            return "UNKNOWN"
            
    def _calculate_alert_trend(self, alerts: List[Dict[str, Any]]) -> str:
        """Calculate trend for alert performance"""
        try:
            if len(alerts) < 5:
                return "INSUFFICIENT_DATA"
                
            recent_success_rate = sum(1 for a in alerts[-5:] if a['success']) / 5
            older_success_rate = sum(1 for a in alerts[-10:-5] if a['success']) / 5 if len(alerts) >= 10 else recent_success_rate
            
            if recent_success_rate > older_success_rate + 0.1:
                return "IMPROVING"
            elif recent_success_rate < older_success_rate - 0.1:
                return "DECLINING"
            else:
                return "STABLE"
                
        except Exception:
            return "UNKNOWN"
            
    async def _generate_script_recommendations(self, script_name: str, metrics: List[PineScriptMetrics]) -> List[str]:
        """Generate recommendations for specific script"""
        recommendations = []
        
        try:
            if not metrics:
                return ["No data available for analysis"]
                
            latest = metrics[-1]
            
            # Performance recommendations
            if latest.execution_time_ms > 1000:
                recommendations.append("Consider optimizing script execution time (currently > 1s)")
                
            if latest.performance_score < 0.5:
                recommendations.append("Script performance is below acceptable threshold")
                
            if latest.error_count > 0:
                recommendations.append("Address script errors to improve reliability")
                
            if latest.cpu_usage_percent > 80:
                recommendations.append("High CPU usage detected - consider optimization")
                
            if latest.memory_usage_mb > 100:
                recommendations.append("High memory usage detected - check for memory leaks")
                
            # Trend-based recommendations
            trend = self._calculate_health_trend(metrics)
            if trend == "DECLINING":
                recommendations.append("Performance is declining - investigate recent changes")
                
            if not recommendations:
                recommendations.append("Script is performing well - no immediate action needed")
                
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
            
        return recommendations
        
    async def _generate_alert_recommendations(self, alert_key: str, alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for alert patterns"""
        recommendations = []
        
        try:
            if not alerts:
                return ["No alert data available"]
                
            success_rate = sum(1 for a in alerts if a['success']) / len(alerts)
            avg_execution_time = statistics.mean([a['execution_time'] for a in alerts])
            
            if success_rate < 0.8:
                recommendations.append(f"Alert success rate is low ({success_rate:.1%}) - investigate failures")
                
            if avg_execution_time > 500:
                recommendations.append("Alert execution time is high - consider optimization")
                
            # Check for error patterns
            error_messages = [a['error_message'] for a in alerts if not a['success']]
            if error_messages:
                common_errors = {}
                for error in error_messages:
                    common_errors[error] = common_errors.get(error, 0) + 1
                    
                for error, count in common_errors.items():
                    if count > 1:
                        recommendations.append(f"Recurring error pattern: {error} ({count} times)")
                        
            if not recommendations:
                recommendations.append("Alert performance is satisfactory")
                
        except Exception as e:
            recommendations.append(f"Error analyzing alerts: {str(e)}")
            
        return recommendations
        
    def _calculate_overall_health_score(self, error_rate: float, avg_performance: float, active_scripts: int, total_scripts: int) -> float:
        """Calculate overall health score"""
        try:
            # Base score from performance
            score = avg_performance
            
            # Penalize high error rate
            score -= error_rate * 0.5
            
            # Penalize inactive scripts
            if total_scripts > 0:
                activity_ratio = active_scripts / total_scripts
                score *= activity_ratio
                
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0
            
    def _determine_overall_health_status(self, health_score: float) -> str:
        """Determine overall health status"""
        if health_score >= 0.9:
            return "EXCELLENT"
        elif health_score >= 0.7:
            return "GOOD"
        elif health_score >= 0.5:
            return "FAIR"
        elif health_score >= 0.3:
            return "POOR"
        else:
            return "CRITICAL"
            
    async def _identify_critical_issues(self, metrics: Dict[str, PineScriptMetrics]) -> List[str]:
        """Identify critical issues from metrics"""
        issues = []
        
        try:
            for script_name, metric in metrics.items():
                if metric.health_status == "CRITICAL":
                    issues.append(f"Script {script_name} is in critical condition")
                    
                if metric.error_count > 5:
                    issues.append(f"High error count in {script_name}: {metric.error_count} errors")
                    
                if metric.execution_time_ms > 2000:
                    issues.append(f"Extremely slow execution in {script_name}: {metric.execution_time_ms}ms")
                    
                # Check if script hasn't executed recently
                if (datetime.now() - metric.last_execution).seconds > 3600:  # 1 hour
                    issues.append(f"Script {script_name} hasn't executed in over an hour")
                    
        except Exception as e:
            issues.append(f"Error identifying critical issues: {str(e)}")
            
        return issues
        
    async def _identify_warnings(self, metrics: Dict[str, PineScriptMetrics]) -> List[str]:
        """Identify warnings from metrics"""
        warnings = []
        
        try:
            for script_name, metric in metrics.items():
                if metric.health_status in ["POOR", "FAIR"]:
                    warnings.append(f"Script {script_name} performance is suboptimal")
                    
                if metric.warning_count > 0:
                    warnings.append(f"Script {script_name} has {metric.warning_count} warnings")
                    
                if metric.cpu_usage_percent > 70:
                    warnings.append(f"High CPU usage in {script_name}: {metric.cpu_usage_percent}%")
                    
                if metric.memory_usage_mb > 80:
                    warnings.append(f"High memory usage in {script_name}: {metric.memory_usage_mb}MB")
                    
        except Exception as e:
            warnings.append(f"Error identifying warnings: {str(e)}")
            
        return warnings
        
    async def _generate_overall_recommendations(self, metrics: Dict[str, PineScriptMetrics], error_rate: float, avg_performance: float) -> List[str]:
        """Generate overall system recommendations"""
        recommendations = []
        
        try:
            if error_rate > 0.1:
                recommendations.append("High system error rate detected - review all scripts for issues")
                
            if avg_performance < 0.6:
                recommendations.append("Overall system performance is below optimal - consider optimization")
                
            if len(metrics) == 0:
                recommendations.append("No active Pine Scripts detected - check monitoring configuration")
                
            # Resource usage recommendations
            high_cpu_scripts = [name for name, m in metrics.items() if m.cpu_usage_percent > 80]
            if high_cpu_scripts:
                recommendations.append(f"High CPU usage scripts need optimization: {', '.join(high_cpu_scripts)}")
                
            high_memory_scripts = [name for name, m in metrics.items() if m.memory_usage_mb > 100]
            if high_memory_scripts:
                recommendations.append(f"High memory usage scripts need review: {', '.join(high_memory_scripts)}")
                
            if not recommendations:
                recommendations.append("System is operating within normal parameters")
                
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
            
        return recommendations
        
    async def _generate_trend_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive trend analysis"""
        try:
            trend_analysis = {
                'performance_trends': {},
                'error_trends': {},
                'resource_trends': {},
                'alert_trends': {},
                'overall_trend': 'STABLE'
            }
            
            # Analyze performance trends for each script
            for script_name, metrics_deque in self.metrics_history.items():
                if len(metrics_deque) >= 5:
                    metrics_list = list(metrics_deque)
                    trend_analysis['performance_trends'][script_name] = self._calculate_health_trend(metrics_list)
                    
            # Analyze error trends
            current_time = datetime.now()
            for script_name, metrics_deque in self.metrics_history.items():
                recent_errors = sum(m.error_count for m in metrics_deque if (current_time - m.last_execution).seconds < 3600)
                older_errors = sum(m.error_count for m in metrics_deque if 3600 <= (current_time - m.last_execution).seconds < 7200)
                
                if recent_errors > older_errors:
                    trend_analysis['error_trends'][script_name] = 'INCREASING'
                elif recent_errors < older_errors:
                    trend_analysis['error_trends'][script_name] = 'DECREASING'
                else:
                    trend_analysis['error_trends'][script_name] = 'STABLE'
                    
            # Analyze alert trends
            for alert_key, alerts_deque in self.alert_history.items():
                if len(alerts_deque) >= 5:
                    alerts_list = list(alerts_deque)
                    trend_analysis['alert_trends'][alert_key] = self._calculate_alert_trend(alerts_list)
                    
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {e}")
            return {'error': str(e)}
            
    async def _continuous_monitoring(self):
        """Background task for continuous monitoring"""
        while self.monitoring_active:
            try:
                # Perform periodic health checks
                await asyncio.sleep(60)  # Check every minute
                
                # Clean old data
                await self._cleanup_old_data()
                
                # Update performance baselines
                await self._update_all_baselines()
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)
                
    async def _performance_analysis(self):
        """Background task for performance analysis"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
                # Perform performance analysis
                for script_name in self.metrics_history.keys():
                    analysis = await self.analyze_script_performance(script_name)
                    
                    # Log significant performance changes
                    if 'error' not in analysis:
                        if analysis.get('performance_trend') == 'DECLINING':
                            logger.warning(f"Performance declining for script: {script_name}")
                            
            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
                await asyncio.sleep(300)
                
    async def _error_pattern_detection(self):
        """Background task for error pattern detection"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(600)  # Check every 10 minutes
                
                # Analyze error patterns
                for error_msg, count in self.error_patterns.items():
                    if count > 5:  # Threshold for concerning pattern
                        logger.warning(f"Recurring error pattern detected: {error_msg} (occurred {count} times)")
                        
            except Exception as e:
                logger.error(f"Error in pattern detection: {e}")
                await asyncio.sleep(600)
                
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of data
            
            # Clean metrics history
            for script_name, metrics_deque in self.metrics_history.items():
                # Convert to list, filter, and convert back
                filtered_metrics = [m for m in metrics_deque if m.last_execution > cutoff_time]
                metrics_deque.clear()
                metrics_deque.extend(filtered_metrics)
                
            # Clean alert history
            for alert_key, alerts_deque in self.alert_history.items():
                filtered_alerts = [a for a in alerts_deque if a['timestamp'] > cutoff_time]
                alerts_deque.clear()
                alerts_deque.extend(filtered_alerts)
                
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
            
    async def _update_all_baselines(self):
        """Update performance baselines for all scripts"""
        try:
            for script_name, metrics_deque in self.metrics_history.items():
                if metrics_deque:
                    latest_metric = metrics_deque[-1]
                    await self._update_performance_baselines(script_name, latest_metric)
                    
        except Exception as e:
            logger.error(f"Error updating baselines: {e}")

# Global monitor instance
pine_monitor = PineScriptHealthMonitor()

# Convenience functions for easy integration
async def start_pine_monitoring():
    """Start Pine Script monitoring"""
    await pine_monitor.start_monitoring()

async def stop_pine_monitoring():
    """Stop Pine Script monitoring"""
    await pine_monitor.stop_monitoring()

async def record_pine_execution(script_data: Dict[str, Any]):
    """Record Pine Script execution"""
    await pine_monitor.record_script_execution(script_data)

async def record_pine_alert(alert_data: Dict[str, Any]):
    """Record Pine Script alert"""
    await pine_monitor.record_alert(alert_data)

async def get_pine_health_report() -> PineHealthReport:
    """Get comprehensive Pine Script health report"""
    return await pine_monitor.generate_health_report()

async def get_pine_real_time_status() -> Dict[str, Any]:
    """Get real-time Pine Script status"""
    return await pine_monitor.get_real_time_status()

async def analyze_pine_script(script_name: str) -> Dict[str, Any]:
    """Analyze specific Pine Script performance"""
    return await pine_monitor.analyze_script_performance(script_name)

if __name__ == "__main__":
    # Test Pine Script Health Monitor
    async def test_pine_monitor():
        print("🌲 Testing Pine Script Health Monitor...")
        
        # Start monitoring
        await start_pine_monitoring()
        
        # Simulate some script executions
        test_scripts = [
            {
                'script_name': 'RSI_Strategy',
                'timeframe': '5m',
                'symbol': 'BTCUSDT',
                'execution_time': 150.0,
                'memory_usage': 45.0,
                'cpu_usage': 25.0,
                'alert_count': 1,
                'error_count': 0,
                'warning_count': 0
            },
            {
                'script_name': 'MACD_Signal',
                'timeframe': '1h',
                'symbol': 'ETHUSDT',
                'execution_time': 800.0,
                'memory_usage': 120.0,
                'cpu_usage': 85.0,
                'alert_count': 0,
                'error_count': 2,
                'warning_count': 1
            }
        ]
        
        # Record executions
        for script_data in test_scripts:
            await record_pine_execution(script_data)
            
        # Record some alerts
        await record_pine_alert({
            'alert_type': 'BUY',
            'script_name': 'RSI_Strategy',
            'symbol': 'BTCUSDT',
            'timeframe': '5m',
            'execution_time': 50.0,
            'success': True
        })
        
        await record_pine_alert({
            'alert_type': 'SELL',
            'script_name': 'MACD_Signal',
            'symbol': 'ETHUSDT',
            'timeframe': '1h',
            'execution_time': 200.0,
            'success': False,
            'error_message': 'Connection timeout'
        })
        
        # Get real-time status
        status = await get_pine_real_time_status()
        print(f"Real-time status: {status['active_scripts']} active scripts")
        
        # Analyze specific script
        analysis = await analyze_pine_script('RSI_Strategy')
        print(f"RSI Strategy analysis: {analysis.get('avg_performance_score', 'N/A')}")
        
        # Generate health report
        report = await get_pine_health_report()
        print(f"Health report: {report.overall_health} (Score: {report.health_score:.2f})")
        print(f"Critical issues: {len(report.critical_issues)}")
        print(f"Recommendations: {len(report.recommendations)}")
        
        # Stop monitoring
        await stop_pine_monitoring()
        
        print("✅ Pine Script Health Monitor test completed!")

    # Run the test
    import asyncio
    asyncio.run(test_pine_monitor())
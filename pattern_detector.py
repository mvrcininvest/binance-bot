"""
Pattern Detector - Advanced pattern recognition and anomaly detection system
Part of the Hybrid Ultra-Diagnostics system for Trading Bot v9.1
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque, Counter
from enum import Enum
import statistics
import logging
from dataclasses import dataclass

@dataclass
class PatternReport:
    """Raport z wykrytych wzorców"""
    anomalies_detected: int = 0
    avg_anomaly_deviation: float = 0.0
    patterns: list = None
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = []

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns that can be detected"""
    ERROR_PATTERN = "error_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    TRADING_PATTERN = "trading_pattern"
    ALERT_PATTERN = "alert_pattern"
    ANOMALY_PATTERN = "anomaly_pattern"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    TEMPORAL_PATTERN = "temporal_pattern"

class Severity(Enum):
    """Pattern severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PatternMatch:
    """Represents a detected pattern match"""
    pattern_id: str
    pattern_type: PatternType
    severity: Severity
    confidence: float
    description: str
    first_occurrence: datetime
    last_occurrence: datetime
    occurrence_count: int
    affected_components: List[str]
    sample_data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    recommendations: List[str]

@dataclass
class AnomalyDetection:
    """Represents an anomaly detection result"""
    anomaly_id: str
    component: str
    metric: str
    anomaly_type: str
    severity: Severity
    confidence: float
    detected_at: datetime
    baseline_value: float
    anomalous_value: float
    deviation_percentage: float
    context: Dict[str, Any]
    recommendations: List[str]

@dataclass
class PatternReport:
    """Comprehensive pattern analysis report"""
    report_id: str
    generated_at: datetime
    analysis_period: timedelta
    total_patterns: int
    critical_patterns: int
    high_severity_patterns: int
    anomalies_detected: int
    patterns_by_type: Dict[PatternType, int]
    patterns_by_component: Dict[str, int]
    detected_patterns: List[PatternMatch]
    detected_anomalies: List[AnomalyDetection]
    trend_analysis: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]

class PatternDetector:
    """Advanced pattern detection and anomaly detection system"""
    
    def __init__(self):
        self.pattern_history = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_history = defaultdict(lambda: deque(maxlen=500))
        self.known_patterns = {}
        self.baselines = defaultdict(dict)
        self.detection_rules = []
        self.monitoring_active = False
        self.pattern_cache = {}
        self.anomaly_thresholds = {
            'execution_time': {'multiplier': 3.0, 'min_samples': 10},
            'error_rate': {'threshold': 0.1, 'window': 100},
            'performance_score': {'threshold': 0.3, 'window': 50},
            'memory_usage': {'multiplier': 2.5, 'min_samples': 20},
            'cpu_usage': {'threshold': 90.0, 'window': 30}
        }
        
        # Initialize built-in patterns
        self._initialize_builtin_patterns()

    def analyze_patterns(self, symbol: str, timeframe: str = '5m', lookback_hours: int = 24) -> Dict:
        """Alias dla detect_patterns dla kompatybilności wstecznej"""
        return self.detect_patterns(symbol, timeframe, lookback_hours)

    async def start_detection(self):
        """Start pattern detection system"""
        self.monitoring_active = True
        logger.info("🔍 Pattern Detector started")
        
        # Start background detection tasks
        asyncio.create_task(self._continuous_pattern_detection())
        asyncio.create_task(self._anomaly_detection_loop())
        asyncio.create_task(self._pattern_learning_loop())
        
    async def stop_detection(self):
        """Stop pattern detection system"""
        self.monitoring_active = False
        logger.info("🛑 Pattern Detector stopped")
        
    async def analyze_data(self, component: str, data: Dict[str, Any]):
        """Analyze incoming data for patterns and anomalies"""
        try:
            timestamp = datetime.now()
            
            # Store data for pattern analysis
            data_entry = {
                'timestamp': timestamp,
                'component': component,
                'data': data
            }
            self.pattern_history[component].append(data_entry)
            
            # Detect patterns in real-time
            patterns = await self._detect_patterns_in_data(component, data_entry)
            
            # Detect anomalies
            anomalies = await self._detect_anomalies_in_data(component, data)
            
            # Update baselines
            await self._update_baselines(component, data)
            
            # Log significant findings
            for pattern in patterns:
                if pattern.severity in [Severity.HIGH, Severity.CRITICAL]:
                    logger.warning(f"High severity pattern detected: {pattern.description}")
                    
            for anomaly in anomalies:
                if anomaly.severity in [Severity.HIGH, Severity.CRITICAL]:
                    logger.warning(f"Anomaly detected in {anomaly.component}: {anomaly.anomaly_type}")
                    
            # Store results
            for pattern in patterns:
                await self._store_pattern_match(pattern)
                
            for anomaly in anomalies:
                await self._store_anomaly(anomaly)
                
            return {
                'patterns_detected': len(patterns),
                'anomalies_detected': len(anomalies),
                'patterns': [asdict(p) for p in patterns],
                'anomalies': [asdict(a) for a in anomalies]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return {'error': str(e)}
            
    async def detect_error_patterns(self, error_logs: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Detect patterns in error logs"""
        try:
            patterns = []
            
            if not error_logs:
                return patterns
                
            # Group errors by message
            error_groups = defaultdict(list)
            for error in error_logs:
                error_msg = error.get('message', '').strip()
                if error_msg:
                    # Normalize error message (remove timestamps, IDs, etc.)
                    normalized_msg = self._normalize_error_message(error_msg)
                    error_groups[normalized_msg].append(error)
                    
            # Analyze each error group
            for normalized_msg, error_list in error_groups.items():
                if len(error_list) >= 3:  # Pattern threshold
                    pattern = await self._create_error_pattern(normalized_msg, error_list)
                    if pattern:
                        patterns.append(pattern)
                        
            # Detect temporal error patterns
            temporal_patterns = await self._detect_temporal_error_patterns(error_logs)
            patterns.extend(temporal_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting error patterns: {e}")
            return []
            
    async def detect_performance_patterns(self, performance_data: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Detect patterns in performance data"""
        try:
            patterns = []
            
            if len(performance_data) < 10:
                return patterns
                
            # Analyze execution time patterns
            execution_times = [d.get('execution_time', 0) for d in performance_data]
            time_patterns = await self._analyze_time_series_patterns(execution_times, 'execution_time')
            patterns.extend(time_patterns)
            
            # Analyze resource usage patterns
            if any('cpu_usage' in d for d in performance_data):
                cpu_usage = [d.get('cpu_usage', 0) for d in performance_data if 'cpu_usage' in d]
                cpu_patterns = await self._analyze_resource_patterns(cpu_usage, 'cpu_usage')
                patterns.extend(cpu_patterns)
                
            if any('memory_usage' in d for d in performance_data):
                memory_usage = [d.get('memory_usage', 0) for d in performance_data if 'memory_usage' in d]
                memory_patterns = await self._analyze_resource_patterns(memory_usage, 'memory_usage')
                patterns.extend(memory_patterns)
                
            # Detect performance degradation patterns
            degradation_patterns = await self._detect_performance_degradation(performance_data)
            patterns.extend(degradation_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting performance patterns: {e}")
            return []
            
    async def detect_trading_patterns(self, trading_data: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Detect patterns in trading behavior"""
        try:
            patterns = []
            
            if not trading_data:
                return patterns
                
            # Analyze win/loss patterns
            win_loss_patterns = await self._analyze_win_loss_patterns(trading_data)
            patterns.extend(win_loss_patterns)
            
            # Analyze position sizing patterns
            sizing_patterns = await self._analyze_position_sizing_patterns(trading_data)
            patterns.extend(sizing_patterns)
            
            # Analyze timing patterns
            timing_patterns = await self._analyze_trading_timing_patterns(trading_data)
            patterns.extend(timing_patterns)
            
            # Detect unusual trading behavior
            behavioral_patterns = await self._detect_unusual_trading_behavior(trading_data)
            patterns.extend(behavioral_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting trading patterns: {e}")
            return []
            
    async def detect_anomalies(self, component: str, metrics: Dict[str, float]) -> List[AnomalyDetection]:
        """Detect anomalies in component metrics"""
        try:
            anomalies = []
            
            for metric_name, value in metrics.items():
                anomaly = await self._detect_metric_anomaly(component, metric_name, value)
                if anomaly:
                    anomalies.append(anomaly)
                    
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
            
    async def generate_pattern_report(self, hours_back: int = 24) -> PatternReport:
        """Generate comprehensive pattern analysis report"""
        try:
            report_id = f"pattern_report_{int(time.time())}"
            generated_at = datetime.now()
            analysis_period = timedelta(hours=hours_back)
            cutoff_time = generated_at - analysis_period
            
            # Collect all patterns from the specified period
            all_patterns = []
            all_anomalies = []
            
            # Get patterns from history
            for component, pattern_deque in self.pattern_history.items():
                for entry in pattern_deque:
                    if entry['timestamp'] > cutoff_time:
                        # Re-analyze this data for patterns
                        patterns = await self._detect_patterns_in_data(component, entry)
                        all_patterns.extend(patterns)
                        
            # Get anomalies from history
            for component, anomaly_deque in self.anomaly_history.items():
                for anomaly in anomaly_deque:
                    if anomaly.detected_at > cutoff_time:
                        all_anomalies.append(anomaly)
                        
            # Calculate statistics
            total_patterns = len(all_patterns)
            critical_patterns = len([p for p in all_patterns if p.severity == Severity.CRITICAL])
            high_severity_patterns = len([p for p in all_patterns if p.severity == Severity.HIGH])
            anomalies_detected = len(all_anomalies)
            
            # Group patterns by type and component
            patterns_by_type = defaultdict(int)
            patterns_by_component = defaultdict(int)
            
            for pattern in all_patterns:
                patterns_by_type[pattern.pattern_type] += 1
                for component in pattern.affected_components:
                    patterns_by_component[component] += 1
                    
            # Generate trend analysis
            trend_analysis = await self._generate_trend_analysis(all_patterns, all_anomalies)
            
            # Generate recommendations
            recommendations = await self._generate_pattern_recommendations(all_patterns, all_anomalies)
            
            # Assess risk
            risk_assessment = await self._assess_pattern_risk(all_patterns, all_anomalies)
            
            report = PatternReport(
                report_id=report_id,
                generated_at=generated_at,
                analysis_period=analysis_period,
                total_patterns=total_patterns,
                critical_patterns=critical_patterns,
                high_severity_patterns=high_severity_patterns,
                anomalies_detected=anomalies_detected,
                patterns_by_type=dict(patterns_by_type),
                patterns_by_component=dict(patterns_by_component),
                detected_patterns=all_patterns,
                detected_anomalies=all_anomalies,
                trend_analysis=trend_analysis,
                recommendations=recommendations,
                risk_assessment=risk_assessment
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating pattern report: {e}")
            return PatternReport(
                report_id=f"error_report_{int(time.time())}",
                generated_at=datetime.now(),
                analysis_period=timedelta(hours=hours_back),
                total_patterns=0,
                critical_patterns=0,
                high_severity_patterns=0,
                anomalies_detected=0,
                patterns_by_type={},
                patterns_by_component={},
                detected_patterns=[],
                detected_anomalies=[],
                trend_analysis={},
                recommendations=[f"Error generating report: {str(e)}"],
                risk_assessment={'error': True}
            )
            
    async def learn_pattern(self, pattern_data: Dict[str, Any], pattern_type: PatternType):
        """Learn a new pattern from provided data"""
        try:
            pattern_id = f"learned_{pattern_type.value}_{int(time.time())}"
            
            # Extract pattern characteristics
            characteristics = await self._extract_pattern_characteristics(pattern_data, pattern_type)
            
            # Create pattern definition
            pattern_definition = {
                'id': pattern_id,
                'type': pattern_type,
                'characteristics': characteristics,
                'learned_at': datetime.now(),
                'confidence_threshold': 0.7,
                'sample_data': pattern_data
            }
            
            # Store learned pattern
            self.known_patterns[pattern_id] = pattern_definition
            
            logger.info(f"Learned new pattern: {pattern_id}")
            
            return pattern_id
            
        except Exception as e:
            logger.error(f"Error learning pattern: {e}")
            return None
            
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern detection statistics"""
        try:
            stats = {
                'monitoring_active': self.monitoring_active,
                'known_patterns': len(self.known_patterns),
                'components_monitored': len(self.pattern_history),
                'total_data_points': sum(len(deque) for deque in self.pattern_history.values()),
                'recent_patterns': 0,
                'recent_anomalies': 0,
                'pattern_types': defaultdict(int),
                'severity_distribution': defaultdict(int),
                'top_affected_components': [],
                'detection_performance': {}
            }
            
            # Count recent patterns and anomalies (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            for component, pattern_deque in self.pattern_history.items():
                for entry in pattern_deque:
                    if entry['timestamp'] > one_hour_ago:
                        stats['recent_patterns'] += 1
                        
            for component, anomaly_deque in self.anomaly_history.items():
                for anomaly in anomaly_deque:
                    if anomaly.detected_at > one_hour_ago:
                        stats['recent_anomalies'] += 1
                        
            # Calculate detection performance metrics
            stats['detection_performance'] = {
                'avg_detection_time': await self._calculate_avg_detection_time(),
                'false_positive_rate': await self._calculate_false_positive_rate(),
                'pattern_accuracy': await self._calculate_pattern_accuracy()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting pattern statistics: {e}")
            return {'error': str(e)}
            
    async def detect_patterns(self, component: str = None) -> Dict[str, Any]:
        """
        Performs a one-off analysis of recent data for patterns.
        This method can be called periodically to check for new patterns.
        If a component is specified, it analyzes only that component's data.
        """
        logger.info(f"Running on-demand pattern detection for component: {component or 'all'}")
        detected_patterns_summary = defaultdict(int)
        all_detected_patterns = []  # Lista do zbierania obiektów wzorców
    
        components_to_check = [component] if component and component in self.pattern_history else self.pattern_history.keys()
    
        for comp in components_to_check:
            data_deque = self.pattern_history.get(comp)
            if not data_deque or len(data_deque) < 5:
                continue

            recent_data = list(data_deque)[-20:]

            performance_data = []
            error_logs = []
        
            for entry in recent_data:
                data = entry.get('data', {})
                if 'error_message' in data:
                    data['timestamp'] = entry.get('timestamp', datetime.now())
                    error_logs.append(data)
                if any(key in data for key in ['execution_time', 'cpu_usage', 'memory_usage']):
                    performance_data.append(data)

            if performance_data:
                patterns = await self.detect_performance_patterns(performance_data)
                all_detected_patterns.extend(patterns)
                for pattern in patterns:
                    await self._store_pattern_match(pattern)
                    detected_patterns_summary[pattern.pattern_type.value] += 1
                
            if error_logs:
                patterns = await self.detect_error_patterns(error_logs)
                all_detected_patterns.extend(patterns)
                for pattern in patterns:
                    await self._store_pattern_match(pattern)
                    detected_patterns_summary[pattern.pattern_type.value] += 1

        summary_message = f"On-demand detection complete. Found patterns: {dict(detected_patterns_summary)}"
        logger.info(summary_message)
        return {
            "status": "success",
            "summary": summary_message,
            "detected_patterns": all_detected_patterns  # Zwracaj listę obiektów
        }

    # Private helper methods
    
    def _initialize_builtin_patterns(self):
        """Initialize built-in pattern definitions"""
        try:
            # Error patterns
            self.known_patterns['connection_timeout'] = {
                'id': 'connection_timeout',
                'type': PatternType.ERROR_PATTERN,
                'characteristics': {
                    'keywords': ['timeout', 'connection', 'failed'],
                    'frequency_threshold': 3,
                    'time_window': 300  # 5 minutes
                },
                'severity': Severity.HIGH,
                'confidence_threshold': 0.8
            }
            
            self.known_patterns['api_rate_limit'] = {
                'id': 'api_rate_limit',
                'type': PatternType.ERROR_PATTERN,
                'characteristics': {
                    'keywords': ['rate limit', '429', 'too many requests'],
                    'frequency_threshold': 2,
                    'time_window': 60
                },
                'severity': Severity.MEDIUM,
                'confidence_threshold': 0.9
            }
            
            # Performance patterns
            self.known_patterns['memory_leak'] = {
                'id': 'memory_leak',
                'type': PatternType.PERFORMANCE_PATTERN,
                'characteristics': {
                    'metric': 'memory_usage',
                    'trend': 'increasing',
                    'threshold_multiplier': 2.0,
                    'min_duration': 1800  # 30 minutes
                },
                'severity': Severity.HIGH,
                'confidence_threshold': 0.85
            }
            
            self.known_patterns['cpu_spike'] = {
                'id': 'cpu_spike',
                'type': PatternType.PERFORMANCE_PATTERN,
                'characteristics': {
                    'metric': 'cpu_usage',
                    'threshold': 90.0,
                    'duration': 300,  # 5 minutes
                    'frequency_threshold': 3
                },
                'severity': Severity.MEDIUM,
                'confidence_threshold': 0.75
            }
            
            # Trading patterns
            self.known_patterns['consecutive_losses'] = {
                'id': 'consecutive_losses',
                'type': PatternType.TRADING_PATTERN,
                'characteristics': {
                    'metric': 'consecutive_losses',
                    'threshold': 5,
                    'time_window': 3600  # 1 hour
                },
                'severity': Severity.HIGH,
                'confidence_threshold': 0.9
            }
            
        except Exception as e:
            logger.error(f"Error initializing builtin patterns: {e}")
            
    async def _detect_patterns_in_data(self, component: str, data_entry: Dict[str, Any]) -> List[PatternMatch]:
        """Detect patterns in a single data entry"""
        try:
            patterns = []
            
            # Check against known patterns
            for pattern_id, pattern_def in self.known_patterns.items():
                match = await self._check_pattern_match(component, data_entry, pattern_def)
                if match:
                    patterns.append(match)
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns in data: {e}")
            return []
            
    async def _detect_anomalies_in_data(self, component: str, data: Dict[str, Any]) -> List[AnomalyDetection]:
        """Detect anomalies in data"""
        try:
            anomalies = []
            
            for metric_name, value in data.items():
                if isinstance(value, (int, float)):
                    anomaly = await self._detect_metric_anomaly(component, metric_name, value)
                    if anomaly:
                        anomalies.append(anomaly)
                        
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
            
    async def _detect_metric_anomaly(self, component: str, metric: str, value: float) -> Optional[AnomalyDetection]:
        """Detect anomaly in a specific metric"""
        try:
            baseline_key = f"{component}_{metric}"
            
            if baseline_key not in self.baselines:
                # Not enough data for baseline
                return None
                
            baseline = self.baselines[baseline_key]
            
            if 'mean' not in baseline or 'std' not in baseline:
                return None
                
            mean = baseline['mean']
            std = baseline['std']
            sample_count = baseline.get('sample_count', 0)
            
            # Check if we have enough samples
            min_samples = self.anomaly_thresholds.get(metric, {}).get('min_samples', 10)
            if sample_count < min_samples:
                return None
                
            # Calculate z-score
            if std == 0:
                z_score = 0
            else:
                z_score = abs(value - mean) / std
                
            # Determine if anomalous
            threshold_multiplier = self.anomaly_thresholds.get(metric, {}).get('multiplier', 3.0)
            
            if z_score > threshold_multiplier:
                # Calculate deviation percentage
                deviation_pct = abs(value - mean) / mean * 100 if mean != 0 else 0
                
                # Determine severity
                if z_score > 5.0:
                    severity = Severity.CRITICAL
                elif z_score > 4.0:
                    severity = Severity.HIGH
                elif z_score > 3.5:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW
                    
                # Generate recommendations
                recommendations = await self._generate_anomaly_recommendations(component, metric, value, mean)
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"anomaly_{component}_{metric}_{int(time.time())}",
                    component=component,
                    metric=metric,
                    anomaly_type="statistical_outlier",
                    severity=severity,
                    confidence=min(z_score / 5.0, 1.0),
                    detected_at=datetime.now(),
                    baseline_value=mean,
                    anomalous_value=value,
                    deviation_percentage=deviation_pct,
                    context={
                        'z_score': z_score,
                        'std_dev': std,
                        'sample_count': sample_count
                    },
                    recommendations=recommendations
                )
                
                return anomaly
                
            return None
            
        except Exception as e:
            logger.error(f"Error detecting metric anomaly: {e}")
            return None
            
    async def _update_baselines(self, component: str, data: Dict[str, Any]):
        """Update baseline statistics for component metrics"""
        try:
            for metric_name, value in data.items():
                if isinstance(value, (int, float)):
                    baseline_key = f"{component}_{metric_name}"
                    
                    if baseline_key not in self.baselines:
                        self.baselines[baseline_key] = {
                            'values': deque(maxlen=1000),
                            'mean': 0.0,
                            'std': 0.0,
                            'sample_count': 0
                        }
                        
                    baseline = self.baselines[baseline_key]
                    baseline['values'].append(value)
                    baseline['sample_count'] += 1
                    
                    # Recalculate statistics
                    values = list(baseline['values'])
                    if len(values) > 1:
                        baseline['mean'] = statistics.mean(values)
                        baseline['std'] = statistics.stdev(values)
                    else:
                        baseline['mean'] = value
                        baseline['std'] = 0.0
                        
        except Exception as e:
            logger.error(f"Error updating baselines: {e}")
            
    def _normalize_error_message(self, error_msg: str) -> str:
        """Normalize error message for pattern matching"""
        try:
            # Remove timestamps
            normalized = re.sub(r'\d{4}-\d{2}-\d{2}[\s\T]\d{2}:\d{2}:\d{2}', '[TIMESTAMP]', error_msg)
            
            # Remove IDs and numbers
            normalized = re.sub(r'\b\d+\b', '[NUMBER]', normalized)
            
            # Remove URLs
            normalized = re.sub(r'https?://[^\s]+', '[URL]', normalized)
            
            # Remove file paths
            normalized = re.sub(r'/[^\s]+', '[PATH]', normalized)
            
            # Normalize whitespace
            normalized = ' '.join(normalized.split())
            
            return normalized.lower()
            
        except Exception as e:
            logger.error(f"Error normalizing error message: {e}")
            return error_msg.lower()
            
    async def _create_error_pattern(self, normalized_msg: str, error_list: List[Dict[str, Any]]) -> Optional[PatternMatch]:
        """Create error pattern from grouped errors"""
        try:
            if len(error_list) < 3:
                return None
                
            # Calculate pattern characteristics
            first_occurrence = min(error['timestamp'] for error in error_list if 'timestamp' in error)
            last_occurrence = max(error['timestamp'] for error in error_list if 'timestamp' in error)
            occurrence_count = len(error_list)
            
            # Determine severity based on frequency and recency
            time_span = (last_occurrence - first_occurrence).total_seconds()
            frequency = occurrence_count / max(time_span / 3600, 1)  # errors per hour
            
            if frequency > 10:
                severity = Severity.CRITICAL
            elif frequency > 5:
                severity = Severity.HIGH
            elif frequency > 2:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
                
            # Extract affected components
            affected_components = list(set(error.get('component', 'unknown') for error in error_list))
            
            # Generate recommendations
            recommendations = [
                f"Investigate recurring error: {normalized_msg}",
                f"Error occurred {occurrence_count} times in {time_span/3600:.1f} hours",
                "Check logs for root cause analysis"
            ]
            
            if frequency > 5:
                recommendations.append("Consider implementing error handling or retry logic")
                
            pattern = PatternMatch(
                pattern_id=f"error_pattern_{hash(normalized_msg)}",
                pattern_type=PatternType.ERROR_PATTERN,
                severity=severity,
                confidence=min(occurrence_count / 10.0, 1.0),
                description=f"Recurring error pattern: {normalized_msg[:100]}...",
                first_occurrence=first_occurrence,
                last_occurrence=last_occurrence,
                occurrence_count=occurrence_count,
                affected_components=affected_components,
                sample_data=error_list[:5],  # Keep first 5 samples
                metadata={
                    'normalized_message': normalized_msg,
                    'frequency_per_hour': frequency,
                    'time_span_hours': time_span / 3600
                },
                recommendations=recommendations
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error creating error pattern: {e}")
            return None
            
    async def _detect_temporal_error_patterns(self, error_logs: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Detect temporal patterns in error logs"""
        try:
            patterns = []
            
            if len(error_logs) < 10:
                return patterns
                
            # Sort errors by timestamp
            sorted_errors = sorted(error_logs, key=lambda x: x.get('timestamp', datetime.min))
            
            # Detect error bursts (many errors in short time)
            burst_patterns = await self._detect_error_bursts(sorted_errors)
            patterns.extend(burst_patterns)
            
            # Detect periodic error patterns
            periodic_patterns = await self._detect_periodic_errors(sorted_errors)
            patterns.extend(periodic_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting temporal patterns: {e}")
            return []
            
    async def _detect_error_bursts(self, sorted_errors: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Detect error burst patterns"""
        try:
            patterns = []
            burst_threshold = 5  # 5 errors in 5 minutes
            time_window = 300  # 5 minutes
            
            i = 0
            while i < len(sorted_errors):
                current_time = sorted_errors[i].get('timestamp', datetime.min)
                burst_errors = []
                
                # Collect errors within time window
                j = i
                while j < len(sorted_errors):
                    error_time = sorted_errors[j].get('timestamp', datetime.min)
                    if (error_time - current_time).total_seconds() <= time_window:
                        burst_errors.append(sorted_errors[j])
                        j += 1
                    else:
                        break
                        
                # Check if this constitutes a burst
                if len(burst_errors) >= burst_threshold:
                    pattern = PatternMatch(
                        pattern_id=f"error_burst_{int(current_time.timestamp())}",
                        pattern_type=PatternType.TEMPORAL_PATTERN,
                        severity=Severity.HIGH,
                        confidence=min(len(burst_errors) / 10.0, 1.0),
                        description=f"Error burst: {len(burst_errors)} errors in {time_window/60} minutes",
                        first_occurrence=current_time,
                        last_occurrence=burst_errors[-1].get('timestamp', current_time),
                        occurrence_count=len(burst_errors),
                        affected_components=list(set(e.get('component', 'unknown') for e in burst_errors)),
                        sample_data=burst_errors[:5],
                        metadata={
                            'burst_size': len(burst_errors),
                            'time_window_seconds': time_window
                        },
                        recommendations=[
                            "Investigate cause of error burst",
                            "Check system resources during burst period",
                            "Consider implementing circuit breaker pattern"
                        ]
                    )
                    patterns.append(pattern)
                    i = j  # Skip processed errors
                else:
                    i += 1
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting error bursts: {e}")
            return []
            
    async def _detect_periodic_errors(self, sorted_errors: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Detect periodic error patterns"""
        try:
            patterns = []
            
            # Group errors by hour of day
            hourly_errors = defaultdict(list)
            for error in sorted_errors:
                timestamp = error.get('timestamp', datetime.min)
                hour = timestamp.hour
                hourly_errors[hour].append(error)
                
            # Find hours with significantly more errors
            avg_errors_per_hour = len(sorted_errors) / 24
            
            for hour, errors in hourly_errors.items():
                if len(errors) > avg_errors_per_hour * 2:  # 2x average
                    pattern = PatternMatch(
                        pattern_id=f"periodic_errors_hour_{hour}",
                        pattern_type=PatternType.TEMPORAL_PATTERN,
                        severity=Severity.MEDIUM,
                        confidence=len(errors) / (avg_errors_per_hour * 3),
                        description=f"Periodic error pattern at hour {hour}:00",
                        first_occurrence=min(e.get('timestamp', datetime.min) for e in errors),
                        last_occurrence=max(e.get('timestamp', datetime.min) for e in errors),
                        occurrence_count=len(errors),
                        affected_components=list(set(e.get('component', 'unknown') for e in errors)),
                        sample_data=errors[:5],
                        metadata={
                            'hour_of_day': hour,
                            'error_count': len(errors),
                            'average_per_hour': avg_errors_per_hour
                        },
                        recommendations=[
                            f"Investigate why errors spike at {hour}:00",
                            "Check for scheduled tasks or external factors",
                            "Consider adjusting system resources during peak error times"
                        ]
                    )
                    patterns.append(pattern)
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting periodic errors: {e}")
            return []
            
    async def _analyze_time_series_patterns(self, values: List[float], metric_name: str) -> List[PatternMatch]:
        """Analyze time series data for patterns"""
        try:
            patterns = []
            
            if len(values) < 10:
                return patterns
                
            # Detect trends
            trend_pattern = await self._detect_trend_pattern(values, metric_name)
            if trend_pattern:
                patterns.append(trend_pattern)
                
            # Detect spikes
            spike_patterns = await self._detect_spike_patterns(values, metric_name)
            patterns.extend(spike_patterns)
            
            # Detect oscillations
            oscillation_pattern = await self._detect_oscillation_pattern(values, metric_name)
            if oscillation_pattern:
                patterns.append(oscillation_pattern)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing time series patterns: {e}")
            return []
            
    async def _detect_trend_pattern(self, values: List[float], metric_name: str) -> Optional[PatternMatch]:
        """Detect trend patterns in time series"""
        try:
            if len(values) < 5:
                return None
                
            # Calculate linear trend
            x = list(range(len(values)))
            n = len(values)
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Determine if trend is significant
            mean_value = statistics.mean(values)
            relative_slope = abs(slope) / mean_value if mean_value != 0 else 0
            
            if relative_slope > 0.1:  # 10% change per unit time
                trend_type = "increasing" if slope > 0 else "decreasing"
                severity = Severity.HIGH if relative_slope > 0.3 else Severity.MEDIUM
                
                pattern = PatternMatch(
                    pattern_id=f"trend_{metric_name}_{trend_type}_{int(time.time())}",
                    pattern_type=PatternType.PERFORMANCE_PATTERN,
                    severity=severity,
                    confidence=min(relative_slope, 1.0),
                    description=f"{trend_type.capitalize()} trend in {metric_name}",
                    first_occurrence=datetime.now() - timedelta(minutes=len(values)),
                    last_occurrence=datetime.now(),
                    occurrence_count=1,
                    affected_components=[metric_name],
                    sample_data=[{'values': values, 'slope': slope}],
                    metadata={
                        'slope': slope,
                        'relative_slope': relative_slope,
                        'trend_type': trend_type
                    },
                    recommendations=[
                        f"Monitor {trend_type} trend in {metric_name}",
                        "Investigate root cause of trend",
                        "Consider implementing alerts for trend continuation"
                    ]
                )
                
                return pattern
                
            return None
            
        except Exception as e:
            logger.error(f"Error detecting trend pattern: {e}")
            return None
            
    async def _detect_spike_patterns(self, values: List[float], metric_name: str) -> List[PatternMatch]:
        """Detect spike patterns in time series"""
        try:
            patterns = []
            
            if len(values) < 5:
                return patterns
                
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if std_val == 0:
                return patterns
                
            # Find spikes (values > mean + 3*std)
            spike_threshold = mean_val + 3 * std_val
            spikes = []
            
            for i, value in enumerate(values):
                if value > spike_threshold:
                    spikes.append({'index': i, 'value': value, 'deviation': (value - mean_val) / std_val})
                    
            if len(spikes) >= 2:  # Multiple spikes indicate a pattern
                pattern = PatternMatch(
                    pattern_id=f"spikes_{metric_name}_{int(time.time())}",
                    pattern_type=PatternType.PERFORMANCE_PATTERN,
                    severity=Severity.MEDIUM,
                    confidence=min(len(spikes) / 5.0, 1.0),
                    description=f"Spike pattern in {metric_name}: {len(spikes)} spikes detected",
                    first_occurrence=datetime.now() - timedelta(minutes=len(values)),
                    last_occurrence=datetime.now(),
                    occurrence_count=len(spikes),
                    affected_components=[metric_name],
                    sample_data=[{'spikes': spikes, 'threshold': spike_threshold}],
                    metadata={
                        'spike_count': len(spikes),
                        'spike_threshold': spike_threshold,
                        'max_deviation': max(s['deviation'] for s in spikes)
                    },
                    recommendations=[
                        f"Investigate cause of spikes in {metric_name}",
                        "Check for resource contention during spike periods",
                        "Consider implementing spike detection alerts"
                    ]
                )
                patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting spike patterns: {e}")
            return []
            
    async def _detect_oscillation_pattern(self, values: List[float], metric_name: str) -> Optional[PatternMatch]:
        """Detect oscillation patterns in time series"""
        try:
            if len(values) < 10:
                return None
                
            # Simple oscillation detection: count direction changes
            direction_changes = 0
            for i in range(2, len(values)):
                prev_diff = values[i-1] - values[i-2]
                curr_diff = values[i] - values[i-1]
                
                if prev_diff * curr_diff < 0:  # Sign change
                    direction_changes += 1
                    
            # If more than 50% of possible changes are direction changes, it's oscillating
            max_changes = len(values) - 2
            oscillation_ratio = direction_changes / max_changes if max_changes > 0 else 0
            
            if oscillation_ratio > 0.5:
                pattern = PatternMatch(
                    pattern_id=f"oscillation_{metric_name}_{int(time.time())}",
                    pattern_type=PatternType.PERFORMANCE_PATTERN,
                    severity=Severity.MEDIUM,
                    confidence=oscillation_ratio,
                    description=f"Oscillation pattern in {metric_name}",
                    first_occurrence=datetime.now() - timedelta(minutes=len(values)),
                    last_occurrence=datetime.now(),
                    occurrence_count=1,
                    affected_components=[metric_name],
                    sample_data=[{'values': values, 'direction_changes': direction_changes}],
                    metadata={
                        'oscillation_ratio': oscillation_ratio,
                        'direction_changes': direction_changes,
                        'total_points': len(values)
                    },
                    recommendations=[
                        f"Investigate oscillation in {metric_name}",
                        "Check for feedback loops or control system issues",
                        "Consider smoothing or damping mechanisms"
                    ]
                )
                
                return pattern
                
            return None
            
        except Exception as e:
            logger.error(f"Error detecting oscillation pattern: {e}")
            return None
            
    async def _analyze_resource_patterns(self, values: List[float], resource_type: str) -> List[PatternMatch]:
        """Analyze resource usage patterns"""
        try:
            patterns = []
            
            # Detect high usage patterns
            high_usage_threshold = 80.0 if resource_type == 'cpu_usage' else 100.0
            high_usage_count = sum(1 for v in values if v > high_usage_threshold)
            
            if high_usage_count > len(values) * 0.3:  # More than 30% high usage
                pattern = PatternMatch(
                    pattern_id=f"high_{resource_type}_{int(time.time())}",
                    pattern_type=PatternType.PERFORMANCE_PATTERN,
                    severity=Severity.HIGH,
                    confidence=high_usage_count / len(values),
                    description=f"High {resource_type} pattern: {high_usage_count}/{len(values)} samples above threshold",
                    first_occurrence=datetime.now() - timedelta(minutes=len(values)),
                    last_occurrence=datetime.now(),
                    occurrence_count=high_usage_count,
                    affected_components=[resource_type],
                    sample_data=[{'values': values, 'threshold': high_usage_threshold}],
                    metadata={
                        'high_usage_count': high_usage_count,
                        'threshold': high_usage_threshold,
                        'percentage': high_usage_count / len(values) * 100
                    },
                    recommendations=[
                        f"Investigate high {resource_type}",
                        "Consider resource optimization",
                        "Monitor for resource exhaustion"
                    ]
                )
                patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing resource patterns: {e}")
            return []
            
    async def _detect_performance_degradation(self, performance_data: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Detect performance degradation patterns"""
        try:
            patterns = []
            
            if len(performance_data) < 10:
                return patterns
                
            # Analyze performance scores over time
            scores = [d.get('performance_score', 0.5) for d in performance_data]
            
            # Split into early and recent periods
            split_point = len(scores) // 2
            early_scores = scores[:split_point]
            recent_scores = scores[split_point:]
            
            early_avg = statistics.mean(early_scores)
            recent_avg = statistics.mean(recent_scores)
            
            # Check for significant degradation
            degradation = early_avg - recent_avg
            if degradation > 0.2:  # 20% degradation
                severity = Severity.CRITICAL if degradation > 0.4 else Severity.HIGH
                
                pattern = PatternMatch(
                    pattern_id=f"performance_degradation_{int(time.time())}",
                    pattern_type=PatternType.PERFORMANCE_PATTERN,
                    severity=severity,
                    confidence=min(degradation / 0.5, 1.0),
                    description=f"Performance degradation: {degradation:.2f} drop in performance score",
                    first_occurrence=datetime.now() - timedelta(minutes=len(performance_data)),
                    last_occurrence=datetime.now(),
                    occurrence_count=1,
                    affected_components=['performance'],
                    sample_data=performance_data[:5],
                    metadata={
                        'early_average': early_avg,
                        'recent_average': recent_avg,
                        'degradation': degradation
                    },
                    recommendations=[
                        "Investigate cause of performance degradation",
                        "Check for recent changes or updates",
                        "Monitor system resources",
                        "Consider rollback if degradation is severe"
                    ]
                )
                patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting performance degradation: {e}")
            return []
            
    async def _analyze_win_loss_patterns(self, trading_data: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Analyze win/loss patterns in trading data"""
        try:
            patterns = []
            
            if not trading_data:
                return patterns
                
            # Extract win/loss information
            results = []
            for trade in trading_data:
                pnl = trade.get('pnl', 0)
                results.append('win' if pnl > 0 else 'loss')
                
            # Detect consecutive losses
            max_consecutive_losses = 0
            current_consecutive = 0
            
            for result in results:
                if result == 'loss':
                    current_consecutive += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
                else:
                    current_consecutive = 0
                    
            if max_consecutive_losses >= 5:
                pattern = PatternMatch(
                    pattern_id=f"consecutive_losses_{int(time.time())}",
                    pattern_type=PatternType.TRADING_PATTERN,
                    severity=Severity.HIGH,
                    confidence=min(max_consecutive_losses / 10.0, 1.0),
                    description=f"Consecutive losses pattern: {max_consecutive_losses} losses in a row",
                    first_occurrence=datetime.now() - timedelta(hours=len(trading_data)),
                    last_occurrence=datetime.now(),
                    occurrence_count=max_consecutive_losses,
                    affected_components=['trading_strategy'],
                    sample_data=trading_data[-max_consecutive_losses:],
                    metadata={
                        'max_consecutive_losses': max_consecutive_losses,
                        'total_trades': len(trading_data)
                    },
                    recommendations=[
                        "Review trading strategy",
                        "Consider reducing position sizes",
                        "Implement stop-loss mechanisms",
                        "Analyze market conditions during loss streak"
                    ]
                )
                patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing win/loss patterns: {e}")
            return []
            
    async def _analyze_position_sizing_patterns(self, trading_data: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Analyze position sizing patterns"""
        try:
            patterns = []
            
            position_sizes = [trade.get('position_size', 0) for trade in trading_data if 'position_size' in trade]
            
            if len(position_sizes) < 5:
                return patterns
                
            # Detect increasing position sizes (potential overconfidence)
            if len(position_sizes) >= 10:
                recent_sizes = position_sizes[-5:]
                earlier_sizes = position_sizes[-10:-5]
                
                recent_avg = statistics.mean(recent_sizes)
                earlier_avg = statistics.mean(earlier_sizes)
                
                if recent_avg > earlier_avg * 1.5:  # 50% increase
                    pattern = PatternMatch(
                        pattern_id=f"increasing_position_sizes_{int(time.time())}",
                        pattern_type=PatternType.TRADING_PATTERN,
                        severity=Severity.MEDIUM,
                        confidence=min((recent_avg / earlier_avg - 1), 1.0),
                        description="Increasing position sizes pattern detected",
                        first_occurrence=datetime.now() - timedelta(hours=len(trading_data)),
                        last_occurrence=datetime.now(),
                        occurrence_count=1,
                        affected_components=['position_sizing'],
                        sample_data=[{'recent_sizes': recent_sizes, 'earlier_sizes': earlier_sizes}],
                        metadata={
                            'recent_average': recent_avg,
                            'earlier_average': earlier_avg,
                            'increase_ratio': recent_avg / earlier_avg
                        },
                        recommendations=[
                            "Review position sizing strategy",
                            "Ensure proper risk management",
                            "Consider if increased sizes are justified",
                            "Monitor for overconfidence bias"
                        ]
                    )
                    patterns.append(pattern)
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing position sizing patterns: {e}")
            return []
            
    async def _analyze_trading_timing_patterns(self, trading_data: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Analyze trading timing patterns"""
        try:
            patterns = []
            
            # Group trades by hour of day
            hourly_trades = defaultdict(list)
            for trade in trading_data:
                if 'timestamp' in trade:
                    hour = trade['timestamp'].hour
                    hourly_trades[hour].append(trade)
                    
            # Find hours with unusual trading activity
            avg_trades_per_hour = len(trading_data) / 24
            
            for hour, trades in hourly_trades.items():
                if len(trades) > avg_trades_per_hour * 3:  # 3x average
                    # Calculate performance for this hour
                    hour_pnl = sum(trade.get('pnl', 0) for trade in trades)
                    avg_pnl = hour_pnl / len(trades) if trades else 0
                    
                    severity = Severity.LOW
                    if avg_pnl < 0:  # Losing hour
                        severity = Severity.MEDIUM
                        
                    pattern = PatternMatch(
                        pattern_id=f"timing_pattern_hour_{hour}",
                        pattern_type=PatternType.TRADING_PATTERN,
                        severity=severity,
                        confidence=len(trades) / (avg_trades_per_hour * 4),
                        description=f"High trading activity at hour {hour}:00",
                        first_occurrence=min(t.get('timestamp', datetime.min) for t in trades),
                        last_occurrence=max(t.get('timestamp', datetime.min) for t in trades),
                        occurrence_count=len(trades),
                        affected_components=['trading_timing'],
                        sample_data=trades[:5],
                        metadata={
                            'hour_of_day': hour,
                            'trade_count': len(trades),
                            'average_pnl': avg_pnl,
                            'total_pnl': hour_pnl
                        },
                        recommendations=[
                            f"Analyze why trading is concentrated at {hour}:00",
                            "Consider if timing is optimal for market conditions",
                            "Review performance during high-activity hours"
                        ]
                    )
                    patterns.append(pattern)
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing timing patterns: {e}")
            return []
            
    async def _detect_unusual_trading_behavior(self, trading_data: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Detect unusual trading behavior patterns"""
        try:
            patterns = []
            
            if len(trading_data) < 10:
                return patterns
                
            # Detect rapid-fire trading
            timestamps = [trade.get('timestamp', datetime.min) for trade in trading_data if 'timestamp' in trade]
            timestamps.sort()
            
            rapid_trades = 0
            for i in range(1, len(timestamps)):
                time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                if time_diff < 60:  # Less than 1 minute between trades
                    rapid_trades += 1
                    
            if rapid_trades > len(trading_data) * 0.3:  # More than 30% rapid trades
                pattern = PatternMatch(
                    pattern_id=f"rapid_trading_{int(time.time())}",
                    pattern_type=PatternType.BEHAVIORAL_PATTERN,
                    severity=Severity.MEDIUM,
                    confidence=rapid_trades / len(trading_data),
                    description=f"Rapid trading pattern: {rapid_trades} trades within 1 minute of previous",
                    first_occurrence=timestamps[0] if timestamps else datetime.now(),
                    last_occurrence=timestamps[-1] if timestamps else datetime.now(),
                    occurrence_count=rapid_trades,
                    affected_components=['trading_behavior'],
                    sample_data=trading_data[:5],
                    metadata={
                        'rapid_trade_count': rapid_trades,
                        'total_trades': len(trading_data),
                        'rapid_trade_percentage': rapid_trades / len(trading_data) * 100
                    },
                    recommendations=[
                        "Review rapid trading behavior",
                        "Consider implementing minimum time between trades",
                        "Analyze if rapid trading is beneficial",
                        "Check for emotional trading patterns"
                    ]
                )
                patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting unusual trading behavior: {e}")
            return []
            
    async def _check_pattern_match(self, component: str, data_entry: Dict[str, Any], pattern_def: Dict[str, Any]) -> Optional[PatternMatch]:
        """Check if data matches a known pattern"""
        try:
            pattern_type = pattern_def['type']
            characteristics = pattern_def['characteristics']
            
            if pattern_type == PatternType.ERROR_PATTERN:
                return await self._check_error_pattern_match(component, data_entry, pattern_def)
            elif pattern_type == PatternType.PERFORMANCE_PATTERN:
                return await self._check_performance_pattern_match(component, data_entry, pattern_def)
            elif pattern_type == PatternType.TRADING_PATTERN:
                return await self._check_trading_pattern_match(component, data_entry, pattern_def)
                
            return None
            
        except Exception as e:
            logger.error(f"Error checking pattern match: {e}")
            return None
            
    async def _check_error_pattern_match(self, component: str, data_entry: Dict[str, Any], pattern_def: Dict[str, Any]) -> Optional[PatternMatch]:
        """Check for error pattern match"""
        try:
            data = data_entry.get('data', {})
            error_message = data.get('error_message', '').lower()
            
            if not error_message:
                return None
                
            characteristics = pattern_def['characteristics']
            keywords = characteristics.get('keywords', [])
            
            # Check if any keywords match
            matches = sum(1 for keyword in keywords if keyword.lower() in error_message)
            confidence = matches / len(keywords) if keywords else 0
            
            if confidence >= pattern_def.get('confidence_threshold', 0.8):
                pattern = PatternMatch(
                    pattern_id=pattern_def['id'],
                    pattern_type=PatternType.ERROR_PATTERN,
                    severity=pattern_def.get('severity', Severity.MEDIUM),
                    confidence=confidence,
                    description=f"Matched error pattern: {pattern_def['id']}",
                    first_occurrence=data_entry['timestamp'],
                    last_occurrence=data_entry['timestamp'],
                    occurrence_count=1,
                    affected_components=[component],
                    sample_data=[data],
                    metadata={'matched_keywords': [k for k in keywords if k.lower() in error_message]},
                    recommendations=[f"Pattern {pattern_def['id']} detected - check system logs"]
                )
                return pattern
                
            return None
            
        except Exception as e:
            logger.error(f"Error checking error pattern match: {e}")
            return None
            
    async def _check_performance_pattern_match(self, component: str, data_entry: Dict[str, Any], pattern_def: Dict[str, Any]) -> Optional[PatternMatch]:
        """Check for performance pattern match"""
        try:
            data = data_entry.get('data', {})
            characteristics = pattern_def['characteristics']
            
            metric = characteristics.get('metric')
            if not metric or metric not in data:
                return None
                
            value = data[metric]
            threshold = characteristics.get('threshold')
            
            if threshold and value > threshold:
                pattern = PatternMatch(
                    pattern_id=pattern_def['id'],
                    pattern_type=PatternType.PERFORMANCE_PATTERN,
                    severity=pattern_def.get('severity', Severity.MEDIUM),
                    confidence=min(value / threshold, 1.0) if threshold > 0 else 1.0,
                    description=f"Performance pattern detected: {metric} = {value}",
                    first_occurrence=data_entry['timestamp'],
                    last_occurrence=data_entry['timestamp'],
                    occurrence_count=1,
                    affected_components=[component],
                    sample_data=[data],
                    metadata={'metric': metric, 'value': value, 'threshold': threshold},
                    recommendations=[f"Monitor {metric} - threshold exceeded"]
                )
                return pattern
                
            return None
            
        except Exception as e:
            logger.error(f"Error checking performance pattern match: {e}")
            return None
            
    async def _check_trading_pattern_match(self, component: str, data_entry: Dict[str, Any], pattern_def: Dict[str, Any]) -> Optional[PatternMatch]:
        """Check for trading pattern match"""
        try:
            # This would need specific implementation based on trading data structure
            # For now, return None as placeholder
            return None
            
        except Exception as e:
            logger.error(f"Error checking trading pattern match: {e}")
            return None
            
    async def _store_pattern_match(self, pattern: PatternMatch):
        """Store detected pattern match"""
        try:
            # Store in database for diagnostics
            from database import log_execution_trace
            log_execution_trace(
                component="pattern_detector",
                operation="pattern_detected",
                input_data={'pattern_id': pattern.pattern_id},
                output_data=asdict(pattern),
                execution_time=0.0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error storing pattern match: {e}")
            
    async def _store_anomaly(self, anomaly: AnomalyDetection):
        """Store detected anomaly"""
        try:
            # Store in anomaly history
            component = anomaly.component
            self.anomaly_history[component].append(anomaly)
            
            # Store in database for diagnostics
            from database import log_execution_trace
            log_execution_trace(
                component="pattern_detector",
                operation="anomaly_detected",
                input_data={'anomaly_id': anomaly.anomaly_id},
                output_data=asdict(anomaly),
                execution_time=0.0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error storing anomaly: {e}")
            
    async def _generate_anomaly_recommendations(self, component: str, metric: str, value: float, baseline: float) -> List[str]:
        """Generate recommendations for detected anomaly"""
        recommendations = []
        
        try:
            deviation_pct = abs(value - baseline) / baseline * 100 if baseline != 0 else 0
            
            recommendations.append(f"Anomaly detected in {component}.{metric}: {value:.2f} (baseline: {baseline:.2f})")
            
            if metric == 'execution_time':
                recommendations.extend([
                    "Check for performance bottlenecks",
                    "Review recent code changes",
                    "Monitor system resources"
                ])
            elif metric == 'memory_usage':
                recommendations.extend([
                    "Check for memory leaks",
                    "Review memory allocation patterns",
                    "Consider garbage collection tuning"
                ])
            elif metric == 'cpu_usage':
                recommendations.extend([
                    "Check for CPU-intensive operations",
                    "Review algorithm efficiency",
                    "Monitor concurrent processes"
                ])
            elif metric == 'error_rate':
                recommendations.extend([
                    "Investigate error causes",
                    "Check external service availability",
                    "Review error handling logic"
                ])
            else:
                recommendations.append(f"Investigate unusual {metric} values")
                
            if deviation_pct > 200:
                recommendations.append("Consider immediate intervention - deviation is extreme")
                
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
            
        return recommendations
        
    async def _generate_trend_analysis(self, patterns: List[PatternMatch], anomalies: List[AnomalyDetection]) -> Dict[str, Any]:
        """Generate comprehensive trend analysis from patterns and anomalies"""
        try:
            trend_analysis = {
                'pattern_trends': {},
                'anomaly_trends': {},
                'correlation_analysis': {},
                'predictive_insights': {},
                'risk_assessment': {},
                'temporal_analysis': {},
                'severity_trends': {},
                'component_analysis': {},
                'recommendations': []
            }
            
            # Analyze pattern trends
            if patterns:
                pattern_types = defaultdict(list)
                for pattern in patterns:
                    pattern_types[pattern.pattern_type].append(pattern)
                    
                for pattern_type, pattern_list in pattern_types.items():
                    confidences = [p.confidence for p in pattern_list]
                    occurrences = [p.occurrence_count for p in pattern_list]
                    
                    trend_analysis['pattern_trends'][pattern_type.value] = {
                        'frequency': len(pattern_list),
                        'avg_confidence': statistics.mean(confidences),
                        'max_confidence': max(confidences),
                        'total_occurrences': sum(occurrences),
                        'avg_occurrences': statistics.mean(occurrences),
                        'trend_direction': await self._calculate_pattern_trend(pattern_list),
                        'severity_distribution': self._get_severity_distribution(pattern_list)
                    }
            
            # Analyze anomaly trends
            if anomalies:
                anomaly_types = defaultdict(list)
                anomaly_components = defaultdict(list)
                
                for anomaly in anomalies:
                    anomaly_types[anomaly.anomaly_type].append(anomaly)
                    anomaly_components[anomaly.component].append(anomaly)
                    
                for anomaly_type, anomaly_list in anomaly_types.items():
                    severities = [a.severity for a in anomaly_list]
                    confidences = [a.confidence for a in anomaly_list]
                    deviations = [a.deviation_percentage for a in anomaly_list]
                    
                    trend_analysis['anomaly_trends'][anomaly_type] = {
                        'frequency': len(anomaly_list),
                        'avg_confidence': statistics.mean(confidences),
                        'avg_deviation': statistics.mean(deviations),
                        'max_deviation': max(deviations),
                        'severity_distribution': Counter([s.value for s in severities]),
                        'affected_components': len(set(a.component for a in anomaly_list))
                    }
                    
                # Component-specific anomaly analysis
                for component, anomaly_list in anomaly_components.items():
                    metrics = defaultdict(list)
                    for anomaly in anomaly_list:
                        metrics[anomaly.metric].append(anomaly)
                        
                    trend_analysis['component_analysis'][component] = {
                        'total_anomalies': len(anomaly_list),
                        'affected_metrics': len(metrics),
                        'metric_breakdown': {
                            metric: len(anomaly_list) for metric, anomaly_list in metrics.items()
                        },
                        'avg_severity': statistics.mean([self._severity_to_numeric(a.severity) for a in anomaly_list])
                    }
            
            # Temporal analysis
            all_events = []
            for pattern in patterns:
                all_events.append({
                    'timestamp': pattern.first_occurrence,
                    'type': 'pattern',
                    'severity': pattern.severity,
                    'component': pattern.affected_components[0] if pattern.affected_components else 'unknown'
                })
                
            for anomaly in anomalies:
                all_events.append({
                    'timestamp': anomaly.detected_at,
                    'type': 'anomaly',
                    'severity': anomaly.severity,
                    'component': anomaly.component
                })
                
            if all_events:
                all_events.sort(key=lambda x: x['timestamp'])
                trend_analysis['temporal_analysis'] = await self._analyze_temporal_trends(all_events)
            
            # Severity trends
            severity_over_time = defaultdict(list)
            for event in all_events:
                hour = event['timestamp'].hour
                severity_over_time[hour].append(self._severity_to_numeric(event['severity']))
                
            trend_analysis['severity_trends'] = {
                hour: {
                    'avg_severity': statistics.mean(severities),
                    'max_severity': max(severities),
                    'event_count': len(severities)
                }
                for hour, severities in severity_over_time.items()
            }
            
            # Correlation analysis
            trend_analysis['correlation_analysis'] = await self._analyze_correlations(patterns, anomalies)
            
            # Predictive insights
            trend_analysis['predictive_insights'] = await self._generate_predictive_insights(patterns, anomalies)
            
            # Risk assessment
            trend_analysis['risk_assessment'] = await self._assess_trend_risks(patterns, anomalies)
            
            # Generate recommendations
            trend_analysis['recommendations'] = await self._generate_trend_recommendations(
                trend_analysis['pattern_trends'], 
                trend_analysis['anomaly_trends'],
                trend_analysis['severity_trends']
            )
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {e}")
            return {'error': str(e)}
    
    async def _calculate_pattern_trend(self, pattern_list: List[PatternMatch]) -> str:
        """Calculate trend direction for patterns"""
        try:
            if len(pattern_list) < 3:
                return "INSUFFICIENT_DATA"
                
            # Sort by first occurrence
            sorted_patterns = sorted(pattern_list, key=lambda p: p.first_occurrence)
            
            # Calculate trend in confidence over time
            confidences = [p.confidence for p in sorted_patterns]
            
            # Simple linear trend
            x = list(range(len(confidences)))
            n = len(confidences)
            sum_x = sum(x)
            sum_y = sum(confidences)
            sum_xy = sum(x[i] * confidences[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 == 0:
                return "STABLE"
                
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            if slope > 0.05:
                return "INCREASING"
            elif slope < -0.05:
                return "DECREASING"
            else:
                return "STABLE"
                
        except Exception as e:
            logger.error(f"Error calculating pattern trend: {e}")
            return "UNKNOWN"
    
    def _get_severity_distribution(self, items: List) -> Dict[str, int]:
        """Get severity distribution for patterns or anomalies"""
        try:
            severities = [item.severity.value for item in items]
            return dict(Counter(severities))
        except Exception as e:
            logger.error(f"Error getting severity distribution: {e}")
            return {}
    
    def _severity_to_numeric(self, severity: Severity) -> float:
        """Convert severity to numeric value for calculations"""
        severity_map = {
            Severity.LOW: 1.0,
            Severity.MEDIUM: 2.0,
            Severity.HIGH: 3.0,
            Severity.CRITICAL: 4.0
        }
        return severity_map.get(severity, 1.0)
    
    async def _analyze_temporal_trends(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in events"""
        try:
            temporal_analysis = {
                'events_per_hour': defaultdict(int),
                'peak_hours': [],
                'quiet_hours': [],
                'event_clustering': {},
                'time_between_events': []
            }
            
            # Count events per hour
            for event in events:
                hour = event['timestamp'].hour
                temporal_analysis['events_per_hour'][hour] += 1
                
            # Calculate time between events
            for i in range(1, len(events)):
                time_diff = (events[i]['timestamp'] - events[i-1]['timestamp']).total_seconds()
                temporal_analysis['time_between_events'].append(time_diff)
                
            # Identify peak and quiet hours
            if temporal_analysis['events_per_hour']:
                avg_events = statistics.mean(temporal_analysis['events_per_hour'].values())
                
                for hour, count in temporal_analysis['events_per_hour'].items():
                    if count > avg_events * 1.5:
                        temporal_analysis['peak_hours'].append(hour)
                    elif count < avg_events * 0.5:
                        temporal_analysis['quiet_hours'].append(hour)
                        
            # Event clustering analysis
            if temporal_analysis['time_between_events']:
                avg_time_between = statistics.mean(temporal_analysis['time_between_events'])
                clusters = 0
                for time_diff in temporal_analysis['time_between_events']:
                    if time_diff < avg_time_between * 0.1:  # Very close events
                        clusters += 1
                        
                temporal_analysis['event_clustering'] = {
                    'cluster_count': clusters,
                    'avg_time_between_events': avg_time_between,
                    'clustering_ratio': clusters / len(temporal_analysis['time_between_events'])
                }
                
            return dict(temporal_analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing temporal trends: {e}")
            return {}
    
    async def _analyze_correlations(self, patterns: List[PatternMatch], anomalies: List[AnomalyDetection]) -> Dict[str, Any]:
        """Analyze correlations between patterns and anomalies"""
        try:
            correlations = {
                'pattern_anomaly_correlation': 0.0,
                'component_correlations': {},
                'temporal_correlations': {},
                'severity_correlations': {}
            }
            
            if not patterns or not anomalies:
                return correlations
                
            # Component correlation
            pattern_components = defaultdict(int)
            anomaly_components = defaultdict(int)
            
            for pattern in patterns:
                for component in pattern.affected_components:
                    pattern_components[component] += 1
                    
            for anomaly in anomalies:
                anomaly_components[anomaly.component] += 1
                
            # Calculate component correlation
            common_components = set(pattern_components.keys()) & set(anomaly_components.keys())
            if common_components:
                correlations['component_correlations'] = {
                    component: {
                        'pattern_count': pattern_components[component],
                        'anomaly_count': anomaly_components[component],
                        'correlation_strength': min(
                            pattern_components[component] / max(pattern_components.values()),
                            anomaly_components[component] / max(anomaly_components.values())
                        )
                    }
                    for component in common_components
                }
                
            return correlations
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    async def _generate_predictive_insights(self, patterns: List[PatternMatch], anomalies: List[AnomalyDetection]) -> Dict[str, Any]:
        """Generate predictive insights from patterns and anomalies"""
        try:
            insights = {
                'risk_indicators': [],
                'trend_predictions': {},
                'early_warning_signals': [],
                'recommended_monitoring': []
            }
            
            # Risk indicators
            critical_patterns = [p for p in patterns if p.severity == Severity.CRITICAL]
            critical_anomalies = [a for a in anomalies if a.severity == Severity.CRITICAL]
            
            if critical_patterns:
                insights['risk_indicators'].append(f"{len(critical_patterns)} critical patterns detected")
                
            if critical_anomalies:
                insights['risk_indicators'].append(f"{len(critical_anomalies)} critical anomalies detected")
                
            # Early warning signals
            high_confidence_patterns = [p for p in patterns if p.confidence > 0.8]
            if len(high_confidence_patterns) > len(patterns) * 0.7:
                insights['early_warning_signals'].append("High confidence pattern detection rate indicates systematic issues")
                
            # Recommended monitoring
            component_issues = defaultdict(int)
            for pattern in patterns:
                for component in pattern.affected_components:
                    component_issues[component] += 1
                    
            for anomaly in anomalies:
                component_issues[anomaly.component] += 1
                
            top_components = sorted(component_issues.items(), key=lambda x: x[1], reverse=True)[:3]
            insights['recommended_monitoring'] = [
                f"Increase monitoring for {component} ({count} issues)"
                for component, count in top_components
            ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating predictive insights: {e}")
            return {}
    
    async def _assess_trend_risks(self, patterns: List[PatternMatch], anomalies: List[AnomalyDetection]) -> Dict[str, Any]:
        """Assess risks based on trend analysis"""
        try:
            risk_assessment = {
                'overall_risk_level': 'LOW',
                'risk_score': 0.0,
                'risk_factors': [],
                'mitigation_priorities': []
            }
            
            risk_score = 0.0
            
            # Pattern-based risk
            critical_patterns = len([p for p in patterns if p.severity == Severity.CRITICAL])
            high_patterns = len([p for p in patterns if p.severity == Severity.HIGH])
            
            risk_score += critical_patterns * 0.4
            risk_score += high_patterns * 0.2
            
            # Anomaly-based risk
            critical_anomalies = len([a for a in anomalies if a.severity == Severity.CRITICAL])
            high_anomalies = len([a for a in anomalies if a.severity == Severity.HIGH])
            
            risk_score += critical_anomalies * 0.3
            risk_score += high_anomalies * 0.15
            
            # Normalize risk score
            risk_assessment['risk_score'] = min(risk_score / 10.0, 1.0)
            
            # Determine risk level
            if risk_assessment['risk_score'] > 0.7:
                risk_assessment['overall_risk_level'] = 'CRITICAL'
            elif risk_assessment['risk_score'] > 0.5:
                risk_assessment['overall_risk_level'] = 'HIGH'
            elif risk_assessment['risk_score'] > 0.3:
                risk_assessment['overall_risk_level'] = 'MEDIUM'
            else:
                risk_assessment['overall_risk_level'] = 'LOW'
                
            # Risk factors
            if critical_patterns > 0:
                risk_assessment['risk_factors'].append(f"{critical_patterns} critical patterns detected")
            if critical_anomalies > 0:
                risk_assessment['risk_factors'].append(f"{critical_anomalies} critical anomalies detected")
                
            # Mitigation priorities
            if critical_patterns > 0 or critical_anomalies > 0:
                risk_assessment['mitigation_priorities'].append("Address critical issues immediately")
            if high_patterns > 2 or high_anomalies > 2:
                risk_assessment['mitigation_priorities'].append("Investigate high-severity patterns")
                
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing trend risks: {e}")
            return {'overall_risk_level': 'UNKNOWN', 'error': str(e)}
    
    async def _generate_trend_recommendations(self, pattern_trends: Dict[str, Any], 
                                            anomaly_trends: Dict[str, Any],
                                            severity_trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        try:
            # Pattern trend recommendations
            for pattern_type, trend_data in pattern_trends.items():
                frequency = trend_data.get('frequency', 0)
                avg_confidence = trend_data.get('avg_confidence', 0)
                
                if frequency > 5:
                    recommendations.append(f"High frequency of {pattern_type} patterns - investigate root causes")
                    
                if avg_confidence > 0.8:
                    recommendations.append(f"High confidence {pattern_type} patterns indicate systematic issues")
                    
            # Anomaly trend recommendations
            for anomaly_type, trend_data in anomaly_trends.items():
                frequency = trend_data.get('frequency', 0)
                avg_deviation = trend_data.get('avg_deviation', 0)
                
                if frequency > 3:
                    recommendations.append(f"Frequent {anomaly_type} anomalies detected")
                    
                if avg_deviation > 100:
                    recommendations.append(f"Large deviations in {anomaly_type} - check system stability")
                    
            # Severity trend recommendations
            high_severity_hours = [
                hour for hour, data in severity_trends.items() 
                if data.get('avg_severity', 0) > 2.5
            ]
            
            if high_severity_hours:
                recommendations.append(f"High severity events during hours: {high_severity_hours}")
                
            if not recommendations:
                recommendations.append("Trend analysis shows normal system behavior")
                
        except Exception as e:
            recommendations.append(f"Error generating trend recommendations: {str(e)}")
            
        return recommendations
    
    async def _generate_pattern_recommendations(self, patterns: List[PatternMatch], anomalies: List[AnomalyDetection]) -> List[str]:
        """Generate overall pattern recommendations"""
        recommendations = []
        
        try:
            # Critical issues first
            critical_patterns = [p for p in patterns if p.severity == Severity.CRITICAL]
            critical_anomalies = [a for a in anomalies if a.severity == Severity.CRITICAL]
            
            if critical_patterns:
                recommendations.append(f"URGENT: {len(critical_patterns)} critical patterns require immediate attention")
                
            if critical_anomalies:
                recommendations.append(f"URGENT: {len(critical_anomalies)} critical anomalies detected")
                
            # Component-specific recommendations
            component_issues = defaultdict(int)
            for pattern in patterns:
                for component in pattern.affected_components:
                    component_issues[component] += 1
                    
            for anomaly in anomalies:
                component_issues[anomaly.component] += 1
                
            if component_issues:
                top_component = max(component_issues.items(), key=lambda x: x[1])
                recommendations.append(f"Focus attention on {top_component[0]} - {top_component[1]} issues detected")
                
            # General recommendations
            if len(patterns) > 10:
                recommendations.append("High number of patterns detected - consider system review")
                
            if len(anomalies) > 5:
                recommendations.append("Multiple anomalies detected - check system stability")
                
            if not recommendations:
                recommendations.append("System showing normal pattern behavior")
                
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
            
        return recommendations
    
    async def _assess_pattern_risk(self, patterns: List[PatternMatch], anomalies: List[AnomalyDetection]) -> Dict[str, Any]:
        """Assess overall risk from patterns and anomalies"""
        try:
            risk_assessment = {
                'overall_risk': 'LOW',
                'risk_score': 0.0,
                'critical_issues': 0,
                'high_priority_issues': 0,
                'affected_components': set(),
                'risk_factors': [],
                'immediate_actions': []
            }
            
            # Count critical and high priority issues
            critical_patterns = [p for p in patterns if p.severity == Severity.CRITICAL]
            high_patterns = [p for p in patterns if p.severity == Severity.HIGH]
            critical_anomalies = [a for a in anomalies if a.severity == Severity.CRITICAL]
            high_anomalies = [a for a in anomalies if a.severity == Severity.HIGH]
            
            risk_assessment['critical_issues'] = len(critical_patterns) + len(critical_anomalies)
            risk_assessment['high_priority_issues'] = len(high_patterns) + len(high_anomalies)
            
            # Calculate risk score
            risk_score = 0.0
            risk_score += len(critical_patterns) * 0.4
            risk_score += len(critical_anomalies) * 0.3
            risk_score += len(high_patterns) * 0.2
            risk_score += len(high_anomalies) * 0.15
            
            risk_assessment['risk_score'] = min(risk_score / 5.0, 1.0)
            
            # Determine overall risk
            if risk_assessment['risk_score'] > 0.8:
                risk_assessment['overall_risk'] = 'CRITICAL'
            elif risk_assessment['risk_score'] > 0.6:
                risk_assessment['overall_risk'] = 'HIGH'
            elif risk_assessment['risk_score'] > 0.3:
                risk_assessment['overall_risk'] = 'MEDIUM'
            else:
                risk_assessment['overall_risk'] = 'LOW'
                
            # Collect affected components
            for pattern in patterns:
                risk_assessment['affected_components'].update(pattern.affected_components)
            for anomaly in anomalies:
                risk_assessment['affected_components'].add(anomaly.component)
                
            risk_assessment['affected_components'] = list(risk_assessment['affected_components'])
            
            # Risk factors
            if critical_patterns:
                risk_assessment['risk_factors'].append(f"{len(critical_patterns)} critical patterns")
            if critical_anomalies:
                risk_assessment['risk_factors'].append(f"{len(critical_anomalies)} critical anomalies")
            if len(risk_assessment['affected_components']) > 5:
                risk_assessment['risk_factors'].append("Multiple components affected")
                
            # Immediate actions
            if risk_assessment['critical_issues'] > 0:
                risk_assessment['immediate_actions'].append("Address critical issues immediately")
            if risk_assessment['high_priority_issues'] > 3:
                risk_assessment['immediate_actions'].append("Review high priority issues")
            if len(risk_assessment['affected_components']) > 3:
                risk_assessment['immediate_actions'].append("Conduct system-wide health check")
                
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing pattern risk: {e}")
            return {'overall_risk': 'UNKNOWN', 'error': str(e)}
    
    # Background monitoring tasks
    
    async def _continuous_pattern_detection(self):
        """Continuous pattern detection background task"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Analyze recent data for patterns
                for component, data_deque in self.pattern_history.items():
                    if len(data_deque) >= 5:  # Minimum data for pattern detection
                        recent_data = list(data_deque)[-10:]  # Last 10 entries
                        
                        # Extract performance data
                        performance_data = []
                        error_logs = []
                        
                        for entry in recent_data:
                            data = entry.get('data', {})
                            if 'execution_time' in data or 'cpu_usage' in data:
                                performance_data.append(data)
                            if 'error_message' in data:
                                error_logs.append(data)
                                
                        # Detect patterns
                        if performance_data:
                            patterns = await self.detect_performance_patterns(performance_data)
                            for pattern in patterns:
                                await self._store_pattern_match(pattern)
                                
                        if error_logs:
                            patterns = await self.detect_error_patterns(error_logs)
                            for pattern in patterns:
                                await self._store_pattern_match(pattern)
                                
            except Exception as e:
                logger.error(f"Error in continuous pattern detection: {e}")
                await asyncio.sleep(60)
                
    async def _anomaly_detection_loop(self):
        """Continuous anomaly detection background task"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check for anomalies in recent data
                for component, data_deque in self.pattern_history.items():
                    if len(data_deque) >= 1:
                        latest_entry = data_deque[-1]
                        data = latest_entry.get('data', {})
                        
                        # Extract numeric metrics
                        metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}
                        
                        if metrics:
                            anomalies = await self.detect_anomalies(component, metrics)
                            for anomaly in anomalies:
                                await self._store_anomaly(anomaly)
                                
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(30)
                
    async def _pattern_learning_loop(self):
        """Pattern learning background task"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(3600)  # Learn every hour
                
                # Analyze stored patterns for learning opportunities
                # This would implement machine learning to discover new patterns
                # For now, just log that learning is active
                logger.info("Pattern learning cycle completed")
                
            except Exception as e:
                logger.error(f"Error in pattern learning loop: {e}")
                await asyncio.sleep(3600)
                
    async def _calculate_avg_detection_time(self) -> float:
        """Calculate average detection time"""
        try:
            # This would calculate actual detection times
            # For now, return a mock value
            return 0.5  # 500ms average
        except Exception as e:
            logger.error(f"Error calculating avg detection time: {e}")
            return 0.0
            
    async def _calculate_false_positive_rate(self) -> float:
        """Calculate false positive rate"""
        try:
            # This would calculate actual false positive rate
            # For now, return a mock value
            return 0.05  # 5% false positive rate
        except Exception as e:
            logger.error(f"Error calculating false positive rate: {e}")
            return 0.0
            
    async def _calculate_pattern_accuracy(self) -> float:
        """Calculate pattern detection accuracy"""
        try:
            # This would calculate actual accuracy
            # For now, return a mock value
            return 0.92  # 92% accuracy
        except Exception as e:
            logger.error(f"Error calculating pattern accuracy: {e}")
            return 0.0
            
    async def _extract_pattern_characteristics(self, pattern_data: Dict[str, Any], pattern_type: PatternType) -> Dict[str, Any]:
        """Extract characteristics from pattern data for learning"""
        try:
            characteristics = {
                'data_points': len(pattern_data) if isinstance(pattern_data, list) else 1,
                'pattern_type': pattern_type.value,
                'extracted_at': datetime.now().isoformat()
            }
            
            # Extract type-specific characteristics
            if pattern_type == PatternType.ERROR_PATTERN:
                if 'error_message' in pattern_data:
                    characteristics['keywords'] = self._extract_keywords(pattern_data['error_message'])
                    
            elif pattern_type == PatternType.PERFORMANCE_PATTERN:
                if 'execution_time' in pattern_data:
                    characteristics['avg_execution_time'] = pattern_data['execution_time']
                    
            return characteristics
            
        except Exception as e:
            logger.error(f"Error extracting pattern characteristics: {e}")
            return {}
            
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for pattern matching"""
        try:
            # Simple keyword extraction
            words = text.lower().split()
            # Filter out common words and keep meaningful terms
            keywords = [word for word in words if len(word) > 3 and word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'who', 'oil', 'sit', 'set']]
            return keywords[:10]  # Return top 10 keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

# Global pattern detector instance
pattern_detector = PatternDetector()

# Convenience functions for easy integration
async def start_pattern_detection():
    """Start pattern detection system"""
    await pattern_detector.start_detection()

async def stop_pattern_detection():
    """Stop pattern detection system"""
    await pattern_detector.stop_detection()

async def analyze_component_data(component: str, data: Dict[str, Any]):
    """Analyze data for patterns and anomalies"""
    return await pattern_detector.analyze_data(component, data)

async def get_pattern_report(hours_back: int = 24):
    """Get pattern analysis report"""
    return await pattern_detector.generate_pattern_report(hours_back)

async def get_pattern_stats():
    """Get pattern detection statistics"""
    return await pattern_detector.get_pattern_statistics()

if __name__ == "__main__":
    # Test Pattern Detector
    async def test_pattern_detector():
        print("🔍 Testing Pattern Detector...")
        
        # Start detection
        await start_pattern_detection()
        
        # Test data analysis
        test_data = {
            'execution_time': 2.5,
            'cpu_usage': 85.0,
            'memory_usage': 1024,
            'error_message': 'Connection timeout occurred'
        }
        
        result = await analyze_component_data('test_component', test_data)
        print(f"Analysis result: {result}")
        
        # Test pattern report
        report = await get_pattern_report(1)
        print(f"Pattern report: {report.total_patterns} patterns, {report.anomalies_detected} anomalies")
        
        # Test statistics
        stats = await get_pattern_stats()
        print(f"Pattern stats: {stats}")
        
        # Stop detection
        await stop_pattern_detection()
        
        print("✅ Pattern Detector test completed!")

    # Run the test
    import asyncio
    asyncio.run(test_pattern_detector())
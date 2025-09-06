"""
Health Checker - System Health Monitoring for Trading Bot v9.1
Comprehensive system health monitoring with real-time diagnostics
"""

import asyncio
import psutil
import aioredis
import asyncpg
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
import subprocess
import socket
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    name: str
    value: float
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    unit: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

@dataclass
class SystemHealth:
    overall_status: HealthStatus
    metrics: List[HealthMetric]
    services: Dict[str, HealthStatus]
    timestamp: datetime
    uptime: float
    alerts: List[str]

class HealthChecker:
    """Comprehensive system health monitoring"""
    
    def __init__(self, db_pool, redis_client, config):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.config = config
        self.start_time = datetime.utcnow()
        self.last_check = None
        self.health_history = []
        self.alert_thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 85.0},
            'memory_usage': {'warning': 75.0, 'critical': 90.0},
            'disk_usage': {'warning': 80.0, 'critical': 95.0},
            'response_time': {'warning': 1000.0, 'critical': 3000.0},
            'error_rate': {'warning': 5.0, 'critical': 10.0},
            'connection_count': {'warning': 80.0, 'critical': 95.0}
        }
        
    async def get_comprehensive_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        try:
            logger.info("🔍 Starting comprehensive health check...")
            
            # Collect all health metrics
            system_metrics = await self._get_system_metrics()
            service_health = await self._check_services_health()
            network_metrics = await self._get_network_metrics()
            database_metrics = await self._get_database_metrics()
            application_metrics = await self._get_application_metrics()
            
            # Combine all metrics
            all_metrics = (
                system_metrics + 
                network_metrics + 
                database_metrics + 
                application_metrics
            )
            
            # Determine overall status
            overall_status = self._calculate_overall_status(all_metrics, service_health)
            
            # Generate alerts
            alerts = self._generate_alerts(all_metrics, service_health)
            
            # Calculate uptime
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            health = SystemHealth(
                overall_status=overall_status,
                metrics=all_metrics,
                services=service_health,
                timestamp=datetime.utcnow(),
                uptime=uptime,
                alerts=alerts
            )
            
            # Store health data
            await self._store_health_data(health)
            
            self.last_check = datetime.utcnow()
            logger.info(f"✅ Health check completed - Status: {overall_status.value}")
            
            return health
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                metrics=[],
                services={},
                timestamp=datetime.utcnow(),
                uptime=0,
                alerts=[f"Health check failed: {str(e)}"]
            )
    
    async def _get_system_metrics(self) -> List[HealthMetric]:
        """Get system resource metrics"""
        metrics = []
        
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._get_status_for_value(
                cpu_percent, 
                self.alert_thresholds['cpu_usage']
            )
            metrics.append(HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                status=cpu_status,
                threshold_warning=self.alert_thresholds['cpu_usage']['warning'],
                threshold_critical=self.alert_thresholds['cpu_usage']['critical'],
                unit="%",
                timestamp=datetime.utcnow(),
                details={
                    "cpu_count": psutil.cpu_count(),
                    "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                }
            ))
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_status = self._get_status_for_value(
                memory_percent,
                self.alert_thresholds['memory_usage']
            )
            metrics.append(HealthMetric(
                name="memory_usage",
                value=memory_percent,
                status=memory_status,
                threshold_warning=self.alert_thresholds['memory_usage']['warning'],
                threshold_critical=self.alert_thresholds['memory_usage']['critical'],
                unit="%",
                timestamp=datetime.utcnow(),
                details={
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "free": memory.free
                }
            ))
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._get_status_for_value(
                disk_percent,
                self.alert_thresholds['disk_usage']
            )
            metrics.append(HealthMetric(
                name="disk_usage",
                value=disk_percent,
                status=disk_status,
                threshold_warning=self.alert_thresholds['disk_usage']['warning'],
                threshold_critical=self.alert_thresholds['disk_usage']['critical'],
                unit="%",
                timestamp=datetime.utcnow(),
                details={
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free
                }
            ))
            
            # Load Average (Linux/Unix only)
            try:
                load_avg = os.getloadavg()
                metrics.append(HealthMetric(
                    name="load_average",
                    value=load_avg[0],
                    status=HealthStatus.HEALTHY if load_avg[0] < psutil.cpu_count() else HealthStatus.WARNING,
                    threshold_warning=float(psutil.cpu_count()),
                    threshold_critical=float(psutil.cpu_count() * 2),
                    unit="",
                    timestamp=datetime.utcnow(),
                    details={
                        "1min": load_avg[0],
                        "5min": load_avg[1],
                        "15min": load_avg[2]
                    }
                ))
            except (OSError, AttributeError):
                # Windows doesn't have load average
                pass
                
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            
        return metrics
    
    async def _check_services_health(self) -> Dict[str, HealthStatus]:
        """Check health of external services"""
        services = {}
        
        # Check PostgreSQL
        try:
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            services['postgresql'] = HealthStatus.HEALTHY
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            services['postgresql'] = HealthStatus.CRITICAL
        
        # Check Redis
        try:
            await self.redis_client.ping()
            services['redis'] = HealthStatus.HEALTHY
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            services['redis'] = HealthStatus.CRITICAL
        
        # Check Binance API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api.binance.com/api/v3/ping',
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        services['binance_api'] = HealthStatus.HEALTHY
                    else:
                        services['binance_api'] = HealthStatus.WARNING
        except Exception as e:
            logger.error(f"Binance API health check failed: {e}")
            services['binance_api'] = HealthStatus.CRITICAL
        
        # Check Discord Webhook
        try:
            webhook_url = self.config.get('DISCORD_WEBHOOK_URL')
            if webhook_url:
                async with aiohttp.ClientSession() as session:
                    # Just check if webhook URL is reachable
                    async with session.get(
                        webhook_url,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        # Discord webhook returns 405 for GET, which is expected
                        if response.status in [200, 405]:
                            services['discord_webhook'] = HealthStatus.HEALTHY
                        else:
                            services['discord_webhook'] = HealthStatus.WARNING
            else:
                services['discord_webhook'] = HealthStatus.WARNING
        except Exception as e:
            logger.error(f"Discord webhook health check failed: {e}")
            services['discord_webhook'] = HealthStatus.CRITICAL
        
        return services
    
    async def _get_network_metrics(self) -> List[HealthMetric]:
        """Get network-related metrics"""
        metrics = []
        
        try:
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.append(HealthMetric(
                    name="network_bytes_sent",
                    value=float(net_io.bytes_sent),
                    status=HealthStatus.HEALTHY,
                    threshold_warning=0,
                    threshold_critical=0,
                    unit="bytes",
                    timestamp=datetime.utcnow(),
                    details={
                        "bytes_recv": net_io.bytes_recv,
                        "packets_sent": net_io.packets_sent,
                        "packets_recv": net_io.packets_recv
                    }
                ))
            
            # Connection count
            connections = len(psutil.net_connections())
            conn_status = self._get_status_for_value(
                connections,
                self.alert_thresholds['connection_count']
            )
            metrics.append(HealthMetric(
                name="network_connections",
                value=float(connections),
                status=conn_status,
                threshold_warning=self.alert_thresholds['connection_count']['warning'],
                threshold_critical=self.alert_thresholds['connection_count']['critical'],
                unit="count",
                timestamp=datetime.utcnow()
            ))
            
        except Exception as e:
            logger.error(f"Error getting network metrics: {e}")
            
        return metrics
    
    async def _get_database_metrics(self) -> List[HealthMetric]:
        """Get database performance metrics"""
        metrics = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Database size
                db_size = await conn.fetchval("""
                    SELECT pg_database_size(current_database())
                """)
                
                metrics.append(HealthMetric(
                    name="database_size",
                    value=float(db_size),
                    status=HealthStatus.HEALTHY,
                    threshold_warning=0,
                    threshold_critical=0,
                    unit="bytes",
                    timestamp=datetime.utcnow()
                ))
                
                # Active connections
                active_connections = await conn.fetchval("""
                    SELECT count(*) FROM pg_stat_activity 
                    WHERE state = 'active'
                """)
                
                metrics.append(HealthMetric(
                    name="database_active_connections",
                    value=float(active_connections),
                    status=HealthStatus.HEALTHY if active_connections < 50 else HealthStatus.WARNING,
                    threshold_warning=50,
                    threshold_critical=100,
                    unit="count",
                    timestamp=datetime.utcnow()
                ))
                
        except Exception as e:
            logger.error(f"Error getting database metrics: {e}")
            
        return metrics
    
    async def _get_application_metrics(self) -> List[HealthMetric]:
        """Get application-specific metrics"""
        metrics = []
        
        try:
            # Process info
            process = psutil.Process()
            
            # Memory usage by process
            memory_info = process.memory_info()
            metrics.append(HealthMetric(
                name="process_memory",
                value=float(memory_info.rss),
                status=HealthStatus.HEALTHY,
                threshold_warning=0,
                threshold_critical=0,
                unit="bytes",
                timestamp=datetime.utcnow(),
                details={
                    "vms": memory_info.vms,
                    "percent": process.memory_percent()
                }
            ))
            
            # CPU usage by process
            cpu_percent = process.cpu_percent()
            metrics.append(HealthMetric(
                name="process_cpu",
                value=cpu_percent,
                status=HealthStatus.HEALTHY if cpu_percent < 50 else HealthStatus.WARNING,
                threshold_warning=50,
                threshold_critical=80,
                unit="%",
                timestamp=datetime.utcnow()
            ))
            
            # File descriptors (Unix only)
            try:
                num_fds = process.num_fds()
                metrics.append(HealthMetric(
                    name="file_descriptors",
                    value=float(num_fds),
                    status=HealthStatus.HEALTHY if num_fds < 1000 else HealthStatus.WARNING,
                    threshold_warning=1000,
                    threshold_critical=2000,
                    unit="count",
                    timestamp=datetime.utcnow()
                ))
            except (AttributeError, psutil.AccessDenied):
                # Windows doesn't have num_fds
                pass
                
        except Exception as e:
            logger.error(f"Error getting application metrics: {e}")
            
        return metrics
    
    def _get_status_for_value(self, value: float, thresholds: Dict[str, float]) -> HealthStatus:
        """Determine health status based on value and thresholds"""
        if value >= thresholds['critical']:
            return HealthStatus.CRITICAL
        elif value >= thresholds['warning']:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _calculate_overall_status(self, metrics: List[HealthMetric], services: Dict[str, HealthStatus]) -> HealthStatus:
        """Calculate overall system health status"""
        # Check for critical issues
        critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
        critical_services = [s for s in services.values() if s == HealthStatus.CRITICAL]
        
        if critical_metrics or critical_services:
            return HealthStatus.CRITICAL
        
        # Check for warnings
        warning_metrics = [m for m in metrics if m.status == HealthStatus.WARNING]
        warning_services = [s for s in services.values() if s == HealthStatus.WARNING]
        
        if warning_metrics or warning_services:
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def _generate_alerts(self, metrics: List[HealthMetric], services: Dict[str, HealthStatus]) -> List[str]:
        """Generate alert messages based on health status"""
        alerts = []
        
        # Metric alerts
        for metric in metrics:
            if metric.status == HealthStatus.CRITICAL:
                alerts.append(f"🚨 CRITICAL: {metric.name} is {metric.value}{metric.unit} (threshold: {metric.threshold_critical}{metric.unit})")
            elif metric.status == HealthStatus.WARNING:
                alerts.append(f"⚠️ WARNING: {metric.name} is {metric.value}{metric.unit} (threshold: {metric.threshold_warning}{metric.unit})")
        
        # Service alerts
        for service, status in services.items():
            if status == HealthStatus.CRITICAL:
                alerts.append(f"🚨 CRITICAL: {service} is not responding")
            elif status == HealthStatus.WARNING:
                alerts.append(f"⚠️ WARNING: {service} has issues")
        
        return alerts
    
    async def _store_health_data(self, health: SystemHealth):
        """Store health data in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store overall health
                await conn.execute("""
                    INSERT INTO system_health_log (
                        timestamp, overall_status, uptime, alert_count
                    ) VALUES ($1, $2, $3, $4)
                """, health.timestamp, health.overall_status.value, health.uptime, len(health.alerts))
                
                # Store individual metrics
                for metric in health.metrics:
                    await conn.execute("""
                        INSERT INTO health_metrics_log (
                            timestamp, metric_name, value, status, unit, details
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """, 
                    metric.timestamp, 
                    metric.name, 
                    metric.value, 
                    metric.status.value, 
                    metric.unit,
                    json.dumps(metric.details) if metric.details else None
                    )
                
        except Exception as e:
            logger.error(f"Error storing health data: {e}")
    
    async def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health history for specified hours"""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT timestamp, overall_status, uptime, alert_count
                    FROM system_health_log
                    WHERE timestamp >= $1
                    ORDER BY timestamp DESC
                """, since)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting health history: {e}")
            return []
    
    async def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get specific metric history"""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT timestamp, value, status, unit
                    FROM health_metrics_log
                    WHERE metric_name = $1 AND timestamp >= $2
                    ORDER BY timestamp DESC
                """, metric_name, since)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting metric history: {e}")
            return []
    
    async def cleanup_old_data(self, days: int = 7):
        """Clean up old health data"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            async with self.db_pool.acquire() as conn:
                deleted_health = await conn.fetchval("""
                    DELETE FROM system_health_log 
                    WHERE timestamp < $1
                    RETURNING count(*)
                """, cutoff)
                
                deleted_metrics = await conn.fetchval("""
                    DELETE FROM health_metrics_log 
                    WHERE timestamp < $1
                    RETURNING count(*)
                """, cutoff)
                
                logger.info(f"🧹 Cleaned up {deleted_health} health records and {deleted_metrics} metric records")
                
        except Exception as e:
            logger.error(f"Error cleaning up health data: {e}")

# Health check utilities
async def quick_health_check(db_pool, redis_client) -> Dict[str, Any]:
    """Quick health check for API endpoints"""
    try:
        # Check database
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # Check Redis
    try:
        await redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    
    return {
        "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": db_status,
            "redis": redis_status
        },
        "system": {
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent
        }
    }
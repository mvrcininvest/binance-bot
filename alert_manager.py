"""
Alert Manager - Comprehensive Alert Management for Trading Bot v9.1
Manages all types of alerts: system, trading, performance, and diagnostic alerts
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertCategory(Enum):
    SYSTEM = "system"
    TRADING = "trading"
    PERFORMANCE = "performance"
    SECURITY = "security"
    NETWORK = "network"
    DATABASE = "database"
    API = "api"
    DIAGNOSTIC = "diagnostic"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    id: str
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    source: str
    metadata: Dict[str, Any]
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    notification_sent: bool = False
    suppressed_until: Optional[datetime] = None

@dataclass
class AlertRule:
    name: str
    condition: str
    severity: AlertSeverity
    category: AlertCategory
    enabled: bool
    cooldown_minutes: int
    escalation_minutes: int
    max_escalations: int
    notification_channels: List[str]
    metadata: Dict[str, Any]

class AlertManager:
    """Comprehensive alert management system"""
    
    def __init__(self, db_pool, discord_client, config):
        self.db_pool = db_pool
        self.discord_client = discord_client
        self.config = config
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.suppressed_alerts: Set[str] = set()
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Rate limiting
        self.alert_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_notifications: Dict[str, datetime] = {}
        
        # Escalation tracking
        self.escalation_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize default rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                condition="cpu_usage > 85",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.SYSTEM,
                enabled=True,
                cooldown_minutes=5,
                escalation_minutes=15,
                max_escalations=3,
                notification_channels=["discord", "log"],
                metadata={"threshold": 85}
            ),
            AlertRule(
                name="critical_memory_usage",
                condition="memory_usage > 90",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.SYSTEM,
                enabled=True,
                cooldown_minutes=2,
                escalation_minutes=10,
                max_escalations=5,
                notification_channels=["discord", "log"],
                metadata={"threshold": 90}
            ),
            AlertRule(
                name="database_connection_failure",
                condition="database_status == 'critical'",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.DATABASE,
                enabled=True,
                cooldown_minutes=1,
                escalation_minutes=5,
                max_escalations=10,
                notification_channels=["discord", "log"],
                metadata={}
            ),
            AlertRule(
                name="trading_position_loss",
                condition="position_pnl < -1000",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.TRADING,
                enabled=True,
                cooldown_minutes=10,
                escalation_minutes=30,
                max_escalations=2,
                notification_channels=["discord"],
                metadata={"loss_threshold": -1000}
            ),
            AlertRule(
                name="api_rate_limit_exceeded",
                condition="api_rate_limit_exceeded == true",
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.API,
                enabled=True,
                cooldown_minutes=15,
                escalation_minutes=60,
                max_escalations=1,
                notification_channels=["log"],
                metadata={}
            ),
            AlertRule(
                name="security_suspicious_activity",
                condition="suspicious_activity_detected == true",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.SECURITY,
                enabled=True,
                cooldown_minutes=0,
                escalation_minutes=5,
                max_escalations=10,
                notification_channels=["discord", "log"],
                metadata={}
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
    
    async def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        category: AlertCategory,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new alert"""
        try:
            # Generate alert ID
            alert_id = self._generate_alert_id(title, source, metadata or {})
            
            # Check if alert already exists and is active
            if alert_id in self.active_alerts:
                existing_alert = self.active_alerts[alert_id]
                if existing_alert.status == AlertStatus.ACTIVE:
                    # Update existing alert
                    existing_alert.updated_at = datetime.utcnow()
                    existing_alert.escalation_level += 1
                    await self._update_alert_in_db(existing_alert)
                    return alert_id
            
            # Check rate limiting
            if self._is_rate_limited(alert_id, severity):
                logger.debug(f"Alert {alert_id} is rate limited")
                return alert_id
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                title=title,
                message=message,
                severity=severity,
                category=category,
                status=AlertStatus.ACTIVE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                source=source,
                metadata=metadata or {}
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Save to database
            await self._save_alert_to_db(alert)
            
            # Send notifications
            await self._send_alert_notifications(alert)
            
            # Schedule escalation if needed
            await self._schedule_escalation(alert)
            
            logger.info(f"🚨 Alert created: {title} [{severity.value}]")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return ""
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            # Cancel escalation
            if alert_id in self.escalation_tasks:
                self.escalation_tasks[alert_id].cancel()
                del self.escalation_tasks[alert_id]
            
            # Update in database
            await self._update_alert_in_db(alert)
            
            # Send acknowledgment notification
            await self._send_acknowledgment_notification(alert, acknowledged_by)
            
            logger.info(f"✅ Alert acknowledged: {alert.title} by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None) -> bool:
        """Resolve an alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            # Cancel escalation
            if alert_id in self.escalation_tasks:
                self.escalation_tasks[alert_id].cancel()
                del self.escalation_tasks[alert_id]
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Update in database
            await self._update_alert_in_db(alert)
            
            # Send resolution notification
            await self._send_resolution_notification(alert, resolved_by)
            
            logger.info(f"✅ Alert resolved: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    async def suppress_alert(self, alert_id: str, duration_minutes: int) -> bool:
        """Suppress an alert for specified duration"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
            alert.updated_at = datetime.utcnow()
            
            self.suppressed_alerts.add(alert_id)
            
            # Cancel escalation
            if alert_id in self.escalation_tasks:
                self.escalation_tasks[alert_id].cancel()
                del self.escalation_tasks[alert_id]
            
            # Update in database
            await self._update_alert_in_db(alert)
            
            logger.info(f"🔇 Alert suppressed: {alert.title} for {duration_minutes} minutes")
            return True
            
        except Exception as e:
            logger.error(f"Error suppressing alert: {e}")
            return False
    
    async def get_active_alerts(self, category: Optional[AlertCategory] = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by category"""
        alerts = list(self.active_alerts.values())
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        # Sort by severity and creation time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3
        }
        
        alerts.sort(key=lambda a: (severity_order[a.severity], a.created_at))
        return alerts
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        try:
            now = datetime.utcnow()
            last_24h = now - timedelta(hours=24)
            last_7d = now - timedelta(days=7)
            
            # Count alerts by severity
            severity_counts = defaultdict(int)
            category_counts = defaultdict(int)
            recent_24h = 0
            recent_7d = 0
            
            for alert in self.alert_history:
                severity_counts[alert.severity.value] += 1
                category_counts[alert.category.value] += 1
                
                if alert.created_at >= last_24h:
                    recent_24h += 1
                if alert.created_at >= last_7d:
                    recent_7d += 1
            
            return {
                "active_alerts": len(self.active_alerts),
                "total_alerts": len(self.alert_history),
                "alerts_24h": recent_24h,
                "alerts_7d": recent_7d,
                "by_severity": dict(severity_counts),
                "by_category": dict(category_counts),
                "suppressed_alerts": len(self.suppressed_alerts),
                "escalation_tasks": len(self.escalation_tasks)
            }
            
        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {}
    
    async def process_system_metrics(self, metrics: Dict[str, Any]):
        """Process system metrics and create alerts based on rules"""
        try:
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Evaluate rule condition
                if self._evaluate_rule_condition(rule.condition, metrics):
                    await self.create_alert(
                        title=f"System Alert: {rule_name.replace('_', ' ').title()}",
                        message=f"Rule '{rule_name}' triggered: {rule.condition}",
                        severity=rule.severity,
                        category=rule.category,
                        source="system_monitor",
                        metadata={
                            "rule": rule_name,
                            "metrics": metrics,
                            "condition": rule.condition
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error processing system metrics: {e}")
    
    def _generate_alert_id(self, title: str, source: str, metadata: Dict[str, Any]) -> str:
        """Generate unique alert ID"""
        content = f"{title}:{source}:{json.dumps(metadata, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _is_rate_limited(self, alert_id: str, severity: AlertSeverity) -> bool:
        """Check if alert is rate limited"""
        now = datetime.utcnow()
        
        # Get rate limit based on severity
        rate_limits = {
            AlertSeverity.CRITICAL: timedelta(minutes=1),
            AlertSeverity.HIGH: timedelta(minutes=5),
            AlertSeverity.MEDIUM: timedelta(minutes=15),
            AlertSeverity.LOW: timedelta(minutes=30)
        }
        
        rate_limit = rate_limits.get(severity, timedelta(minutes=15))
        
        if alert_id in self.last_notifications:
            time_since_last = now - self.last_notifications[alert_id]
            if time_since_last < rate_limit:
                return True
        
        self.last_notifications[alert_id] = now
        return False
    
    def _evaluate_rule_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate rule condition against metrics"""
        try:
            # Simple condition evaluation
            # In production, you might want to use a more sophisticated expression evaluator
            
            # Replace metric names with values
            eval_condition = condition
            for key, value in metrics.items():
                eval_condition = eval_condition.replace(key, str(value))
            
            # Basic safety check - only allow simple comparisons
            allowed_operators = ['>', '<', '>=', '<=', '==', '!=']
            if not any(op in eval_condition for op in allowed_operators):
                return False
            
            # Evaluate condition (be careful with eval in production!)
            return eval(eval_condition)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        try:
            # Get notification channels for alert category
            rule = self.alert_rules.get(f"{alert.category.value}_alert")
            channels = rule.notification_channels if rule else ["discord", "log"]
            
            # Discord notification
            if "discord" in channels:
                await self._send_discord_notification(alert)
            
            # Log notification
            if "log" in channels:
                self._send_log_notification(alert)
            
            alert.notification_sent = True
            
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")
    
    async def _send_discord_notification(self, alert: Alert):
        """Send Discord notification"""
        try:
            # Format message based on severity
            emoji_map = {
                AlertSeverity.CRITICAL: "🚨",
                AlertSeverity.HIGH: "⚠️",
                AlertSeverity.MEDIUM: "⚡",
                AlertSeverity.LOW: "ℹ️"
            }
            
            emoji = emoji_map.get(alert.severity, "📢")
            color_map = {
                AlertSeverity.CRITICAL: 0xFF0000,  # Red
                AlertSeverity.HIGH: 0xFF8C00,      # Orange
                AlertSeverity.MEDIUM: 0xFFD700,    # Gold
                AlertSeverity.LOW: 0x00CED1        # Turquoise
            }
            
            embed = {
                "title": f"{emoji} {alert.title}",
                "description": alert.message,
                "color": color_map.get(alert.severity, 0x808080),
                "fields": [
                    {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
                    {"name": "Category", "value": alert.category.value.upper(), "inline": True},
                    {"name": "Source", "value": alert.source, "inline": True},
                    {"name": "Time", "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"), "inline": False}
                ],
                "footer": {"text": f"Alert ID: {alert.id}"}
            }
            
            if alert.metadata:
                metadata_str = "\n".join([f"**{k}:** {v}" for k, v in alert.metadata.items()])
                embed["fields"].append({"name": "Details", "value": metadata_str, "inline": False})
            
            await self.discord_client.send_embed(embed)
            
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
    
    def _send_log_notification(self, alert: Alert):
        """Send log notification"""
        log_level_map = {
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.HIGH: logging.ERROR,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.LOW: logging.INFO
        }
        
        log_level = log_level_map.get(alert.severity, logging.INFO)
        logger.log(log_level, f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")
    
    async def _schedule_escalation(self, alert: Alert):
        """Schedule alert escalation"""
        try:
            rule = self.alert_rules.get(f"{alert.category.value}_alert")
            if not rule or alert.escalation_level >= rule.max_escalations:
                return
            
            escalation_delay = rule.escalation_minutes * 60  # Convert to seconds
            
            async def escalate():
                await asyncio.sleep(escalation_delay)
                if alert.id in self.active_alerts and alert.status == AlertStatus.ACTIVE:
                    alert.escalation_level += 1
                    alert.updated_at = datetime.utcnow()
                    
                    # Send escalation notification
                    escalated_alert = Alert(
                        id=f"{alert.id}_escalation_{alert.escalation_level}",
                        title=f"ESCALATED: {alert.title}",
                        message=f"Alert has been escalated (Level {alert.escalation_level}): {alert.message}",
                        severity=alert.severity,
                        category=alert.category,
                        status=AlertStatus.ACTIVE,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        source=alert.source,
                        metadata={**alert.metadata, "escalation_level": alert.escalation_level}
                    )
                    
                    await self._send_alert_notifications(escalated_alert)
                    
                    # Schedule next escalation
                    if alert.escalation_level < rule.max_escalations:
                        await self._schedule_escalation(alert)
            
            task = asyncio.create_task(escalate())
            self.escalation_tasks[alert.id] = task
            
        except Exception as e:
            logger.error(f"Error scheduling escalation: {e}")
    
    async def _send_acknowledgment_notification(self, alert: Alert, acknowledged_by: str):
        """Send acknowledgment notification"""
        try:
            embed = {
                "title": "✅ Alert Acknowledged",
                "description": f"Alert '{alert.title}' has been acknowledged by {acknowledged_by}",
                "color": 0x00FF00,
                "fields": [
                    {"name": "Alert ID", "value": alert.id, "inline": True},
                    {"name": "Acknowledged By", "value": acknowledged_by, "inline": True},
                    {"name": "Time", "value": alert.acknowledged_at.strftime("%Y-%m-%d %H:%M:%S UTC"), "inline": False}
                ]
            }
            
            await self.discord_client.send_embed(embed)
            
        except Exception as e:
            logger.error(f"Error sending acknowledgment notification: {e}")
    
    async def _send_resolution_notification(self, alert: Alert, resolved_by: Optional[str]):
        """Send resolution notification"""
        try:
            embed = {
                "title": "✅ Alert Resolved",
                "description": f"Alert '{alert.title}' has been resolved",
                "color": 0x00FF00,
                "fields": [
                    {"name": "Alert ID", "value": alert.id, "inline": True},
                    {"name": "Duration", "value": str(alert.resolved_at - alert.created_at), "inline": True},
                    {"name": "Time", "value": alert.resolved_at.strftime("%Y-%m-%d %H:%M:%S UTC"), "inline": False}
                ]
            }
            
            if resolved_by:
                embed["fields"].append({"name": "Resolved By", "value": resolved_by, "inline": True})
            
            await self.discord_client.send_embed(embed)
            
        except Exception as e:
            logger.error(f"Error sending resolution notification: {e}")
    
    async def _save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO alerts (
                        id, title, message, severity, category, status,
                        created_at, updated_at, source, metadata,
                        escalation_level, notification_sent
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, 
                alert.id, alert.title, alert.message, alert.severity.value,
                alert.category.value, alert.status.value, alert.created_at,
                alert.updated_at, alert.source, json.dumps(alert.metadata),
                alert.escalation_level, alert.notification_sent
                )
                
        except Exception as e:
            logger.error(f"Error saving alert to database: {e}")
    
    async def _update_alert_in_db(self, alert: Alert):
        """Update alert in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE alerts SET
                        status = $2, updated_at = $3, acknowledged_by = $4,
                        acknowledged_at = $5, resolved_at = $6, escalation_level = $7,
                        suppressed_until = $8
                    WHERE id = $1
                """,
                alert.id, alert.status.value, alert.updated_at,
                alert.acknowledged_by, alert.acknowledged_at, alert.resolved_at,
                alert.escalation_level, alert.suppressed_until
                )
                
        except Exception as e:
            logger.error(f"Error updating alert in database: {e}")
    
    async def cleanup_old_alerts(self, days: int = 30):
        """Clean up old resolved alerts"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            async with self.db_pool.acquire() as conn:
                deleted = await conn.fetchval("""
                    DELETE FROM alerts 
                    WHERE status = 'resolved' AND resolved_at < $1
                    RETURNING count(*)
                """, cutoff)
                
                logger.info(f"🧹 Cleaned up {deleted} old alerts")
                
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    async def load_alerts_from_db(self):
        """Load active alerts from database on startup"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM alerts 
                    WHERE status IN ('active', 'acknowledged', 'suppressed')
                    ORDER BY created_at DESC
                """)
                
                for row in rows:
                    alert = Alert(
                        id=row['id'],
                        title=row['title'],
                        message=row['message'],
                        severity=AlertSeverity(row['severity']),
                        category=AlertCategory(row['category']),
                        status=AlertStatus(row['status']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        source=row['source'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                        acknowledged_by=row['acknowledged_by'],
                        acknowledged_at=row['acknowledged_at'],
                        resolved_at=row['resolved_at'],
                        escalation_level=row['escalation_level'],
                        notification_sent=row['notification_sent'],
                        suppressed_until=row['suppressed_until']
                    )
                    
                    self.active_alerts[alert.id] = alert
                    
                logger.info(f"📥 Loaded {len(self.active_alerts)} active alerts from database")
                
        except Exception as e:
            logger.error(f"Error loading alerts from database: {e}")

# Utility functions
async def create_system_alert(alert_manager: AlertManager, title: str, message: str, severity: AlertSeverity = AlertSeverity.MEDIUM):
    """Quick function to create system alerts"""
    return await alert_manager.create_alert(
        title=title,
        message=message,
        severity=severity,
        category=AlertCategory.SYSTEM,
        source="system"
    )

async def create_trading_alert(alert_manager: AlertManager, title: str, message: str, severity: AlertSeverity = AlertSeverity.HIGH):
    """Quick function to create trading alerts"""
    return await alert_manager.create_alert(
        title=title,
        message=message,
        severity=severity,
        category=AlertCategory.TRADING,
        source="trading_engine"
    )
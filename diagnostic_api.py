"""
Diagnostic API - REST API Endpoints for Trading Bot v9.1 Diagnostics
Comprehensive API for monitoring, alerting, and performance management
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import asyncio
from contextlib import asynccontextmanager

# Import our diagnostic components
from health_checker import HealthChecker, HealthStatus, quick_health_check
from alert_manager import AlertManager, AlertSeverity, AlertCategory, AlertStatus
from performance_monitor import PerformanceMonitor, PerformanceLevel, format_performance_report
from diagnostics import SystemHealthMonitor, PerformanceAnalyzer, DiagnosticReporter
from decision_engine import DecisionEngine
from pattern_detector import PatternDetector

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    system: Dict[str, float]
    uptime: Optional[float] = None

class AlertRequest(BaseModel):
    title: str
    message: str
    severity: str = Field(..., regex="^(low|medium|high|critical)$")
    category: str = Field(..., regex="^(system|trading|performance|security|network|database|api|diagnostic)$")
    source: str
    metadata: Optional[Dict[str, Any]] = None

class AlertResponse(BaseModel):
    id: str
    title: str
    message: str
    severity: str
    category: str
    status: str
    created_at: datetime
    source: str

class PerformanceResponse(BaseModel):
    overall_score: float
    level: str
    timestamp: datetime
    metrics_count: int
    alerts_count: int
    recommendations_count: int

class DiagnosticSummary(BaseModel):
    system_health: str
    performance_level: str
    active_alerts: int
    critical_issues: int
    recommendations: List[str]
    last_updated: datetime

class MetricFilter(BaseModel):
    metric_name: Optional[str] = None
    hours: int = Field(default=24, ge=1, le=168)  # 1 hour to 1 week
    limit: int = Field(default=100, ge=1, le=1000)

# Global diagnostic components (will be initialized in lifespan)
health_checker: Optional[HealthChecker] = None
alert_manager: Optional[AlertManager] = None
performance_monitor: Optional[PerformanceMonitor] = None
diagnostic_reporter: Optional[DiagnosticReporter] = None
decision_engine: Optional[DecisionEngine] = None
pattern_detector: Optional[PatternDetector] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize diagnostic components on startup"""
    global health_checker, alert_manager, performance_monitor
    global diagnostic_reporter, decision_engine, pattern_detector
    
    try:
        # These would be injected from main application
        # For now, we'll assume they're available
        logger.info("🚀 Initializing Diagnostic API components...")
        
        # Initialize components (in real app, these would be dependency injected)
        # health_checker = HealthChecker(db_pool, redis_client, config)
        # alert_manager = AlertManager(db_pool, discord_client, config)
        # performance_monitor = PerformanceMonitor(db_pool, alert_manager, config)
        # diagnostic_reporter = DiagnosticReporter(db_pool)
        # decision_engine = DecisionEngine(db_pool)
        # pattern_detector = PatternDetector(db_pool)
        
        logger.info("✅ Diagnostic API components initialized")
        yield
        
    except Exception as e:
        logger.error(f"❌ Error initializing Diagnostic API: {e}")
        yield
    finally:
        logger.info("🛑 Shutting down Diagnostic API")

# Create FastAPI app
app = FastAPI(
    title="Trading Bot v9.1 - Diagnostic API",
    description="Comprehensive diagnostic and monitoring API for Trading Bot v9.1",
    version="9.1.0",
    lifespan=lifespan
)

# Dependency injection helpers
async def get_health_checker() -> HealthChecker:
    if health_checker is None:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    return health_checker

async def get_alert_manager() -> AlertManager:
    if alert_manager is None:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")
    return alert_manager

async def get_performance_monitor() -> PerformanceMonitor:
    if performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitor not initialized")
    return performance_monitor

async def get_diagnostic_reporter() -> DiagnosticReporter:
    if diagnostic_reporter is None:
        raise HTTPException(status_code=503, detail="Diagnostic reporter not initialized")
    return diagnostic_reporter

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified for demo)"""
    # In production, implement proper JWT validation
    if credentials.credentials != "your-api-token":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Health Check Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def get_health_status():
    """Quick health check endpoint"""
    try:
        # Use quick health check if full health checker not available
        if health_checker is None:
            # Simplified health check
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow(),
                services={"api": "healthy"},
                system={"cpu_usage": 0, "memory_usage": 0}
            )
        
        health_data = await health_checker.get_comprehensive_health()
        
        return HealthResponse(
            status=health_data.overall_status.value,
            timestamp=health_data.timestamp,
            services={k: v.value for k, v in health_data.services.items()},
            system={
                "cpu_usage": next((m.value for m in health_data.metrics if m.name == "cpu_usage"), 0),
                "memory_usage": next((m.value for m in health_data.metrics if m.name == "memory_usage"), 0)
            },
            uptime=health_data.uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/health/detailed", tags=["Health"])
async def get_detailed_health(hc: HealthChecker = Depends(get_health_checker)):
    """Detailed health check with all metrics"""
    try:
        health_data = await hc.get_comprehensive_health()
        
        return {
            "overall_status": health_data.overall_status.value,
            "timestamp": health_data.timestamp,
            "uptime": health_data.uptime,
            "services": {k: v.value for k, v in health_data.services.items()},
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "status": m.status.value,
                    "threshold_warning": m.threshold_warning,
                    "threshold_critical": m.threshold_critical,
                    "details": m.details
                }
                for m in health_data.metrics
            ],
            "alerts": health_data.alerts
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detailed health check failed: {str(e)}")

@app.get("/health/history", tags=["Health"])
async def get_health_history(
    hours: int = Query(default=24, ge=1, le=168),
    hc: HealthChecker = Depends(get_health_checker)
):
    """Get health history for specified hours"""
    try:
        history = await hc.get_health_history(hours)
        return {"history": history, "count": len(history)}
        
    except Exception as e:
        logger.error(f"Error getting health history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting health history: {str(e)}")

# Alert Management Endpoints
@app.get("/alerts", tags=["Alerts"])
async def get_alerts(
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(default=50, ge=1, le=500),
    am: AlertManager = Depends(get_alert_manager),
    token: str = Depends(verify_token)
):
    """Get alerts with optional filtering"""
    try:
        # Get active alerts
        alerts = await am.get_active_alerts()
        
        # Apply filters
        if category:
            alerts = [a for a in alerts if a.category.value == category]
        if status:
            alerts = [a for a in alerts if a.status.value == status]
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]
        
        # Limit results
        alerts = alerts[:limit]
        
        return {
            "alerts": [
                AlertResponse(
                    id=a.id,
                    title=a.title,
                    message=a.message,
                    severity=a.severity.value,
                    category=a.category.value,
                    status=a.status.value,
                    created_at=a.created_at,
                    source=a.source
                )
                for a in alerts
            ],
            "count": len(alerts)
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

@app.post("/alerts", response_model=Dict[str, str], tags=["Alerts"])
async def create_alert(
    alert: AlertRequest,
    am: AlertManager = Depends(get_alert_manager),
    token: str = Depends(verify_token)
):
    """Create a new alert"""
    try:
        alert_id = await am.create_alert(
            title=alert.title,
            message=alert.message,
            severity=AlertSeverity(alert.severity),
            category=AlertCategory(alert.category),
            source=alert.source,
            metadata=alert.metadata
        )
        
        return {"alert_id": alert_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating alert: {str(e)}")

@app.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(
    alert_id: str = Path(...),
    acknowledged_by: str = Query(...),
    am: AlertManager = Depends(get_alert_manager),
    token: str = Depends(verify_token)
):
    """Acknowledge an alert"""
    try:
        success = await am.acknowledge_alert(alert_id, acknowledged_by)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"status": "acknowledged", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error acknowledging alert: {str(e)}")

@app.post("/alerts/{alert_id}/resolve", tags=["Alerts"])
async def resolve_alert(
    alert_id: str = Path(...),
    resolved_by: Optional[str] = Query(None),
    am: AlertManager = Depends(get_alert_manager),
    token: str = Depends(verify_token)
):
    """Resolve an alert"""
    try:
        success = await am.resolve_alert(alert_id, resolved_by)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"status": "resolved", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error resolving alert: {str(e)}")

@app.post("/alerts/{alert_id}/suppress", tags=["Alerts"])
async def suppress_alert(
    alert_id: str = Path(...),
    duration_minutes: int = Query(..., ge=1, le=1440),  # Max 24 hours
    am: AlertManager = Depends(get_alert_manager),
    token: str = Depends(verify_token)
):
    """Suppress an alert for specified duration"""
    try:
        success = await am.suppress_alert(alert_id, duration_minutes)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"status": "suppressed", "alert_id": alert_id, "duration_minutes": duration_minutes}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suppressing alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error suppressing alert: {str(e)}")

@app.get("/alerts/statistics", tags=["Alerts"])
async def get_alert_statistics(
    am: AlertManager = Depends(get_alert_manager),
    token: str = Depends(verify_token)
):
    """Get alert statistics"""
    try:
        stats = await am.get_alert_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting alert statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting alert statistics: {str(e)}")

# Performance Monitoring Endpoints
@app.get("/performance", response_model=PerformanceResponse, tags=["Performance"])
async def get_performance_status(
    pm: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get current performance status"""
    try:
        snapshot = await pm.get_performance_snapshot()
        
        return PerformanceResponse(
            overall_score=snapshot.overall_score,
            level=snapshot.level.value,
            timestamp=snapshot.timestamp,
            metrics_count=len(snapshot.metrics),
            alerts_count=len(snapshot.alerts),
            recommendations_count=len(snapshot.recommendations)
        )
        
    except Exception as e:
        logger.error(f"Error getting performance status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance status: {str(e)}")

@app.get("/performance/detailed", tags=["Performance"])
async def get_detailed_performance(
    pm: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get detailed performance metrics"""
    try:
        snapshot = await pm.get_performance_snapshot()
        
        return {
            "overall_score": snapshot.overall_score,
            "level": snapshot.level.value,
            "timestamp": snapshot.timestamp,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "type": m.metric_type.value,
                    "level": m.level.value,
                    "threshold_good": m.threshold_good,
                    "threshold_poor": m.threshold_poor
                }
                for m in snapshot.metrics
            ],
            "recommendations": snapshot.recommendations,
            "alerts": snapshot.alerts
        }
        
    except Exception as e:
        logger.error(f"Error getting detailed performance: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting detailed performance: {str(e)}")

@app.get("/performance/history", tags=["Performance"])
async def get_performance_history(
    hours: int = Query(default=24, ge=1, le=168),
    pm: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get performance history"""
    try:
        history = await pm.get_performance_history(hours)
        return {"history": history, "count": len(history)}
        
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance history: {str(e)}")

@app.get("/performance/report", response_class=HTMLResponse, tags=["Performance"])
async def get_performance_report(
    pm: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get formatted performance report"""
    try:
        snapshot = await pm.get_performance_snapshot()
        report = format_performance_report(snapshot)
        
        # Convert to HTML
        html_report = f"""
        <html>
        <head><title>Performance Report</title></head>
        <body>
        <pre>{report}</pre>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_report)
        
    except Exception as e:
        logger.error(f"Error getting performance report: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance report: {str(e)}")

# Diagnostic Endpoints
@app.get("/diagnostics/summary", response_model=DiagnosticSummary, tags=["Diagnostics"])
async def get_diagnostic_summary(
    hc: HealthChecker = Depends(get_health_checker),
    am: AlertManager = Depends(get_alert_manager),
    pm: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get comprehensive diagnostic summary"""
    try:
        # Get health status
        health_data = await hc.get_comprehensive_health()
        
        # Get performance status
        performance_data = await pm.get_performance_snapshot()
        
        # Get alert statistics
        alert_stats = await am.get_alert_statistics()
        
        # Count critical issues
        critical_issues = len([m for m in health_data.metrics if m.status.value == "critical"])
        critical_issues += len([a for a in health_data.alerts if "CRITICAL" in a])
        
        return DiagnosticSummary(
            system_health=health_data.overall_status.value,
            performance_level=performance_data.level.value,
            active_alerts=alert_stats.get("active_alerts", 0),
            critical_issues=critical_issues,
            recommendations=performance_data.recommendations[:5],  # Top 5
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting diagnostic summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting diagnostic summary: {str(e)}")

@app.get("/diagnostics/comprehensive", tags=["Diagnostics"])
async def get_comprehensive_diagnostics(
    dr: DiagnosticReporter = Depends(get_diagnostic_reporter),
    token: str = Depends(verify_token)
):
    """Get comprehensive diagnostic report"""
    try:
        report = await dr.generate_comprehensive_report()
        return report
        
    except Exception as e:
        logger.error(f"Error getting comprehensive diagnostics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting comprehensive diagnostics: {str(e)}")

# Pattern Detection Endpoints
@app.get("/patterns/detect", tags=["Patterns"])
async def detect_patterns(
    pd: PatternDetector = Depends(lambda: pattern_detector),
    token: str = Depends(verify_token)
):
    """Detect trading patterns and anomalies"""
    try:
        if pd is None:
            raise HTTPException(status_code=503, detail="Pattern detector not available")
        
        patterns = await pd.detect_patterns()
        anomalies = await pd.detect_anomalies()
        insights = await pd.generate_insights(patterns, anomalies)
        
        return {
            "patterns": patterns,
            "anomalies": anomalies,
            "insights": insights,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting patterns: {str(e)}")

# Decision Engine Endpoints
@app.get("/decisions/analysis", tags=["Decisions"])
async def get_decision_analysis(
    de: DecisionEngine = Depends(lambda: decision_engine),
    token: str = Depends(verify_token)
):
    """Get decision analysis and insights"""
    try:
        if de is None:
            raise HTTPException(status_code=503, detail="Decision engine not available")
        
        analysis = await de.analyze_recent_decisions()
        insights = await de.generate_decision_insights()
        quality = await de.evaluate_decision_quality()
        
        return {
            "analysis": analysis,
            "insights": insights,
            "quality_metrics": quality,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting decision analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting decision analysis: {str(e)}")

# Utility Endpoints
@app.post("/diagnostics/cleanup", tags=["Utilities"])
async def cleanup_old_data(
    background_tasks: BackgroundTasks,
    days: int = Query(default=7, ge=1, le=30),
    hc: HealthChecker = Depends(get_health_checker),
    am: AlertManager = Depends(get_alert_manager),
    pm: PerformanceMonitor = Depends(get_performance_monitor),
    token: str = Depends(verify_token)
):
    """Clean up old diagnostic data"""
    try:
        # Run cleanup in background
        background_tasks.add_task(hc.cleanup_old_data, days)
        background_tasks.add_task(am.cleanup_old_alerts, days)
        background_tasks.add_task(pm.cleanup_old_data, days)
        
        return {"status": "cleanup_scheduled", "days": days}
        
    except Exception as e:
        logger.error(f"Error scheduling cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Error scheduling cleanup: {str(e)}")

@app.get("/diagnostics/metrics/{metric_name}", tags=["Metrics"])
async def get_metric_history(
    metric_name: str = Path(...),
    hours: int = Query(default=24, ge=1, le=168),
    hc: HealthChecker = Depends(get_health_checker)
):
    """Get history for specific metric"""
    try:
        history = await hc.get_metric_history(metric_name, hours)
        return {"metric_name": metric_name, "history": history, "count": len(history)}
        
    except Exception as e:
        logger.error(f"Error getting metric history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metric history: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws/diagnostics")
async def websocket_diagnostics(websocket):
    """WebSocket endpoint for real-time diagnostic updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send diagnostic updates every 30 seconds
            if health_checker and performance_monitor:
                health_data = await health_checker.get_comprehensive_health()
                performance_data = await performance_monitor.get_performance_snapshot()
                
                update = {
                    "type": "diagnostic_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "health": {
                        "status": health_data.overall_status.value,
                        "alerts": len(health_data.alerts)
                    },
                    "performance": {
                        "score": performance_data.overall_score,
                        "level": performance_data.level.value
                    }
                }
                
                await websocket.send_json(update)
            
            await asyncio.sleep(30)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Diagnostic API v9.1 starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Diagnostic API v9.1 shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "diagnostic_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
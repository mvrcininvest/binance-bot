"""
Diagnostic Dashboard - Interactive Web Dashboard for Trading Bot v9.1
Real-time monitoring, visualization, and control interface
"""

from fastapi import FastAPI, Request, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import asyncio
import logging
from pathlib import Path

# Import diagnostic components
from health_checker import HealthChecker, HealthStatus
from alert_manager import AlertManager, AlertSeverity, AlertCategory
from performance_monitor import PerformanceMonitor, PerformanceLevel
from diagnostics import SystemHealthMonitor, PerformanceAnalyzer, DiagnosticReporter
from decision_engine import DecisionEngine
from pattern_detector import PatternDetector

logger = logging.getLogger(__name__)

class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_data: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_info: Dict = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_data[websocket] = client_info or {}
        logger.info(f"📱 Dashboard client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_data.pop(websocket, None)
            logger.info(f"📱 Dashboard client disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

# Global components (will be injected from main app)
health_checker: Optional[HealthChecker] = None
alert_manager: Optional[AlertManager] = None
performance_monitor: Optional[PerformanceMonitor] = None
diagnostic_reporter: Optional[DiagnosticReporter] = None
decision_engine: Optional[DecisionEngine] = None
pattern_detector: Optional[PatternDetector] = None

# Connection manager for WebSocket connections
manager = ConnectionManager()

# Create FastAPI app for dashboard
dashboard_app = FastAPI(
    title="Trading Bot v9.1 - Diagnostic Dashboard",
    description="Interactive web dashboard for comprehensive bot monitoring",
    version="9.1.0"
)

# Setup templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

# Create directories if they don't exist
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
dashboard_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Dashboard Routes
@dashboard_app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Trading Bot v9.1 - Diagnostic Dashboard",
        "version": "9.1.0"
    })

@dashboard_app.get("/health", response_class=HTMLResponse)
async def health_dashboard(request: Request):
    """Health monitoring dashboard"""
    return templates.TemplateResponse("health_dashboard.html", {
        "request": request,
        "title": "System Health Dashboard"
    })

@dashboard_app.get("/performance", response_class=HTMLResponse)
async def performance_dashboard(request: Request):
    """Performance monitoring dashboard"""
    return templates.TemplateResponse("performance_dashboard.html", {
        "request": request,
        "title": "Performance Dashboard"
    })

@dashboard_app.get("/alerts", response_class=HTMLResponse)
async def alerts_dashboard(request: Request):
    """Alerts management dashboard"""
    return templates.TemplateResponse("alerts_dashboard.html", {
        "request": request,
        "title": "Alerts Dashboard"
    })

@dashboard_app.get("/patterns", response_class=HTMLResponse)
async def patterns_dashboard(request: Request):
    """Pattern detection dashboard"""
    return templates.TemplateResponse("patterns_dashboard.html", {
        "request": request,
        "title": "Pattern Analysis Dashboard"
    })

@dashboard_app.get("/decisions", response_class=HTMLResponse)
async def decisions_dashboard(request: Request):
    """Decision analysis dashboard"""
    return templates.TemplateResponse("decisions_dashboard.html", {
        "request": request,
        "title": "Decision Analysis Dashboard"
    })

# API Endpoints for Dashboard Data
@dashboard_app.get("/api/dashboard/summary")
async def get_dashboard_summary():
    """Get comprehensive dashboard summary"""
    try:
        summary = {
            "timestamp": datetime.utcnow(),
            "system_status": "unknown",
            "performance_level": "unknown",
            "active_alerts": 0,
            "critical_issues": 0,
            "uptime": 0,
            "trading_active": False,
            "last_signal": None,
            "total_trades": 0,
            "success_rate": 0.0,
            "current_pnl": 0.0
        }
        
        # Get health data
        if health_checker:
            try:
                health_data = await health_checker.get_comprehensive_health()
                summary.update({
                    "system_status": health_data.overall_status.value,
                    "uptime": health_data.uptime or 0,
                    "services": {k: v.value for k, v in health_data.services.items()},
                    "system_metrics": {
                        m.name: {"value": m.value, "unit": m.unit, "status": m.status.value}
                        for m in health_data.metrics
                    }
                })
            except Exception as e:
                logger.error(f"Error getting health data: {e}")
        
        # Get performance data
        if performance_monitor:
            try:
                perf_data = await performance_monitor.get_performance_snapshot()
                summary.update({
                    "performance_level": perf_data.level.value,
                    "performance_score": perf_data.overall_score,
                    "recommendations": perf_data.recommendations[:3]  # Top 3
                })
            except Exception as e:
                logger.error(f"Error getting performance data: {e}")
        
        # Get alert data
        if alert_manager:
            try:
                alert_stats = await alert_manager.get_alert_statistics()
                summary.update({
                    "active_alerts": alert_stats.get("active_alerts", 0),
                    "critical_alerts": alert_stats.get("critical_alerts", 0),
                    "alert_trends": alert_stats.get("trends", {})
                })
            except Exception as e:
                logger.error(f"Error getting alert data: {e}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dashboard summary: {str(e)}")

@dashboard_app.get("/api/dashboard/health")
async def get_health_data():
    """Get detailed health data for dashboard"""
    try:
        if not health_checker:
            return {"error": "Health checker not available"}
        
        health_data = await health_checker.get_comprehensive_health()
        
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
        logger.error(f"Error getting health data: {e}")
        return {"error": str(e)}

@dashboard_app.get("/api/dashboard/performance")
async def get_performance_data():
    """Get performance data for dashboard"""
    try:
        if not performance_monitor:
            return {"error": "Performance monitor not available"}
        
        perf_data = await performance_monitor.get_performance_snapshot()
        
        return {
            "overall_score": perf_data.overall_score,
            "level": perf_data.level.value,
            "timestamp": perf_data.timestamp,
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
                for m in perf_data.metrics
            ],
            "recommendations": perf_data.recommendations,
            "alerts": perf_data.alerts
        }
        
    except Exception as e:
        logger.error(f"Error getting performance data: {e}")
        return {"error": str(e)}

@dashboard_app.get("/api/dashboard/alerts")
async def get_alerts_data():
    """Get alerts data for dashboard"""
    try:
        if not alert_manager:
            return {"error": "Alert manager not available"}
        
        # Get active alerts
        active_alerts = await alert_manager.get_active_alerts()
        
        # Get alert statistics
        alert_stats = await alert_manager.get_alert_statistics()
        
        return {
            "active_alerts": [
                {
                    "id": a.id,
                    "title": a.title,
                    "message": a.message,
                    "severity": a.severity.value,
                    "category": a.category.value,
                    "status": a.status.value,
                    "created_at": a.created_at,
                    "source": a.source
                }
                for a in active_alerts[:20]  # Limit to 20 most recent
            ],
            "statistics": alert_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts data: {e}")
        return {"error": str(e)}

@dashboard_app.get("/api/dashboard/patterns")
async def get_patterns_data():
    """Get pattern detection data for dashboard"""
    try:
        if not pattern_detector:
            return {"error": "Pattern detector not available"}
        
        patterns = await pattern_detector.detect_patterns()
        anomalies = await pattern_detector.detect_anomalies()
        insights = await pattern_detector.generate_insights(patterns, anomalies)
        
        return {
            "patterns": patterns,
            "anomalies": anomalies,
            "insights": insights,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting patterns data: {e}")
        return {"error": str(e)}

@dashboard_app.get("/api/dashboard/decisions")
async def get_decisions_data():
    """Get decision analysis data for dashboard"""
    try:
        if not decision_engine:
            return {"error": "Decision engine not available"}
        
        analysis = await decision_engine.analyze_recent_decisions()
        insights = await decision_engine.generate_decision_insights()
        quality = await decision_engine.evaluate_decision_quality()
        
        return {
            "analysis": analysis,
            "insights": insights,
            "quality_metrics": quality,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting decisions data: {e}")
        return {"error": str(e)}

# WebSocket endpoint for real-time updates
@dashboard_app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await manager.connect(websocket, {"type": "dashboard", "connected_at": datetime.utcnow()})
    
    try:
        while True:
            # Send dashboard updates every 10 seconds
            try:
                summary = await get_dashboard_summary()
                
                update = {
                    "type": "dashboard_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": summary
                }
                
                await manager.send_personal_message(json.dumps(update, default=str), websocket)
                
            except Exception as e:
                logger.error(f"Error sending dashboard update: {e}")
            
            await asyncio.sleep(10)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@dashboard_app.websocket("/ws/health")
async def websocket_health(websocket: WebSocket):
    """WebSocket endpoint for real-time health updates"""
    await manager.connect(websocket, {"type": "health", "connected_at": datetime.utcnow()})
    
    try:
        while True:
            try:
                health_data = await get_health_data()
                
                update = {
                    "type": "health_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": health_data
                }
                
                await manager.send_personal_message(json.dumps(update, default=str), websocket)
                
            except Exception as e:
                logger.error(f"Error sending health update: {e}")
            
            await asyncio.sleep(5)  # More frequent health updates
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@dashboard_app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alert updates"""
    await manager.connect(websocket, {"type": "alerts", "connected_at": datetime.utcnow()})
    
    try:
        while True:
            try:
                alerts_data = await get_alerts_data()
                
                update = {
                    "type": "alerts_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": alerts_data
                }
                
                await manager.send_personal_message(json.dumps(update, default=str), websocket)
                
            except Exception as e:
                logger.error(f"Error sending alerts update: {e}")
            
            await asyncio.sleep(15)  # Alert updates every 15 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Control endpoints
@dashboard_app.post("/api/dashboard/control/restart")
async def restart_system():
    """Restart system components (placeholder)"""
    try:
        # This would trigger a system restart
        # Implementation depends on your deployment setup
        
        await manager.broadcast({
            "type": "system_notification",
            "message": "System restart initiated",
            "severity": "warning",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"status": "restart_initiated", "timestamp": datetime.utcnow()}
        
    except Exception as e:
        logger.error(f"Error initiating restart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_app.post("/api/dashboard/control/pause_trading")
async def pause_trading():
    """Pause trading operations (placeholder)"""
    try:
        # This would pause trading
        # Implementation depends on your bot's architecture
        
        await manager.broadcast({
            "type": "trading_notification",
            "message": "Trading paused via dashboard",
            "severity": "info",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"status": "trading_paused", "timestamp": datetime.utcnow()}
        
    except Exception as e:
        logger.error(f"Error pausing trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_app.post("/api/dashboard/control/resume_trading")
async def resume_trading():
    """Resume trading operations (placeholder)"""
    try:
        # This would resume trading
        # Implementation depends on your bot's architecture
        
        await manager.broadcast({
            "type": "trading_notification",
            "message": "Trading resumed via dashboard",
            "severity": "info",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"status": "trading_resumed", "timestamp": datetime.utcnow()}
        
    except Exception as e:
        logger.error(f"Error resuming trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility functions
def initialize_dashboard_components(
    hc: HealthChecker,
    am: AlertManager,
    pm: PerformanceMonitor,
    dr: DiagnosticReporter,
    de: DecisionEngine,
    pd: PatternDetector
):
    """Initialize dashboard with diagnostic components"""
    global health_checker, alert_manager, performance_monitor
    global diagnostic_reporter, decision_engine, pattern_detector
    
    health_checker = hc
    alert_manager = am
    performance_monitor = pm
    diagnostic_reporter = dr
    decision_engine = de
    pattern_detector = pd
    
    logger.info("🎛️ Dashboard components initialized")

async def broadcast_notification(notification: Dict):
    """Broadcast notification to all connected dashboard clients"""
    await manager.broadcast({
        "type": "notification",
        "data": notification,
        "timestamp": datetime.utcnow().isoformat()
    })

# Background task for periodic updates
async def dashboard_background_task():
    """Background task for periodic dashboard updates"""
    while True:
        try:
            # Broadcast system status every 30 seconds
            if health_checker and performance_monitor:
                summary = await get_dashboard_summary()
                
                await manager.broadcast({
                    "type": "system_status",
                    "data": summary,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in dashboard background task: {e}")
            await asyncio.sleep(60)  # Wait longer on error

# HTML Templates (basic versions - you can enhance these)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .metric-card { transition: all 0.3s ease; }
        .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .real-time-indicator { 
            display: inline-block; 
            width: 10px; 
            height: 10px; 
            background: #28a745; 
            border-radius: 50%; 
            animation: pulse 2s infinite; 
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">
                <i class="fas fa-robot"></i> Trading Bot v9.1 Dashboard
            </a>
            <div class="navbar-nav ms-auto">
                <span class="real-time-indicator me-2"></span>
                <span class="text-light">Live</span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-md-3 mb-4">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <i class="fas fa-heartbeat fa-2x mb-2 status-healthy"></i>
                        <h5>System Health</h5>
                        <h3 id="system-status">Loading...</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-4">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <i class="fas fa-tachometer-alt fa-2x mb-2 status-warning"></i>
                        <h5>Performance</h5>
                        <h3 id="performance-level">Loading...</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-4">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <i class="fas fa-exclamation-triangle fa-2x mb-2 status-critical"></i>
                        <h5>Active Alerts</h5>
                        <h3 id="active-alerts">Loading...</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-4">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <i class="fas fa-chart-line fa-2x mb-2 text-info"></i>
                        <h5>Trading Status</h5>
                        <h3 id="trading-status">Loading...</h3>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-server"></i> System Metrics</h5>
                    </div>
                    <div class="card-body">
                        <div id="system-metrics">Loading system metrics...</div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-bell"></i> Recent Alerts</h5>
                    </div>
                    <div class="card-body">
                        <div id="recent-alerts">Loading recent alerts...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws/dashboard`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'dashboard_update') {
                updateDashboard(data.data);
            }
        };
        
        function updateDashboard(data) {
            document.getElementById('system-status').textContent = data.system_status || 'Unknown';
            document.getElementById('performance-level').textContent = data.performance_level || 'Unknown';
            document.getElementById('active-alerts').textContent = data.active_alerts || '0';
            document.getElementById('trading-status').textContent = data.trading_active ? 'Active' : 'Inactive';
            
            // Update system metrics
            if (data.system_metrics) {
                let metricsHtml = '';
                for (const [name, metric] of Object.entries(data.system_metrics)) {
                    const statusClass = metric.status === 'healthy' ? 'text-success' : 
                                       metric.status === 'warning' ? 'text-warning' : 'text-danger';
                    metricsHtml += `
                        <div class="d-flex justify-content-between mb-2">
                            <span>${name}:</span>
                            <span class="${statusClass}">${metric.value} ${metric.unit}</span>
                        </div>
                    `;
                }
                document.getElementById('system-metrics').innerHTML = metricsHtml;
            }
        }
        
        // Initial load
        fetch('/api/dashboard/summary')
            .then(response => response.json())
            .then(data => updateDashboard(data))
            .catch(error => console.error('Error loading dashboard data:', error));
    </script>
</body>
</html>
"""

# Create template files
def create_dashboard_templates():
    """Create basic HTML templates for dashboard"""
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Main dashboard template
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(DASHBOARD_HTML)
    
    logger.info("📄 Dashboard templates created")

if __name__ == "__main__":
    import uvicorn
    
    # Create templates
    create_dashboard_templates()
    
    # Run dashboard
    uvicorn.run(
        "diagnostic_dashboard:dashboard_app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
# analytics.py (v1.1 - Zintegrowany z Centrum Dowodzenia)
import logging
from datetime import datetime, timedelta
from typing import Any

from database import Session, Trade

logger = logging.getLogger("analytics")


class AnalyticsEngine:
    def __init__(self, bot_instance=None):
        self.bot = bot_instance
        self.session = Session()
        self.market_state = {"volatility": "normal", "trend": "neutral"}
        self.performance_metrics = {}
        self.mode_scores = {"Conservative": 0.0, "Balanced": 50.0, "Aggressive": 0.0}
        logger.info("Advanced Analytics Engine initialized")
        self.analyze_recent_performance()

    def analyze_market_conditions(self):
        # Implementacja bez zmian
        pass

    def calculate_mode_scores(self) -> str:
        # Implementacja bez zmian
        pass

    def analyze_recent_performance(self):
        # Implementacja bez zmian
        pass

    def get_pnl_report_for_period(self, period: str) -> dict[str, Any]:
        """Oblicza PnL i statystyki dla zadanego okresu."""
        days = {"daily": 1, "weekly": 7, "monthly": 30}.get(period, 7)
        start_date = datetime.utcnow() - timedelta(days=days)

        trades = (
            self.session.query(Trade)
            .filter(Trade.entry_time >= start_date, Trade.status == "closed")
            .all()
        )

        if not trades:
            return {"period": period, "total_trades": 0, "total_pnl": 0, "win_rate": 0}

        total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
        winning_trades = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)

        return {
            "period": period,
            "total_trades": len(trades),
            "total_pnl": total_pnl,
            "win_rate": (winning_trades / len(trades)) * 100 if len(trades) > 0 else 0,
        }

    def get_performance_by_symbol(self, symbol: str) -> dict[str, Any]:
        """Generuje raport performance dla pojedynczego symbolu."""
        trades = (
            self.session.query(Trade).filter(Trade.symbol == symbol, Trade.status == "closed").all()
        )
        if not trades:
            return {
                "symbol": symbol,
                "message": "Brak zamkniętych transakcji dla tego symbolu.",
            }

        total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
        winning_trades = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)

        return {
            "symbol": symbol,
            "total_trades": len(trades),
            "total_pnl": total_pnl,
            "win_rate": (winning_trades / len(trades)) * 100 if len(trades) > 0 else 0,
        }

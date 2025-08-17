import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import desc

from database import Session, Trade

logger = logging.getLogger("predictive_analytics")


class PredictiveAnalytics:
    """System predykcyjnej analityki - zbiera dane nawet gdy ML jest wyłączone."""

    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.min_trades_for_prediction = 50
        logger.info("Predictive Analytics System initialized (passive mode)")

    def can_make_predictions(self) -> bool:
        """Sprawdza czy mamy wystarczająco danych."""
        with Session() as session:
            trade_count = (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.is_dry_run.is_(False))
                .count()
            )
            return trade_count >= self.min_trades_for_prediction

    def predict_market_movement(self, symbol: str, timeframe_hours: int = 4) -> dict[str, Any]:
        """Przewiduje ruch rynku na podstawie wzorców."""
        trade_count = self._get_trade_count()

        if not self.can_make_predictions():
            return {
                "direction": "neutral",
                "confidence": 0.0,
                "reason": f"Potrzeba minimum {self.min_trades_for_prediction} transakcji. Mamy: {trade_count}",
                "trades_available": trade_count,
            }

        try:
            patterns = self._analyze_historical_patterns(symbol, timeframe_hours)
            prediction_score = self._calculate_prediction_score(patterns)

            direction = (
                "bullish"
                if prediction_score > 0.6
                else "bearish" if prediction_score < 0.4 else "neutral"
            )
            confidence = abs(prediction_score - 0.5) * 2

            return {
                "direction": direction,
                "confidence": confidence,
                "prediction_score": prediction_score,
                "time_horizon_hours": timeframe_hours,
                "patterns_analyzed": len(patterns),
                "trades_available": trade_count,
            }

        except Exception as e:
            logger.error(f"Błąd predykcji dla {symbol}: {e}")
            return {"direction": "neutral", "confidence": 0.0, "error": str(e)}

    def _get_trade_count(self) -> int:
        with Session() as session:
            return (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.is_dry_run.is_(False))
                .count()
            )

    def _analyze_historical_patterns(self, symbol: str, hours: int) -> list[dict]:
        with Session() as session:
            cutoff_time = datetime.utcnow() - timedelta(days=7)

            trades = (
                session.query(Trade)
                .filter(
                    Trade.symbol == symbol,
                    Trade.status == "closed",
                    Trade.entry_time >= cutoff_time,
                    Trade.is_dry_run.is_(False),
                )
                .order_by(desc(Trade.entry_time))
                .limit(20)
                .all()
            )

            patterns = []
            for trade in trades:
                if trade.pnl is not None:
                    patterns.append(
                        {
                            "action": trade.action,
                            "tier": trade.tier,
                            "strength": trade.signal_strength,
                            "pnl": trade.pnl,
                            "profitable": trade.pnl > 0,
                        }
                    )

            return patterns

    def _calculate_prediction_score(self, patterns: list[dict]) -> float:
        if not patterns:
            return 0.5

        profitable_count = sum(1 for p in patterns if p.get("profitable", False))
        total_count = len(patterns)

        if total_count == 0:
            return 0.5

        base_score = profitable_count / total_count
        avg_strength = np.mean([p.get("strength", 0.5) for p in patterns])
        strength_modifier = (avg_strength - 0.5) * 0.2

        final_score = np.clip(base_score + strength_modifier, 0.0, 1.0)
        return float(final_score)


class AutoOptimizer:
    """System automatycznej optymalizacji parametrów."""

    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.min_trades_for_optimization = 30
        logger.info("Auto Optimizer initialized")

    def analyze_and_suggest_optimizations(self) -> dict[str, Any]:
        """Analizuje performance i sugeruje zmiany."""
        with Session() as session:
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_trades = (
                session.query(Trade)
                .filter(
                    Trade.status == "closed",
                    Trade.entry_time >= recent_cutoff,
                    Trade.is_dry_run.is_(False),
                )
                .count()
            )

            if recent_trades < self.min_trades_for_optimization:
                return {
                    "can_optimize": False,
                    "reason": f"Potrzeba minimum {self.min_trades_for_optimization} transakcji z ostatnich 7 dni. Mamy: {recent_trades}",
                    "suggestions": [],
                }

        try:
            performance = self._analyze_recent_performance()
            suggestions = self._generate_suggestions(performance)

            return {
                "can_optimize": True,
                "performance_analysis": performance,
                "suggestions": suggestions,
                "analysis_date": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Błąd analizy optymalizacji: {e}")
            return {"can_optimize": False, "error": str(e), "suggestions": []}

    def _analyze_recent_performance(self) -> dict[str, Any]:
        with Session() as session:
            cutoff_time = datetime.utcnow() - timedelta(days=7)

            trades = (
                session.query(Trade)
                .filter(
                    Trade.status == "closed",
                    Trade.entry_time >= cutoff_time,
                    Trade.is_dry_run.is_(False),
                    Trade.pnl.is_not(None),
                )
                .all()
            )

            if not trades:
                return {"error": "Brak transakcji do analizy"}

            profitable_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]

            total_pnl = sum(t.pnl for t in trades)
            win_rate = len(profitable_trades) / len(trades) if trades else 0

            avg_win = np.mean([t.pnl for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0

            profit_factor = (
                (sum(t.pnl for t in profitable_trades) / sum(abs(t.pnl) for t in losing_trades))
                if losing_trades
                else float("inf")
            )

            return {
                "total_trades": len(trades),
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
            }

    def _generate_suggestions(self, performance: dict[str, Any]) -> list[dict[str, str]]:
        suggestions = []

        win_rate = performance.get("win_rate", 0)
        profit_factor = performance.get("profit_factor", 0)

        if win_rate < 0.45:
            suggestions.append(
                {
                    "type": "signal_filtering",
                    "priority": "high",
                    "suggestion": "Rozważ zwiększenie minimalnej siły sygnału - niska win rate",
                    "current_value": f"{win_rate:.1%}",
                    "target": ">45%",
                }
            )

        if profit_factor < 1.2:
            suggestions.append(
                {
                    "type": "risk_reward",
                    "priority": "high",
                    "suggestion": "Rozważ optymalizację take profit lub stop loss",
                    "current_value": f"{profit_factor:.2f}",
                    "target": ">1.2",
                }
            )

        return suggestions

# mode_switcher.py (v2.0 - Naprawiony)
import logging
from datetime import datetime, timedelta

from database import Session, Trade

logger = logging.getLogger("mode_switcher")


class IntelligentModeSwitcher:
    """Automatyczne przełączanie trybów na podstawie performance."""

    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.last_mode_check = datetime.utcnow()
        self.mode_switch_cooldown = timedelta(hours=4)

    def should_switch_mode(self) -> tuple[bool, str, str]:
        """Sprawdza czy należy zmienić tryb."""
        if datetime.utcnow() - self.last_mode_check < self.mode_switch_cooldown:
            return False, self.bot.current_mode, "Cooldown aktywny"

        performance_24h = self._get_performance_metrics(hours=24)
        performance_7d = self._get_performance_metrics(hours=168)

        current_mode = self.bot.current_mode
        suggested_mode = self._analyze_and_suggest_mode(performance_24h, performance_7d)

        if suggested_mode != current_mode:
            reason = self._get_switch_reason(performance_24h, performance_7d, suggested_mode)
            return True, suggested_mode, reason

        return False, current_mode, "Obecny tryb optymalny"

    def _get_performance_metrics(self, hours: int) -> dict:
        """Pobiera metryki z ostatnich N godzin."""
        start_time = datetime.utcnow() - timedelta(hours=hours)

        with Session() as session:
            trades = (
                session.query(Trade)
                .filter(
                    Trade.entry_time >= start_time,
                    Trade.status == "closed",
                    Trade.is_dry_run.is_(False),
                )
                .all()
            )

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
            }

        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades if t.pnl)
        win_rate = len(winning_trades) / len(trades) if trades else 0

        gross_profit = sum(t.pnl for t in winning_trades if t.pnl)
        gross_loss = abs(sum(t.pnl for t in losing_trades if t.pnl))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max Drawdown calculation
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        for trade in sorted(trades, key=lambda x: x.entry_time):
            if trade.pnl:
                cumulative_pnl += trade.pnl
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                drawdown = (peak - cumulative_pnl) / abs(peak) * 100 if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_trades": len(trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
        }

    def _analyze_and_suggest_mode(self, perf_24h: dict, perf_7d: dict) -> str:
        """Sugeruje tryb na podstawie performance."""
        # Conservative conditions
        if (
            perf_24h["win_rate"] < 0.4
            or perf_24h["max_drawdown"] > 5.0
            or perf_7d["profit_factor"] < 1.0
        ):
            return "Conservative"

        # Aggressive conditions
        if (
            perf_24h["win_rate"] > 0.65
            and perf_24h["profit_factor"] > 1.8
            and perf_24h["max_drawdown"] < 2.0
            and perf_7d["total_pnl"] > 0
        ):
            return "Aggressive"

        return "Balanced"

    def _get_switch_reason(self, perf_24h: dict, perf_7d: dict, new_mode: str) -> str:
        """Generuje powód zmiany trybu."""
        if new_mode == "Conservative":
            reasons = []
            if perf_24h["win_rate"] < 0.4:
                reasons.append(f"Niska win rate: {perf_24h['win_rate']:.1%}")
            if perf_24h["max_drawdown"] > 5.0:
                reasons.append(f"Wysoki drawdown: {perf_24h['max_drawdown']:.1f}%")
            if perf_7d["profit_factor"] < 1.0:
                reasons.append(f"PF < 1.0: {perf_7d['profit_factor']:.2f}")
            return "Ochrona kapitału: " + ", ".join(reasons)

        elif new_mode == "Aggressive":
            return (
                f"Świetna forma: WR={perf_24h['win_rate']:.1%}, "
                f"PF={perf_24h['profit_factor']:.2f}, "
                f"DD={perf_24h['max_drawdown']:.1f}%"
            )

        return "Powrót do stabilnego trybu"

    def execute_mode_switch(self, new_mode: str, reason: str):
        """Wykonuje zmianę trybu."""
        old_mode = self.bot.current_mode
        self.bot.current_mode = new_mode
        self.bot.mode_manager.current_mode = new_mode
        self.last_mode_check = datetime.utcnow()

        logger.warning(f"🔄 ZMIANA TRYBU: {old_mode} → {new_mode}")
        logger.warning(f"Powód: {reason}")

        # Bezpieczne powiadomienie Discord
        try:
            if hasattr(self.bot, "notifications") and hasattr(self.bot.notifications, "send_alert"):
                self.bot.notifications.send_alert(
                    "mode_change",
                    f"Tryb zmieniony: {old_mode} → {new_mode}",
                    {"Powód": reason, "Czas": datetime.utcnow().strftime("%H:%M:%S")},
                )
        except Exception as e:
            logger.error(f"Nie udało się wysłać alertu o zmianie trybu: {e}")

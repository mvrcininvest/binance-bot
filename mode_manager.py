# mode_manager.py (v2.0 - Zarządca Ryzyka)
import logging
from typing import Any

logger = logging.getLogger("mode_manager")


class ModeManager:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.analytics = bot_instance.analytics
        self.current_mode = "Balanced"

        self.mode_configs = {
            "Conservative": {
                "max_leverage": 20,
                "risk_multiplier": 0.7,
                "allow_counter_htf_trend": False,
            },
            "Balanced": {
                "max_leverage": 50,
                "risk_multiplier": 1.0,
                "allow_counter_htf_trend": False,
            },
            "Aggressive": {
                "max_leverage": 100,
                "risk_multiplier": 1.3,
                "allow_counter_htf_trend": True,
            },
        }
        logger.info("ModeManager initialized with mode: %s", self.current_mode)

    def get_trade_parameters(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Zwraca parametry ryzyka i logiki na podstawie aktualnego trybu."""
        config = self.mode_configs[self.current_mode]
        params: dict[str, Any] = {}

        # Ustalenie dźwigni
        if self.bot.leverage_override_enabled:
            original_leverage = int(signal.get("leverage", 10))
            max_leverage = config["max_leverage"]
            if original_leverage > max_leverage:
                logger.info(
                    "Dźwignia dostosowana z %sx do %sx (tryb: %s).",
                    original_leverage,
                    max_leverage,
                    self.current_mode,
                )
                params["leverage"] = max_leverage
            else:
                params["leverage"] = original_leverage
        else:
            params["leverage"] = int(signal.get("leverage", 10))

        # Ustalenie finalnego ryzyka
        params["risk_percent"] = self.bot.runtime_risk * config["risk_multiplier"]

        # Ustalenie, czy można grać przeciwko trendowi HTF
        params["allow_counter_htf"] = config["allow_counter_htf_trend"]

        return params

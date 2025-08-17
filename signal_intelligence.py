# signal_intelligence.py (v3.0 FINAL - Obsługa v8.0 + ML)
import logging
from datetime import datetime
from typing import Any

from config import Config
from database import Session  # optional: keep as factory for future DB use
from ml_predictor import TradingMLPredictor

logger = logging.getLogger("signal_intelligence")


class SignalIntelligence:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.Session = Session  # session factory; use with: `with self.Session() as s:`
        self.ml_predictor = TradingMLPredictor()
        logger.info("Signal Intelligence v3.0 (v8.0 Indicator + ML) initialized")

    def _parse_alert_data(self, alert_data: dict[str, Any]) -> dict[str, Any]:
        """Parsuje i waliduje dane z alertu wskaźnika v8.0."""
        try:
            action = str(alert_data.get("action", "")).lower()
            is_emergency = "emergency" in action

            parsed = {
                "action": "buy" if "buy" in action else "sell",
                "is_emergency_signal": is_emergency,
                "symbol": alert_data.get("symbol", ""),
                "price": float(alert_data.get("price", 0)),
                "sl": float(alert_data.get("sl", 0)),
                "tp1": float(alert_data.get("tp1", 0)),
                "tp2": float(alert_data.get("tp2", 0)),
                "tp3": float(alert_data.get("tp3", 0)),
                "strength": float(alert_data.get("strength", 0)),
                "tier": str(alert_data.get("tier", "Quick")),
                "position_size_multiplier": float(alert_data.get("position_size_multiplier", 1.0)),
                "leverage": int(alert_data.get("leverage", 10)),
                "market_regime": str(alert_data.get("market_regime", "NEUTRAL")),
                "market_condition": str(alert_data.get("market_condition", "NORMAL")),
                "confidence_penalty": float(alert_data.get("confidence_penalty", 0.0)),
                "pair_tier": int(alert_data.get("pair_tier", 3)),
                "session": str(alert_data.get("session", "Unknown")),
                "mfi": float(alert_data.get("mfi", 50)),
                "adx": float(alert_data.get("adx", 0)),
                "htf_trend": str(alert_data.get("htf_trend", "neutral")),
                "btc_correlation": float(alert_data.get("btc_correlation", 0)),
                "volume_spike": str(alert_data.get("volume_spike", "false")).lower() == "true",
                "near_key_level": str(alert_data.get("near_key_level", "false")).lower() == "true",
                "timestamp": alert_data.get("timestamp", datetime.utcnow().timestamp()),
            }

            if not parsed["symbol"] or parsed["price"] <= 0 or parsed["sl"] <= 0:
                raise ValueError("Brak kluczowych danych: symbol, cena lub sl.")

            return parsed

        except (ValueError, TypeError) as e:
            logger.error(f"Błąd parsowania alertu v8.0: {e}. Dane: {alert_data}")
            raise

    def analyze_signal(self, alert_data: dict[str, Any]) -> dict[str, Any]:
        """Analizuje sygnał v8.0 z opcjonalną logiką ML."""
        try:
            parsed_data = self._parse_alert_data(alert_data)

            # Filtry bezpieczeństwa
            if not parsed_data["is_emergency_signal"]:
                if parsed_data.get("market_condition") in [
                    "EXTREME_MOVE",
                    "MARKET_STRESS",
                ]:
                    return {
                        "action": "reject",
                        "reason": f"Ekstremalne warunki: {parsed_data['market_condition']}",
                    }
                if parsed_data.get("confidence_penalty", 0.0) > 0.25:
                    return {
                        "action": "reject",
                        "reason": f"Wysoka kara pewności: {parsed_data['confidence_penalty']:.2f}",
                    }

            signal_tier = parsed_data.get("tier", "None")
            if signal_tier == "None":
                return {"action": "reject", "reason": "Sygnał zbyt słaby"}

            # Minimalny próg siły
            min_strength = 0.15 if parsed_data["is_emergency_signal"] else 0.20
            if parsed_data.get("strength", 0) < min_strength:
                return {
                    "action": "reject",
                    "reason": f"Sygnał zbyt słaby: {parsed_data.get('strength', 0):.3f} < {min_strength}",
                }

            # Priority Enhancements
            priority_multiplier = self._calculate_priority_multiplier(parsed_data)
            base_multiplier = parsed_data["position_size_multiplier"]
            final_multiplier = base_multiplier * priority_multiplier

            if parsed_data["is_emergency_signal"]:
                final_multiplier *= 0.9

            # ML predykcje (zawsze obliczane)
            try:
                ml_predictions = self.ml_predictor.predict(parsed_data)
                parsed_data["ml_predictions"] = ml_predictions

                # Decyzja ML (włącz/wyłącz)
                use_ml_for_decision = getattr(
                    self.bot,
                    "use_ml_for_decision",
                    getattr(Config, "USE_ML_FOR_DECISION", False),
                )
                if use_ml_for_decision:
                    min_win_prob_threshold = 0.50 if parsed_data["is_emergency_signal"] else 0.55
                    should_take, ml_reason = self.ml_predictor.should_take_trade(
                        ml_predictions, min_win_prob=min_win_prob_threshold
                    )
                    if not should_take:
                        logger.info(f"ML odrzuciło {parsed_data['symbol']}: {ml_reason}")
                        return {"action": "reject", "reason": f"ML: {ml_reason}"}

                # Sizing ML (włącz/wyłącz)
                use_ml_for_sizing = getattr(
                    self.bot,
                    "use_ml_for_sizing",
                    getattr(Config, "USE_ML_FOR_SIZING", False),
                )
                if use_ml_for_sizing:
                    ml_multiplier = self._calculate_ml_risk_multiplier(ml_predictions)
                    final_multiplier *= ml_multiplier
                    parsed_data["ml_multiplier_used"] = ml_multiplier
                else:
                    parsed_data["ml_multiplier_used"] = 1.0

            except Exception as e:
                logger.warning(f"Nie udało się obliczyć predykcji ML: {e}")
                parsed_data["ml_predictions"] = {}
                parsed_data["ml_multiplier_used"] = 1.0

            return {
                "action": "execute",
                "symbol": parsed_data["symbol"],
                "side": parsed_data["action"],
                "price": parsed_data["price"],
                "entry_price": parsed_data["price"],
                "stop_loss": parsed_data["sl"],
                "take_profits": {
                    "tp1": parsed_data["tp1"],
                    "tp2": parsed_data["tp2"],
                    "tp3": parsed_data["tp3"],
                },
                "tp_distribution": Config.TP_DISTRIBUTION,
                "position_multiplier": final_multiplier,
                "leverage": parsed_data["leverage"],
                "risk_percent": self.bot.runtime_risk * final_multiplier,
                "tier": signal_tier,
                "metadata": parsed_data,
            }

        except Exception as e:
            logger.error(f"Błąd analizy: {e}", exc_info=True)
            return {"action": "reject", "reason": str(e)}

    def _calculate_priority_multiplier(self, parsed_data: dict[str, Any]) -> float:
        """Oblicza mnożnik priorytetów z wskaźnika v8.0"""
        multiplier = 1.0

        # Priority Enhancements z Pine Script
        priority_enhancements = parsed_data.get("priority_enhancements", {})

        if priority_enhancements.get("mitigation_mode_active"):
            multiplier *= 1.15
        if priority_enhancements.get("dynamic_opposite_zone_active"):
            multiplier *= 1.12
        if priority_enhancements.get("rising_adx_active"):
            multiplier *= 1.10
        if priority_enhancements.get("emergency_mode_active"):
            multiplier *= 1.08

        # Emergency conditions bonus
        if parsed_data.get("is_emergency_signal"):
            emergency_conditions = parsed_data.get("emergency_conditions", {})
            volume_multiplier = emergency_conditions.get("volume_multiplier", 1.0)
            if volume_multiplier > 5.0:
                multiplier *= 1.20
            elif volume_multiplier > 3.0:
                multiplier *= 1.10

        return multiplier

    def _calculate_ml_risk_multiplier(self, predictions: dict[str, float]) -> float:
        """Oblicza mnożnik ryzyka na podstawie ML."""
        win_prob = predictions.get("win_probability", 0.5)
        expected_value = predictions.get("expected_value", 0)
        confidence = predictions.get("confidence", 0.5)

        if win_prob > 0.70:
            base = 1.3
        elif win_prob > 0.60:
            base = 1.1
        else:
            base = 0.9

        if expected_value > 2.0:
            ev_boost = 1.2
        elif expected_value > 1.0:
            ev_boost = 1.1
        elif expected_value < 0:
            ev_boost = 0.7
        else:
            ev_boost = 1.0

        confidence_mult = 0.8 + (confidence * 0.4)

        final_multiplier = base * ev_boost * confidence_mult
        return max(0.5, min(1.5, final_multiplier))

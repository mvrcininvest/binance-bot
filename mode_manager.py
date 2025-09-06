"""
Mode Manager for Trading Bot v9.1
Handles different trading modes and their specific rules
Enhanced with v9.1 features: Signal Intelligence, ML Integration, Advanced Analytics
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any
import json

from config import Config
from database import Session, get_setting, set_setting, get_open_positions, Trade

logger = logging.getLogger(__name__)


class ModeManager:
    """Manages trading modes and their rules with v9.1 enhancements"""

    def __init__(self, bot_instance=None):
        """Initialize mode manager with v9.1 features"""
        self.bot_instance = bot_instance
        self.modes = Config.MODE_CONFIGS
        self.current_mode = None
        self.mode_rules = {
            "conservative": ConservativeMode(),
            "balanced": BalancedMode(),
            "aggressive": AggressiveMode(),
            "scalping": ScalpingMode(),
            "swing": SwingMode(),
            "emergency": EmergencyMode(),
            "ml_adaptive": MLAdaptiveMode(),  # v9.1 NEW
            "institutional": InstitutionalMode(),  # v9.1 NEW
        }

        # v9.1 Enhanced tracking
        self.mode_performance = {}
        self.last_mode_switch = None
        self.mode_switch_cooldown = 300  # 5 minutes

        logger.info("🎯 Mode Manager v9.1 initialized with enhanced features")

    def get_current_mode(self) -> str:
        """Get current trading mode with v9.1 enhancements"""
        with Session() as session:
            mode = get_setting(session, "mode", Config.DEFAULT_MODE)

            # v9.1 CORE: Check for emergency override
            if get_setting(session, "emergency_enabled", False):
                return "emergency"

            # v9.1 CORE: Auto mode switching based on performance
            if Config.AUTO_MODE_SWITCH:
                suggested_mode = self.should_auto_switch_mode()
                if suggested_mode and suggested_mode != mode:
                    if self._can_switch_mode():
                        logger.info(
                            f"🔄 Auto-switching from {mode} to {suggested_mode}"
                        )
                        self.set_mode(
                            suggested_mode, "Auto-switch based on performance"
                        )
                        return suggested_mode

            self.current_mode = mode
            return mode

    def set_mode(self, mode: str, reason: str = "Manual") -> bool:
        """Set trading mode with v9.1 enhancements"""
        if mode not in self.mode_rules:
            logger.error(f"❌ Invalid mode: {mode}")
            return False

        old_mode = self.current_mode

        with Session() as session:
            set_setting(session, "mode", mode, "string", reason)
            set_setting(
                session,
                "mode_changed_at",
                datetime.utcnow().isoformat(),
                "string",
                "system",
            )
            set_setting(
                session, "previous_mode", old_mode or "unknown", "string", "system"
            )

        self.current_mode = mode
        self.last_mode_switch = datetime.utcnow()

        logger.info(f"🎯 Mode changed: {old_mode} → {mode} | Reason: {reason}")

        # v9.1 CORE: Send Discord notification
        if self.bot_instance and hasattr(self.bot_instance, "discord"):
            import asyncio

            asyncio.create_task(
                self.bot_instance.discord.send_mode_change_notification(
                    old_mode, mode, reason
                )
            )

        return True

    def get_trade_parameters(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Get trade parameters based on current mode and signal analysis"""
        mode = self.get_current_mode()
        mode_handler = self.mode_rules.get(mode)

        if not mode_handler:
            logger.error(f"❌ No handler for mode: {mode}")
            return {}

        # Get base parameters from mode
        params = mode_handler.get_trade_parameters(decision)

        # v9.1 CORE: Apply signal intelligence adjustments
        params = self._apply_signal_intelligence_adjustments(params, decision)

        # v9.1 CORE: Apply ML adjustments if available
        if decision.get("ml_prediction") and Config.USE_ML_FOR_SIZING:
            params = self._apply_ml_adjustments(params, decision)

        logger.info(f"📊 Trade parameters for {mode} mode: {params}")
        return params

    def _apply_signal_intelligence_adjustments(
        self, params: Dict, decision: Dict
    ) -> Dict:
        """Apply signal intelligence adjustments to trade parameters"""
        adjustments = {}

        # Fake breakout detection adjustment
        if decision.get("fake_breakout_detected"):
            adjustments["position_size_multiplier"] = (
                params.get("position_size_multiplier", 1.0) * 0.5
            )
            adjustments["leverage"] = max(1, int(params.get("leverage", 10) * 0.7))
            logger.info("🎯 Applied fake breakout adjustments")

        # Institutional flow adjustment
        institutional_flow = decision.get("institutional_flow", 0.0)
        if institutional_flow > 0.7:
            adjustments["position_size_multiplier"] = (
                params.get("position_size_multiplier", 1.0) * 1.2
            )
            logger.info(
                f"📈 Strong institutional flow ({institutional_flow:.2f}) - increased position size"
            )
        elif institutional_flow < 0.3:
            adjustments["position_size_multiplier"] = (
                params.get("position_size_multiplier", 1.0) * 0.8
            )
            logger.info(
                f"📉 Weak institutional flow ({institutional_flow:.2f}) - reduced position size"
            )

        # MTF agreement adjustment
        mtf_agreement = decision.get("mtf_agreement_ratio", 0.5)
        if mtf_agreement > 0.8:
            adjustments["leverage"] = min(
                Config.MAX_LEVERAGE, int(params.get("leverage", 10) * 1.1)
            )
            logger.info(
                f"🎯 Strong MTF agreement ({mtf_agreement:.2f}) - increased leverage"
            )
        elif mtf_agreement < 0.4:
            adjustments["leverage"] = max(1, int(params.get("leverage", 10) * 0.8))
            logger.info(
                f"⚠️ Weak MTF agreement ({mtf_agreement:.2f}) - reduced leverage"
            )

        # Regime confidence adjustment
        regime_confidence = decision.get("regime_confidence", 0.5)
        if regime_confidence > 0.8:
            adjustments["stop_loss_multiplier"] = (
                params.get("stop_loss_multiplier", 1.0) * 1.1
            )
            logger.info(
                f"📊 High regime confidence ({regime_confidence:.2f}) - wider stops"
            )

        # Apply adjustments
        params.update(adjustments)
        return params

    def _apply_ml_adjustments(self, params: Dict, decision: Dict) -> Dict:
        """Apply ML-based adjustments to trade parameters"""
        ml_prediction = decision.get("ml_prediction", {})

        win_probability = ml_prediction.get("win_probability", 0.5)
        confidence = ml_prediction.get("confidence", 0.5)

        # Adjust position size based on ML confidence
        if confidence > 0.8 and win_probability > 0.7:
            params["position_size_multiplier"] = (
                params.get("position_size_multiplier", 1.0) * 1.3
            )
            logger.info(
                f"🤖 High ML confidence ({confidence:.2f}) & win prob ({win_probability:.2f}) - increased position"
            )
        elif confidence < 0.4 or win_probability < 0.4:
            params["position_size_multiplier"] = (
                params.get("position_size_multiplier", 1.0) * 0.7
            )
            logger.info(
                f"🤖 Low ML confidence ({confidence:.2f}) or win prob ({win_probability:.2f}) - reduced position"
            )

        # Adjust leverage based on predicted volatility
        predicted_volatility = ml_prediction.get("volatility", 0.5)
        if predicted_volatility > 0.7:
            params["leverage"] = max(1, int(params.get("leverage", 10) * 0.8))
            logger.info(
                f"🤖 High predicted volatility ({predicted_volatility:.2f}) - reduced leverage"
            )

        return params

    async def evaluate_signal(self, alert: Dict) -> Dict:
        """Evaluate signal based on current mode with v9.1 enhancements"""
        mode = self.get_current_mode()
        mode_handler = self.mode_rules.get(mode)

        if not mode_handler:
            logger.error(f"❌ No handler for mode: {mode}")
            return {"should_trade": False, "reason": f"Invalid mode: {mode}"}

        # Check if paused
        with Session() as session:
            if get_setting(session, "is_paused", False):
                return {"should_trade": False, "reason": "Bot is paused"}

        # v9.1 CORE: Enhanced evaluation with mode-specific rules
        evaluation = await mode_handler.evaluate(alert)

        # v9.1 CORE: Track mode performance
        self._track_mode_performance(mode, evaluation)

        return evaluation

    def _track_mode_performance(self, mode: str, evaluation: Dict):
        """Track performance metrics for each mode"""
        if mode not in self.mode_performance:
            self.mode_performance[mode] = {
                "signals_evaluated": 0,
                "signals_accepted": 0,
                "last_used": datetime.utcnow(),
            }

        self.mode_performance[mode]["signals_evaluated"] += 1
        self.mode_performance[mode]["last_used"] = datetime.utcnow()

        if evaluation.get("should_trade", False):
            self.mode_performance[mode]["signals_accepted"] += 1

    def get_mode_config(self, mode: Optional[str] = None) -> Dict:
        """Get configuration for mode"""
        if not mode:
            mode = self.get_current_mode()

        return self.modes.get(mode, self.modes[Config.DEFAULT_MODE])

    def should_auto_switch_mode(self) -> Optional[str]:
        """Check if mode should be auto-switched based on performance"""
        if not Config.AUTO_MODE_SWITCH:
            return None

        with Session() as session:
            # Get recent performance
            from database import get_performance_stats

            stats = get_performance_stats(session, "daily")

            # v9.1 CORE: Enhanced auto-switching logic
            current_mode = self.get_current_mode()

            # Emergency triggers
            if stats["total_pnl"] < Config.EMERGENCY_CLOSE_THRESHOLD:
                return "emergency"

            # Check losing streak
            losing_streak = self._get_losing_streak(session)
            if losing_streak >= 5 and current_mode != "conservative":
                return "conservative"

            # Check daily loss
            if stats["total_pnl"] < -Config.MAX_DAILY_LOSS * 0.5:
                return "conservative"

            # Check win rate for mode switching
            if stats["total_trades"] >= 10:
                if stats["win_rate"] < 30 and current_mode != "conservative":
                    return "conservative"
                elif stats["win_rate"] > 75 and current_mode not in [
                    "aggressive",
                    "ml_adaptive",
                ]:
                    return "ml_adaptive" if Config.USE_ML_FOR_DECISION else "aggressive"
                elif 45 <= stats["win_rate"] <= 65 and current_mode in [
                    "conservative",
                    "aggressive",
                ]:
                    return "balanced"

            # v9.1 CORE: ML-based mode switching
            if Config.USE_ML_FOR_DECISION and stats["total_trades"] >= 5:
                if stats["win_rate"] > 60 and current_mode != "ml_adaptive":
                    return "ml_adaptive"

            return None

    def _can_switch_mode(self) -> bool:
        """Check if mode can be switched (cooldown check)"""
        if not self.last_mode_switch:
            return True

        time_since_switch = (datetime.utcnow() - self.last_mode_switch).total_seconds()
        return time_since_switch >= self.mode_switch_cooldown

    def _get_losing_streak(self, session: Session) -> int:
        """Get current losing streak (bazuje na Trade)"""
        recent_trades = (
            session.query(Trade)
            .filter(Trade.status == "closed")
            .order_by(Trade.exit_time.desc())
            .limit(10)
            .all()
        )

        streak = 0
        for t in recent_trades:
            if (t.pnl_usdt or 0) < 0:
                streak += 1
            else:
                break

        return streak

    def get_mode_statistics(self) -> Dict[str, Any]:
        """Get statistics for all modes"""
        return {
            "current_mode": self.get_current_mode(),
            "mode_performance": self.mode_performance,
            "last_mode_switch": (
                self.last_mode_switch.isoformat() if self.last_mode_switch else None
            ),
            "auto_switch_enabled": Config.AUTO_MODE_SWITCH,
            "available_modes": list(self.mode_rules.keys()),
        }


class BaseModeHandler:
    """Base class for mode handlers with v9.1 enhancements"""

    async def evaluate(self, alert: Dict) -> Dict:
        """Evaluate signal for this mode"""
        raise NotImplementedError

    def get_trade_parameters(self, decision: Dict) -> Dict:
        """Get trade parameters for this mode"""
        raise NotImplementedError

    def check_tier_allowed(self, tier: str, allowed_tiers: list) -> bool:
        """Check if tier is allowed"""
        return tier in allowed_tiers

    def check_strength_threshold(self, strength: float, min_strength: float) -> bool:
        """Check if signal strength meets threshold"""
        return strength >= min_strength

    def check_max_positions(self, session: Session, max_positions: int) -> bool:
        """Check if max positions reached"""
        open_positions = get_open_positions(session)
        return len(open_positions) < max_positions

    def calculate_leverage_adjustment(
        self, base_leverage: int, tier: str, strength: float
    ) -> int:
        """Calculate adjusted leverage based on signal quality"""
        # v9.1 Enhanced tier multipliers
        tier_multipliers = {"Premium": 1.2, "Standard": 1.0, "Basic": 0.8}

        # Strength adjustment (0.8 to 1.2)
        strength_multiplier = 0.8 + (strength / 100) * 0.4

        # Calculate final leverage
        adjusted = int(
            base_leverage * tier_multipliers.get(tier, 1.0) * strength_multiplier
        )

        # Apply limits
        return max(1, min(adjusted, Config.MAX_LEVERAGE))


class ConservativeMode(BaseModeHandler):
    """Conservative trading mode - low risk, high quality signals only"""

    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get("tier", "Standard")
        strength = alert.get("strength", 0)

        # Only accept high quality signals
        allowed_tiers = ["Premium", "Standard"]
        if not self.check_tier_allowed(tier, allowed_tiers):
            return {
                "should_trade": False,
                "reason": f"Conservative mode only accepts {allowed_tiers} tiers",
            }

        # High strength requirement
        min_strength = 70
        if not self.check_strength_threshold(strength, min_strength):
            return {
                "should_trade": False,
                "reason": f"Signal strength {strength:.1f}% below conservative threshold {min_strength}%",
            }

        # v9.1 CORE: Check for fake breakout
        if alert.get("fake_breakout_detected"):
            return {
                "should_trade": False,
                "reason": "Fake breakout detected - rejected in conservative mode",
            }

        # v9.1 CORE: Check institutional flow
        if alert.get("institutional_flow", 0) < 0.3:
            return {
                "should_trade": False,
                "reason": "Insufficient institutional flow for conservative mode",
            }

        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 3):
                return {
                    "should_trade": False,
                    "reason": "Max positions (3) reached in conservative mode",
                }

        return {
            "should_trade": True,
            "reason": "Signal meets conservative criteria",
            "mode": "conservative",
        }

    def get_trade_parameters(self, decision: Dict) -> Dict:
        """Get conservative trade parameters"""
        tier = decision.get("tier", "Standard")
        strength = decision.get("strength", 50)

        leverage = self.calculate_leverage_adjustment(5, tier, strength)

        return {
            "leverage": leverage,
            "position_size_multiplier": 0.5,
            "stop_loss_multiplier": 0.8,
            "risk_percent": Config.RISK_PER_TRADE * 0.5,
        }


class BalancedMode(BaseModeHandler):
    """Balanced trading mode - moderate risk/reward"""

    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get("tier", "Standard")
        strength = alert.get("strength", 0)

        # Accept most tiers
        allowed_tiers = ["Premium", "Standard", "Basic"]
        if not self.check_tier_allowed(tier, allowed_tiers):
            return {
                "should_trade": False,
                "reason": f"Balanced mode does not accept {tier} tier",
            }

        # Moderate strength requirement
        min_strength = 50
        if not self.check_strength_threshold(strength, min_strength):
            return {
                "should_trade": False,
                "reason": f"Signal strength {strength:.1f}% below balanced threshold {min_strength}%",
            }

        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 5):
                return {
                    "should_trade": False,
                    "reason": "Max positions (5) reached in balanced mode",
                }

        return {
            "should_trade": True,
            "reason": "Signal meets balanced criteria",
            "mode": "balanced",
        }

    def get_trade_parameters(self, decision: Dict) -> Dict:
        """Get balanced trade parameters"""
        tier = decision.get("tier", "Standard")
        strength = decision.get("strength", 50)

        leverage = self.calculate_leverage_adjustment(10, tier, strength)

        # Reduce position on fake breakout
        fake_breakout_multiplier = (
            0.5 if decision.get("fake_breakout_detected") else 1.0
        )

        return {
            "leverage": leverage,
            "position_size_multiplier": 0.75 * fake_breakout_multiplier,
            "stop_loss_multiplier": 1.0,
            "risk_percent": Config.RISK_PER_TRADE,
        }


class AggressiveMode(BaseModeHandler):
    """Aggressive trading mode - higher risk, more signals"""

    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get("tier", "Standard")
        strength = alert.get("strength", 0)

        # Lower strength requirement
        min_strength = 30
        if not self.check_strength_threshold(strength, min_strength):
            return {
                "should_trade": False,
                "reason": f"Signal strength {strength:.1f}% below aggressive threshold {min_strength}%",
            }

        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 8):
                return {
                    "should_trade": False,
                    "reason": "Max positions (8) reached in aggressive mode",
                }

        return {
            "should_trade": True,
            "reason": "Signal accepted in aggressive mode",
            "mode": "aggressive",
        }

    def get_trade_parameters(self, decision: Dict) -> Dict:
        """Get aggressive trade parameters"""
        tier = decision.get("tier", "Standard")
        strength = decision.get("strength", 50)

        leverage = self.calculate_leverage_adjustment(15, tier, strength)

        # Boost leverage for strong signals
        if strength > 80 and tier == "Premium":
            leverage = min(leverage + 5, Config.MAX_LEVERAGE)

        return {
            "leverage": leverage,
            "position_size_multiplier": 1.0,
            "stop_loss_multiplier": 1.2,
            "risk_percent": Config.RISK_PER_TRADE * 1.5,
        }


class ScalpingMode(BaseModeHandler):
    """Scalping mode - quick trades, small profits"""

    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get("tier", "Standard")
        strength = alert.get("strength", 0)
        timeframe = alert.get("timeframe", "15m")

        # Check timeframe (prefer lower timeframes)
        if timeframe not in ["1m", "5m", "15m"]:
            return {
                "should_trade": False,
                "reason": f"Scalping mode requires lower timeframe, got {timeframe}",
            }

        # Moderate strength requirement
        min_strength = 40
        if not self.check_strength_threshold(strength, min_strength):
            return {
                "should_trade": False,
                "reason": f"Signal strength {strength:.1f}% below scalping threshold {min_strength}%",
            }

        # v9.1 CORE: Check MTF agreement for scalping
        if alert.get("mtf_agreement_ratio", 0) < 0.6:
            return {
                "should_trade": False,
                "reason": "Insufficient multi-timeframe agreement for scalping",
            }

        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 10):
                return {
                    "should_trade": False,
                    "reason": "Max positions (10) reached in scalping mode",
                }

        return {
            "should_trade": True,
            "reason": "Signal meets scalping criteria",
            "mode": "scalping",
        }

    def get_trade_parameters(self, decision: Dict) -> Dict:
        """Get scalping trade parameters"""
        tier = decision.get("tier", "Standard")
        strength = decision.get("strength", 50)

        leverage = self.calculate_leverage_adjustment(20, tier, strength)

        return {
            "leverage": leverage,
            "position_size_multiplier": 0.5,
            "stop_loss_multiplier": 0.5,
            "take_profit_multiplier": 0.3,
            "use_only_tp1": True,
            "risk_percent": Config.RISK_PER_TRADE * 0.5,
        }


class SwingMode(BaseModeHandler):
    """Swing trading mode - longer term positions"""

    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get("tier", "Standard")
        strength = alert.get("strength", 0)
        regime = alert.get("enhanced_regime", "NEUTRAL")

        # Prefer high quality signals for swing
        allowed_tiers = ["Premium", "Standard"]
        if not self.check_tier_allowed(tier, allowed_tiers):
            return {
                "should_trade": False,
                "reason": f"Swing mode requires {allowed_tiers} tiers",
            }

        # High strength requirement
        min_strength = 60
        if not self.check_strength_threshold(strength, min_strength):
            return {
                "should_trade": False,
                "reason": f"Signal strength {strength:.1f}% below swing threshold {min_strength}%",
            }

        # v9.1 CORE: Check market regime
        if regime == "NEUTRAL":
            return {
                "should_trade": False,
                "reason": "Swing mode requires trending market (TRENDING_UP/TRENDING_DOWN)",
            }

        # v9.1 CORE: Check regime confidence
        if alert.get("regime_confidence", 0) < 0.7:
            return {
                "should_trade": False,
                "reason": "Insufficient regime confidence for swing trading",
            }

        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 3):
                return {
                    "should_trade": False,
                    "reason": "Max positions (3) reached in swing mode",
                }

        return {
            "should_trade": True,
            "reason": "Signal meets swing trading criteria",
            "mode": "swing",
        }

    def get_trade_parameters(self, decision: Dict) -> Dict:
        """Get swing trade parameters"""
        tier = decision.get("tier", "Standard")
        strength = decision.get("strength", 50)

        leverage = self.calculate_leverage_adjustment(5, tier, strength)

        return {
            "leverage": leverage,
            "position_size_multiplier": 1.5,
            "stop_loss_multiplier": 2.0,
            "hold_for_all_targets": True,
            "disable_break_even": True,
            "risk_percent": Config.RISK_PER_TRADE * 2.0,
        }


class EmergencyMode(BaseModeHandler):
    """Emergency mode - only emergency signals or close positions"""

    async def evaluate(self, alert: Dict) -> Dict:
        action = alert.get("action", "").upper()

        # Only accept CLOSE actions in emergency mode
        if action == "CLOSE":
            return {
                "should_trade": True,
                "reason": "Close signal accepted in emergency mode",
                "immediate_close": True,
            }

        # Very restrictive for new positions
        with Session() as session:
            open_positions = get_open_positions(session)
            if len(open_positions) > 0:
                return {
                    "should_trade": False,
                    "reason": "No new positions in emergency mode with existing positions",
                }

        # Only accept very strong signals
        if alert.get("strength", 0) < 90:
            return {
                "should_trade": False,
                "reason": "Emergency mode requires strength > 90%",
            }

        return {
            "should_trade": True,
            "reason": "Emergency signal accepted",
            "mode": "emergency",
        }

    def get_trade_parameters(self, decision: Dict) -> Dict:
        """Get emergency trade parameters"""
        return {
            "leverage": 1,
            "position_size_multiplier": 0.25,
            "stop_loss_multiplier": 0.5,
            "emergency_mode": True,
            "risk_percent": Config.RISK_PER_TRADE * 0.25,
        }


class MLAdaptiveMode(BaseModeHandler):
    """v9.1 NEW: ML Adaptive mode - uses ML predictions for decisions"""

    async def evaluate(self, alert: Dict) -> Dict:
        # Require ML prediction
        if not Config.USE_ML_FOR_DECISION:
            return {
                "should_trade": False,
                "reason": "ML not enabled for ML Adaptive mode",
            }

        # Basic signal quality check
        if alert.get("strength", 0) < 40:
            return {
                "should_trade": False,
                "reason": "Signal strength too low for ML Adaptive mode",
            }

        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 6):
                return {
                    "should_trade": False,
                    "reason": "Max positions (6) reached in ML Adaptive mode",
                }

        return {
            "should_trade": True,
            "reason": "Signal accepted for ML evaluation",
            "mode": "ml_adaptive",
        }

    def get_trade_parameters(self, decision: Dict) -> Dict:
        """Get ML adaptive trade parameters"""
        tier = decision.get("tier", "Standard")
        strength = decision.get("strength", 50)

        # Base leverage adjusted by ML confidence
        base_leverage = 12
        ml_prediction = decision.get("ml_prediction", {})
        confidence = ml_prediction.get("confidence", 0.5)

        leverage = int(base_leverage * (0.5 + confidence * 0.5))
        leverage = max(1, min(leverage, Config.MAX_LEVERAGE))

        return {
            "leverage": leverage,
            "position_size_multiplier": 0.8 + (confidence * 0.4),
            "stop_loss_multiplier": 1.0,
            "risk_percent": Config.RISK_PER_TRADE * (0.8 + confidence * 0.4),
        }


class InstitutionalMode(BaseModeHandler):
    """v9.1 NEW: Institutional mode - follows institutional flow"""

    async def evaluate(self, alert: Dict) -> Dict:
        # Require institutional flow data
        institutional_flow = alert.get("institutional_flow", 0)
        if institutional_flow < 0.6:
            return {
                "should_trade": False,
                "reason": f"Insufficient institutional flow ({institutional_flow:.2f}) for institutional mode",
            }

        # High strength requirement
        if alert.get("strength", 0) < 60:
            return {
                "should_trade": False,
                "reason": "Signal strength too low for institutional mode",
            }

        # Prefer premium signals
        tier = alert.get("tier", "Standard")
        if tier not in ["Premium", "Standard"]:
            return {
                "should_trade": False,
                "reason": "Institutional mode requires Premium or Standard tier",
            }

        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 4):
                return {
                    "should_trade": False,
                    "reason": "Max positions (4) reached in institutional mode",
                }

        return {
            "should_trade": True,
            "reason": "Signal meets institutional criteria",
            "mode": "institutional",
        }

    def get_trade_parameters(self, decision: Dict) -> Dict:
        """Get institutional trade parameters"""
        tier = decision.get("tier", "Standard")
        strength = decision.get("strength", 50)
        institutional_flow = decision.get("institutional_flow", 0.5)

        leverage = self.calculate_leverage_adjustment(8, tier, strength)

        # Adjust based on institutional flow strength
        flow_multiplier = 0.8 + (institutional_flow * 0.4)

        return {
            "leverage": leverage,
            "position_size_multiplier": 1.0 * flow_multiplier,
            "stop_loss_multiplier": 1.1,
            "risk_percent": Config.RISK_PER_TRADE * flow_multiplier,
        }


# v9.1 Enhanced Mode transition rules
class ModeTransitionManager:
    """Manages automatic mode transitions based on performance with v9.1 enhancements"""

    @staticmethod
    def check_transition_rules() -> Optional[Tuple[str, str]]:
        """Check if mode should be transitioned with v9.1 logic"""
        with Session() as session:
            current_mode = get_setting(session, "mode", Config.DEFAULT_MODE)

            # Get performance metrics
            from database import get_performance_stats

            daily_stats = get_performance_stats(session, "daily")

            # v9.1 CORE: Emergency triggers
            if daily_stats["total_pnl"] < Config.EMERGENCY_CLOSE_THRESHOLD:
                return (
                    "emergency",
                    f"Daily loss exceeded emergency threshold: ${daily_stats['total_pnl']:.2f}",
                )

            # v9.1 CORE: Conservative triggers
            if daily_stats["win_rate"] < 25 and daily_stats["total_trades"] >= 5:
                return (
                    "conservative",
                    f"Very low win rate: {daily_stats['win_rate']:.1f}%",
                )

            # v9.1 CORE: ML Adaptive triggers
            if Config.USE_ML_FOR_DECISION and current_mode != "ml_adaptive":
                if daily_stats["win_rate"] > 65 and daily_stats["total_trades"] >= 8:
                    return (
                        "ml_adaptive",
                        f"Good performance - switching to ML adaptive",
                    )

            # v9.1 CORE: Institutional mode triggers
            if Config.ENABLE_INSTITUTIONAL_FLOW and current_mode != "institutional":
                # Check if recent signals had strong institutional flow
                recent_positions = (
                    session.query(Trade)
                    .filter(
                        Trade.entry_time >= datetime.utcnow() - timedelta(hours=24),
                        Trade.status == "open"
                    )
                    .all()
                )

                if len(recent_trades) >= 3:
                    avg_institutional_flow = sum(
                        p.institutional_flow or 0 for p in recent_positions
                    (t.institutional_flow or 0) for t in recent_trades
                    ) / len(recent_trades)

                    if avg_institutional_flow > 0.7:
                        return (
                            "institutional",
                            f"Strong institutional flow detected: {avg_institutional_flow:.2f}",
                        )

            # Return to balanced from extremes
            if current_mode in ["conservative", "aggressive", "emergency"]:
                if (
                    40 <= daily_stats["win_rate"] <= 60
                    and daily_stats["total_trades"] >= 5
                ):
                    return (
                        "balanced",
                        "Performance normalized - returning to balanced",
                    )

            return None

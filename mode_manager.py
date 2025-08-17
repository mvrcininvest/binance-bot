"""
Mode Manager for Trading Bot v9.1
Handles different trading modes and their specific rules
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

from config import Config
from database import Session, get_setting, set_setting, get_open_positions

logger = logging.getLogger(__name__)


class ModeManager:
    """Manages trading modes and their rules"""
    
    def __init__(self):
        """Initialize mode manager"""
        self.modes = Config.MODES
        self.current_mode = None
        self.mode_rules = {
            'conservative': ConservativeMode(),
            'balanced': BalancedMode(),
            'aggressive': AggressiveMode(),
            'scalping': ScalpingMode(),
            'swing': SwingMode(),
            'emergency': EmergencyMode()
        }
        
        logger.info("Mode Manager initialized")
    
    def get_current_mode(self) -> str:
        """Get current trading mode"""
        with Session() as session:
            mode = get_setting(session, "mode", Config.DEFAULT_MODE)
            
            # Check for emergency override
            if get_setting(session, "emergency_enabled", False):
                return "emergency"
            
            return mode
    
    def set_mode(self, mode: str, reason: str = "Manual") -> bool:
        """Set trading mode"""
        if mode not in self.modes:
            logger.error(f"Invalid mode: {mode}")
            return False
        
        with Session() as session:
            set_setting(session, "mode", mode, "string", reason)
            set_setting(session, "mode_changed_at", datetime.utcnow().isoformat(), "string", "system")
            
            logger.info(f"Mode changed to {mode} - Reason: {reason}")
            return True
    
    async def evaluate_signal(self, alert: Dict) -> Dict:
        """Evaluate signal based on current mode"""
        mode = self.get_current_mode()
        mode_handler = self.mode_rules.get(mode)
        
        if not mode_handler:
            logger.error(f"No handler for mode: {mode}")
            return {'accept': False, 'reason': f'Invalid mode: {mode}'}
        
        # Check if paused
        with Session() as session:
            if get_setting(session, "is_paused", False):
                return {'accept': False, 'reason': 'Bot is paused'}
        
        # Evaluate with mode-specific rules
        return await mode_handler.evaluate(alert)
    
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
            
            # Check losing streak
            losing_streak = self._get_losing_streak(session)
            if losing_streak >= 5:
                logger.warning(f"Losing streak: {losing_streak} - Switching to conservative")
                return "conservative"
            
            # Check daily loss
            if stats['total_pnl'] < -Config.DAILY_LOSS_LIMIT:
                logger.warning(f"Daily loss limit reached: ${stats['total_pnl']:.2f}")
                return "conservative"
            
            # Check win rate
            if stats['total_trades'] >= 10:
                if stats['win_rate'] < 30:
                    return "conservative"
                elif stats['win_rate'] > 70:
                    return "aggressive"
            
            return None
    
    def _get_losing_streak(self, session: Session) -> int:
        """Get current losing streak"""
        from database import Trade
        recent_trades = session.query(Trade).filter(
            Trade.status == 'closed'
        ).order_by(Trade.closed_at.desc()).limit(10).all()
        
        streak = 0
        for trade in recent_trades:
            if trade.realized_pnl < 0:
                streak += 1
            else:
                break
        
        return streak


class BaseModeHandler:
    """Base class for mode handlers"""
    
    async def evaluate(self, alert: Dict) -> Dict:
        """Evaluate signal for this mode"""
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
    
    def calculate_leverage_adjustment(self, base_leverage: int, tier: str, strength: float) -> int:
        """Calculate adjusted leverage based on signal quality"""
        # Tier multipliers
        tier_multipliers = {
            'Platinum': 1.2,
            'Premium': 1.1,
            'Standard': 1.0,
            'Quick': 0.9,
            'Emergency': 0.5
        }
        
        # Strength adjustment (0.8 to 1.2)
        strength_multiplier = 0.8 + (strength / 100) * 0.4
        
        # Calculate final leverage
        adjusted = int(base_leverage * tier_multipliers.get(tier, 1.0) * strength_multiplier)
        
        # Apply limits
        return max(1, min(adjusted, Config.MAX_LEVERAGE))


class ConservativeMode(BaseModeHandler):
    """Conservative trading mode - low risk, high quality signals only"""
    
    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get('tier', 'Standard')
        strength = alert.get('strength', 0)
        
        # Only accept high quality signals
        allowed_tiers = ['Platinum', 'Premium']
        if not self.check_tier_allowed(tier, allowed_tiers):
            return {
                'accept': False,
                'reason': f'Conservative mode only accepts {allowed_tiers} tiers'
            }
        
        # High strength requirement
        min_strength = 70
        if not self.check_strength_threshold(strength, min_strength):
            return {
                'accept': False,
                'reason': f'Signal strength {strength:.1f}% below conservative threshold {min_strength}%'
            }
        
        # Check for fake breakout
        if alert.get('fake_breakout_detected'):
            return {
                'accept': False,
                'reason': 'Fake breakout detected - rejected in conservative mode'
            }
        
        # Check institutional flow
        if alert.get('institutional_flow', 0) < 0.3:
            return {
                'accept': False,
                'reason': 'Insufficient institutional flow for conservative mode'
            }
        
        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 3):  # Max 3 positions in conservative
                return {
                    'accept': False,
                    'reason': 'Max positions (3) reached in conservative mode'
                }
        
        # Calculate conservative adjustments
        leverage = self.calculate_leverage_adjustment(5, tier, strength)  # Base leverage 5x
        
        return {
            'accept': True,
            'reason': 'Signal meets conservative criteria',
            'adjustments': {
                'leverage': leverage,
                'position_size_multiplier': 0.5,  # Half position size
                'stop_loss_multiplier': 0.8  # Tighter stop loss
            }
        }


class BalancedMode(BaseModeHandler):
    """Balanced trading mode - moderate risk/reward"""
    
    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get('tier', 'Standard')
        strength = alert.get('strength', 0)
        
        # Accept most tiers except Emergency
        allowed_tiers = ['Platinum', 'Premium', 'Standard', 'Quick']
        if not self.check_tier_allowed(tier, allowed_tiers):
            return {
                'accept': False,
                'reason': f'Balanced mode does not accept {tier} tier'
            }
        
        # Moderate strength requirement
        min_strength = 50
        if not self.check_strength_threshold(strength, min_strength):
            return {
                'accept': False,
                'reason': f'Signal strength {strength:.1f}% below balanced threshold {min_strength}%'
            }
        
        # Reduce position on fake breakout
        fake_breakout_multiplier = 1.0
        if alert.get('fake_breakout_detected'):
            fake_breakout_multiplier = 0.5
        
        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 5):  # Max 5 positions
                return {
                    'accept': False,
                    'reason': 'Max positions (5) reached in balanced mode'
                }
        
        # Calculate balanced adjustments
        leverage = self.calculate_leverage_adjustment(10, tier, strength)  # Base leverage 10x
        
        return {
            'accept': True,
            'reason': 'Signal meets balanced criteria',
            'adjustments': {
                'leverage': leverage,
                'position_size_multiplier': 0.75 * fake_breakout_multiplier,
                'stop_loss_multiplier': 1.0
            }
        }


class AggressiveMode(BaseModeHandler):
    """Aggressive trading mode - higher risk, more signals"""
    
    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get('tier', 'Standard')
        strength = alert.get('strength', 0)
        
        # Accept all tiers
        allowed_tiers = ['Platinum', 'Premium', 'Standard', 'Quick', 'Emergency']
        
        # Lower strength requirement
        min_strength = 30
        if not self.check_strength_threshold(strength, min_strength):
            return {
                'accept': False,
                'reason': f'Signal strength {strength:.1f}% below aggressive threshold {min_strength}%'
            }
        
        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 8):  # Max 8 positions
                return {
                    'accept': False,
                    'reason': 'Max positions (8) reached in aggressive mode'
                }
        
        # Calculate aggressive adjustments
        leverage = self.calculate_leverage_adjustment(15, tier, strength)  # Base leverage 15x
        
        # Boost leverage for strong signals
        if strength > 80 and tier in ['Platinum', 'Premium']:
            leverage = min(leverage + 5, Config.MAX_LEVERAGE)
        
        return {
            'accept': True,
            'reason': 'Signal accepted in aggressive mode',
            'adjustments': {
                'leverage': leverage,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.2  # Wider stop loss
            }
        }


class ScalpingMode(BaseModeHandler):
    """Scalping mode - quick trades, small profits"""
    
    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get('tier', 'Standard')
        strength = alert.get('strength', 0)
        timeframe = alert.get('timeframe', '15m')
        
        # Prefer Quick tier for scalping
        preferred_tiers = ['Quick', 'Standard', 'Premium']
        if tier not in preferred_tiers:
            return {
                'accept': False,
                'reason': f'Scalping mode prefers {preferred_tiers} tiers'
            }
        
        # Check timeframe (prefer lower timeframes)
        if timeframe not in ['1m', '5m', '15m']:
            return {
                'accept': False,
                'reason': f'Scalping mode requires lower timeframe, got {timeframe}'
            }
        
        # Moderate strength requirement
        min_strength = 40
        if not self.check_strength_threshold(strength, min_strength):
            return {
                'accept': False,
                'reason': f'Signal strength {strength:.1f}% below scalping threshold {min_strength}%'
            }
        
        # Check MTF agreement for scalping
        if alert.get('mtf_agreement', 0) < 0.6:
            return {
                'accept': False,
                'reason': 'Insufficient multi-timeframe agreement for scalping'
            }
        
        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 10):  # Max 10 positions for scalping
                return {
                    'accept': False,
                    'reason': 'Max positions (10) reached in scalping mode'
                }
        
        # Calculate scalping adjustments
        leverage = self.calculate_leverage_adjustment(20, tier, strength)  # Higher leverage for scalping
        
        return {
            'accept': True,
            'reason': 'Signal meets scalping criteria',
            'adjustments': {
                'leverage': leverage,
                'position_size_multiplier': 0.5,  # Smaller positions
                'stop_loss_multiplier': 0.5,  # Very tight stop loss
                'take_profit_multiplier': 0.3,  # Quick profits
                'use_only_tp1': True  # Only use first take profit
            }
        }


class SwingMode(BaseModeHandler):
    """Swing trading mode - longer term positions"""
    
    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get('tier', 'Standard')
        strength = alert.get('strength', 0)
        regime = alert.get('regime', 'NEUTRAL')
        
        # Prefer high quality signals for swing
        allowed_tiers = ['Platinum', 'Premium', 'Standard']
        if not self.check_tier_allowed(tier, allowed_tiers):
            return {
                'accept': False,
                'reason': f'Swing mode requires {allowed_tiers} tiers'
            }
        
        # High strength requirement
        min_strength = 60
        if not self.check_strength_threshold(strength, min_strength):
            return {
                'accept': False,
                'reason': f'Signal strength {strength:.1f}% below swing threshold {min_strength}%'
            }
        
        # Check market regime
        if regime == 'NEUTRAL':
            return {
                'accept': False,
                'reason': 'Swing mode requires trending market (BULLISH/BEARISH)'
            }
        
        # Check regime confidence
        if alert.get('regime_confidence', 0) < 0.7:
            return {
                'accept': False,
                'reason': 'Insufficient regime confidence for swing trading'
            }
        
        # Check max positions
        with Session() as session:
            if not self.check_max_positions(session, 3):  # Max 3 positions for swing
                return {
                    'accept': False,
                    'reason': 'Max positions (3) reached in swing mode'
                }
        
        # Calculate swing adjustments
        leverage = self.calculate_leverage_adjustment(5, tier, strength)  # Lower leverage for swing
        
        return {
            'accept': True,
            'reason': 'Signal meets swing trading criteria',
            'adjustments': {
                'leverage': leverage,
                'position_size_multiplier': 1.5,  # Larger positions for swing
                'stop_loss_multiplier': 2.0,  # Wider stop loss
                'hold_for_all_targets': True,  # Hold for all take profit levels
                'disable_break_even': True  # Don't move to break even quickly
            }
        }


class EmergencyMode(BaseModeHandler):
    """Emergency mode - only emergency signals or close positions"""
    
    async def evaluate(self, alert: Dict) -> Dict:
        tier = alert.get('tier', 'Standard')
        action = alert.get('action', '').upper()
        
        # Only accept Emergency tier or CLOSE actions
        if action == 'CLOSE':
            return {
                'accept': True,
                'reason': 'Close signal accepted in emergency mode',
                'adjustments': {
                    'immediate_close': True
                }
            }
        
        if tier != 'Emergency':
            return {
                'accept': False,
                'reason': 'Only Emergency tier signals accepted in emergency mode'
            }
        
        # Check if we should open new positions at all
        with Session() as session:
            open_positions = get_open_positions(session)
            if len(open_positions) > 0:
                return {
                    'accept': False,
                    'reason': 'No new positions in emergency mode with existing positions'
                }
        
        # Very conservative settings for emergency trades
        return {
            'accept': True,
            'reason': 'Emergency signal accepted',
            'adjustments': {
                'leverage': 1,  # Minimum leverage
                'position_size_multiplier': 0.25,  # Very small position
                'stop_loss_multiplier': 0.5,  # Very tight stop
                'emergency_mode': True
            }
        }


# Mode transition rules
class ModeTransitionManager:
    """Manages automatic mode transitions based on performance"""
    
    @staticmethod
    def check_transition_rules() -> Optional[Tuple[str, str]]:
        """Check if mode should be transitioned
        Returns: (new_mode, reason) or None
        """
        with Session() as session:
            current_mode = get_setting(session, "mode", Config.DEFAULT_MODE)
            
            # Get performance metrics
            from database import get_performance_stats
            daily_stats = get_performance_stats(session, "daily")
            weekly_stats = get_performance_stats(session, "weekly")
            
            # Emergency triggers
            if daily_stats['total_pnl'] < -Config.DAILY_LOSS_LIMIT * 1.5:
                return ("emergency", f"Daily loss exceeded 150% of limit: ${daily_stats['total_pnl']:.2f}")
            
            # Conservative triggers
            if daily_stats['win_rate'] < 25 and daily_stats['total_trades'] >= 5:
                return ("conservative", f"Very low win rate: {daily_stats['win_rate']:.1f}%")
            
            # Aggressive triggers (only if performing well)
            if current_mode != "aggressive":
                if daily_stats['win_rate'] > 75 and daily_stats['total_trades'] >= 10:
                    if daily_stats['total_pnl'] > Config.DAILY_PROFIT_TARGET:
                        return ("aggressive", f"Excellent performance - WR: {daily_stats['win_rate']:.1f}%, PnL: ${daily_stats['total_pnl']:.2f}")
            
            # Return to balanced from extremes
            if current_mode in ["conservative", "aggressive"]:
                if 40 <= daily_stats['win_rate'] <= 60 and daily_stats['total_trades'] >= 5:
                    return ("balanced", "Performance normalized")
            
            return None
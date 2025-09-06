"""
Signal Intelligence System for Trading Bot v9.1
Enhanced with comprehensive v9.1 features:
- Institutional flow analysis
- Fake breakout detection
- Enhanced regime detection
- Multi-timeframe agreement
- Volume context analysis
- Retest confidence scoring
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass

from config import Config
from database import Session, get_setting
from ml_predictor import TradingMLPredictor

logger = logging.getLogger(__name__)


@dataclass
class VolumeContext:
    """Volume context data structure - v9.1 NEW"""

    current_volume: float
    avg_volume_20: float
    volume_ratio: float
    volume_spike: bool
    volume_trend: str  # increasing/decreasing/stable
    institutional_flow: float
    retail_flow: float


@dataclass
class RegimeAnalysis:
    """Market regime analysis - v9.1 ENHANCED"""

    regime: str  # TRENDING_UP/TRENDING_DOWN/RANGING/VOLATILE/BREAKOUT
    confidence: float
    strength: float
    duration_bars: int
    volatility_percentile: float
    trend_consistency: float


class SignalIntelligence:
    """Enhanced Signal Intelligence System v9.1"""

    def __init__(self):
        """Initialize Signal Intelligence with v9.1 enhancements"""
        self.ml_predictor = TradingMLPredictor()

        # v9.1 CORE: Enhanced caching
        self.regime_cache = {}  # symbol -> RegimeAnalysis
        self.volume_cache = {}  # symbol -> VolumeContext
        self.fake_breakout_cache = {}  # symbol -> detection_data
        self.institutional_flow_cache = {}  # symbol -> flow_data

        # v9.1 CORE: Analysis parameters
        self.min_strength_thresholds = {
            "Emergency": 0.15,
            "Platinum": 0.25,
            "Premium": 0.30,
            "Standard": 0.35,
            "Quick": 0.40,
        }

        # v9.1 CORE: Fake breakout detection parameters
        self.fake_breakout_params = {
            "volume_threshold": 1.5,  # Minimum volume ratio for valid breakout
            "price_follow_through": 0.002,  # 0.2% minimum follow-through
            "time_window_minutes": 15,  # Time window for validation
            "retest_strength_threshold": 0.7,  # Minimum strength for retest signals
        }

        logger.info("Signal Intelligence v9.1 initialized with enhanced features")
        # v9.1 ADAPTIVE: Trade counting and adaptive thresholds
        self.total_trades_count = 0
        self.adaptive_learning_threshold = 1000  # First 1000 trades
        self.is_adaptive_mode = True
        self.base_strength_thresholds = self.min_strength_thresholds.copy()

    async def initialize(self):
        """Initialize async components - v9.1"""
        try:
            # Initialize ML predictor
            await self.ml_predictor.initialize()

            # Load settings
            with Session() as session:
                self.use_ml_for_decision = get_setting(
                    session, "use_ml_for_decision", Config.USE_ML_FOR_DECISION
                )
                self.use_ml_for_sizing = get_setting(
                    session, "use_ml_for_sizing", Config.USE_ML_FOR_SIZING
                )
                self.enable_fake_breakout_detection = get_setting(
                    session, "enable_fake_breakout_detection", True
                )
                self.enable_institutional_flow = get_setting(
                    session, "enable_institutional_flow", True
                )
                # Load trade count for adaptive system
                self.total_trades_count = get_setting(session, "total_trades_count", 0)
                self.is_adaptive_mode = self.total_trades_count < self.adaptive_learning_threshold
                
                if self.is_adaptive_mode:
                    logger.info(f"Adaptive mode active: {self.total_trades_count}/{self.adaptive_learning_threshold} trades")
                else:
                    logger.info(f"Standard mode active: {self.total_trades_count} trades completed")

            logger.info("Signal Intelligence initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Signal Intelligence: {e}")
            raise

    async def analyze_signal(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive signal analysis with v9.1 enhancements"""
        start_time = time.time()

        try:
            # v9.1 CORE: Parse and validate alert data
            parsed_data = self._parse_alert_data_v91(alert_data)

            # v9.1 CORE: Enhanced regime analysis
            regime_analysis = await self._analyze_market_regime(parsed_data)
            parsed_data["regime_analysis"] = regime_analysis

            # v9.1 CORE: Volume context analysis
            volume_context = await self._analyze_volume_context(parsed_data)
            parsed_data["volume_context"] = volume_context

            # v9.1 CORE: Institutional flow analysis
            institutional_flow = await self._analyze_institutional_flow(parsed_data)
            parsed_data["institutional_flow"] = institutional_flow

            # v9.1 CORE: Fake breakout detection
            fake_breakout_result = await self._detect_fake_breakout(parsed_data)
            parsed_data["fake_breakout_detected"] = fake_breakout_result["detected"]
            parsed_data["fake_breakout_penalty"] = fake_breakout_result["penalty"]

            # v9.1 CORE: Retest confidence scoring
            retest_confidence = await self._calculate_retest_confidence(parsed_data)
            parsed_data["retest_confidence"] = retest_confidence

            # v9.1 CORE: Multi-timeframe agreement
            mtf_agreement = await self._calculate_mtf_agreement(parsed_data)
            parsed_data["mtf_agreement_ratio"] = mtf_agreement

            # v9.1 CORE: Enhanced filtering logic
            filter_result = await self._apply_enhanced_filters(parsed_data)
            if not filter_result["accept"]:
                return {
                    "accept": False,
                    "reason": filter_result["reason"],
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "analysis_data": parsed_data,
                }

            # v9.1 CORE: ML analysis (if enabled)
            if self.use_ml_for_decision or self.use_ml_for_sizing:
                ml_result = await self._apply_ml_analysis(parsed_data)
                parsed_data["ml_predictions"] = ml_result

                if self.use_ml_for_decision and not ml_result["should_take"]:
                    return {
                        "accept": False,
                        "reason": f"ML rejection: {ml_result['reason']}",
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                        "analysis_data": parsed_data,
                    }

            # v9.1 CORE: Calculate final parameters
            final_params = await self._calculate_final_parameters(parsed_data)

            processing_time = int((time.time() - start_time) * 1000)
            # Update trade count for adaptive system
            await self.update_trade_count()

            return {
                "accept": True,
                "should_trade": True,
                "reason": "Signal accepted after comprehensive analysis",
                "processing_time_ms": processing_time,
                "analysis_data": parsed_data,
                "trade_parameters": final_params,
                "institutional_flow": institutional_flow,
                "fake_breakout_detected": fake_breakout_result["detected"],
                "enhanced_regime": regime_analysis.regime,
                "regime_confidence": regime_analysis.confidence,
                "retest_confidence": retest_confidence,
                "mtf_agreement_ratio": mtf_agreement,
                "volume_context": volume_context.__dict__ if volume_context else None,
            }

        except Exception as e:
            logger.error(f"Error in signal analysis: {e}")
            return {
                "accept": False,
                "reason": f"Analysis error: {str(e)}",
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }
    async def update_trade_count(self):
        """Update trade count and check if adaptive mode should be disabled"""
        try:
            self.total_trades_count += 1
            
            # Check if we should exit adaptive mode
            if self.is_adaptive_mode and self.total_trades_count >= self.adaptive_learning_threshold:
                self.is_adaptive_mode = False
                logger.info(f"Exiting adaptive mode after {self.total_trades_count} trades")
                
                # Update thresholds with ML predictions if available
                if hasattr(self, 'ml_predictor') and self.ml_predictor:
                    await self._update_thresholds_with_ml()
            
            # Save to database every 10 trades
            if self.total_trades_count % 10 == 0:
                with Session() as session:
                    from database import update_setting
                    update_setting(session, "total_trades_count", self.total_trades_count)
                    session.commit()
                    
        except Exception as e:
            logger.error(f"Error updating trade count: {e}")
    async def _update_thresholds_with_ml(self):
        """Update strength thresholds based on ML learning"""
        try:
            if not hasattr(self, 'ml_predictor') or not self.ml_predictor:
                return
                
            # Get ML recommendations for optimal thresholds
            ml_recommendations = await self.ml_predictor.get_optimal_thresholds()
            
            if ml_recommendations:
                for tier, recommended_threshold in ml_recommendations.items():
                    if tier in self.min_strength_thresholds:
                        old_threshold = self.min_strength_thresholds[tier]
                        self.min_strength_thresholds[tier] = recommended_threshold
                        logger.info(f"Updated {tier} threshold: {old_threshold:.3f} -> {recommended_threshold:.3f}")
                        
        except Exception as e:
            logger.error(f"Error updating thresholds with ML: {e}")

    def _parse_alert_data_v91(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse alert data with v9.1 enhancements"""
        try:
            action = str(alert_data.get("action", "")).lower()
            is_emergency = (
                "emergency" in action or alert_data.get("tier") == "Emergency"
            )

            parsed = {
                # Basic fields
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
                "leverage": int(alert_data.get("leverage", 10)),
                "position_size_multiplier": float(
                    alert_data.get("position_size_multiplier", 1.0)
                ),
                # v9.1 CORE: Enhanced fields
                "indicator_version": alert_data.get("indicator_version", "8.0"),
                "timeframe": alert_data.get("timeframe", "15m"),
                "session": str(alert_data.get("session", "Unknown")),
                "timestamp": alert_data.get("timestamp", datetime.utcnow().timestamp()),
                # Market context
                "market_regime": str(alert_data.get("market_regime", "NEUTRAL")),
                "market_condition": str(alert_data.get("market_condition", "NORMAL")),
                "confidence_penalty": float(alert_data.get("confidence_penalty", 0.0)),
                # Technical indicators
                "mfi": float(alert_data.get("mfi", 50)),
                "adx": float(alert_data.get("adx", 0)),
                "rsi": float(alert_data.get("rsi", 50)),
                "htf_trend": str(alert_data.get("htf_trend", "neutral")),
                "btc_correlation": float(alert_data.get("btc_correlation", 0)),
                # Volume and flow data
                "volume_spike": str(alert_data.get("volume_spike", "false")).lower()
                == "true",
                "volume_ratio": float(alert_data.get("volume_ratio", 1.0)),
                "institutional_volume": float(
                    alert_data.get("institutional_volume", 0)
                ),
                "retail_volume": float(alert_data.get("retail_volume", 0)),
                # Key levels and structure
                "near_key_level": str(alert_data.get("near_key_level", "false")).lower()
                == "true",
                "key_level_distance": float(alert_data.get("key_level_distance", 0)),
                "structure_break": str(
                    alert_data.get("structure_break", "false")
                ).lower()
                == "true",
                # v9.1 NEW: Enhanced context
                "liquidity_grab": str(alert_data.get("liquidity_grab", "false")).lower()
                == "true",
                "order_block_retest": str(
                    alert_data.get("order_block_retest", "false")
                ).lower()
                == "true",
                "fair_value_gap": str(alert_data.get("fair_value_gap", "false")).lower()
                == "true",
                "imbalance_fill": str(alert_data.get("imbalance_fill", "false")).lower()
                == "true",
                # Priority enhancements
                "priority_enhancements": alert_data.get("priority_enhancements", {}),
                "emergency_conditions": alert_data.get("emergency_conditions", {}),
            }

            # Validation
            if not parsed["symbol"] or parsed["price"] <= 0 or parsed["sl"] <= 0:
                raise ValueError("Missing critical data: symbol, price, or stop loss")
            return parsed

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing v9.1 alert data: {e}")
            raise

    async def _analyze_market_regime(
        self, parsed_data: Dict[str, Any]
    ) -> RegimeAnalysis:
        """Analyze market regime with v9.1 enhancements"""
        try:
            symbol = parsed_data["symbol"]

            # Check cache first
            if symbol in self.regime_cache:
                cached_regime = self.regime_cache[symbol]
                if (
                    time.time() - cached_regime.get("timestamp", 0) < 300
                ):  # 5 minutes cache
                    return cached_regime["analysis"]

            # v9.1 CORE: Enhanced regime detection
            market_regime = parsed_data.get("market_regime", "NEUTRAL")
            adx = parsed_data.get("adx", 0)
            mfi = parsed_data.get("mfi", 50)
            htf_trend = parsed_data.get("htf_trend", "neutral")

            # Determine regime
            if market_regime == "TRENDING_UP" or (adx > 25 and htf_trend == "bullish"):
                regime = "TRENDING_UP"
                confidence = min(0.9, adx / 50 + 0.3)
                strength = adx / 100
            elif market_regime == "TRENDING_DOWN" or (
                adx > 25 and htf_trend == "bearish"
            ):
                regime = "TRENDING_DOWN"
                confidence = min(0.9, adx / 50 + 0.3)
                strength = adx / 100
            elif market_regime == "RANGING" or adx < 20:
                regime = "RANGING"
                confidence = 0.7
                strength = 0.3
            elif market_regime == "VOLATILE":
                regime = "VOLATILE"
                confidence = 0.6
                strength = 0.8
            else:
                regime = "BREAKOUT"
                confidence = 0.8
                strength = 0.7

            # Calculate additional metrics
            volatility_percentile = min(100, max(0, (abs(mfi - 50) / 50) * 100))
            trend_consistency = confidence * 0.8 + (strength * 0.2)

            analysis = RegimeAnalysis(
                regime=regime,
                confidence=confidence,
                strength=strength,
                duration_bars=parsed_data.get("regime_duration", 10),
                volatility_percentile=volatility_percentile,
                trend_consistency=trend_consistency,
            )

            # Cache result
            self.regime_cache[symbol] = {"analysis": analysis, "timestamp": time.time()}

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return RegimeAnalysis(
                regime="NEUTRAL",
                confidence=0.5,
                strength=0.5,
                duration_bars=5,
                volatility_percentile=50,
                trend_consistency=0.5,
            )

    async def _analyze_volume_context(
        self, parsed_data: Dict[str, Any]
    ) -> Optional[VolumeContext]:
        """Analyze volume context with v9.1 enhancements"""
        try:
            symbol = parsed_data["symbol"]

            # Get volume data from alert
            volume_ratio = parsed_data.get("volume_ratio", 1.0)
            volume_spike = parsed_data.get("volume_spike", False)
            institutional_volume = parsed_data.get("institutional_volume", 0)
            retail_volume = parsed_data.get("retail_volume", 0)

            # Calculate volume metrics
            current_volume = volume_ratio * 1000000  # Normalized volume
            avg_volume_20 = (
                current_volume / volume_ratio if volume_ratio > 0 else current_volume
            )

            # Determine volume trend
            if volume_ratio > 1.5:
                volume_trend = "increasing"
            elif volume_ratio < 0.7:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"

            # Calculate institutional vs retail flow
            total_volume = institutional_volume + retail_volume
            if total_volume > 0:
                institutional_flow = institutional_volume / total_volume
                retail_flow = retail_volume / total_volume
            else:
                institutional_flow = 0.5
                retail_flow = 0.5

            context = VolumeContext(
                current_volume=current_volume,
                avg_volume_20=avg_volume_20,
                volume_ratio=volume_ratio,
                volume_spike=volume_spike,
                volume_trend=volume_trend,
                institutional_flow=institutional_flow,
                retail_flow=retail_flow,
            )

            # Cache result
            self.volume_cache[symbol] = {"context": context, "timestamp": time.time()}

            return context

        except Exception as e:
            logger.error(f"Error analyzing volume context: {e}")
            return None

    async def _analyze_institutional_flow(self, parsed_data: Dict[str, Any]) -> float:
        """Analyze institutional flow with v9.1 enhancements"""
        try:
            if not self.enable_institutional_flow:
                return 0.0

            symbol = parsed_data["symbol"]

            # Get institutional indicators
            institutional_volume = parsed_data.get("institutional_volume", 0)
            retail_volume = parsed_data.get("retail_volume", 0)
            volume_ratio = parsed_data.get("volume_ratio", 1.0)

            # Calculate institutional flow score
            total_volume = institutional_volume + retail_volume
            if total_volume > 0:
                institutional_ratio = institutional_volume / total_volume
            else:
                institutional_ratio = 0.5

            # Adjust for volume spike
            volume_adjustment = min(2.0, volume_ratio) if volume_ratio > 1.0 else 1.0

            # Calculate final flow score (-1 to 1)
            flow_score = (institutional_ratio - 0.5) * 2 * volume_adjustment
            flow_score = max(-1.0, min(1.0, flow_score))

            # Cache result
            self.institutional_flow_cache[symbol] = {
                "flow_score": flow_score,
                "timestamp": time.time(),
            }

            return flow_score

        except Exception as e:
            logger.error(f"Error analyzing institutional flow: {e}")
            return 0.0

    async def _detect_fake_breakout(
        self, parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect fake breakouts with v9.1 enhancements"""
        try:
            if not self.enable_fake_breakout_detection:
                return {"detected": False, "penalty": 1.0}

            symbol = parsed_data["symbol"]

            # Get breakout indicators
            structure_break = parsed_data.get("structure_break", False)
            volume_ratio = parsed_data.get("volume_ratio", 1.0)
            near_key_level = parsed_data.get("near_key_level", False)
            strength = parsed_data.get("strength", 0)

            fake_breakout_detected = False
            penalty = 1.0

            # v9.1 CORE: Fake breakout detection logic
            if structure_break:
                # Check volume confirmation
                if volume_ratio < self.fake_breakout_params["volume_threshold"]:
                    fake_breakout_detected = True
                    penalty = 0.7  # Reduce position size by 30%

                # Check if near key level without strong volume
                if near_key_level and volume_ratio < 2.0 and strength < 0.6:
                    fake_breakout_detected = True
                    penalty = 0.8  # Reduce position size by 20%

                # Check for weak follow-through
                price = parsed_data.get("price", 0)
                sl = parsed_data.get("sl", 0)
                if price > 0 and sl > 0:
                    follow_through = abs(price - sl) / price
                    if (
                        follow_through
                        < self.fake_breakout_params["price_follow_through"]
                    ):
                        fake_breakout_detected = True
                        penalty = 0.9  # Reduce position size by 10%

            # Cache result
            self.fake_breakout_cache[symbol] = {
                "detected": fake_breakout_detected,
                "penalty": penalty,
                "timestamp": time.time(),
            }

            return {"detected": fake_breakout_detected, "penalty": penalty}

        except Exception as e:
            logger.error(f"Error detecting fake breakout: {e}")
            return {"detected": False, "penalty": 1.0}

    async def _calculate_retest_confidence(self, parsed_data: Dict[str, Any]) -> float:
        """Calculate retest confidence with v9.1 enhancements"""
        try:
            # Get retest indicators
            order_block_retest = parsed_data.get("order_block_retest", False)
            fair_value_gap = parsed_data.get("fair_value_gap", False)
            liquidity_grab = parsed_data.get("liquidity_grab", False)
            near_key_level = parsed_data.get("near_key_level", False)
            strength = parsed_data.get("strength", 0)

            confidence = 0.5  # Base confidence

            # v9.1 CORE: Retest confidence calculation
            if order_block_retest:
                confidence += 0.2

            if fair_value_gap:
                confidence += 0.15

            if liquidity_grab:
                confidence += 0.1

            if near_key_level:
                confidence += 0.1

            # Adjust for signal strength
            strength_adjustment = (strength - 0.5) * 0.3
            confidence += strength_adjustment

            # Clamp between 0 and 1
            confidence = max(0.0, min(1.0, confidence))

            return confidence

        except Exception as e:
            logger.error(f"Error calculating retest confidence: {e}")
            return 0.5

    async def _calculate_mtf_agreement(self, parsed_data: Dict[str, Any]) -> float:
        """Calculate multi-timeframe agreement with v9.1 enhancements"""
        try:
            # Get timeframe data
            htf_trend = parsed_data.get("htf_trend", "neutral")
            current_action = parsed_data.get("action", "buy")
            adx = parsed_data.get("adx", 0)

            agreement_score = 0.5  # Base score

            # v9.1 CORE: MTF agreement calculation
            if htf_trend == "bullish" and current_action == "buy":
                agreement_score = 0.8
            elif htf_trend == "bearish" and current_action == "sell":
                agreement_score = 0.8
            elif htf_trend == "neutral":
                agreement_score = 0.6
            else:
                agreement_score = 0.3  # Divergence

            # Adjust for trend strength (ADX)
            if adx > 25:
                agreement_score += 0.1
            elif adx < 15:
                agreement_score -= 0.1

            # Clamp between 0 and 1
            agreement_score = max(0.0, min(1.0, agreement_score))

            return agreement_score

        except Exception as e:
            logger.error(f"Error calculating MTF agreement: {e}")
            return 0.5

    async def _apply_enhanced_filters(
        self, parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply enhanced filtering logic with v9.1 enhancements"""
        try:
            tier = parsed_data.get("tier", "Quick")
            strength = parsed_data.get("strength", 0)
            is_emergency = parsed_data.get("is_emergency_signal", False)

            # v9.1 CORE: Enhanced strength filtering
            min_strength = self._get_adaptive_strength_threshold(tier)
            if is_emergency:
                min_strength *= 0.8  # Lower threshold for emergency signals

            if strength < min_strength:
                return {
                    "accept": False,
                    "reason": f"Signal strength {strength:.3f} below minimum {min_strength:.3f} for tier {tier}",
                }

            # v9.1 CORE: Market condition filters
            market_condition = parsed_data.get("market_condition", "NORMAL")
            if not is_emergency and market_condition in [
                "EXTREME_MOVE",
                "MARKET_STRESS",
            ]:
                return {
                    "accept": False,
                    "reason": f"Extreme market conditions: {market_condition}",
                }

            # v9.1 CORE: Confidence penalty filter
            confidence_penalty = parsed_data.get("confidence_penalty", 0.0)
            max_penalty = 0.3 if is_emergency else 0.25
            if confidence_penalty > max_penalty:
                return {
                    "accept": False,
                    "reason": f"High confidence penalty: {confidence_penalty:.2f} > {max_penalty}",
                }

            # v9.1 CORE: Fake breakout filter
            if parsed_data.get("fake_breakout_detected", False):
                fake_breakout_penalty = parsed_data.get("fake_breakout_penalty", 1.0)
                if fake_breakout_penalty < 0.8:  # Too risky
                    return {
                        "accept": False,
                        "reason": f"High fake breakout risk (penalty: {fake_breakout_penalty:.2f})",
                    }

            # v9.1 CORE: MTF agreement filter
            mtf_agreement = parsed_data.get("mtf_agreement_ratio", 0.5)
            if mtf_agreement < 0.4:
                return {
                    "accept": False,
                    "reason": f"Poor multi-timeframe agreement: {mtf_agreement:.2f}",
                }

            return {"accept": True, "reason": "All filters passed"}

        except Exception as e:
            logger.error(f"Error applying enhanced filters: {e}")
            return {"accept": False, "reason": f"Filter error: {str(e)}"}

    async def _apply_ml_analysis(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ML analysis with v9.1 enhancements"""
        try:
            # Prepare ML features
            ml_features = {
                "strength": parsed_data.get("strength", 0),
                "tier": parsed_data.get("tier", "Quick"),
                "adx": parsed_data.get("adx", 0),
                "mfi": parsed_data.get("mfi", 50),
                "rsi": parsed_data.get("rsi", 50),
                "volume_ratio": parsed_data.get("volume_ratio", 1.0),
                "institutional_flow": parsed_data.get("institutional_flow", 0),
                "regime_confidence": parsed_data.get("regime_analysis", {}).get(
                    "confidence", 0.5
                ),
                "retest_confidence": parsed_data.get("retest_confidence", 0.5),
                "mtf_agreement": parsed_data.get("mtf_agreement_ratio", 0.5),
            }

            # Get ML predictions
            predictions = await self.ml_predictor.predict(ml_features)

            # Determine if should take trade
            min_win_prob = 0.50 if parsed_data.get("is_emergency_signal") else 0.55
            should_take, reason = self.ml_predictor.should_take_trade(
                predictions, min_win_prob=min_win_prob
            )

            # Calculate ML risk multiplier
            ml_multiplier = self._calculate_ml_risk_multiplier(predictions)

            return {
                "predictions": predictions,
                "should_take": should_take,
                "reason": reason,
                "risk_multiplier": ml_multiplier,
            }

        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
            return {
                "predictions": {},
                "should_take": True,
                "reason": "ML analysis failed",
                "risk_multiplier": 1.0,
            }

    async def _calculate_final_parameters(
        self, parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate final trade parameters with v9.1 enhancements"""
        try:
            # Base multiplier from indicator
            base_multiplier = parsed_data.get("position_size_multiplier", 1.0)

            # v9.1 CORE: Priority multiplier
            priority_multiplier = self._calculate_priority_multiplier(parsed_data)

            # v9.1 CORE: Fake breakout penalty
            fake_breakout_penalty = parsed_data.get("fake_breakout_penalty", 1.0)

            # v9.1 CORE: ML multiplier (if enabled)
            ml_multiplier = 1.0
            if self.use_ml_for_sizing and "ml_predictions" in parsed_data:
                ml_multiplier = parsed_data["ml_predictions"].get(
                    "risk_multiplier", 1.0
                )

            # v9.1 CORE: Regime adjustment
            regime_analysis = parsed_data.get("regime_analysis")
            regime_multiplier = 1.0
            if regime_analysis:
                if regime_analysis.regime in ["TRENDING_UP", "TRENDING_DOWN"]:
                    regime_multiplier = 1.1  # Boost trending signals
                elif regime_analysis.regime == "VOLATILE":
                    regime_multiplier = 0.9  # Reduce volatile signals

            # Calculate final multiplier
            final_multiplier = (
                base_multiplier
                * priority_multiplier
                * fake_breakout_penalty
                * ml_multiplier
                * regime_multiplier
            )

            # Clamp multiplier
            final_multiplier = max(0.5, min(2.0, final_multiplier))

            return {
                "position_multiplier": final_multiplier,
                "base_multiplier": base_multiplier,
                "priority_multiplier": priority_multiplier,
                "fake_breakout_penalty": fake_breakout_penalty,
                "ml_multiplier": ml_multiplier,
                "regime_multiplier": regime_multiplier,
                "leverage": parsed_data.get("leverage", 10),
                "tier": parsed_data.get("tier", "Quick"),
            }

        except Exception as e:
            logger.error(f"Error calculating final parameters: {e}")
            return {"position_multiplier": 1.0, "leverage": 10, "tier": "Quick"}

    def _calculate_priority_multiplier(self, parsed_data: Dict[str, Any]) -> float:
        """Calculate priority multiplier with v9.1 enhancements"""
        try:
            multiplier = 1.0

            # Priority enhancements from indicator
            priority_enhancements = parsed_data.get("priority_enhancements", {})

            if priority_enhancements.get("mitigation_mode_active"):
                multiplier *= 1.15
            if priority_enhancements.get("dynamic_opposite_zone_active"):
                multiplier *= 1.12
            if priority_enhancements.get("rising_adx_active"):
                multiplier *= 1.10
            if priority_enhancements.get("emergency_mode_active"):
                multiplier *= 1.08

            # v9.1 CORE: Enhanced priority factors
            if parsed_data.get("order_block_retest"):
                multiplier *= 1.05
            if parsed_data.get("liquidity_grab"):
                multiplier *= 1.03
            if parsed_data.get("fair_value_gap"):
                multiplier *= 1.02

            # Emergency conditions bonus
            if parsed_data.get("is_emergency_signal"):
                emergency_conditions = parsed_data.get("emergency_conditions", {})
                volume_multiplier = emergency_conditions.get("volume_multiplier", 1.0)
                if volume_multiplier > 5.0:
                    multiplier *= 1.20
                elif volume_multiplier > 3.0:
                    multiplier *= 1.10

            return multiplier

        except Exception as e:
            logger.error(f"Error calculating priority multiplier: {e}")
            return 1.0

    def _calculate_ml_risk_multiplier(self, predictions: Dict[str, float]) -> float:
        """Calculate ML risk multiplier with v9.1 enhancements"""
        try:
            win_prob = predictions.get("win_probability", 0.5)
            expected_value = predictions.get("expected_value", 0)
            confidence = predictions.get("confidence", 0.5)

            # Base multiplier from win probability
            if win_prob > 0.70:
                base = 1.3
            elif win_prob > 0.60:
                base = 1.1
            else:
                base = 0.9

            # Expected value adjustment
            if expected_value > 2.0:
                ev_boost = 1.2
            elif expected_value > 1.0:
                ev_boost = 1.1
            elif expected_value < 0:
                ev_boost = 0.7
            else:
                ev_boost = 1.0

            # Confidence adjustment
            confidence_mult = 0.8 + (confidence * 0.4)

            # Calculate final multiplier
            final_multiplier = base * ev_boost * confidence_mult

            # Clamp between 0.5 and 1.5
            return max(0.5, min(1.5, final_multiplier))

        except Exception as e:
            logger.error(f"Error calculating ML risk multiplier: {e}")
            return 1.0

    def _get_adaptive_strength_threshold(self, tier: str) -> float:
        """Get adaptive strength threshold based on trade count"""
        try:
            base_threshold = self.base_strength_thresholds.get(tier, 0.4)
            
            if not self.is_adaptive_mode:
                # Use ML-adjusted thresholds after 1000 trades
                return base_threshold
            
            # Adaptive mode: significantly lower thresholds for learning
            adaptive_multiplier = 0.3  # 70% reduction for first 1000 trades
            adaptive_threshold = base_threshold * adaptive_multiplier
            
            logger.debug(f"Adaptive threshold for {tier}: {adaptive_threshold:.3f} (base: {base_threshold:.3f})")
            return adaptive_threshold
            
        except Exception as e:
            logger.error(f"Error calculating adaptive threshold: {e}")
            return 0.15  # Very low fallback for adaptive mode

    def clear_caches(self):
        """Clear all caches - v9.1 UTILITY"""
        self.regime_cache.clear()
        self.volume_cache.clear()
        self.fake_breakout_cache.clear()
        self.institutional_flow_cache.clear()
        logger.info("Signal Intelligence caches cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics - v9.1 UTILITY"""
        return {
            "regime_cache": len(self.regime_cache),
            "volume_cache": len(self.volume_cache),
            "fake_breakout_cache": len(self.fake_breakout_cache),
            "institutional_flow_cache": len(self.institutional_flow_cache),
        }

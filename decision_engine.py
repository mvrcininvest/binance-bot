"""
Explainable Decision Engine v9.1 - HYBRID ULTRA-DIAGNOSTICS
Kompletny system podejmowania decyzji z pełną transparentnością i wyjaśnialnością
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import math

from database import (
    Session, Trade, Position, AlertHistory, ExecutionTrace, MLPrediction, MLModelMetrics,
    log_execution_trace, complete_execution_trace, log_ml_prediction
)
from config import Config

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Typy decyzji w systemie"""
    SIGNAL_ACCEPTANCE = "signal_acceptance"
    POSITION_SIZING = "position_sizing"
    RISK_MANAGEMENT = "risk_management"
    ML_PREDICTION = "ml_prediction"
    EMERGENCY_ACTION = "emergency_action"
    MODE_CHANGE = "mode_change"
    FAKE_BREAKOUT = "fake_breakout"
    TIER_VALIDATION = "tier_validation"


class DecisionOutcome(Enum):
    """Możliwe wyniki decyzji"""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MODIFIED = "modified"
    DEFERRED = "deferred"
    ERROR = "error"


class ConfidenceLevel(Enum):
    """Poziomy pewności decyzji"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 75-89%
    MEDIUM = "medium"       # 50-74%
    LOW = "low"            # 25-49%
    VERY_LOW = "very_low"  # 0-24%


@dataclass
class DecisionFactor:
    """Pojedynczy czynnik wpływający na decyzję"""
    name: str
    value: Union[float, str, bool]
    weight: float
    impact: float  # -1.0 do 1.0
    explanation: str
    source: str  # ML, rule, config, etc.
    confidence: float  # 0.0 do 1.0


@dataclass
class DecisionPath:
    """Ścieżka podejmowania decyzji"""
    step_number: int
    step_name: str
    input_data: Dict[str, Any]
    factors_evaluated: List[DecisionFactor]
    intermediate_result: Any
    reasoning: str
    execution_time_ms: int


@dataclass
class DecisionResult:
    """Kompletny wynik procesu decyzyjnego"""
    decision_id: str
    decision_type: DecisionType
    outcome: DecisionOutcome
    confidence: ConfidenceLevel
    confidence_score: float  # 0.0 do 1.0
    
    # Dane wejściowe
    input_signal: Dict[str, Any]
    context: Dict[str, Any]
    
    # Proces decyzyjny
    decision_path: List[DecisionPath]
    all_factors: List[DecisionFactor]
    
    # Wynik
    final_decision: Dict[str, Any]
    reasoning_summary: str
    recommendations: List[str]
    
    # Metadane
    timestamp: datetime
    execution_time_ms: int
    model_versions: Dict[str, str]
    
    # Risk assessment
    risk_score: float
    risk_factors: List[str]
    
    # Performance tracking
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    outcome_timestamp: Optional[datetime] = None


class DecisionEngine:
    """
    Explainable Decision Engine v9.1
    Zapewnia pełną transparentność i wyjaśnialność wszystkich decyzji systemu
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decision_history = []
        self.factor_weights = self._initialize_factor_weights()
        self.decision_rules = self._initialize_decision_rules()
        self.performance_tracker = {}
        
        # Thresholds for decision making
        self.confidence_thresholds = {
            DecisionType.SIGNAL_ACCEPTANCE: 0.6,
            DecisionType.POSITION_SIZING: 0.5,
            DecisionType.RISK_MANAGEMENT: 0.8,
            DecisionType.ML_PREDICTION: 0.4,
            DecisionType.EMERGENCY_ACTION: 0.9,
            DecisionType.MODE_CHANGE: 0.7,
            DecisionType.FAKE_BREAKOUT: 0.6,
            DecisionType.TIER_VALIDATION: 0.5
        }

    async def initialize(self):
        """Initialize the decision engine."""
        self.logger.info("Decision Engine initialized.")
        # W przyszłości można tu dodać logikę ładowania modeli, wag, itp.
        pass

    async def optimize_parameters(
        self, 
        signal_data: Dict[str, Any], 
        decision: Dict[str, Any], 
        trade_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize trade parameters based on signal, context, and ML insights.
        Na razie zwraca parametry bez modyfikacji.
        """
        self.logger.info(f"Running parameter optimization for {signal_data.get('symbol')}")
        
        # Tutaj w przyszłości znajdzie się zaawansowana logika optymalizacji
        # np. z użyciem ML, analizy ryzyka, itp.
        
        # Zwracamy oryginalne parametry, aby bot mógł kontynuować
        optimized_params = trade_params.copy()
        optimized_params["parameter_decisions"] = {
            "reason": "No optimization logic implemented yet. Using default parameters."
        }
        
        return optimized_params
    
    def _initialize_factor_weights(self) -> Dict[str, float]:
        """Inicjalizuj wagi czynników decyzyjnych"""
        return {
            # Signal factors
            "signal_strength": 0.20,
            "tier_quality": 0.15,
            "ml_prediction": 0.15,
            "fake_breakout_risk": 0.10,
            "market_conditions": 0.10,
            
            # Risk factors
            "position_risk": 0.08,
            "portfolio_risk": 0.07,
            "correlation_risk": 0.05,
            
            # Technical factors
            "technical_confluence": 0.05,
            "volume_confirmation": 0.03,
            "time_of_day": 0.02
        }
    
    def _initialize_decision_rules(self) -> Dict[str, Dict]:
        """Inicjalizuj reguły decyzyjne"""
        return {
            "signal_acceptance": {
                "min_strength": {
                    "Emergency": 0.01,
                    "Platinum": 0.3,
                    "Premium": 0.4,
                    "Standard": 0.5,
                    "Quick": 0.6
                },
                "max_fake_breakout_risk": 0.7,
                "min_ml_confidence": 0.4,
                "max_portfolio_risk": 0.8
            },
            "position_sizing": {
                "base_risk_per_trade": 0.02,  # 2% per trade
                "max_risk_per_trade": 0.05,   # 5% max
                "tier_multipliers": {
                    "Emergency": 1.5,
                    "Platinum": 1.3,
                    "Premium": 1.1,
                    "Standard": 1.0,
                    "Quick": 0.8
                }
            },
            "risk_management": {
                "max_total_risk": 0.15,       # 15% total portfolio
                "max_correlated_risk": 0.10,  # 10% in correlated assets
                "emergency_stop_loss": 0.20   # 20% portfolio loss
            }
        }
    
    async def make_signal_decision(
        self, 
        signal_data: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> DecisionResult:
        """
        Podejmij decyzję o przyjęciu sygnału z pełną transparentnością
        """
        start_time = time.time()
        decision_id = f"signal_{int(time.time() * 1000)}"
        
        # Log execution trace
        trace_id = log_execution_trace("signal_decision", {
            "decision_id": decision_id,
            "symbol": signal_data.get("symbol", "unknown"),
            "action": signal_data.get("action", "unknown"),
            "tier": signal_data.get("tier", "unknown")
        })
        
        try:
            self.logger.info(f"🧠 Making signal decision: {decision_id}")
            
            # Initialize decision context
            if context is None:
                context = await self._gather_decision_context(signal_data)
            
            decision_path = []
            all_factors = []
            
            # Step 1: Basic Signal Validation
            step1_result = await self._evaluate_basic_signal_validation(
                signal_data, context, decision_path, all_factors
            )
            
            if not step1_result["valid"]:
                return self._create_rejection_result(
                    decision_id, DecisionType.SIGNAL_ACCEPTANCE, signal_data, context,
                    decision_path, all_factors, step1_result["reason"], trace_id, start_time
                )
            
            # Step 2: ML Prediction Analysis
            step2_result = await self._evaluate_ml_prediction(
                signal_data, context, decision_path, all_factors
            )
            
            # Step 3: Fake Breakout Detection
            step3_result = await self._evaluate_fake_breakout_risk(
                signal_data, context, decision_path, all_factors
            )
            
            # Step 4: Risk Management Check
            step4_result = await self._evaluate_risk_management(
                signal_data, context, decision_path, all_factors
            )
            
            if not step4_result["acceptable"]:
                return self._create_rejection_result(
                    decision_id, DecisionType.SIGNAL_ACCEPTANCE, signal_data, context,
                    decision_path, all_factors, step4_result["reason"], trace_id, start_time
                )
            
            # Step 5: Market Conditions Analysis
            step5_result = await self._evaluate_market_conditions(
                signal_data, context, decision_path, all_factors
            )
            
            # Step 6: Portfolio Impact Assessment
            step6_result = await self._evaluate_portfolio_impact(
                signal_data, context, decision_path, all_factors
            )
            
            # Step 7: Final Decision Calculation
            final_result = await self._calculate_final_decision(
                signal_data, context, decision_path, all_factors
            )
            
            # Create decision result
            confidence_score = final_result["confidence_score"]
            confidence_level = self._get_confidence_level(confidence_score)
            
            # Determine outcome
            min_confidence = self.confidence_thresholds[DecisionType.SIGNAL_ACCEPTANCE]
            
            if confidence_score >= min_confidence:
                outcome = DecisionOutcome.ACCEPTED
                reasoning = f"Signal accepted with {confidence_level.value} confidence ({confidence_score:.3f})"
            else:
                outcome = DecisionOutcome.REJECTED
                reasoning = f"Signal rejected - confidence too low ({confidence_score:.3f} < {min_confidence})"
            
            # Calculate risk assessment
            risk_score, risk_factors = self._calculate_risk_assessment(all_factors)
            
            # Generate recommendations
            recommendations = self._generate_decision_recommendations(
                all_factors, outcome, confidence_score
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            decision_result = DecisionResult(
                decision_id=decision_id,
                decision_type=DecisionType.SIGNAL_ACCEPTANCE,
                outcome=outcome,
                confidence=confidence_level,
                confidence_score=confidence_score,
                input_signal=signal_data,
                context=context,
                decision_path=decision_path,
                all_factors=all_factors,
                final_decision=final_result,
                reasoning_summary=reasoning,
                recommendations=recommendations,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                model_versions=self._get_model_versions(),
                risk_score=risk_score,
                risk_factors=risk_factors
            )
            
            # Store decision
            self.decision_history.append(decision_result)
            if len(self.decision_history) > 1000:  # Keep last 1000 decisions
                self.decision_history = self.decision_history[-1000:]
            
            # Save to database
            await self._save_decision_to_database(decision_result)
            
            # Complete execution trace
            complete_execution_trace(
                trace_id, outcome == DecisionOutcome.ACCEPTED, execution_time,
                f"Decision: {outcome.value} (confidence: {confidence_score:.3f})"
            )
            
            self.logger.info(
                f"✅ Signal decision completed: {outcome.value} "
                f"(confidence: {confidence_score:.3f}, time: {execution_time}ms)"
            )
            
            return decision_result
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            complete_execution_trace(trace_id, False, execution_time, f"Decision error: {str(e)}")
            self.logger.error(f"❌ Signal decision failed: {e}", exc_info=True)
            
            # Return error result
            return DecisionResult(
                decision_id=decision_id,
                decision_type=DecisionType.SIGNAL_ACCEPTANCE,
                outcome=DecisionOutcome.ERROR,
                confidence=ConfidenceLevel.VERY_LOW,
                confidence_score=0.0,
                input_signal=signal_data,
                context=context or {},
                decision_path=[],
                all_factors=[],
                final_decision={"error": str(e)},
                reasoning_summary=f"Decision process failed: {str(e)}",
                recommendations=["Investigate decision engine error", "Check system health"],
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                model_versions={},
                risk_score=1.0,
                risk_factors=["System error"]
            )
    
    async def _gather_decision_context(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Zbierz kontekst dla podejmowania decyzji"""
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "market_session": self._get_market_session(),
            "system_mode": getattr(Config, "DEFAULT_MODE", "NORMAL"),
        }
        
        try:
            with Session() as session:
                # Current positions
                from database import get_open_positions
                open_positions = get_open_positions(session)
                context["open_positions_count"] = len(open_positions)
                context["open_positions"] = [
                    {
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "size": float(pos.quantity) if pos.quantity else 0,
                        "pnl": float(pos.pnl_usdt) if pos.pnl_usdt else 0
                    }
                    for pos in open_positions
                ]
                
                # Recent performance
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                recent_trades = session.query(Trade).filter(
                    Trade.entry_time >= recent_cutoff,
                    Trade.status == "closed"
                ).all()
                
                if recent_trades:
                    recent_pnl = sum(t.pnl_usdt for t in recent_trades if t.pnl_usdt)
                    winning_trades = len([t for t in recent_trades if t.pnl_usdt and t.pnl_usdt > 0])
                    
                    context["recent_performance"] = {
                        "trades_24h": len(recent_trades),
                        "pnl_24h": recent_pnl,
                        "win_rate_24h": winning_trades / len(recent_trades) if recent_trades else 0
                    }
                
                # Symbol-specific history
                symbol = signal_data.get("symbol", "")
                if symbol:
                    symbol_trades = session.query(Trade).filter(
                        Trade.symbol == symbol,
                        Trade.entry_time >= datetime.utcnow() - timedelta(days=7)
                    ).all()
                    
                    context["symbol_history"] = {
                        "trades_7d": len(symbol_trades),
                        "avg_pnl": statistics.mean([t.pnl_usdt for t in symbol_trades if t.pnl_usdt]) if symbol_trades else 0,
                        "last_trade_hours_ago": None
                    }
                    
                    if symbol_trades:
                        last_trade = max(symbol_trades, key=lambda t: t.entry_time)
                        hours_ago = (datetime.utcnow() - last_trade.entry_time).total_seconds() / 3600
                        context["symbol_history"]["last_trade_hours_ago"] = hours_ago
        
        except Exception as e:
            self.logger.error(f"Failed to gather decision context: {e}")
            context["context_error"] = str(e)
        
        return context
    
    async def _evaluate_basic_signal_validation(
        self, 
        signal_data: Dict[str, Any], 
        context: Dict[str, Any],
        decision_path: List[DecisionPath],
        all_factors: List[DecisionFactor]
    ) -> Dict[str, Any]:
        """Podstawowa walidacja sygnału"""
        step_start = time.time()
        factors_evaluated = []
        
        # Check required fields
        required_fields = ["symbol", "action", "price", "sl", "tp1", "tier"]
        missing_fields = [field for field in required_fields if not signal_data.get(field)]
        
        if missing_fields:
            factor = DecisionFactor(
                name="required_fields",
                value=f"Missing: {', '.join(missing_fields)}",
                weight=1.0,
                impact=-1.0,
                explanation=f"Signal missing required fields: {missing_fields}",
                source="validation",
                confidence=1.0
            )
            factors_evaluated.append(factor)
            all_factors.append(factor)
            
            decision_path.append(DecisionPath(
                step_number=1,
                step_name="Basic Signal Validation",
                input_data={"required_fields": required_fields, "signal_data": signal_data},
                factors_evaluated=factors_evaluated,
                intermediate_result={"valid": False, "reason": f"Missing fields: {missing_fields}"},
                reasoning="Signal validation failed due to missing required fields",
                execution_time_ms=int((time.time() - step_start) * 1000)
            ))
            
            return {"valid": False, "reason": f"Missing required fields: {missing_fields}"}
        
        # Validate price levels
        price = float(signal_data["price"])
        sl = float(signal_data["sl"])
        tp1 = float(signal_data["tp1"])
        action = signal_data["action"].lower()
        
        price_validation_ok = True
        price_validation_reason = ""
        
        if action in ["buy", "long", "emergency_buy"]:
            if sl >= price:
                price_validation_ok = False
                price_validation_reason = "Stop loss must be below entry price for long positions"
            elif tp1 <= price:
                price_validation_ok = False
                price_validation_reason = "Take profit must be above entry price for long positions"
        elif action in ["sell", "short", "emergency_sell"]:
            if sl <= price:
                price_validation_ok = False
                price_validation_reason = "Stop loss must be above entry price for short positions"
            elif tp1 >= price:
                price_validation_ok = False
                price_validation_reason = "Take profit must be below entry price for short positions"
        
        price_factor = DecisionFactor(
            name="price_levels_validation",
            value=price_validation_ok,
            weight=0.8,
            impact=1.0 if price_validation_ok else -1.0,
            explanation="Price levels validation" if price_validation_ok else price_validation_reason,
            source="validation",
            confidence=1.0
        )
        factors_evaluated.append(price_factor)
        all_factors.append(price_factor)
        
        # Validate tier
        tier = signal_data.get("tier", "Standard")
        valid_tiers = ["Emergency", "Platinum", "Premium", "Standard", "Quick"]
        tier_valid = tier in valid_tiers
        
        tier_factor = DecisionFactor(
            name="tier_validation",
            value=tier,
            weight=0.3,
            impact=1.0 if tier_valid else -0.5,
            explanation=f"Tier '{tier}' is valid" if tier_valid else f"Invalid tier '{tier}'",
            source="validation",
            confidence=1.0
        )
        factors_evaluated.append(tier_factor)
        all_factors.append(tier_factor)
        
        # Validate signal strength
        strength = signal_data.get("strength", 0.0)
        min_strength = self.decision_rules["signal_acceptance"]["min_strength"].get(tier, 0.5)
        strength_ok = strength >= min_strength
        
        strength_factor = DecisionFactor(
            name="signal_strength",
            value=strength,
            weight=self.factor_weights["signal_strength"],
            impact=min(strength * 2 - 1, 1.0),  # Scale to -1 to 1
            explanation=f"Signal strength {strength:.3f} {'meets' if strength_ok else 'below'} minimum {min_strength:.3f} for {tier}",
            source="signal",
            confidence=0.8
        )
        factors_evaluated.append(strength_factor)
        all_factors.append(strength_factor)
        
        execution_time = int((time.time() - step_start) * 1000)
        
        # Overall validation result
        validation_ok = price_validation_ok and tier_valid and strength_ok
        
        decision_path.append(DecisionPath(
            step_number=1,
            step_name="Basic Signal Validation",
            input_data={"signal_data": signal_data},
            factors_evaluated=factors_evaluated,
            intermediate_result={
                "valid": validation_ok,
                "price_validation": price_validation_ok,
                "tier_validation": tier_valid,
                "strength_validation": strength_ok
            },
            reasoning="Basic signal validation completed" if validation_ok else "Signal validation failed",
            execution_time_ms=execution_time
        ))
        
        return {
            "valid": validation_ok,
            "reason": price_validation_reason if not price_validation_ok else "Validation passed"
        }
    
    async def _evaluate_ml_prediction(
        self,
        signal_data: Dict[str, Any],
        context: Dict[str, Any],
        decision_path: List[DecisionPath],
        all_factors: List[DecisionFactor]
    ) -> Dict[str, Any]:
        """Ocena predykcji ML"""
        step_start = time.time()
        factors_evaluated = []
        
        try:
            # Import ML predictor
            from ml_predictor import MLPredictor
            predictor = MLPredictor()
            
            # Get ML prediction
            prediction = await predictor.predict(signal_data)
            
            if prediction:
                # Win probability factor
                win_prob = prediction.get("win_probability", 0.5)
                win_prob_factor = DecisionFactor(
                    name="ml_win_probability",
                    value=win_prob,
                    weight=self.factor_weights["ml_prediction"] * 0.4,
                    impact=(win_prob - 0.5) * 2,  # Scale to -1 to 1
                    explanation=f"ML predicts {win_prob:.1%} win probability",
                    source="ml_model",
                    confidence=prediction.get("confidence", 0.5)
                )
                factors_evaluated.append(win_prob_factor)
                all_factors.append(win_prob_factor)
                
                # Expected value factor
                expected_value = prediction.get("expected_value", 0.0)
                ev_factor = DecisionFactor(
                    name="ml_expected_value",
                    value=expected_value,
                    weight=self.factor_weights["ml_prediction"] * 0.4,
                    impact=min(max(expected_value / 2, -1.0), 1.0),  # Scale to -1 to 1
                    explanation=f"ML expected value: {expected_value:.3f}",
                    source="ml_model",
                    confidence=prediction.get("confidence", 0.5)
                )
                factors_evaluated.append(ev_factor)
                all_factors.append(ev_factor)
                
                # Model confidence factor
                model_confidence = prediction.get("confidence", 0.5)
                confidence_factor = DecisionFactor(
                    name="ml_model_confidence",
                    value=model_confidence,
                    weight=self.factor_weights["ml_prediction"] * 0.2,
                    impact=(model_confidence - 0.5) * 2,
                    explanation=f"ML model confidence: {model_confidence:.3f}",
                    source="ml_model",
                    confidence=1.0
                )
                factors_evaluated.append(confidence_factor)
                all_factors.append(confidence_factor)
                
                ml_result = {
                    "available": True,
                    "prediction": prediction,
                    "win_probability": win_prob,
                    "expected_value": expected_value,
                    "confidence": model_confidence
                }
                
            else:
                # ML not available
                ml_unavailable_factor = DecisionFactor(
                    name="ml_availability",
                    value=False,
                    weight=0.1,
                    impact=-0.2,
                    explanation="ML prediction not available",
                    source="ml_model",
                    confidence=1.0
                )
                factors_evaluated.append(ml_unavailable_factor)
                all_factors.append(ml_unavailable_factor)
                
                ml_result = {"available": False, "reason": "ML predictor unavailable"}
        
        except Exception as e:
            # ML error
            ml_error_factor = DecisionFactor(
                name="ml_error",
                value=str(e),
                weight=0.1,
                impact=-0.3,
                explanation=f"ML prediction error: {str(e)}",
                source="ml_model",
                confidence=1.0
            )
            factors_evaluated.append(ml_error_factor)
            all_factors.append(ml_error_factor)
            
            ml_result = {"available": False, "error": str(e)}
        
        execution_time = int((time.time() - step_start) * 1000)
        
        decision_path.append(DecisionPath(
            step_number=2,
            step_name="ML Prediction Analysis",
            input_data={"signal_data": signal_data},
            factors_evaluated=factors_evaluated,
            intermediate_result=ml_result,
            reasoning="ML prediction analysis completed",
            execution_time_ms=execution_time
        ))
        
        return ml_result
    
    async def _evaluate_fake_breakout_risk(
        self,
        signal_data: Dict[str, Any],
        context: Dict[str, Any],
        decision_path: List[DecisionPath],
        all_factors: List[DecisionFactor]
    ) -> Dict[str, Any]:
        """Ocena ryzyka fake breakout"""
        step_start = time.time()
        factors_evaluated = []
        
        # Check v9.1 fake breakout data
        v91_data = signal_data.get("v91_enhancements", {})
        fake_breakout_data = v91_data.get("fake_breakout", {}) if isinstance(v91_data, dict) else {}
        
        if isinstance(fake_breakout_data, dict):
            # Fake breakout detection
            fake_detected = fake_breakout_data.get("detected", False)
            penalty_multiplier = fake_breakout_data.get("penalty_multiplier", 1.0)
            emergency_bypass = fake_breakout_data.get("emergency_bypass", False)
            
            fake_breakout_factor = DecisionFactor(
                name="fake_breakout_detected",
                value=fake_detected,
                weight=self.factor_weights["fake_breakout_risk"],
                impact=-0.8 if fake_detected and not emergency_bypass else 0.0,
                explanation=f"Fake breakout {'detected' if fake_detected else 'not detected'}" + 
                           (f" (emergency bypass active)" if emergency_bypass else ""),
                source="pine_script_v91",
                confidence=0.8
            )
            factors_evaluated.append(fake_breakout_factor)
            all_factors.append(fake_breakout_factor)
            
            # Penalty multiplier
            if penalty_multiplier != 1.0:
                penalty_factor = DecisionFactor(
                    name="fake_breakout_penalty",
                    value=penalty_multiplier,
                    weight=0.1,
                    impact=-(penalty_multiplier - 1.0),  # Higher penalty = negative impact
                    explanation=f"Fake breakout penalty multiplier: {penalty_multiplier:.2f}",
                    source="pine_script_v91",
                    confidence=0.7
                )
                factors_evaluated.append(penalty_factor)
                all_factors.append(penalty_factor)
            
            fake_breakout_result = {
                "detected": fake_detected,
                "penalty_multiplier": penalty_multiplier,
                "emergency_bypass": emergency_bypass,
                "risk_level": "high" if fake_detected and not emergency_bypass else "low"
            }
        else:
            # No fake breakout data available
            no_data_factor = DecisionFactor(
                name="fake_breakout_data_availability",
                value=False,
                weight=0.05,
                impact=-0.1,
                explanation="No fake breakout detection data available",
                source="pine_script_v91",
                confidence=1.0
            )
            factors_evaluated.append(no_data_factor)
            all_factors.append(no_data_factor)
            
            fake_breakout_result = {"available": False, "risk_level": "unknown"}
        
        # Additional fake breakout heuristics
        strength = signal_data.get("strength", 0.0)
        if strength < 0.3:  # Low strength signals more likely to be fake breakouts
            low_strength_factor = DecisionFactor(
                name="low_strength_fake_risk",
                value=strength,
                weight=0.05,
                impact=-0.3,
                explanation=f"Low signal strength ({strength:.3f}) increases fake breakout risk",
                source="heuristic",
                confidence=0.6
            )
            factors_evaluated.append(low_strength_factor)
            all_factors.append(low_strength_factor)
        
        execution_time = int((time.time() - step_start) * 1000)
        
        decision_path.append(DecisionPath(
            step_number=3,
            step_name="Fake Breakout Risk Assessment",
            input_data={"v91_data": v91_data, "signal_strength": strength},
            factors_evaluated=factors_evaluated,
            intermediate_result=fake_breakout_result,
            reasoning="Fake breakout risk assessment completed",
            execution_time_ms=execution_time
        ))
        
        return fake_breakout_result
    
    async def _evaluate_risk_management(
        self,
        signal_data: Dict[str, Any],
        context: Dict[str, Any],
        decision_path: List[DecisionPath],
        all_factors: List[DecisionFactor]
    ) -> Dict[str, Any]:
        """Ocena zarządzania ryzykiem"""
        step_start = time.time()
        factors_evaluated = []
        
        # Calculate position risk
        price = float(signal_data["price"])
        sl = float(signal_data["sl"])
        risk_per_share = abs(price - sl)
        risk_percentage = risk_per_share / price
        
        position_risk_factor = DecisionFactor(
            name="position_risk_percentage",
            value=risk_percentage,
            weight=self.factor_weights["position_risk"],
            impact=max(-risk_percentage * 10, -1.0),  # Higher risk = more negative impact
            explanation=f"Position risk: {risk_percentage:.1%} per share",
            source="calculation",
            confidence=1.0
        )
        factors_evaluated.append(position_risk_factor)
        all_factors.append(position_risk_factor)
        
        # Portfolio risk assessment
        open_positions = context.get("open_positions", [])
        current_portfolio_risk = sum(abs(pos.get("pnl", 0)) for pos in open_positions)
        
        # Estimate new position size (simplified)
        account_balance = getattr(Config, "ACCOUNT_BALANCE", 10000)  # Default
        base_risk = self.decision_rules["position_sizing"]["base_risk_per_trade"]
        tier = signal_data.get("tier", "Standard")
        tier_multiplier = self.decision_rules["position_sizing"]["tier_multipliers"].get(tier, 1.0)
        
        position_size_usd = account_balance * base_risk * tier_multiplier
        new_position_risk = position_size_usd * risk_percentage
        
        portfolio_risk_factor = DecisionFactor(
            name="portfolio_risk_impact",
            value=new_position_risk,
            weight=self.factor_weights["portfolio_risk"],
            impact=max(-new_position_risk / 1000, -1.0),  # Scale impact
            explanation=f"New position would add ${new_position_risk:.2f} risk to portfolio",
            source="calculation",
            confidence=0.8
        )
        factors_evaluated.append(portfolio_risk_factor)
        all_factors.append(portfolio_risk_factor)
        
        # Check maximum risk limits
        max_total_risk = self.decision_rules["risk_management"]["max_total_risk"] * account_balance
        projected_total_risk = current_portfolio_risk + new_position_risk
        
        risk_limit_ok = projected_total_risk <= max_total_risk
        
        risk_limit_factor = DecisionFactor(
            name="risk_limit_compliance",
            value=risk_limit_ok,
            weight=0.3,
            impact=1.0 if risk_limit_ok else -1.0,
            explanation=f"Risk limit {'compliant' if risk_limit_ok else 'exceeded'}: ${projected_total_risk:.2f} / ${max_total_risk:.2f}",
            source="risk_management",
            confidence=1.0
        )
        factors_evaluated.append(risk_limit_factor)
        all_factors.append(risk_limit_factor)
        
        # Symbol correlation risk (simplified)
        symbol = signal_data.get("symbol", "")
        correlated_positions = [
            pos for pos in open_positions 
            if pos.get("symbol", "").startswith(symbol[:3])  # Same base currency
        ]
        
        correlation_risk = len(correlated_positions) * 0.1  # Simplified correlation risk
        
        correlation_factor = DecisionFactor(
            name="correlation_risk",
            value=correlation_risk,
            weight=self.factor_weights["correlation_risk"],
            impact=max(-correlation_risk * 2, -1.0),
            explanation=f"Correlation risk: {len(correlated_positions)} related positions",
            source="calculation",
            confidence=0.6
        )
        factors_evaluated.append(correlation_factor)
        all_factors.append(correlation_factor)
        
        execution_time = int((time.time() - step_start) * 1000)
        
        risk_result = {
            "acceptable": risk_limit_ok,
            "position_risk_pct": risk_percentage,
            "new_position_risk_usd": new_position_risk,
            "projected_total_risk": projected_total_risk,
            "max_allowed_risk": max_total_risk,
            "correlation_risk": correlation_risk,
            "reason": "Risk limits exceeded" if not risk_limit_ok else "Risk acceptable"
        }
        
        decision_path.append(DecisionPath(
            step_number=4,
            step_name="Risk Management Assessment",
            input_data={
                "position_risk": risk_percentage,
                "portfolio_risk": current_portfolio_risk,
                "open_positions": len(open_positions)
            },
            factors_evaluated=factors_evaluated,
            intermediate_result=risk_result,
            reasoning="Risk management assessment completed",
            execution_time_ms=execution_time
        ))
        
        return risk_result
    
    async def _evaluate_market_conditions(
        self,
        signal_data: Dict[str, Any],
        context: Dict[str, Any],
        decision_path: List[DecisionPath],
        all_factors: List[DecisionFactor]
    ) -> Dict[str, Any]:
        """Ocena warunków rynkowych"""
        step_start = time.time()
        factors_evaluated = []
        
        # Market session analysis
        market_session = context.get("market_session", "unknown")
        session_quality = {
            "london": 0.8,
            "new_york": 0.9,
            "asian": 0.6,
            "overlap": 1.0,
            "unknown": 0.5
        }
        
        session_score = session_quality.get(market_session.lower(), 0.5)
        session_factor = DecisionFactor(
            name="market_session_quality",
            value=market_session,
            weight=self.factor_weights["time_of_day"],
            impact=(session_score - 0.5) * 2,
            explanation=f"Market session: {market_session} (quality: {session_score:.1f})",
            source="market_analysis",
            confidence=0.7
        )
        factors_evaluated.append(session_factor)
        all_factors.append(session_factor)
        
        # Recent performance impact
        recent_perf = context.get("recent_performance", {})
        if recent_perf:
            win_rate_24h = recent_perf.get("win_rate_24h", 0.5)
            pnl_24h = recent_perf.get("pnl_24h", 0.0)
            
            performance_factor = DecisionFactor(
                name="recent_performance",
                value={"win_rate": win_rate_24h, "pnl": pnl_24h},
                weight=0.1,
                impact=min(max((win_rate_24h - 0.5) * 2 + pnl_24h / 1000, -1.0), 1.0),
                explanation=f"Recent performance: {win_rate_24h:.1%} win rate, ${pnl_24h:.2f} PnL",
                source="performance_tracking",
                confidence=0.6
            )
            factors_evaluated.append(performance_factor)
            all_factors.append(performance_factor)
        
        # System mode consideration
        system_mode = context.get("system_mode", "NORMAL")
        mode_impact = {
            "CONSERVATIVE": -0.2,
            "NORMAL": 0.0,
            "AGGRESSIVE": 0.2,
            "EMERGENCY": -0.5
        }
        
        mode_factor = DecisionFactor(
            name="system_mode",
            value=system_mode,
            weight=0.15,
            impact=mode_impact.get(system_mode, 0.0),
            explanation=f"System mode: {system_mode}",
            source="system_config",
            confidence=1.0
        )
        factors_evaluated.append(mode_factor)
        all_factors.append(mode_factor)
        
        # Volume context (if available)
        v91_data = signal_data.get("v91_enhancements", {})
        if isinstance(v91_data, dict):
            volume_context = v91_data.get("volume_context", {})
            if isinstance(volume_context, dict):
                institutional_flow = volume_context.get("institutional_flow")
                if institutional_flow is not None:
                    volume_factor = DecisionFactor(
                        name="institutional_volume_flow",
                        value=institutional_flow,
                        weight=self.factor_weights["volume_confirmation"],
                        impact=(institutional_flow - 0.5) * 2,
                        explanation=f"Institutional flow: {institutional_flow:.3f}",
                        source="pine_script_v91",
                        confidence=0.7
                    )
                    factors_evaluated.append(volume_factor)
                    all_factors.append(volume_factor)
        
        execution_time = int((time.time() - step_start) * 1000)
        
        market_result = {
            "session": market_session,
            "session_quality": session_score,
            "system_mode": system_mode,
            "recent_performance": recent_perf,
            "overall_conditions": "favorable" if session_score > 0.6 else "neutral"
        }
        
        decision_path.append(DecisionPath(
            step_number=5,
            step_name="Market Conditions Analysis",
            input_data={
                "market_session": market_session,
                "system_mode": system_mode,
                "recent_performance": recent_perf
            },
            factors_evaluated=factors_evaluated,
            intermediate_result=market_result,
            reasoning="Market conditions analysis completed",
            execution_time_ms=execution_time
        ))
        
        return market_result
    
    async def _evaluate_portfolio_impact(
        self,
        signal_data: Dict[str, Any],
        context: Dict[str, Any],
        decision_path: List[DecisionPath],
        all_factors: List[DecisionFactor]
    ) -> Dict[str, Any]:
        """Ocena wpływu na portfel"""
        step_start = time.time()
        factors_evaluated = []
        
        open_positions = context.get("open_positions", [])
        symbol = signal_data.get("symbol", "")
        
        # Portfolio diversification
        unique_symbols = set(pos.get("symbol", "") for pos in open_positions)
        diversification_score = min(len(unique_symbols) / 10, 1.0)  # Max score at 10+ symbols
        
        diversification_factor = DecisionFactor(
            name="portfolio_diversification",
            value=len(unique_symbols),
            weight=0.1,
            impact=(diversification_score - 0.5) * 2,
            explanation=f"Portfolio diversification: {len(unique_symbols)} unique symbols",
            source="portfolio_analysis",
            confidence=0.8
        )
        factors_evaluated.append(diversification_factor)
        all_factors.append(diversification_factor)
        
        # Position concentration
        total_positions = len(open_positions)
        concentration_risk = 1.0 if total_positions > 15 else total_positions / 15
        
        concentration_factor = DecisionFactor(
            name="position_concentration",
            value=total_positions,
            weight=0.1,
            impact=-concentration_risk * 0.5,
            explanation=f"Position concentration: {total_positions} open positions",
            source="portfolio_analysis",
            confidence=0.8
        )
        factors_evaluated.append(concentration_factor)
        all_factors.append(concentration_factor)
        
        # Symbol-specific analysis
        symbol_history = context.get("symbol_history", {})
        if symbol_history:
            recent_trades = symbol_history.get("trades_7d", 0)
            avg_pnl = symbol_history.get("avg_pnl", 0.0)
            last_trade_hours = symbol_history.get("last_trade_hours_ago")
            
            # Recent symbol performance
            symbol_perf_factor = DecisionFactor(
                name="symbol_recent_performance",
                value=avg_pnl,
                weight=0.1,
                impact=min(max(avg_pnl / 100, -1.0), 1.0),  # Scale to -1 to 1
                explanation=f"Symbol avg PnL (7d): ${avg_pnl:.2f} from {recent_trades} trades",
                source="historical_analysis",
                confidence=0.7 if recent_trades > 0 else 0.3
            )
            factors_evaluated.append(symbol_perf_factor)
            all_factors.append(symbol_perf_factor)
            
            # Overtrading check
            if last_trade_hours is not None and last_trade_hours < 4:  # Less than 4 hours ago
                overtrading_factor = DecisionFactor(
                    name="symbol_overtrading_risk",
                    value=last_trade_hours,
                    weight=0.1,
                    impact=-0.3,
                    explanation=f"Recent trade on {symbol} only {last_trade_hours:.1f} hours ago",
                    source="overtrading_protection",
                    confidence=0.8
                )
                factors_evaluated.append(overtrading_factor)
                all_factors.append(overtrading_factor)
        
        # Portfolio balance
        long_positions = len([pos for pos in open_positions if pos.get("side") == "long"])
        short_positions = len([pos for pos in open_positions if pos.get("side") == "short"])
        
        if total_positions > 0:
            balance_ratio = abs(long_positions - short_positions) / total_positions
            balance_factor = DecisionFactor(
                name="portfolio_balance",
                value={"long": long_positions, "short": short_positions},
                weight=0.05,
                impact=-balance_ratio * 0.5,  # Penalty for imbalance
                explanation=f"Portfolio balance: {long_positions} long, {short_positions} short",
                source="portfolio_analysis",
                confidence=0.6
            )
            factors_evaluated.append(balance_factor)
            all_factors.append(balance_factor)
        
        execution_time = int((time.time() - step_start) * 1000)
        
        portfolio_result = {
            "diversification_score": diversification_score,
            "concentration_risk": concentration_risk,
            "total_positions": total_positions,
            "symbol_history": symbol_history,
            "portfolio_balance": {"long": long_positions, "short": short_positions}
        }
        
        decision_path.append(DecisionPath(
            step_number=6,
            step_name="Portfolio Impact Assessment",
            input_data={
                "open_positions": total_positions,
                "symbol": symbol,
                "symbol_history": symbol_history
            },
            factors_evaluated=factors_evaluated,
            intermediate_result=portfolio_result,
            reasoning="Portfolio impact assessment completed",
            execution_time_ms=execution_time
        ))
        
        return portfolio_result
    
    async def _calculate_final_decision(
        self,
        signal_data: Dict[str, Any],
        context: Dict[str, Any],
        decision_path: List[DecisionPath],
        all_factors: List[DecisionFactor]
    ) -> Dict[str, Any]:
        """Oblicz ostateczną decyzję"""
        step_start = time.time()
        
        # Calculate weighted score
        total_weight = sum(factor.weight for factor in all_factors)
        if total_weight == 0:
            weighted_score = 0.0
        else:
            weighted_score = sum(factor.weight * factor.impact for factor in all_factors) / total_weight
        
        # Normalize to 0-1 range
        confidence_score = max(0.0, min(1.0, (weighted_score + 1.0) / 2.0))
        
        # Apply tier-specific adjustments
        tier = signal_data.get("tier", "Standard")
        tier_adjustments = {
            "Emergency": 0.1,    # Boost emergency signals
            "Platinum": 0.05,    # Slight boost for premium tiers
            "Premium": 0.02,
            "Standard": 0.0,
            "Quick": -0.05       # Slight penalty for quick signals
        }
        
        tier_adjustment = tier_adjustments.get(tier, 0.0)
        confidence_score = max(0.0, min(1.0, confidence_score + tier_adjustment))
        
        # Calculate position sizing recommendation
        base_risk = self.decision_rules["position_sizing"]["base_risk_per_trade"]
        tier_multiplier = self.decision_rules["position_sizing"]["tier_multipliers"].get(tier, 1.0)
        confidence_multiplier = confidence_score  # Higher confidence = larger position
        
        recommended_risk = base_risk * tier_multiplier * confidence_multiplier
        max_risk = self.decision_rules["position_sizing"]["max_risk_per_trade"]
        recommended_risk = min(recommended_risk, max_risk)
        
        execution_time = int((time.time() - step_start) * 1000)
        
        final_result = {
            "confidence_score": confidence_score,
            "weighted_score": weighted_score,
            "tier_adjustment": tier_adjustment,
            "recommended_position_risk": recommended_risk,
            "factors_summary": {
                "total_factors": len(all_factors),
                "positive_factors": len([f for f in all_factors if f.impact > 0]),
                "negative_factors": len([f for f in all_factors if f.impact < 0]),
                "neutral_factors": len([f for f in all_factors if f.impact == 0])
            },
            "top_positive_factors": sorted(
                [f for f in all_factors if f.impact > 0],
                key=lambda x: x.weight * x.impact,
                reverse=True
            )[:3],
            "top_negative_factors": sorted(
                [f for f in all_factors if f.impact < 0],
                key=lambda x: abs(x.weight * x.impact),
                reverse=True
            )[:3]
        }
        
        decision_path.append(DecisionPath(
            step_number=7,
            step_name="Final Decision Calculation",
            input_data={
                "total_factors": len(all_factors),
                "tier": tier,
                "tier_adjustment": tier_adjustment
            },
            factors_evaluated=[],
            intermediate_result=final_result,
            reasoning=f"Final confidence score: {confidence_score:.3f} (weighted: {weighted_score:.3f}, tier adj: {tier_adjustment:+.3f})",
            execution_time_ms=execution_time
        ))
        
        return final_result
    
    def _create_rejection_result(
        self,
        decision_id: str,
        decision_type: DecisionType,
        signal_data: Dict[str, Any],
        context: Dict[str, Any],
        decision_path: List[DecisionPath],
        all_factors: List[DecisionFactor],
        reason: str,
        trace_id: str,
        start_time: float
    ) -> DecisionResult:
        """Utwórz wynik odrzucenia"""
        execution_time = int((time.time() - start_time) * 1000)
        
        complete_execution_trace(trace_id, False, execution_time, f"Signal rejected: {reason}")
        
        return DecisionResult(
            decision_id=decision_id,
            decision_type=decision_type,
            outcome=DecisionOutcome.REJECTED,
            confidence=ConfidenceLevel.VERY_LOW,
            confidence_score=0.0,
            input_signal=signal_data,
            context=context,
            decision_path=decision_path,
            all_factors=all_factors,
            final_decision={"rejected": True, "reason": reason},
            reasoning_summary=f"Signal rejected: {reason}",
            recommendations=["Review signal quality", "Check validation rules"],
            timestamp=datetime.utcnow(),
            execution_time_ms=execution_time,
            model_versions=self._get_model_versions(),
            risk_score=1.0,
            risk_factors=[reason]
        )
    
    def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Konwertuj score na poziom pewności"""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_risk_assessment(self, factors: List[DecisionFactor]) -> Tuple[float, List[str]]:
        """Oblicz ocenę ryzyka"""
        risk_factors = []
        risk_score = 0.0
        
        for factor in factors:
            if factor.impact < -0.5:  # High negative impact
                risk_factors.append(f"{factor.name}: {factor.explanation}")
                risk_score += abs(factor.impact) * factor.weight
        
        # Normalize risk score
        total_weight = sum(f.weight for f in factors if f.impact < -0.5)
        if total_weight > 0:
            risk_score = min(risk_score / total_weight, 1.0)
        
        return risk_score, risk_factors
    
    def _generate_decision_recommendations(
        self,
        factors: List[DecisionFactor],
        outcome: DecisionOutcome,
        confidence_score: float
    ) -> List[str]:
        """Generuj rekomendacje"""
        recommendations = []
        
        if outcome == DecisionOutcome.REJECTED:
            # Find main rejection reasons
            negative_factors = [f for f in factors if f.impact < -0.5]
            for factor in negative_factors[:3]:  # Top 3 issues
                if "fake_breakout" in factor.name:
                    recommendations.append("Wait for stronger confirmation to avoid fake breakouts")
                elif "risk" in factor.name:
                    recommendations.append("Reduce position size or wait for better risk/reward")
                elif "strength" in factor.name:
                    recommendations.append("Wait for higher signal strength")
                elif "ml" in factor.name:
                    recommendations.append("ML model suggests unfavorable conditions")
        
        elif outcome == DecisionOutcome.ACCEPTED:
            if confidence_score < 0.7:
                recommendations.append("Consider reduced position size due to moderate confidence")
            
            # Check for specific risks
            risk_factors = [f for f in factors if f.impact < -0.2 and f.impact > -0.5]
            if risk_factors:
                recommendations.append("Monitor position closely due to identified risk factors")
            
            # Positive reinforcement
            positive_factors = [f for f in factors if f.impact > 0.5]
            if len(positive_factors) >= 3:
                recommendations.append("Strong signal with multiple confirming factors")
        
        # General recommendations
        if confidence_score > 0.8:
            recommendations.append("High confidence signal - consider standard position sizing")
        elif confidence_score < 0.6:
            recommendations.append("Lower confidence - consider paper trading or reduced size")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Pobierz wersje modeli"""
        versions = {
            "decision_engine": "9.1",
            "pine_script": "9.1"
        }
        
        try:
            from ml_predictor import MLPredictor
            predictor = MLPredictor()
            versions["ml_predictor"] = getattr(predictor, "model_version", "unknown")
        except:
            versions["ml_predictor"] = "unavailable"
        
        return versions
    
    def _get_market_session(self) -> str:
        """Określ sesję rynkową"""
        now = datetime.utcnow()
        hour = now.hour
        
        # Simplified session detection (UTC)
        if 8 <= hour < 16:
            return "london"
        elif 13 <= hour < 21:
            return "new_york"
        elif 13 <= hour < 16:  # Overlap
            return "overlap"
        elif 22 <= hour or hour < 8:
            return "asian"
        else:
            return "unknown"
    
    async def _save_decision_to_database(self, decision: DecisionResult) -> None:
        """Zapisz decyzję do bazy danych"""
        try:
            # For now, we'll use the execution trace system
            # In a full implementation, you'd create a dedicated Decision table
            log_execution_trace(
                f"decision_{decision.decision_type.value}",
                {
                    "decision_id": decision.decision_id,
                    "outcome": decision.outcome.value,
                    "confidence": decision.confidence_score,
                    "factors_count": len(decision.all_factors),
                    "execution_time_ms": decision.execution_time_ms
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to save decision to database: {e}")
    
    async def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Pobierz historię decyzji"""
        recent_decisions = self.decision_history[-limit:] if self.decision_history else []
        
        return [
            {
                "decision_id": d.decision_id,
                "timestamp": d.timestamp.isoformat(),
                "decision_type": d.decision_type.value,
                "outcome": d.outcome.value,
                "confidence": d.confidence.value,
                "confidence_score": d.confidence_score,
                "symbol": d.input_signal.get("symbol", "unknown"),
                "execution_time_ms": d.execution_time_ms,
                "factors_count": len(d.all_factors),
                "risk_score": d.risk_score
            }
            for d in recent_decisions
        ]
    
    async def get_decision_analytics(self) -> Dict[str, Any]:
        """Pobierz analitykę decyzji"""
        if not self.decision_history:
            return {"message": "No decision history available"}
        
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        
        # Basic stats
        total_decisions = len(recent_decisions)
        accepted_decisions = len([d for d in recent_decisions if d.outcome == DecisionOutcome.ACCEPTED])
        rejected_decisions = len([d for d in recent_decisions if d.outcome == DecisionOutcome.REJECTED])
        
        # Confidence distribution
        confidence_scores = [d.confidence_score for d in recent_decisions]
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        # Performance by tier
        tier_stats = {}
        for decision in recent_decisions:
            tier = decision.input_signal.get("tier", "Unknown")
            if tier not in tier_stats:
                tier_stats[tier] = {"total": 0, "accepted": 0, "avg_confidence": []}
            
            tier_stats[tier]["total"] += 1
            if decision.outcome == DecisionOutcome.ACCEPTED:
                tier_stats[tier]["accepted"] += 1
            tier_stats[tier]["avg_confidence"].append(decision.confidence_score)
        
        # Calculate tier acceptance rates
        for tier in tier_stats:
            stats = tier_stats[tier]
            stats["acceptance_rate"] = stats["accepted"] / stats["total"] if stats["total"] > 0 else 0
            stats["avg_confidence"] = statistics.mean(stats["avg_confidence"]) if stats["avg_confidence"] else 0
        
        # Execution time stats
        execution_times = [d.execution_time_ms for d in recent_decisions]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        
        return {
            "summary": {
                "total_decisions": total_decisions,
                "accepted_decisions": accepted_decisions,
                "rejected_decisions": rejected_decisions,
                "acceptance_rate": accepted_decisions / total_decisions if total_decisions > 0 else 0,
                "avg_confidence": avg_confidence,
                "avg_execution_time_ms": avg_execution_time
            },
            "tier_performance": tier_stats,
            "confidence_distribution": {
                "very_high": len([d for d in recent_decisions if d.confidence == ConfidenceLevel.VERY_HIGH]),
                "high": len([d for d in recent_decisions if d.confidence == ConfidenceLevel.HIGH]),
                "medium": len([d for d in recent_decisions if d.confidence == ConfidenceLevel.MEDIUM]),
                "low": len([d for d in recent_decisions if d.confidence == ConfidenceLevel.LOW]),
                "very_low": len([d for d in recent_decisions if d.confidence == ConfidenceLevel.VERY_LOW])
            },
            "recent_trends": {
                "last_10_acceptance_rate": len([d for d in recent_decisions[-10:] if d.outcome == DecisionOutcome.ACCEPTED]) / min(10, len(recent_decisions)),
                "confidence_trend": "improving" if len(confidence_scores) > 10 and statistics.mean(confidence_scores[-10:]) > statistics.mean(confidence_scores[-20:-10]) else "stable"
            }
        }

# --- ALIAS DODANY DLA KOMPATYBILNOŚCI Z DISCORD_CLIENT ---

    async def get_recent_decision_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Alias dla get_decision_history() dla komendy /decision_trace."""
        self.logger.info("Wywołano alias: get_recent_decision_traces -> get_decision_history")
        return await self.get_decision_history(limit=limit)

# Global decision engine instance
decision_engine = DecisionEngine()


# Convenience functions
async def make_signal_decision(signal_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> DecisionResult:
    """
    Make a trading decision based on signal data with full explainability.
    """
    try:
        # Initialize decision engine
        engine = DecisionEngine()
        
        # Process the signal
        decision = await engine.process_signal(signal_data, context or {})
        
        # Log the decision for diagnostics
        from database import log_execution_trace
        await log_execution_trace(
            component="decision_engine",
            operation="make_signal_decision",
            input_data=signal_data,
            output_data=decision.to_dict(),
            execution_time=0.0,  # You can measure this if needed
            success=True
        )
        
        return decision
        
    except Exception as e:
        # Log error for diagnostics
        from database import log_execution_trace
        await log_execution_trace(
            component="decision_engine",
            operation="make_signal_decision",
            input_data=signal_data,
            output_data={"error": str(e)},
            execution_time=0.0,
            success=False
        )
        
        # Return safe default decision
        return DecisionResult(
            decision="hold",
            confidence=0.0,
            reasoning=f"Error in decision making: {str(e)}",
            factors={},
            risk_assessment={"error": True},
            metadata={"error": str(e)}
        )
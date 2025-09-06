import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import desc

from database import Session, Trade
from database import (
    log_execution_trace, complete_execution_trace,
    MLPrediction, MLModelMetrics, log_ml_prediction
)
logger = logging.getLogger("ml_predictor")


class TradingMLPredictor:
    """
    ML Predictor dla systemu tradingowego.
    Przewiduje: P(win), PnL%, Expected Value
    """

    def __init__(self):
        self.win_prob_model = None
        self.pnl_model = None
        self.feature_columns = None
        self.model_version = "v1.0"
        self.min_trades_for_training = 100  # Minimum dla pierwszego modelu

        # Spróbuj załadować istniejące modele
        self.load_models()
        # v9.1 HYBRID ULTRA-DIAGNOSTICS
        self.diagnostics_enabled = True
        self.prediction_history = []
        self.model_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "last_updated": None
        }
        self.feature_importance_cache = {}
        self.prediction_count = 0
        self.successful_predictions = 0

    async def initialize(self):
        """Inicjalizuje ML predictor - wymagane dla kompatybilności."""
        try:
            # Sprawdź czy modele są załadowane
            if not self.win_prob_model or not self.pnl_model:
                logger.info("Brak modeli ML - spróbuję wytrenować nowe")
                success = self.train_models()
                if not success:
                    logger.warning("Nie udało się wytrenować modeli ML")
            else:
                logger.info("Modele ML załadowane pomyślnie")
                
        except Exception as e:
            logger.error(f"Błąd inicjalizacji ML predictor: {e}")

    def extract_features(self, signal_data: dict[str, Any]) -> dict[str, float]:
        """
        Wyciąga cechy z sygnału Pine Script v7.7 dla ML.
        """

        features = {
            # Signal strength & tier
            "signal_strength": float(signal_data.get("strength", 0)),
            "tier_premium": 1.0 if signal_data.get("tier") == "Premium" else 0.0,
            "tier_standard": 1.0 if signal_data.get("tier") == "Standard" else 0.0,
            "tier_quick": 1.0 if signal_data.get("tier") == "Quick" else 0.0,
            "position_multiplier": float(
                signal_data.get("position_size_multiplier", 1.0)
            ),
            # Risk & confidence
            "confidence_penalty": float(signal_data.get("confidence_penalty", 0.0)),
            "leverage": float(signal_data.get("leverage", 10)),
            # Market regime
            "regime_fear": 1.0 if signal_data.get("market_regime") == "FEAR" else 0.0,
            "regime_neutral": (
                1.0 if signal_data.get("market_regime") == "NEUTRAL" else 0.0
            ),
            "regime_greed": 1.0 if signal_data.get("market_regime") == "GREED" else 0.0,
            # Market condition
            "condition_normal": (
                1.0 if signal_data.get("market_condition") == "NORMAL" else 0.0
            ),
            "condition_high_vol": (
                1.0 if signal_data.get("market_condition") == "HIGH_VOLATILITY" else 0.0
            ),
            "condition_extreme": (
                1.0
                if signal_data.get("market_condition")
                in ["EXTREME_MOVE", "MARKET_STRESS"]
                else 0.0
            ),
            # Technical indicators
            "mfi": float(signal_data.get("mfi", 50)) / 100.0,  # Normalize 0-1
            "adx": float(signal_data.get("adx", 0)) / 100.0,
            "btc_correlation": float(signal_data.get("btc_correlation", 0)),
            # Binary flags
            "volume_spike": (
                1.0
                if str(signal_data.get("volume_spike", "false")).lower() == "true"
                else 0.0
            ),
            "near_key_level": (
                1.0
                if str(signal_data.get("near_key_level", "false")).lower() == "true"
                else 0.0
            ),
            "liquidity_sweep": (
                1.0
                if str(signal_data.get("liquidity_sweep", "false")).lower() == "true"
                else 0.0
            ),
            "fresh_bos": (
                1.0
                if str(signal_data.get("fresh_bos", "false")).lower() == "true"
                else 0.0
            ),
            # HTF trend
            "htf_bullish": 1.0 if signal_data.get("htf_trend") == "bullish" else 0.0,
            "htf_bearish": 1.0 if signal_data.get("htf_trend") == "bearish" else 0.0,
            "htf_neutral": 1.0 if signal_data.get("htf_trend") == "neutral" else 0.0,
            # Session
            "session_london": (
                1.0 if signal_data.get("session") in ["London", "London/NY"] else 0.0
            ),
            "session_ny": (
                1.0 if signal_data.get("session") in ["NY", "London/NY"] else 0.0
            ),
            "session_asia": 1.0 if signal_data.get("session") == "Asia" else 0.0,
            # Pair tier
            "pair_tier": float(signal_data.get("pair_tier", 3)) / 3.0,  # Normalize 0-1
            # Action
            "is_buy": 1.0 if signal_data.get("action") == "buy" else 0.0,
        }

        return features

    def prepare_training_data(
        self,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
        """Przygotowuje dane treningowe z bazy."""
        session = Session()
        try:
            trades = (
                session.query(Trade)
                .filter(
                    Trade.status == "closed",
                    Trade.raw_signal_data.is_not(None),
                    Trade.pnl_usdt.is_not(None),
                    Trade.is_dry_run.is_(False),
                )
                .order_by(desc(Trade.entry_time))
                .limit(1000)
                .all()
            )

            if len(trades) < self.min_trades_for_training:
                logger.warning(
                    f"Za mało danych: {len(trades)} < {self.min_trades_for_training}"
                )
                return None

            features_list = []
            win_labels = []
            pnl_labels = []

            for trade in trades:
                try:
                    if isinstance(trade.raw_signal_data, str):
                        signal_data = json.loads(trade.raw_signal_data)
                    else:
                        signal_data = trade.raw_signal_data or {}

                    features = self.extract_features(signal_data)
                    features_list.append(features)

                    win_labels.append(1.0 if trade.pnl_usdt > 0 else 0.0)
                    pnl_labels.append(trade.pnl_percent or 0.0)

                except Exception as e:
                    logger.warning(f"Błąd trade ID {trade.id}: {e}")
                    continue

            if not features_list:
                return None

            features_df = pd.DataFrame(features_list)
            win_series = pd.Series(win_labels)
            pnl_series = pd.Series(pnl_labels)

            logger.info(
                f"Dane ML: {len(features_df)} próbek, Win rate: {win_series.mean():.2%}"
            )
            return features_df, win_series, pnl_series

        finally:
            session.close()

    def train_models(self) -> bool:
        """Trenuje modele ML."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split

            data = self.prepare_training_data()
            if data is None:
                return False

            X, y_win, y_pnl = data
            self.feature_columns = X.columns.tolist()

            X_train, X_test, y_win_train, y_win_test, y_pnl_train, y_pnl_test = (
                train_test_split(
                    X, y_win, y_pnl, test_size=0.2, random_state=42, stratify=y_win
                )
            )

            # Model prawdopodobieństwa wygranej
            logger.info("Trenuję model P(win)...")
            self.win_prob_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
            )
            self.win_prob_model.fit(X_train, y_win_train)

            y_win_pred = self.win_prob_model.predict(X_test)
            accuracy = accuracy_score(y_win_test, y_win_pred)
            logger.info(f"P(win) Model Accuracy: {accuracy:.3f}")

            # Model PnL%
            logger.info("Trenuję model PnL%...")
            self.pnl_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.pnl_model.fit(X_train, y_pnl_train)

            self.save_models()
            self._log_feature_importance()

            logger.info(f"Modele ML wytrenowane! (v{self.model_version})")
            # v9.1 DIAGNOSTICS: Update model metrics
            self.model_metrics = {
                "accuracy": accuracy,
                "precision": 0.0,  # Calculate if needed
                "recall": 0.0,     # Calculate if needed
                "f1_score": 0.0,   # Calculate if needed
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Update feature importance cache
            if self.win_prob_model and self.feature_columns:
                importances = self.win_prob_model.feature_importances_
                self.feature_importance_cache = dict(zip(self.feature_columns, importances))
            return True

        except Exception as e:
            logger.error(f"Błąd trenowania: {e}", exc_info=True)
            return False

    def predict(self, signal_data: dict[str, Any]) -> dict[str, float]:
        """Główna funkcja predykcji."""
        if not self.win_prob_model or not self.pnl_model:
            return {
                "win_probability": 0.5,
                "predicted_pnl_pct": 0.0,
                "expected_value": 0.0,
                "confidence": 0.0,
            }
        # v9.1 DIAGNOSTICS: Track prediction attempt
        start_time = datetime.utcnow()
        trace_id = None
        if self.diagnostics_enabled:
            trace_id = log_execution_trace("ml_prediction", {
                "symbol": signal_data.get("symbol", "unknown"),
                "tier": signal_data.get("tier", "unknown")
            })
        
        self.prediction_count += 1

        try:
            features = self.extract_features(signal_data)
            features_df = pd.DataFrame([features])

            # Upewnij się o kolejności kolumn
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0.0
            features_df = features_df[self.feature_columns]

            # Predykcje
            win_prob = self.win_prob_model.predict_proba(features_df)[0][1]
            predicted_pnl = self.pnl_model.predict(features_df)[0]

            # Expected Value
            expected_value = win_prob * abs(predicted_pnl) - (1 - win_prob) * abs(
                predicted_pnl * 0.5
            )

            # Confidence (na podstawie entropii)
            prob_entropy = -win_prob * np.log2(win_prob + 1e-10) - (
                1 - win_prob
            ) * np.log2(1 - win_prob + 1e-10)
            confidence = 1.0 - prob_entropy

            return {
                "win_probability": float(win_prob),
                "predicted_pnl_pct": float(predicted_pnl),
                "expected_value": float(expected_value),
                "confidence": float(confidence),
            }

        except Exception as e:
            logger.error(f"Błąd predykcji: {e}", exc_info=True)
            # v9.1 DIAGNOSTICS: Log successful prediction
            self.successful_predictions += 1
            
            result = {
                "win_probability": float(win_prob),
                "predicted_pnl_pct": float(predicted_pnl),
                "expected_value": float(expected_value),
                "confidence": float(confidence),
            }
            
            # Store prediction in history for diagnostics
            prediction_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": signal_data.get("symbol", "unknown"),
                "tier": signal_data.get("tier", "unknown"),
                "win_probability": float(win_prob),
                "expected_value": float(expected_value),
                "confidence": float(confidence)
            }
            self.prediction_history.append(prediction_record)
            
            # Keep only last 100 predictions in memory
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            # Log to database
            if self.diagnostics_enabled:
                try:
                    log_ml_prediction(
                        symbol=signal_data.get("symbol", "unknown"),
                        prediction_data=result,
                        model_version=self.model_version
                    )
                except Exception as e:
                    logger.error(f"Failed to log ML prediction: {e}")
            
            # Complete execution trace
            if trace_id and self.diagnostics_enabled:
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                complete_execution_trace(trace_id, True, duration_ms, "ML prediction completed")
            
            return result

        except Exception as e:
            logger.error(f"Błąd predykcji: {e}", exc_info=True)
            
            # Complete execution trace with error
            if trace_id and self.diagnostics_enabled:
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                complete_execution_trace(trace_id, False, duration_ms, f"Prediction error: {str(e)}")
            
            return {
                "win_probability": 0.5,
                "predicted_pnl_pct": 0.0,
                "expected_value": 0.0,
                "confidence": 0.0,
            }

    def should_take_trade(
        self, predictions: dict[str, float], min_win_prob: float = 0.55
    ) -> tuple[bool, str]:
        """Decyzja ML czy wziąć trade."""
        win_prob = predictions["win_probability"]
        expected_value = predictions["expected_value"]
        confidence = predictions["confidence"]

        if confidence < 0.3:
            return False, f"Niska pewność: {confidence:.2f}"

        if win_prob < min_win_prob:
            return False, f"Za niska P(win): {win_prob:.2%} < {min_win_prob:.2%}"

        if expected_value < 0:
            return False, f"Ujemny EV: {expected_value:.2f}%"

        return True, f"ML OK: P(win)={win_prob:.2%}, EV={expected_value:.2f}%"

    async def get_optimal_thresholds(self) -> dict[str, float] | None:
        """
        Oblicza optymalne progi siły sygnału na podstawie danych ML.
        Zwraca słownik {tier: optimal_threshold} lub None jeśli brak danych.
        """
        try:
            if not self.win_prob_model or not self.feature_columns:
                logger.warning("Brak wytrenowanych modeli ML dla optymalizacji progów")
                return None

            # Przygotuj dane treningowe
            data = self.prepare_training_data()
            if data is None:
                logger.warning("Brak danych treningowych dla optymalizacji progów")
                return None

            X, y_win, y_pnl = data
            
            # Analizuj dla każdego tier
            optimal_thresholds = {}
            tiers = ["Emergency", "Platinum", "Premium", "Standard", "Quick"]
            
            for tier in tiers:
                try:
                    # Filtruj dane dla danego tier
                    tier_mask = self._get_tier_mask(X, tier)
                    if tier_mask.sum() < 20:  # Minimum 20 próbek
                        logger.warning(f"Za mało danych dla tier {tier}: {tier_mask.sum()}")
                        continue
                    
                    X_tier = X[tier_mask]
                    y_win_tier = y_win[tier_mask]
                    
                    # Znajdź optymalny próg siły sygnału
                    optimal_threshold = self._find_optimal_threshold(X_tier, y_win_tier)
                    optimal_thresholds[tier] = optimal_threshold
                    
                    logger.info(f"Optymalny próg dla {tier}: {optimal_threshold:.3f}")
                    
                except Exception as e:
                    logger.error(f"Błąd optymalizacji dla tier {tier}: {e}")
                    continue
            
            if not optimal_thresholds:
                logger.warning("Nie udało się obliczyć żadnych optymalnych progów")
                return None
                
            return optimal_thresholds
            
        except Exception as e:
            logger.error(f"Błąd w get_optimal_thresholds: {e}")
            return None

    def _get_tier_mask(self, X: pd.DataFrame, tier: str) -> pd.Series:
        """Zwraca maskę dla danego tier."""
        if tier == "Emergency":
            # Emergency nie ma dedykowanej kolumny, więc używamy wysokiej siły sygnału
            return X["signal_strength"] > 0.8
        elif tier == "Platinum":
            return X["tier_premium"] == 1.0  # Platinum używa premium w starszych wersjach
        elif tier == "Premium":
            return X["tier_premium"] == 1.0
        elif tier == "Standard":
            return X["tier_standard"] == 1.0
        elif tier == "Quick":
            return X["tier_quick"] == 1.0
        else:
            return pd.Series([False] * len(X))

    def _find_optimal_threshold(self, X_tier: pd.DataFrame, y_win_tier: pd.Series) -> float:
        """
        Znajduje optymalny próg siły sygnału dla danego tier.
        Używa analizy ROC do znalezienia najlepszego balansu precision/recall.
        """
        try:
            signal_strengths = X_tier["signal_strength"].values
            win_rates = y_win_tier.values
            
            # Testuj różne progi od 0.1 do 0.8
            thresholds = np.arange(0.1, 0.81, 0.05)
            best_threshold = 0.4
            best_score = 0.0
            
            for threshold in thresholds:
                # Filtruj sygnały powyżej progu
                above_threshold = signal_strengths >= threshold
                
                if above_threshold.sum() < 5:  # Minimum 5 sygnałów
                    continue
                
                # Oblicz metryki
                win_rate = win_rates[above_threshold].mean()
                signal_count = above_threshold.sum()
                total_signals = len(signal_strengths)
                
                # Score = win_rate * coverage (preferuj wysoką win_rate z rozsądnym coverage)
                coverage = signal_count / total_signals
                score = win_rate * 0.7 + coverage * 0.3
                
                if score > best_score and win_rate > 0.5:  # Minimum 50% win rate
                    best_score = score
                    best_threshold = threshold
            
            # Dodaj margines bezpieczeństwa (obniż próg o 10%)
            best_threshold *= 0.9
            
            return max(0.1, min(0.7, best_threshold))  # Clamp między 0.1-0.7
            
        except Exception as e:
            logger.error(f"Błąd w _find_optimal_threshold: {e}")
            return 0.4  # Fallback

    def save_models(self):
        """Zapisuje modele."""
        models_dir = Path("ml_models")
        models_dir.mkdir(exist_ok=True)

        if self.win_prob_model:
            joblib.dump(self.win_prob_model, models_dir / "win_prob_model.pkl")
        if self.pnl_model:
            joblib.dump(self.pnl_model, models_dir / "pnl_model.pkl")
        if self.feature_columns:
            joblib.dump(self.feature_columns, models_dir / "feature_columns.pkl")

        metadata = {
            "model_version": self.model_version,
            "trained_at": datetime.utcnow().isoformat(),
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
        }
        with open(models_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_models(self):
        """Ładuje modele z plików."""
        models_dir = Path("ml_models")
        try:
            if (models_dir / "win_prob_model.pkl").exists():
                self.win_prob_model = joblib.load(models_dir / "win_prob_model.pkl")
                logger.info("Załadowano win_prob_model")

            if (models_dir / "pnl_model.pkl").exists():
                self.pnl_model = joblib.load(models_dir / "pnl_model.pkl")
                logger.info("Załadowano pnl_model")

            if (models_dir / "feature_columns.pkl").exists():
                self.feature_columns = joblib.load(models_dir / "feature_columns.pkl")
                logger.info(f"Załadowano {len(self.feature_columns)} cech")

        except Exception as e:
            logger.warning(f"Nie udało się załadować modeli: {e}")

    def _log_feature_importance(self):
        """Loguje ważność cech."""
        if self.win_prob_model and self.feature_columns:
            importances = self.win_prob_model.feature_importances_
            feature_importance = list(
                zip(self.feature_columns, importances, strict=False)
            )
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            logger.info("Top 10 najważniejszych cech:")
            for feature, importance in feature_importance[:10]:
                logger.info(f"  {feature}: {importance:.3f}")

    async def get_diagnostics(self) -> dict:
        """Get comprehensive ML diagnostics - v9.1 HYBRID ULTRA-DIAGNOSTICS"""
        try:
            start_time = datetime.utcnow()
            
            # Log execution trace
            trace_id = None
            if self.diagnostics_enabled:
                trace_id = log_execution_trace("ml_diagnostics", {"component": "ml_predictor"})
            
            diagnostics = {
                "model_status": {
                    "win_prob_model_loaded": self.win_prob_model is not None,
                    "pnl_model_loaded": self.pnl_model is not None,
                    "feature_columns_count": len(self.feature_columns) if self.feature_columns else 0,
                    "model_version": self.model_version
                },
                "performance_metrics": self.model_metrics.copy(),
                "prediction_stats": {
                    "total_predictions": self.prediction_count,
                    "successful_predictions": self.successful_predictions,
                    "success_rate": self.successful_predictions / max(1, self.prediction_count)
                },
                "feature_importance": self.feature_importance_cache.copy(),
                "recent_predictions": self.prediction_history[-10:] if self.prediction_history else [],
                "data_quality": await self._assess_data_quality(),
                "model_health": self._assess_model_health(),
                "recommendations": self._generate_recommendations()
            }
            
            # Complete execution trace
            if trace_id and self.diagnostics_enabled:
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                complete_execution_trace(trace_id, True, duration_ms, "ML diagnostics completed")
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error getting ML diagnostics: {e}")
            if trace_id and self.diagnostics_enabled:
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                complete_execution_trace(trace_id, False, duration_ms, f"Error: {str(e)}")
            return {"error": str(e)}

    async def _assess_data_quality(self) -> dict:
        """Assess training data quality"""
        try:
            session = Session()
            
            # Count available trades
            total_trades = session.query(Trade).filter(
                Trade.status == "closed",
                Trade.raw_signal_data.is_not(None),
                Trade.pnl_usdt.is_not(None),
                Trade.is_dry_run.is_(False)
            ).count()
            
            # Count recent trades (last 30 days)
            from datetime import timedelta
            recent_cutoff = datetime.utcnow() - timedelta(days=30)
            recent_trades = session.query(Trade).filter(
                Trade.status == "closed",
                Trade.entry_time >= recent_cutoff,
                Trade.raw_signal_data.is_not(None),
                Trade.pnl_usdt.is_not(None),
                Trade.is_dry_run.is_(False)
            ).count()
            
            session.close()
            
            return {
                "total_trades": total_trades,
                "recent_trades": recent_trades,
                "data_freshness": "good" if recent_trades > 10 else "poor",
                "sufficient_data": total_trades >= self.min_trades_for_training
            }
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {"error": str(e)}

    def _assess_model_health(self) -> dict:
        """Assess model health status"""
        health_score = 0.0
        issues = []
        
        # Check if models are loaded
        if self.win_prob_model and self.pnl_model:
            health_score += 0.3
        else:
            issues.append("Models not loaded")
        
        # Check accuracy
        if self.model_metrics["accuracy"] > 0.6:
            health_score += 0.3
        elif self.model_metrics["accuracy"] > 0.5:
            health_score += 0.15
        else:
            issues.append("Low model accuracy")
        
        # Check prediction success rate
        success_rate = self.successful_predictions / max(1, self.prediction_count)
        if success_rate > 0.8:
            health_score += 0.2
        elif success_rate > 0.6:
            health_score += 0.1
        else:
            issues.append("Low prediction success rate")
        
        # Check data freshness
        if self.model_metrics.get("last_updated"):
            from datetime import timedelta
            last_update = datetime.fromisoformat(self.model_metrics["last_updated"])
            days_old = (datetime.utcnow() - last_update).days
            if days_old < 7:
                health_score += 0.2
            elif days_old < 30:
                health_score += 0.1
            else:
                issues.append("Model needs retraining")
        
        return {
            "health_score": min(1.0, health_score),
            "status": "healthy" if health_score > 0.7 else "warning" if health_score > 0.4 else "critical",
            "issues": issues
        }

    def _generate_recommendations(self) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Model health recommendations
        if not self.win_prob_model or not self.pnl_model:
            recommendations.append("Train ML models - insufficient model coverage")
        
        if self.model_metrics["accuracy"] < 0.6:
            recommendations.append("Retrain models - accuracy below threshold")
        
        # Data recommendations
        if self.prediction_count < 50:
            recommendations.append("Collect more prediction data for better diagnostics")
        
        success_rate = self.successful_predictions / max(1, self.prediction_count)
        if success_rate < 0.7:
            recommendations.append("Review prediction logic - high failure rate")
        
        # Feature recommendations
        if len(self.feature_importance_cache) == 0:
            recommendations.append("Update feature importance analysis")
        
        return recommendations[:5]  # Top 5 recommendations

    async def clear_diagnostic_data(self, days_to_keep: int = 30) -> dict:
        """Clear old diagnostic data - v9.1 DIAGNOSTICS"""
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clear old predictions from history
            self.prediction_history = [
                pred for pred in self.prediction_history 
                if datetime.fromisoformat(pred["timestamp"]) > cutoff_date
            ]
            
            # Clear database diagnostic data
            cleared_count = clear_diagnostic_data(days_to_keep)
            
            return {
                "success": True,
                "cleared_predictions": len(self.prediction_history),
                "cleared_db_records": cleared_count
            }
            
        except Exception as e:
            logger.error(f"Error clearing diagnostic data: {e}")
            return {"success": False, "error": str(e)}
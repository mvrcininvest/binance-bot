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
            "position_multiplier": float(signal_data.get("position_size_multiplier", 1.0)),
            # Risk & confidence
            "confidence_penalty": float(signal_data.get("confidence_penalty", 0.0)),
            "leverage": float(signal_data.get("leverage", 10)),
            # Market regime
            "regime_fear": 1.0 if signal_data.get("market_regime") == "FEAR" else 0.0,
            "regime_neutral": (1.0 if signal_data.get("market_regime") == "NEUTRAL" else 0.0),
            "regime_greed": 1.0 if signal_data.get("market_regime") == "GREED" else 0.0,
            # Market condition
            "condition_normal": (1.0 if signal_data.get("market_condition") == "NORMAL" else 0.0),
            "condition_high_vol": (
                1.0 if signal_data.get("market_condition") == "HIGH_VOLATILITY" else 0.0
            ),
            "condition_extreme": (
                1.0
                if signal_data.get("market_condition") in ["EXTREME_MOVE", "MARKET_STRESS"]
                else 0.0
            ),
            # Technical indicators
            "mfi": float(signal_data.get("mfi", 50)) / 100.0,  # Normalize 0-1
            "adx": float(signal_data.get("adx", 0)) / 100.0,
            "btc_correlation": float(signal_data.get("btc_correlation", 0)),
            # Binary flags
            "volume_spike": (
                1.0 if str(signal_data.get("volume_spike", "false")).lower() == "true" else 0.0
            ),
            "near_key_level": (
                1.0 if str(signal_data.get("near_key_level", "false")).lower() == "true" else 0.0
            ),
            "liquidity_sweep": (
                1.0 if str(signal_data.get("liquidity_sweep", "false")).lower() == "true" else 0.0
            ),
            "fresh_bos": (
                1.0 if str(signal_data.get("fresh_bos", "false")).lower() == "true" else 0.0
            ),
            # HTF trend
            "htf_bullish": 1.0 if signal_data.get("htf_trend") == "bullish" else 0.0,
            "htf_bearish": 1.0 if signal_data.get("htf_trend") == "bearish" else 0.0,
            "htf_neutral": 1.0 if signal_data.get("htf_trend") == "neutral" else 0.0,
            # Session
            "session_london": (
                1.0 if signal_data.get("session") in ["London", "London/NY"] else 0.0
            ),
            "session_ny": (1.0 if signal_data.get("session") in ["NY", "London/NY"] else 0.0),
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
                    Trade.pnl.is_not(None),
                    Trade.is_dry_run.is_(False),
                )
                .order_by(desc(Trade.entry_time))
                .limit(1000)
                .all()
            )

            if len(trades) < self.min_trades_for_training:
                logger.warning(f"Za mało danych: {len(trades)} < {self.min_trades_for_training}")
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

                    win_labels.append(1.0 if trade.pnl > 0 else 0.0)
                    pnl_labels.append(trade.pnl_percent or 0.0)

                except Exception as e:
                    logger.warning(f"Błąd trade ID {trade.id}: {e}")
                    continue

            if not features_list:
                return None

            features_df = pd.DataFrame(features_list)
            win_series = pd.Series(win_labels)
            pnl_series = pd.Series(pnl_labels)

            logger.info(f"Dane ML: {len(features_df)} próbek, Win rate: {win_series.mean():.2%}")
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

            X_train, X_test, y_win_train, y_win_test, y_pnl_train, y_pnl_test = train_test_split(
                X, y_win, y_pnl, test_size=0.2, random_state=42, stratify=y_win
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
            self.pnl_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            self.pnl_model.fit(X_train, y_pnl_train)

            self.save_models()
            self._log_feature_importance()

            logger.info(f"Modele ML wytrenowane! (v{self.model_version})")
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
            prob_entropy = -win_prob * np.log2(win_prob + 1e-10) - (1 - win_prob) * np.log2(
                1 - win_prob + 1e-10
            )
            confidence = 1.0 - prob_entropy

            return {
                "win_probability": float(win_prob),
                "predicted_pnl_pct": float(predicted_pnl),
                "expected_value": float(expected_value),
                "confidence": float(confidence),
            }

        except Exception as e:
            logger.error(f"Błąd predykcji: {e}", exc_info=True)
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
            feature_importance = list(zip(self.feature_columns, importances, strict=False))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            logger.info("Top 10 najważniejszych cech:")
            for feature, importance in feature_importance[:10]:
                logger.info(f"  {feature}: {importance:.3f}")

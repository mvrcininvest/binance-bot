"""
SHAP Explainer - Advanced ML model explainability and interpretability system
Part of the Hybrid Ultra-Diagnostics system for Trading Bot v9.1
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import logging
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock SHAP implementation since we can't install external libraries
# In production, you would use: import shap

class MockSHAPExplainer:
    """Mock SHAP explainer for demonstration purposes"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.expected_value = 0.5
        
    def shap_values(self, X):
        """Mock SHAP values calculation"""
        if isinstance(X, (list, np.ndarray)):
            X = np.array(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        
        # Generate mock SHAP values
        shap_values = []
        for sample in X:
            # Create realistic-looking SHAP values
            values = np.random.normal(0, 0.1, len(sample))
            # Make some features more important
            values[0] *= 2  # First feature more important
            values[1] *= 1.5  # Second feature moderately important
            shap_values.append(values)
            
        return np.array(shap_values)

@dataclass
class FeatureImportance:
    """Feature importance information"""
    feature_name: str
    importance_score: float
    shap_value: float
    contribution_percentage: float
    direction: str  # 'positive' or 'negative'
    confidence: float
    description: str

@dataclass
class ModelExplanation:
    """Complete model explanation"""
    model_name: str
    prediction: float
    confidence: float
    expected_value: float
    feature_importances: List[FeatureImportance]
    top_positive_features: List[str]
    top_negative_features: List[str]
    explanation_summary: str
    risk_factors: List[str]
    supporting_factors: List[str]
    metadata: Dict[str, Any]

@dataclass
class GlobalExplanation:
    """Global model explanation across all predictions"""
    model_name: str
    total_predictions: int
    average_confidence: float
    feature_importance_ranking: List[Tuple[str, float]]
    feature_stability: Dict[str, float]
    model_behavior_patterns: Dict[str, Any]
    common_decision_paths: List[Dict[str, Any]]
    outlier_explanations: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime

@dataclass
class ExplanationReport:
    """Comprehensive explanation report"""
    report_id: str
    generated_at: datetime
    model_explanations: Dict[str, ModelExplanation]
    global_explanations: Dict[str, GlobalExplanation]
    feature_analysis: Dict[str, Any]
    model_comparison: Dict[str, Any]
    interpretability_metrics: Dict[str, float]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]

class SHAPExplainer:
    """Advanced SHAP-based model explainability system"""
    
    def __init__(self):
        self.explainers = {}
        self.explanation_history = defaultdict(lambda: deque(maxlen=1000))
        self.feature_importance_cache = {}
        self.global_explanations = {}
        self.model_baselines = {}
        self.explanation_metadata = {}
        self.monitoring_active = False
        
        # Feature name mappings for better interpretability
        self.feature_mappings = {
            'rsi': 'RSI Indicator',
            'macd': 'MACD Signal',
            'bb_upper': 'Bollinger Band Upper',
            'bb_lower': 'Bollinger Band Lower',
            'volume': 'Trading Volume',
            'price_change': 'Price Change %',
            'volatility': 'Price Volatility',
            'momentum': 'Price Momentum',
            'support_level': 'Support Level',
            'resistance_level': 'Resistance Level',
            'trend_strength': 'Trend Strength',
            'market_sentiment': 'Market Sentiment'
        }
        
    async def start_explanation_service(self):
        """Start the SHAP explanation service"""
        self.monitoring_active = True
        logger.info("🔍 SHAP Explainer service started")
        
        # Start background tasks
        asyncio.create_task(self._update_global_explanations())
        asyncio.create_task(self._monitor_explanation_quality())
        
    async def stop_explanation_service(self):
        """Stop the SHAP explanation service"""
        self.monitoring_active = False
        logger.info("🛑 SHAP Explainer service stopped")
        
    async def register_model(self, model_name: str, model: Any, training_data: np.ndarray, feature_names: List[str]):
        """Register a model for SHAP explanation"""
        try:
            logger.info(f"Registering model for explanation: {model_name}")
            
            # Create SHAP explainer
            explainer = MockSHAPExplainer(model, training_data)
            self.explainers[model_name] = {
                'explainer': explainer,
                'model': model,
                'feature_names': feature_names,
                'training_data': training_data,
                'registered_at': datetime.now()
            }
            
            # Calculate baseline explanations
            await self._calculate_model_baseline(model_name)
            
            logger.info(f"Model {model_name} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            return False
            
    async def explain_prediction(self, model_name: str, input_data: Union[Dict, List, np.ndarray], 
                               prediction: float = None) -> Optional[ModelExplanation]:
        """Explain a single prediction using SHAP"""
        try:
            if model_name not in self.explainers:
                logger.error(f"Model {model_name} not registered")
                return None
                
            explainer_info = self.explainers[model_name]
            explainer = explainer_info['explainer']
            feature_names = explainer_info['feature_names']
            model = explainer_info['model']
            
            # Prepare input data
            if isinstance(input_data, dict):
                # Convert dict to array using feature names
                input_array = np.array([input_data.get(name, 0) for name in feature_names])
            elif isinstance(input_data, list):
                input_array = np.array(input_data)
            else:
                input_array = input_data
                
            # Ensure correct shape
            if input_array.ndim == 1:
                input_array = input_array.reshape(1, -1)
                
            # Get prediction if not provided
            if prediction is None:
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict_proba(input_array)[0][1]  # Probability of positive class
                elif hasattr(model, 'predict'):
                    prediction = model.predict(input_array)[0]
                else:
                    prediction = 0.5  # Default
                    
            # Calculate SHAP values
            shap_values = explainer.shap_values(input_array)
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Take first sample
                
            # Create feature importances
            feature_importances = []
            total_abs_shap = sum(abs(val) for val in shap_values)
            
            for i, (feature_name, shap_val) in enumerate(zip(feature_names, shap_values)):
                importance = FeatureImportance(
                    feature_name=self.feature_mappings.get(feature_name, feature_name),
                    importance_score=abs(shap_val),
                    shap_value=shap_val,
                    contribution_percentage=abs(shap_val) / total_abs_shap * 100 if total_abs_shap > 0 else 0,
                    direction='positive' if shap_val > 0 else 'negative',
                    confidence=min(abs(shap_val) * 2, 1.0),
                    description=await self._generate_feature_description(feature_name, shap_val, input_array[0][i])
                )
                feature_importances.append(importance)
                
            # Sort by importance
            feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Get top positive and negative features
            positive_features = [f.feature_name for f in feature_importances if f.direction == 'positive'][:3]
            negative_features = [f.feature_name for f in feature_importances if f.direction == 'negative'][:3]
            
            # Generate explanation summary
            explanation_summary = await self._generate_explanation_summary(
                prediction, feature_importances, positive_features, negative_features
            )
            
            # Identify risk and supporting factors
            risk_factors = await self._identify_risk_factors(feature_importances, input_data)
            supporting_factors = await self._identify_supporting_factors(feature_importances, input_data)
            
            # Calculate confidence
            confidence = await self._calculate_explanation_confidence(shap_values, prediction)
            
            explanation = ModelExplanation(
                model_name=model_name,
                prediction=prediction,
                confidence=confidence,
                expected_value=explainer.expected_value,
                feature_importances=feature_importances,
                top_positive_features=positive_features,
                top_negative_features=negative_features,
                explanation_summary=explanation_summary,
                risk_factors=risk_factors,
                supporting_factors=supporting_factors,
                metadata={
                    'input_data': input_data if isinstance(input_data, dict) else input_array.tolist(),
                    'total_shap_impact': sum(shap_values),
                    'explanation_timestamp': datetime.now().isoformat()
                }
            )
            
            # Store explanation
            await self._store_explanation(model_name, explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction for {model_name}: {e}")
            return None
            
    async def explain_batch_predictions(self, model_name: str, batch_data: List[Dict], 
                                      predictions: List[float] = None) -> List[Optional[ModelExplanation]]:
        """Explain multiple predictions in batch"""
        try:
            explanations = []
            
            for i, input_data in enumerate(batch_data):
                pred = predictions[i] if predictions and i < len(predictions) else None
                explanation = await self.explain_prediction(model_name, input_data, pred)
                explanations.append(explanation)
                
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining batch predictions: {e}")
            return []
            
    async def generate_global_explanation(self, model_name: str, sample_size: int = 100) -> Optional[GlobalExplanation]:
        """Generate global explanation for model behavior"""
        try:
            if model_name not in self.explainers:
                return None
                
            explainer_info = self.explainers[model_name]
            training_data = explainer_info['training_data']
            feature_names = explainer_info['feature_names']
            
            # Sample data for global explanation
            if len(training_data) > sample_size:
                indices = np.random.choice(len(training_data), sample_size, replace=False)
                sample_data = training_data[indices]
            else:
                sample_data = training_data
                
            # Get SHAP values for sample
            explainer = explainer_info['explainer']
            shap_values = explainer.shap_values(sample_data)
            
            # Calculate global feature importance
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_importance_ranking = list(zip(feature_names, mean_abs_shap))
            feature_importance_ranking.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate feature stability (consistency of importance)
            feature_stability = {}
            for i, feature_name in enumerate(feature_names):
                feature_shap_values = shap_values[:, i]
                stability = 1.0 - (np.std(np.abs(feature_shap_values)) / (np.mean(np.abs(feature_shap_values)) + 1e-8))
                feature_stability[feature_name] = max(0.0, min(1.0, stability))
                
            # Analyze model behavior patterns
            behavior_patterns = await self._analyze_model_behavior_patterns(shap_values, feature_names)
            
            # Find common decision paths
            decision_paths = await self._find_common_decision_paths(shap_values, feature_names)
            
            # Identify outlier explanations
            outlier_explanations = await self._identify_outlier_explanations(shap_values, feature_names)
            
            # Generate recommendations
            recommendations = await self._generate_global_recommendations(
                feature_importance_ranking, feature_stability, behavior_patterns
            )
            
            # Calculate average confidence
            confidences = []
            for explanation_deque in self.explanation_history.values():
                confidences.extend([exp.confidence for exp in explanation_deque if exp.model_name == model_name])
            avg_confidence = statistics.mean(confidences) if confidences else 0.5
            
            global_explanation = GlobalExplanation(
                model_name=model_name,
                total_predictions=len(confidences),
                average_confidence=avg_confidence,
                feature_importance_ranking=feature_importance_ranking,
                feature_stability=feature_stability,
                model_behavior_patterns=behavior_patterns,
                common_decision_paths=decision_paths,
                outlier_explanations=outlier_explanations,
                recommendations=recommendations,
                generated_at=datetime.now()
            )
            
            # Store global explanation
            self.global_explanations[model_name] = global_explanation
            
            return global_explanation
            
        except Exception as e:
            logger.error(f"Error generating global explanation: {e}")
            return None
            
    async def compare_models(self, model_names: List[str], test_data: List[Dict]) -> Dict[str, Any]:
        """Compare explanations across multiple models"""
        try:
            comparison_results = {
                'models_compared': model_names,
                'test_samples': len(test_data),
                'feature_importance_comparison': {},
                'prediction_agreement': {},
                'explanation_consistency': {},
                'model_reliability': {},
                'recommendations': []
            }
            
            # Get explanations for all models
            model_explanations = {}
            for model_name in model_names:
                if model_name in self.explainers:
                    explanations = []
                    for data_point in test_data:
                        explanation = await self.explain_prediction(model_name, data_point)
                        if explanation:
                            explanations.append(explanation)
                    model_explanations[model_name] = explanations
                    
            # Compare feature importance across models
            for model_name, explanations in model_explanations.items():
                if explanations:
                    # Average feature importance across all explanations
                    feature_importance_avg = defaultdict(list)
                    for explanation in explanations:
                        for feature_imp in explanation.feature_importances:
                            feature_importance_avg[feature_imp.feature_name].append(feature_imp.importance_score)
                            
                    # Calculate average importance for each feature
                    avg_importance = {}
                    for feature, scores in feature_importance_avg.items():
                        avg_importance[feature] = statistics.mean(scores)
                        
                    comparison_results['feature_importance_comparison'][model_name] = avg_importance
                    
            # Calculate prediction agreement
            if len(model_explanations) >= 2:
                model_pairs = [(m1, m2) for i, m1 in enumerate(model_names) 
                              for m2 in model_names[i+1:] if m1 in model_explanations and m2 in model_explanations]
                
                for m1, m2 in model_pairs:
                    explanations1 = model_explanations[m1]
                    explanations2 = model_explanations[m2]
                    
                    if explanations1 and explanations2:
                        # Calculate prediction correlation
                        predictions1 = [exp.prediction for exp in explanations1]
                        predictions2 = [exp.prediction for exp in explanations2]
                        
                        min_len = min(len(predictions1), len(predictions2))
                        if min_len > 1:
                            correlation = np.corrcoef(predictions1[:min_len], predictions2[:min_len])[0, 1]
                            comparison_results['prediction_agreement'][f"{m1}_vs_{m2}"] = correlation
                            
            # Calculate explanation consistency
            for model_name, explanations in model_explanations.items():
                if len(explanations) > 1:
                    # Measure consistency of top features across explanations
                    top_features_lists = [exp.top_positive_features[:2] + exp.top_negative_features[:2] 
                                        for exp in explanations]
                    
                    # Calculate Jaccard similarity between feature sets
                    similarities = []
                    for i in range(len(top_features_lists)):
                        for j in range(i+1, len(top_features_lists)):
                            set1 = set(top_features_lists[i])
                            set2 = set(top_features_lists[j])
                            if set1 or set2:
                                similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                                similarities.append(similarity)
                                
                    consistency = statistics.mean(similarities) if similarities else 0.0
                    comparison_results['explanation_consistency'][model_name] = consistency
                    
            # Calculate model reliability
            for model_name, explanations in model_explanations.items():
                if explanations:
                    confidences = [exp.confidence for exp in explanations]
                    avg_confidence = statistics.mean(confidences)
                    confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
                    
                    # Reliability is high confidence with low variance
                    reliability = avg_confidence * (1 - confidence_std)
                    comparison_results['model_reliability'][model_name] = reliability
                    
            # Generate recommendations
            comparison_results['recommendations'] = await self._generate_comparison_recommendations(comparison_results)
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {'error': str(e)}
            
    async def generate_explanation_report(self, hours_back: int = 24) -> ExplanationReport:
        """Generate comprehensive explanation report"""
        try:
            report_id = f"explanation_report_{int(time.time())}"
            generated_at = datetime.now()
            cutoff_time = generated_at - timedelta(hours=hours_back)
            
            # Collect recent explanations
            model_explanations = {}
            for model_name, explanation_deque in self.explanation_history.items():
                recent_explanations = [exp for exp in explanation_deque if 
                                     'explanation_timestamp' in exp.metadata and 
                                     datetime.fromisoformat(exp.metadata['explanation_timestamp']) > cutoff_time]
                
                if recent_explanations:
                    # Get the most representative explanation
                    model_explanations[model_name] = recent_explanations[-1]  # Most recent
                    
            # Get global explanations
            global_explanations = {}
            for model_name in self.explainers.keys():
                global_exp = await self.generate_global_explanation(model_name)
                if global_exp:
                    global_explanations[model_name] = global_exp
                    
            # Analyze features across all models
            feature_analysis = await self._analyze_features_across_models(model_explanations)
            
            # Compare models if multiple exist
            model_comparison = {}
            if len(model_explanations) > 1:
                test_data = []  # Would need actual test data
                model_comparison = await self.compare_models(list(model_explanations.keys()), test_data)
                
            # Calculate interpretability metrics
            interpretability_metrics = await self._calculate_interpretability_metrics(
                model_explanations, global_explanations
            )
            
            # Generate recommendations
            recommendations = await self._generate_report_recommendations(
                model_explanations, global_explanations, interpretability_metrics
            )
            
            # Assess risk
            risk_assessment = await self._assess_explanation_risk(model_explanations, interpretability_metrics)
            
            report = ExplanationReport(
                report_id=report_id,
                generated_at=generated_at,
                model_explanations=model_explanations,
                global_explanations=global_explanations,
                feature_analysis=feature_analysis,
                model_comparison=model_comparison,
                interpretability_metrics=interpretability_metrics,
                recommendations=recommendations,
                risk_assessment=risk_assessment
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating explanation report: {e}")
            return ExplanationReport(
                report_id=f"error_report_{int(time.time())}",
                generated_at=datetime.now(),
                model_explanations={},
                global_explanations={},
                feature_analysis={},
                model_comparison={},
                interpretability_metrics={},
                recommendations=[f"Error generating report: {str(e)}"],
                risk_assessment={'error': True}
            )
            
    async def get_feature_importance_trends(self, model_name: str, feature_name: str, 
                                          days_back: int = 7) -> Dict[str, Any]:
        """Get feature importance trends over time"""
        try:
            if model_name not in self.explanation_history:
                return {'error': f'No explanation history for model {model_name}'}
                
            cutoff_time = datetime.now() - timedelta(days=days_back)
            explanations = self.explanation_history[model_name]
            
            # Filter recent explanations
            recent_explanations = []
            for exp in explanations:
                if 'explanation_timestamp' in exp.metadata:
                    timestamp = datetime.fromisoformat(exp.metadata['explanation_timestamp'])
                    if timestamp > cutoff_time:
                        recent_explanations.append((timestamp, exp))
                        
            if not recent_explanations:
                return {'error': 'No recent explanations found'}
                
            # Sort by timestamp
            recent_explanations.sort(key=lambda x: x[0])
            
            # Extract feature importance over time
            timestamps = []
            importance_scores = []
            shap_values = []
            
            for timestamp, explanation in recent_explanations:
                for feature_imp in explanation.feature_importances:
                    if feature_imp.feature_name == feature_name or \
                       self.feature_mappings.get(feature_name) == feature_imp.feature_name:
                        timestamps.append(timestamp)
                        importance_scores.append(feature_imp.importance_score)
                        shap_values.append(feature_imp.shap_value)
                        break
                        
            if not importance_scores:
                return {'error': f'Feature {feature_name} not found in explanations'}
                
            # Calculate trend statistics
            trend_analysis = {
                'feature_name': feature_name,
                'data_points': len(importance_scores),
                'time_range': {
                    'start': timestamps[0].isoformat() if timestamps else None,
                    'end': timestamps[-1].isoformat() if timestamps else None
                },
                'importance_stats': {
                    'mean': statistics.mean(importance_scores),
                    'median': statistics.median(importance_scores),
                    'std': statistics.stdev(importance_scores) if len(importance_scores) > 1 else 0,
                    'min': min(importance_scores),
                    'max': max(importance_scores)
                },
                'shap_stats': {
                    'mean': statistics.mean(shap_values),
                    'median': statistics.median(shap_values),
                    'std': statistics.stdev(shap_values) if len(shap_values) > 1 else 0,
                    'positive_ratio': sum(1 for v in shap_values if v > 0) / len(shap_values)
                },
                'trend': await self._calculate_feature_trend(importance_scores),
                'stability': await self._calculate_feature_stability(importance_scores),
                'recommendations': await self._generate_feature_trend_recommendations(
                    feature_name, importance_scores, shap_values
                )
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error getting feature importance trends: {e}")
            return {'error': str(e)}
            
    # Private helper methods
    
    async def _calculate_model_baseline(self, model_name: str):
        """Calculate baseline explanation metrics for model"""
        try:
            explainer_info = self.explainers[model_name]
            training_data = explainer_info['training_data']
            
            # Sample some data for baseline
            sample_size = min(50, len(training_data))
            sample_indices = np.random.choice(len(training_data), sample_size, replace=False)
            sample_data = training_data[sample_indices]
            
            # Get baseline SHAP values
            explainer = explainer_info['explainer']
            baseline_shap = explainer.shap_values(sample_data)
            
            # Calculate baseline statistics
            baseline_stats = {
                'mean_abs_shap': np.mean(np.abs(baseline_shap), axis=0),
                'std_shap': np.std(baseline_shap, axis=0),
                'feature_correlations': np.corrcoef(baseline_shap.T) if baseline_shap.shape[1] > 1 else np.array([[1.0]]),
                'calculated_at': datetime.now()
            }
            
            self.model_baselines[model_name] = baseline_stats
            
        except Exception as e:
            logger.error(f"Error calculating model baseline: {e}")
            
    async def _generate_feature_description(self, feature_name: str, shap_value: float, feature_value: float) -> str:
        """Generate human-readable description of feature contribution"""
        try:
            mapped_name = self.feature_mappings.get(feature_name, feature_name)
            direction = "increases" if shap_value > 0 else "decreases"
            strength = "strongly" if abs(shap_value) > 0.1 else "moderately" if abs(shap_value) > 0.05 else "slightly"
            
            description = f"{mapped_name} (value: {feature_value:.3f}) {strength} {direction} the prediction"
            
            # Add context-specific descriptions
            if feature_name == 'rsi':
                if feature_value > 70:
                    description += " - indicating overbought conditions"
                elif feature_value < 30:
                    description += " - indicating oversold conditions"
            elif feature_name == 'volume':
                if feature_value > 1.5:
                    description += " - unusually high trading activity"
                elif feature_value < 0.5:
                    description += " - low trading activity"
            elif feature_name == 'volatility':
                if feature_value > 0.02:
                    description += " - high market volatility"
                elif feature_value < 0.005:
                    description += " - low market volatility"
                    
            return description
            
        except Exception as e:
            return f"{feature_name}: {shap_value:.3f}"
            
    async def _generate_explanation_summary(self, prediction: float, feature_importances: List[FeatureImportance],
                                          positive_features: List[str], negative_features: List[str]) -> str:
        """Generate human-readable explanation summary"""
        try:
            # Determine prediction type
            if prediction > 0.7:
                pred_desc = "strong positive"
            elif prediction > 0.6:
                pred_desc = "positive"
            elif prediction > 0.4:
                pred_desc = "neutral"
            elif prediction > 0.3:
                pred_desc = "negative"
            else:
                pred_desc = "strong negative"
                
            summary = f"The model made a {pred_desc} prediction (confidence: {prediction:.1%}). "
            
            if positive_features:
                summary += f"Key supporting factors: {', '.join(positive_features[:2])}. "
                
            if negative_features:
                summary += f"Key opposing factors: {', '.join(negative_features[:2])}. "
                
            # Add most important feature
            if feature_importances:
                top_feature = feature_importances[0]
                summary += f"The most influential factor was {top_feature.feature_name} "
                summary += f"({top_feature.contribution_percentage:.1f}% of decision)."
                
            return summary
            
        except Exception as e:
            return f"Prediction: {prediction:.1%} (explanation generation error)"
            
    async def _identify_risk_factors(self, feature_importances: List[FeatureImportance], 
                                   input_data: Union[Dict, np.ndarray]) -> List[str]:
        """Identify risk factors from feature importances"""
        risk_factors = []
        
        try:
            for feature_imp in feature_importances[:5]:  # Top 5 features
                if feature_imp.direction == 'negative' and feature_imp.importance_score > 0.05:
                    risk_factors.append(f"{feature_imp.feature_name}: {feature_imp.description}")
                    
            # Add specific risk checks
            if isinstance(input_data, dict):
                if input_data.get('volatility', 0) > 0.03:
                    risk_factors.append("High market volatility increases prediction uncertainty")
                if input_data.get('volume', 1) < 0.3:
                    risk_factors.append("Low trading volume may indicate unreliable signals")
                    
        except Exception as e:
            risk_factors.append(f"Error identifying risk factors: {str(e)}")
            
        return risk_factors
        
    async def _identify_supporting_factors(self, feature_importances: List[FeatureImportance],
                                         input_data: Union[Dict, np.ndarray]) -> List[str]:
        """Identify supporting factors from feature importances"""
        supporting_factors = []
        
        try:
            for feature_imp in feature_importances[:5]:  # Top 5 features
                if feature_imp.direction == 'positive' and feature_imp.importance_score > 0.05:
                    supporting_factors.append(f"{feature_imp.feature_name}: {feature_imp.description}")
                    
            # Add specific support checks
            if isinstance(input_data, dict):
                if input_data.get('trend_strength', 0) > 0.7:
                    supporting_factors.append("Strong trend provides reliable directional signal")
                if input_data.get('volume', 1) > 1.5:
                    supporting_factors.append("High volume confirms signal strength")
                    
        except Exception as e:
            supporting_factors.append(f"Error identifying supporting factors: {str(e)}")
            
        return supporting_factors
        
    async def _calculate_explanation_confidence(self, shap_values: np.ndarray, prediction: float) -> float:
        """Calculate confidence in the explanation"""
        try:
            # Base confidence on SHAP value magnitude and consistency
            total_abs_shap = sum(abs(val) for val in shap_values)
            
            # Higher total SHAP magnitude indicates more confident explanation
            magnitude_confidence = min(total_abs_shap * 2, 1.0)
            
            # Prediction extremity indicates confidence
            prediction_confidence = abs(prediction - 0.5) * 2
            
            # Combine confidences
            overall_confidence = (magnitude_confidence + prediction_confidence) / 2
            
            return min(max(overall_confidence, 0.0), 1.0)
            
        except Exception as e:
            return 0.5  # Default confidence
            
    async def _store_explanation(self, model_name: str, explanation: ModelExplanation):
        """Store explanation in history"""
        try:
            self.explanation_history[model_name].append(explanation)
            
            # Store in database for diagnostics
            from database import log_execution_trace
            await log_execution_trace(
                component="shap_explainer",
                operation="explanation_generated",
                input_data={'model_name': model_name},
                output_data={
                    'prediction': explanation.prediction,
                    'confidence': explanation.confidence,
                    'top_features': explanation.top_positive_features + explanation.top_negative_features
                },
                execution_time=0.0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error storing explanation: {e}")
            
    async def _analyze_model_behavior_patterns(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze patterns in model behavior"""
        try:
            patterns = {
                'feature_interactions': {},
                'decision_boundaries': {},
                'common_patterns': [],
                'unusual_patterns': []
            }
            
            # Analyze feature interactions (correlations in SHAP values)
            if shap_values.shape[1] > 1:
                shap_correlations = np.corrcoef(shap_values.T)
                
                for i, feature1 in enumerate(feature_names):
                    for j, feature2 in enumerate(feature_names[i+1:], i+1):
                        correlation = shap_correlations[i, j]
                        if abs(correlation) > 0.5:  # Strong correlation
                            patterns['feature_interactions'][f"{feature1}_vs_{feature2}"] = correlation
                            
            # Identify common decision patterns
            # Cluster similar SHAP value patterns
            positive_decisions = shap_values[shap_values.sum(axis=1) > 0]
            negative_decisions = shap_values[shap_values.sum(axis=1) < 0]
            
            if len(positive_decisions) > 0:
                avg_positive_pattern = np.mean(positive_decisions, axis=0)
                patterns['common_patterns'].append({
                    'type': 'positive_decisions',
                    'pattern': dict(zip(feature_names, avg_positive_pattern)),
                    'frequency': len(positive_decisions) / len(shap_values)
                })
                
            if len(negative_decisions) > 0:
                avg_negative_pattern = np.mean(negative_decisions, axis=0)
                patterns['common_patterns'].append({
                    'type': 'negative_decisions',
                    'pattern': dict(zip(feature_names, avg_negative_pattern)),
                    'frequency': len(negative_decisions) / len(shap_values)
                })
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing model behavior patterns: {e}")
            return {}
            
    async def _find_common_decision_paths(self, shap_values: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Find common decision paths in the model"""
        try:
            decision_paths = []
            
            # Find samples with similar SHAP patterns
            for i in range(len(shap_values)):
                similar_samples = []
                base_shap = shap_values[i]
                
                for j in range(len(shap_values)):
                    if i != j:
                        # Calculate similarity (cosine similarity)
                        similarity = np.dot(base_shap, shap_values[j]) / (
                            np.linalg.norm(base_shap) * np.linalg.norm(shap_values[j]) + 1e-8
                        )
                        if similarity > 0.8:  # High similarity
                            similar_samples.append(j)
                            
                if len(similar_samples) >= 3:  # At least 3 similar samples
                    # Create decision path
                    avg_shap = np.mean(shap_values[similar_samples + [i]], axis=0)
                    
                    path = {
                        'path_id': f"path_{len(decision_paths)}",
                        'sample_count': len(similar_samples) + 1,
                        'frequency': (len(similar_samples) + 1) / len(shap_values),
                        'key_features': [],
                        'decision_logic': ""
                    }
                    
                    # Identify key features in this path
                    feature_importance = [(feature_names[idx], abs(val)) for idx, val in enumerate(avg_shap)]
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    path['key_features'] = feature_importance[:3]
                    
                    # Generate decision logic description
                    top_features = [f"{name} ({'positive' if avg_shap[feature_names.index(name)] > 0 else 'negative'})" 
                                  for name, _ in feature_importance[:2]]
                    path['decision_logic'] = f"When {' and '.join(top_features)}, model tends to make similar decisions"
                    
                    decision_paths.append(path)
                    
            # Remove duplicate paths
            unique_paths = []
            for path in decision_paths:
                is_duplicate = False
                for existing_path in unique_paths:
                    if set(f[0] for f in path['key_features'][:2]) == set(f[0] for f in existing_path['key_features'][:2]):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_paths.append(path)
                    
            return unique_paths[:5]  # Return top 5 paths
            
        except Exception as e:
            logger.error(f"Error finding decision paths: {e}")
            return []
            
    async def _identify_outlier_explanations(self, shap_values: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Identify outlier explanations that are unusual"""
        try:
            outliers = []
            
            # Calculate mean and std for each feature's SHAP values
            mean_shap = np.mean(shap_values, axis=0)
            std_shap = np.std(shap_values, axis=0)
            
            # Find samples with unusual SHAP patterns
            for i, sample_shap in enumerate(shap_values):
                outlier_features = []
                
                for j, (feature_name, shap_val) in enumerate(zip(feature_names, sample_shap)):
                    if std_shap[j] > 0:
                        z_score = abs(shap_val - mean_shap[j]) / std_shap[j]
                        if z_score > 2.5:  # Outlier threshold
                            outlier_features.append({
                                'feature': feature_name,
                                'shap_value': shap_val,
                                'z_score': z_score,
                                'typical_value': mean_shap[j]
                            })
                            
                if len(outlier_features) >= 2:  # At least 2 outlier features
                    outlier = {
                        'sample_index': i,
                        'outlier_features': outlier_features,
                        'total_outlier_score': sum(f['z_score'] for f in outlier_features),
                        'description': f"Unusual explanation with {len(outlier_features)} atypical features"
                    }
                    outliers.append(outlier)
                    
            # Sort by outlier score and return top outliers
            outliers.sort(key=lambda x: x['total_outlier_score'], reverse=True)
            return outliers[:3]
            
        except Exception as e:
            logger.error(f"Error identifying outlier explanations: {e}")
            return []
            
    async def _generate_global_recommendations(self, feature_importance_ranking: List[Tuple[str, float]],
                                             feature_stability: Dict[str, float],
                                             behavior_patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on global model analysis"""
        recommendations = []
        
        try:
            # Feature importance recommendations
            if feature_importance_ranking:
                top_feature = feature_importance_ranking[0]
                recommendations.append(f"Model relies heavily on {top_feature[0]} - ensure this feature is reliable")
                
                # Check for unstable important features
                for feature_name, importance in feature_importance_ranking[:3]:
                    stability = feature_stability.get(feature_name, 0.5)
                    if stability < 0.3:
                        recommendations.append(f"Feature {feature_name} is important but unstable - investigate data quality")
                        
            # Feature interaction recommendations
            interactions = behavior_patterns.get('feature_interactions', {})
            strong_interactions = [(pair, corr) for pair, corr in interactions.items() if abs(corr) > 0.7]
            
            if strong_interactions:
                recommendations.append("Strong feature interactions detected - consider feature engineering")
                
            # Decision pattern recommendations
            common_patterns = behavior_patterns.get('common_patterns', [])
            if len(common_patterns) < 2:
                recommendations.append("Model shows limited decision diversity - consider expanding training data")
                
            if not recommendations:
                recommendations.append("Model shows healthy global behavior patterns")
                
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
            
        return recommendations
        
    async def _generate_comparison_recommendations(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations from model comparison"""
        recommendations = []
        
        try:
            # Prediction agreement recommendations
            agreements = comparison_results.get('prediction_agreement', {})
            low_agreement = [pair for pair, corr in agreements.items() if corr < 0.5]
            
            if low_agreement:
                recommendations.append("Low prediction agreement between some models - investigate model differences")
                
            # Explanation consistency recommendations
            consistency = comparison_results.get('explanation_consistency', {})
            inconsistent_models = [model for model, cons in consistency.items() if cons < 0.3]
            
            if inconsistent_models:
                recommendations.append(f"Models with inconsistent explanations: {', '.join(inconsistent_models)}")
                
            # Reliability recommendations
            reliability = comparison_results.get('model_reliability', {})
            if reliability:
                most_reliable = max(reliability.items(), key=lambda x: x[1])
                recommendations.append(f"Most reliable model: {most_reliable[0]} (score: {most_reliable[1]:.2f})")
                
        except Exception as e:
            recommendations.append(f"Error in comparison analysis: {str(e)}")
            
        return recommendations
        
    async def _analyze_features_across_models(self, model_explanations: Dict[str, ModelExplanation]) -> Dict[str, Any]:
        """Analyze feature behavior across all models"""
        try:
            feature_analysis = {
                'cross_model_importance': {},
                'feature_consistency': {},
                'unique_features': {},
                'common_features': []
            }
            
            # Collect all features and their importance across models
            all_features = defaultdict(list)
            
            for model_name, explanation in model_explanations.items():
                for feature_imp in explanation.feature_importances:
                    all_features[feature_imp.feature_name].append({
                        'model': model_name,
                        'importance': feature_imp.importance_score,
                        'direction': feature_imp.direction
                    })
                    
            # Analyze each feature
            for feature_name, feature_data in all_features.items():
                if len(feature_data) > 1:  # Feature appears in multiple models
                    importances = [f['importance'] for f in feature_data]
                    directions = [f['direction'] for f in feature_data]
                    
                    feature_analysis['cross_model_importance'][feature_name] = {
                        'avg_importance': statistics.mean(importances),
                        'importance_std': statistics.stdev(importances) if len(importances) > 1 else 0,
                        'models_count': len(feature_data),
                        'direction_consistency': len(set(directions)) == 1
                    }
                    
                    # Check if feature is consistently important
                    if statistics.mean(importances) > 0.1 and len(feature_data) >= len(model_explanations) * 0.7:
                        feature_analysis['common_features'].append(feature_name)
                else:
                    # Feature unique to one model
                    feature_analysis['unique_features'][feature_data[0]['model']] = \
                        feature_analysis['unique_features'].get(feature_data[0]['model'], []) + [feature_name]
                        
            return feature_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing features across models: {e}")
            return {}
            
    async def _calculate_interpretability_metrics(self, model_explanations: Dict[str, ModelExplanation],
                                                global_explanations: Dict[str, GlobalExplanation]) -> Dict[str, float]:
        """Calculate interpretability metrics"""
        try:
            metrics = {
                'explanation_confidence': 0.0,
                'feature_stability': 0.0,
                'decision_consistency': 0.0,
                'coverage': 0.0,
                'complexity': 0.0
            }
            
            if model_explanations:
                # Average explanation confidence
                confidences = [exp.confidence for exp in model_explanations.values()]
                metrics['explanation_confidence'] = statistics.mean(confidences)
                
                # Feature stability (how consistent are important features)
                all_top_features = []
                for exp in model_explanations.values():
                    all_top_features.extend(exp.top_positive_features[:2] + exp.top_negative_features[:2])
                    
                if all_top_features:
                    feature_counts = Counter(all_top_features)
                    most_common_count = feature_counts.most_common(1)[0][1] if feature_counts else 1
                    metrics['feature_stability'] = most_common_count / len(model_explanations)
                    
            if global_explanations:
                # Average feature stability from global explanations
                stabilities = []
                for global_exp in global_explanations.values():
                    stabilities.extend(global_exp.feature_stability.values())
                    
                if stabilities:
                    metrics['decision_consistency'] = statistics.mean(stabilities)
                    
                # Coverage (how many features are consistently important)
                all_important_features = set()
                for global_exp in global_explanations.values():
                    top_features = [f[0] for f in global_exp.feature_importance_ranking[:5]]
                    all_important_features.update(top_features)
                    
                metrics['coverage'] = len(all_important_features) / 10.0  # Normalize by expected feature count
                
            # Complexity (inverse of explanation simplicity)
            if model_explanations:
                avg_features_used = statistics.mean([
                    len([f for f in exp.feature_importances if f.importance_score > 0.05])
                    for exp in model_explanations.values()
                ])
                metrics['complexity'] = min(avg_features_used / 10.0, 1.0)  # Normalize
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating interpretability metrics: {e}")
            return {}
            
    async def _generate_report_recommendations(self, model_explanations: Dict[str, ModelExplanation],
                                             global_explanations: Dict[str, GlobalExplanation],
                                             interpretability_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for the explanation report"""
        recommendations = []
        
        try:
            # Confidence recommendations
            confidence = interpretability_metrics.get('explanation_confidence', 0.5)
            if confidence < 0.5:
                recommendations.append("Low explanation confidence - consider model retraining or feature engineering")
            elif confidence > 0.8:
                recommendations.append("High explanation confidence indicates reliable model interpretability")
                
            # Stability recommendations
            stability = interpretability_metrics.get('feature_stability', 0.5)
            if stability < 0.3:
                recommendations.append("Low feature stability - investigate feature quality and consistency")
                
            # Complexity recommendations
            complexity = interpretability_metrics.get('complexity', 0.5)
            if complexity > 0.8:
                recommendations.append("High model complexity - consider feature selection or model simplification")
            elif complexity < 0.2:
                recommendations.append("Very simple model - ensure sufficient complexity for the problem")
                
            # Model-specific recommendations
            for model_name, explanation in model_explanations.items():
                if explanation.confidence < 0.4:
                    recommendations.append(f"Model {model_name} shows low prediction confidence")
                    
                if len(explanation.risk_factors) > 3:
                    recommendations.append(f"Model {model_name} has multiple risk factors - review carefully")
                    
            if not recommendations:
                recommendations.append("Model explanations show healthy interpretability characteristics")
                
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
            
        return recommendations
        
    async def _assess_explanation_risk(self, model_explanations: Dict[str, ModelExplanation],
                                     interpretability_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess risk based on explanation analysis"""
        try:
            risk_assessment = {
                'overall_risk': 'LOW',
                'risk_factors': [],
                'risk_score': 0.0,
                'mitigation_strategies': []
            }
            
            risk_score = 0.0
            
            # Low confidence risk
            confidence = interpretability_metrics.get('explanation_confidence', 0.5)
            if confidence < 0.3:
                risk_score += 0.3
                risk_assessment['risk_factors'].append("Very low explanation confidence")
                risk_assessment['mitigation_strategies'].append("Retrain model with better data")
                
            # High complexity risk
            complexity = interpretability_metrics.get('complexity', 0.5)
            if complexity > 0.8:
                risk_score += 0.2
                risk_assessment['risk_factors'].append("High model complexity reduces interpretability")
                risk_assessment['mitigation_strategies'].append("Consider feature selection or model simplification")
                
            # Inconsistent explanations risk
            if model_explanations:
                confidences = [exp.confidence for exp in model_explanations.values()]
                if len(confidences) > 1 and statistics.stdev(confidences) > 0.3:
                    risk_score += 0.2
                    risk_assessment['risk_factors'].append("Inconsistent explanation confidence across models")
                    
            # Multiple risk factors
            total_risk_factors = sum(len(exp.risk_factors) for exp in model_explanations.values())
            if total_risk_factors > len(model_explanations) * 2:
                risk_score += 0.1
                risk_assessment['risk_factors'].append("Multiple risk factors identified across models")
                
            # Determine overall risk level
            risk_assessment['risk_score'] = min(risk_score, 1.0)
            
            if risk_score > 0.7:
                risk_assessment['overall_risk'] = 'HIGH'
            elif risk_score > 0.4:
                risk_assessment['overall_risk'] = 'MEDIUM'
            else:
                risk_assessment['overall_risk'] = 'LOW'
                
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing explanation risk: {e}")
            return {'overall_risk': 'UNKNOWN', 'error': str(e)}
            
    async def _calculate_feature_trend(self, importance_scores: List[float]) -> str:
        """Calculate trend in feature importance over time"""
        try:
            if len(importance_scores) < 3:
                return "INSUFFICIENT_DATA"
                
            # Simple linear trend
            x = list(range(len(importance_scores)))
            n = len(importance_scores)
            sum_x = sum(x)
            sum_y = sum(importance_scores)
            sum_xy = sum(x[i] * importance_scores[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            if slope > 0.01:
                return "INCREASING"
            elif slope < -0.01:
                return "DECREASING"
            else:
                return "STABLE"
                
        except Exception:
            return "UNKNOWN"
            
    async def _calculate_feature_stability(self, importance_scores: List[float]) -> float:
        """Calculate stability of feature importance"""
        try:
            if len(importance_scores) < 2:
                return 0.5
                
            mean_importance = statistics.mean(importance_scores)
            std_importance = statistics.stdev(importance_scores)
            
            # Stability is inverse of coefficient of variation
            if mean_importance > 0:
                cv = std_importance / mean_importance
                stability = 1.0 / (1.0 + cv)
            else:
                stability = 0.0
                
            return min(max(stability, 0.0), 1.0)
            
        except Exception:
            return 0.5
            
    async def _generate_feature_trend_recommendations(self, feature_name: str, importance_scores: List[float],
                                                    shap_values: List[float]) -> List[str]:
        """Generate recommendations based on feature trends"""
        recommendations = []
        
        try:
            trend = await self._calculate_feature_trend(importance_scores)
            stability = await self._calculate_feature_stability(importance_scores)
            
            if trend == "DECREASING":
                recommendations.append(f"Feature {feature_name} importance is decreasing - investigate data quality")
            elif trend == "INCREASING":
                recommendations.append(f"Feature {feature_name} importance is increasing - monitor for overfitting")
                
            if stability < 0.3:
                recommendations.append(f"Feature {feature_name} shows high variability - check data consistency")
                
            # Direction consistency
            positive_ratio = sum(1 for v in shap_values if v > 0) / len(shap_values)
            if 0.3 < positive_ratio < 0.7:
                recommendations.append(f"Feature {feature_name} has inconsistent directional impact")
                
            if not recommendations:
                recommendations.append(f"Feature {feature_name} shows stable and consistent behavior")
                
        except Exception as e:
            recommendations.append(f"Error analyzing feature trends: {str(e)}")
            
        return recommendations
        
    async def _update_global_explanations(self):
        """Background task to update global explanations"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(1800)  # Update every 30 minutes
                
                for model_name in self.explainers.keys():
                    await self.generate_global_explanation(model_name)
                    
            except Exception as e:
                logger.error(f"Error updating global explanations: {e}")
                await asyncio.sleep(1800)
                
    async def _monitor_explanation_quality(self):
        """Background task to monitor explanation quality"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check for degrading explanation quality
                for model_name, explanation_deque in self.explanation_history.items():
                    if len(explanation_deque) >= 10:
                        recent_confidences = [exp.confidence for exp in list(explanation_deque)[-10:]]
                        avg_confidence = statistics.mean(recent_confidences)
                        
                        if avg_confidence < 0.3:
                            logger.warning(f"Low explanation confidence for model {model_name}: {avg_confidence:.2f}")
                            
            except Exception as e:
                logger.error(f"Error monitoring explanation quality: {e}")
                await asyncio.sleep(3600)

# Global explainer instance
shap_explainer = SHAPExplainer()

# Convenience functions for easy integration
async def start_shap_service():
    """Start SHAP explanation service"""
    await shap_explainer.start_explanation_service()

async def stop_shap_service():
    """Stop SHAP explanation service"""
    await shap_explainer.stop_explanation_service()

async def register_model_for_explanation(model_name: str, model: Any, training_data: np.ndarray, feature_names: List[str]):
    """Register model for SHAP explanation"""
    return await shap_explainer.register_model(model_name, model, training_data, feature_names)

async def explain_model_prediction(model_name: str, input_data: Union[Dict, List, np.ndarray], prediction: float = None):
    """Explain a model prediction"""
    return await shap_explainer.explain_prediction(model_name, input_data, prediction)

async def get_global_model_explanation(model_name: str):
    """Get global explanation for model"""
    return await shap_explainer.generate_global_explanation(model_name)

async def get_explanation_report(hours_back: int = 24):
    """Get comprehensive explanation report"""
    return await shap_explainer.generate_explanation_report(hours_back)

async def get_feature_trends(model_name: str, feature_name: str, days_back: int = 7):
    """Get feature importance trends"""
    return await shap_explainer.get_feature_importance_trends(model_name, feature_name, days_back)

if __name__ == "__main__":
    # Test SHAP Explainer
    async def test_shap_explainer():
        print("🔍 Testing SHAP Explainer...")
        
        # Start service
        await start_shap_service()
        
        # Create mock model and data
        class MockModel:
            def predict_proba(self, X):
                # Mock prediction
                return np.random.random((len(X), 2))
                
        model = MockModel()
        training_data = np.random.random((100, 5))
        feature_names = ['rsi', 'macd', 'volume', 'volatility', 'trend_strength']
        
        # Register model
        success = await register_model_for_explanation('test_model', model, training_data, feature_names)
        print(f"Model registration: {'Success' if success else 'Failed'}")
        
        # Test prediction explanation
        test_input = {
            'rsi': 65.0,
            'macd': 0.5,
            'volume': 1.2,
            'volatility': 0.015,
            'trend_strength': 0.8
        }
        
        explanation = await explain_model_prediction('test_model', test_input)
        if explanation:
            print(f"Explanation generated: {explanation.explanation_summary}")
            print(f"Top positive features: {explanation.top_positive_features}")
            print(f"Confidence: {explanation.confidence:.2f}")
        
        # Test global explanation
        global_exp = await get_global_model_explanation('test_model')
        if global_exp:
            print(f"Global explanation: {len(global_exp.feature_importance_ranking)} features analyzed")
            print(f"Top feature: {global_exp.feature_importance_ranking[0][0]}")
        
        # Test explanation report
        report = await get_explanation_report(24)
        print(f"Explanation report: {len(report.model_explanations)} models analyzed")
        print(f"Recommendations: {len(report.recommendations)}")
        
        # Stop service
        await stop_shap_service()
        
        print("✅ SHAP Explainer test completed!")

    # Run the test
    import asyncio
    asyncio.run(test_shap_explainer())
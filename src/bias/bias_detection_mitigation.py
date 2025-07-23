"""
Bias Detection and Mitigation Framework for AI Systems
Ensures equitable outcomes across demographic groups
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from scipy import stats
import json
from datetime import datetime


class BiasType(Enum):
    DEMOGRAPHIC = "demographic"
    SOCIOECONOMIC = "socioeconomic"
    GEOGRAPHIC = "geographic"
    BEHAVIORAL = "behavioral"
    HISTORICAL = "historical"
    REPRESENTATION = "representation"
    ALGORITHMIC = "algorithmic"


class FairnessMetric(Enum):
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    DISPARATE_IMPACT = "disparate_impact"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"


class MitigationStrategy(Enum):
    PRE_PROCESSING = "pre_processing"
    IN_PROCESSING = "in_processing"
    POST_PROCESSING = "post_processing"
    ADVERSARIAL = "adversarial"
    REWEIGHTING = "reweighting"
    FAIRNESS_CONSTRAINTS = "fairness_constraints"


@dataclass
class BiasAssessment:
    """Complete bias assessment results"""
    bias_scores: Dict[BiasType, float]
    affected_groups: Dict[str, List[str]]
    fairness_metrics: Dict[FairnessMetric, float]
    recommendations: List[str]
    severity: str  # low, medium, high, critical
    confidence: float


@dataclass
class ProtectedAttribute:
    """Protected attribute definition"""
    name: str
    categories: List[str]
    reference_category: Optional[str] = None
    is_sensitive: bool = True


class BiasDetector:
    """Detect various types of bias in AI systems"""
    
    def __init__(self, variance_threshold: float = 0.05):
        self.variance_threshold = variance_threshold
        self.protected_attributes = self._define_protected_attributes()
        self.detection_history = []
    
    def _define_protected_attributes(self) -> List[ProtectedAttribute]:
        """Define protected attributes to monitor"""
        return [
            ProtectedAttribute(
                name="gender",
                categories=["male", "female", "non-binary", "other"],
                reference_category="male"
            ),
            ProtectedAttribute(
                name="age_group",
                categories=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
                reference_category="35-44"
            ),
            ProtectedAttribute(
                name="ethnicity",
                categories=["white", "black", "hispanic", "asian", "other"],
                reference_category=None  # No default reference
            ),
            ProtectedAttribute(
                name="income_level",
                categories=["low", "medium", "high"],
                reference_category="medium"
            ),
            ProtectedAttribute(
                name="education",
                categories=["high_school", "bachelors", "masters", "doctorate"],
                reference_category="bachelors"
            ),
            ProtectedAttribute(
                name="location_type",
                categories=["urban", "suburban", "rural"],
                reference_category="suburban"
            )
        ]
    
    def detect_bias(self,
                   predictions: pd.DataFrame,
                   ground_truth: Optional[pd.DataFrame] = None,
                   sensitive_features: Optional[pd.DataFrame] = None) -> BiasAssessment:
        """Comprehensive bias detection across multiple dimensions"""
        
        bias_scores = {}
        affected_groups = {}
        fairness_metrics = {}
        
        # Detect different types of bias
        if sensitive_features is not None:
            # Demographic bias
            demographic_bias = self._detect_demographic_bias(
                predictions, sensitive_features
            )
            bias_scores[BiasType.DEMOGRAPHIC] = demographic_bias["score"]
            affected_groups["demographic"] = demographic_bias["affected_groups"]
            
            # Representation bias
            representation_bias = self._detect_representation_bias(
                sensitive_features
            )
            bias_scores[BiasType.REPRESENTATION] = representation_bias["score"]
            affected_groups["representation"] = representation_bias["underrepresented"]
        
        # Algorithmic bias (can be detected without sensitive features)
        algorithmic_bias = self._detect_algorithmic_bias(predictions)
        bias_scores[BiasType.ALGORITHMIC] = algorithmic_bias["score"]
        
        # Calculate fairness metrics if ground truth available
        if ground_truth is not None and sensitive_features is not None:
            fairness_metrics = self._calculate_fairness_metrics(
                predictions, ground_truth, sensitive_features
            )
        
        # Historical bias
        if hasattr(self, 'historical_data'):
            historical_bias = self._detect_historical_bias(predictions)
            bias_scores[BiasType.HISTORICAL] = historical_bias["score"]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            bias_scores, affected_groups, fairness_metrics
        )
        
        # Calculate overall severity
        severity = self._calculate_severity(bias_scores, fairness_metrics)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            len(predictions), 
            len(sensitive_features.columns) if sensitive_features is not None else 0
        )
        
        assessment = BiasAssessment(
            bias_scores=bias_scores,
            affected_groups=affected_groups,
            fairness_metrics=fairness_metrics,
            recommendations=recommendations,
            severity=severity,
            confidence=confidence
        )
        
        # Store in history
        self.detection_history.append({
            "timestamp": datetime.now(),
            "assessment": assessment,
            "data_size": len(predictions)
        })
        
        return assessment
    
    def _detect_demographic_bias(self,
                                predictions: pd.DataFrame,
                                sensitive_features: pd.DataFrame) -> Dict:
        """Detect bias across demographic groups"""
        affected_groups = []
        max_bias_score = 0
        
        for attribute in self.protected_attributes:
            if attribute.name in sensitive_features.columns:
                # Group predictions by attribute
                grouped = predictions.groupby(sensitive_features[attribute.name])
                
                # Calculate metrics per group
                group_metrics = {}
                for group, group_preds in grouped:
                    if 'score' in group_preds.columns:
                        group_metrics[group] = {
                            'mean': group_preds['score'].mean(),
                            'std': group_preds['score'].std(),
                            'count': len(group_preds)
                        }
                
                # Check for significant differences
                if len(group_metrics) > 1:
                    means = [m['mean'] for m in group_metrics.values()]
                    max_diff = max(means) - min(means)
                    avg_mean = np.mean(means)
                    
                    if avg_mean > 0:
                        bias_score = max_diff / avg_mean
                        
                        if bias_score > self.variance_threshold:
                            affected_groups.append(f"{attribute.name}: {max_diff:.3f} difference")
                            max_bias_score = max(max_bias_score, bias_score)
        
        return {
            "score": max_bias_score,
            "affected_groups": affected_groups
        }
    
    def _detect_representation_bias(self, sensitive_features: pd.DataFrame) -> Dict:
        """Detect underrepresentation of groups"""
        underrepresented = []
        total_samples = len(sensitive_features)
        
        for attribute in self.protected_attributes:
            if attribute.name in sensitive_features.columns:
                value_counts = sensitive_features[attribute.name].value_counts()
                
                for category in attribute.categories:
                    if category in value_counts:
                        proportion = value_counts[category] / total_samples
                        expected_proportion = 1.0 / len(attribute.categories)
                        
                        # Check if significantly underrepresented
                        if proportion < expected_proportion * 0.5:
                            underrepresented.append(
                                f"{attribute.name}={category} ({proportion:.1%})"
                            )
        
        bias_score = len(underrepresented) / len(self.protected_attributes)
        
        return {
            "score": bias_score,
            "underrepresented": underrepresented
        }
    
    def _detect_algorithmic_bias(self, predictions: pd.DataFrame) -> Dict:
        """Detect patterns suggesting algorithmic bias"""
        bias_indicators = 0
        
        if 'score' in predictions.columns:
            scores = predictions['score']
            
            # Check for unusual score distributions
            # 1. Extreme clustering
            unique_scores = len(scores.unique())
            if unique_scores < len(scores) * 0.1:  # Less than 10% unique values
                bias_indicators += 1
            
            # 2. Systematic patterns
            # Check for repeating patterns in scores
            if len(scores) > 10:
                diffs = np.diff(sorted(scores))
                if np.std(diffs) < 0.01:  # Very regular intervals
                    bias_indicators += 1
            
            # 3. Range restriction
            score_range = scores.max() - scores.min()
            if score_range < 0.1:  # Very narrow range
                bias_indicators += 1
        
        return {
            "score": bias_indicators / 3.0,
            "indicators": bias_indicators
        }
    
    def _detect_historical_bias(self, current_predictions: pd.DataFrame) -> Dict:
        """Detect bias from historical patterns"""
        # This would compare current predictions to historical data
        # Simplified implementation
        return {"score": 0.0}
    
    def _calculate_fairness_metrics(self,
                                  predictions: pd.DataFrame,
                                  ground_truth: pd.DataFrame,
                                  sensitive_features: pd.DataFrame) -> Dict:
        """Calculate various fairness metrics"""
        metrics = {}
        
        # Ensure we have binary predictions and ground truth
        if 'prediction' in predictions.columns and 'label' in ground_truth.columns:
            y_pred = predictions['prediction']
            y_true = ground_truth['label']
            
            for attribute in self.protected_attributes:
                if attribute.name in sensitive_features.columns:
                    groups = sensitive_features[attribute.name]
                    
                    # Demographic Parity
                    dp_score = self._calculate_demographic_parity(y_pred, groups)
                    metrics[FairnessMetric.DEMOGRAPHIC_PARITY] = dp_score
                    
                    # Equal Opportunity
                    eo_score = self._calculate_equal_opportunity(y_pred, y_true, groups)
                    metrics[FairnessMetric.EQUAL_OPPORTUNITY] = eo_score
                    
                    # Disparate Impact
                    di_score = self._calculate_disparate_impact(y_pred, groups)
                    metrics[FairnessMetric.DISPARATE_IMPACT] = di_score
        
        return metrics
    
    def _calculate_demographic_parity(self,
                                    predictions: pd.Series,
                                    groups: pd.Series) -> float:
        """Calculate demographic parity difference"""
        positive_rates = predictions.groupby(groups).mean()
        
        if len(positive_rates) > 1:
            max_rate = positive_rates.max()
            min_rate = positive_rates.min()
            return max_rate - min_rate
        
        return 0.0
    
    def _calculate_equal_opportunity(self,
                                   predictions: pd.Series,
                                   ground_truth: pd.Series,
                                   groups: pd.Series) -> float:
        """Calculate equal opportunity difference"""
        # True positive rates per group
        tpr_per_group = {}
        
        for group in groups.unique():
            group_mask = groups == group
            group_positives = ground_truth[group_mask] == 1
            
            if group_positives.sum() > 0:
                tpr = (predictions[group_mask] & ground_truth[group_mask]).sum() / group_positives.sum()
                tpr_per_group[group] = tpr
        
        if len(tpr_per_group) > 1:
            tprs = list(tpr_per_group.values())
            return max(tprs) - min(tprs)
        
        return 0.0
    
    def _calculate_disparate_impact(self,
                                  predictions: pd.Series,
                                  groups: pd.Series) -> float:
        """Calculate disparate impact ratio"""
        positive_rates = predictions.groupby(groups).mean()
        
        if len(positive_rates) > 1:
            min_rate = positive_rates.min()
            max_rate = positive_rates.max()
            
            if max_rate > 0:
                return min_rate / max_rate
        
        return 1.0
    
    def _generate_recommendations(self,
                                bias_scores: Dict,
                                affected_groups: Dict,
                                fairness_metrics: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check demographic bias
        if bias_scores.get(BiasType.DEMOGRAPHIC, 0) > self.variance_threshold:
            recommendations.append(
                "Apply demographic reweighting to balance predictions across groups"
            )
            
            if affected_groups.get("demographic"):
                recommendations.append(
                    f"Focus on improving performance for: {', '.join(affected_groups['demographic'][:3])}"
                )
        
        # Check representation bias
        if bias_scores.get(BiasType.REPRESENTATION, 0) > 0.3:
            recommendations.append(
                "Increase data collection for underrepresented groups"
            )
            recommendations.append(
                "Consider synthetic data generation for balance"
            )
        
        # Check fairness metrics
        if fairness_metrics:
            if fairness_metrics.get(FairnessMetric.DISPARATE_IMPACT, 1) < 0.8:
                recommendations.append(
                    "Implement fairness constraints during model training"
                )
            
            if fairness_metrics.get(FairnessMetric.EQUAL_OPPORTUNITY, 0) > 0.1:
                recommendations.append(
                    "Adjust decision thresholds per group for equal opportunity"
                )
        
        # Check algorithmic bias
        if bias_scores.get(BiasType.ALGORITHMIC, 0) > 0.5:
            recommendations.append(
                "Review model architecture for systematic biases"
            )
            recommendations.append(
                "Implement adversarial debiasing techniques"
            )
        
        if not recommendations:
            recommendations.append("Continue monitoring for bias emergence")
        
        return recommendations
    
    def _calculate_severity(self,
                          bias_scores: Dict,
                          fairness_metrics: Dict) -> str:
        """Calculate overall bias severity"""
        max_bias = max(bias_scores.values()) if bias_scores else 0
        
        # Check fairness violations
        fairness_violations = 0
        if fairness_metrics:
            if fairness_metrics.get(FairnessMetric.DISPARATE_IMPACT, 1) < 0.8:
                fairness_violations += 1
            if fairness_metrics.get(FairnessMetric.DEMOGRAPHIC_PARITY, 0) > 0.1:
                fairness_violations += 1
            if fairness_metrics.get(FairnessMetric.EQUAL_OPPORTUNITY, 0) > 0.1:
                fairness_violations += 1
        
        # Determine severity
        if max_bias > 0.3 or fairness_violations >= 2:
            return "critical"
        elif max_bias > 0.2 or fairness_violations >= 1:
            return "high"
        elif max_bias > 0.1:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence(self, 
                            sample_size: int,
                            num_attributes: int) -> float:
        """Calculate confidence in bias detection"""
        # Base confidence on sample size
        sample_confidence = min(1.0, sample_size / 1000)
        
        # Adjust for attribute coverage
        attribute_confidence = min(1.0, num_attributes / len(self.protected_attributes))
        
        return sample_confidence * 0.7 + attribute_confidence * 0.3


class BiasMitigator:
    """Mitigate detected biases using various strategies"""
    
    def __init__(self):
        self.mitigation_history = []
    
    def mitigate(self,
                data: pd.DataFrame,
                bias_assessment: BiasAssessment,
                strategy: MitigationStrategy,
                sensitive_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply bias mitigation strategy"""
        
        if strategy == MitigationStrategy.REWEIGHTING:
            return self._apply_reweighting(data, sensitive_features)
        elif strategy == MitigationStrategy.POST_PROCESSING:
            return self._apply_post_processing(data, bias_assessment, sensitive_features)
        elif strategy == MitigationStrategy.ADVERSARIAL:
            return self._apply_adversarial_debiasing(data, sensitive_features)
        else:
            return data
    
    def _apply_reweighting(self,
                         data: pd.DataFrame,
                         sensitive_features: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Apply sample reweighting for fairness"""
        if sensitive_features is None or 'score' not in data.columns:
            return data
        
        weighted_data = data.copy()
        weights = np.ones(len(data))
        
        # Calculate weights to balance groups
        for column in sensitive_features.columns:
            if column in data.columns or column in sensitive_features.columns:
                groups = sensitive_features[column] if column in sensitive_features.columns else data[column]
                group_counts = groups.value_counts()
                
                # Weight inversely proportional to group size
                for group, count in group_counts.items():
                    group_mask = groups == group
                    weights[group_mask] *= len(groups) / (len(group_counts) * count)
        
        weighted_data['weight'] = weights
        
        # Adjust scores based on weights
        if 'score' in weighted_data.columns:
            weighted_data['adjusted_score'] = weighted_data['score'] * np.sqrt(weights)
            # Normalize to original range
            original_range = data['score'].max() - data['score'].min()
            new_range = weighted_data['adjusted_score'].max() - weighted_data['adjusted_score'].min()
            if new_range > 0:
                weighted_data['adjusted_score'] = (
                    weighted_data['adjusted_score'] - weighted_data['adjusted_score'].min()
                ) * original_range / new_range + data['score'].min()
        
        return weighted_data
    
    def _apply_post_processing(self,
                             data: pd.DataFrame,
                             bias_assessment: BiasAssessment,
                             sensitive_features: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Apply post-processing adjustments"""
        if sensitive_features is None or 'score' not in data.columns:
            return data
        
        adjusted_data = data.copy()
        
        # Adjust thresholds per group for equal opportunity
        if FairnessMetric.EQUAL_OPPORTUNITY in bias_assessment.fairness_metrics:
            for column in sensitive_features.columns:
                if column in sensitive_features.columns:
                    groups = sensitive_features[column]
                    
                    # Calculate group-specific thresholds
                    group_thresholds = {}
                    overall_threshold = data['score'].median()
                    
                    for group in groups.unique():
                        group_mask = groups == group
                        group_scores = data.loc[group_mask, 'score']
                        
                        # Adjust threshold to achieve similar positive rates
                        group_threshold = np.percentile(group_scores, 50)
                        group_thresholds[group] = group_threshold
                    
                    # Apply adjusted thresholds
                    if 'prediction' in adjusted_data.columns:
                        for group, threshold in group_thresholds.items():
                            group_mask = groups == group
                            adjusted_data.loc[group_mask, 'adjusted_prediction'] = (
                                adjusted_data.loc[group_mask, 'score'] > threshold
                            )
        
        return adjusted_data
    
    def _apply_adversarial_debiasing(self,
                                    data: pd.DataFrame,
                                    sensitive_features: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Apply adversarial debiasing (simplified)"""
        if sensitive_features is None or 'score' not in data.columns:
            return data
        
        debiased_data = data.copy()
        
        # Add noise to break correlation with sensitive attributes
        for column in sensitive_features.columns:
            if column in sensitive_features.columns:
                # Calculate correlation with sensitive attribute
                groups = sensitive_features[column]
                
                # Add controlled noise inversely proportional to group size
                for group in groups.unique():
                    group_mask = groups == group
                    group_size = group_mask.sum()
                    
                    # Smaller groups get less noise to preserve signal
                    noise_scale = 0.05 * (1 - group_size / len(data))
                    noise = np.random.normal(0, noise_scale, group_size)
                    
                    if 'score' in debiased_data.columns:
                        debiased_data.loc[group_mask, 'debiased_score'] = (
                            debiased_data.loc[group_mask, 'score'] + noise
                        )
        
        # Ensure scores remain in valid range
        if 'debiased_score' in debiased_data.columns:
            debiased_data['debiased_score'] = np.clip(
                debiased_data['debiased_score'],
                data['score'].min(),
                data['score'].max()
            )
        
        return debiased_data
    
    def select_best_strategy(self,
                           bias_assessment: BiasAssessment) -> MitigationStrategy:
        """Select optimal mitigation strategy based on bias type"""
        severity = bias_assessment.severity
        primary_bias = max(bias_assessment.bias_scores.items(), key=lambda x: x[1])[0]
        
        # Strategy selection logic
        if severity == "critical":
            # For critical bias, use strongest mitigation
            if primary_bias == BiasType.DEMOGRAPHIC:
                return MitigationStrategy.FAIRNESS_CONSTRAINTS
            else:
                return MitigationStrategy.ADVERSARIAL
        
        elif severity == "high":
            if primary_bias == BiasType.REPRESENTATION:
                return MitigationStrategy.REWEIGHTING
            else:
                return MitigationStrategy.POST_PROCESSING
        
        else:
            # For low/medium bias, use lighter touch
            return MitigationStrategy.POST_PROCESSING


class BiasMonitor:
    """Continuous monitoring of bias in production"""
    
    def __init__(self, alert_threshold: float = 0.1):
        self.alert_threshold = alert_threshold
        self.monitoring_history = []
        self.baseline_metrics = None
    
    def set_baseline(self, initial_assessment: BiasAssessment):
        """Set baseline metrics for monitoring"""
        self.baseline_metrics = {
            "bias_scores": initial_assessment.bias_scores.copy(),
            "fairness_metrics": initial_assessment.fairness_metrics.copy(),
            "timestamp": datetime.now()
        }
    
    def monitor(self,
               current_assessment: BiasAssessment) -> Dict[str, Any]:
        """Monitor for bias drift and anomalies"""
        alerts = []
        metrics_comparison = {}
        
        if self.baseline_metrics:
            # Compare to baseline
            for bias_type, current_score in current_assessment.bias_scores.items():
                baseline_score = self.baseline_metrics["bias_scores"].get(bias_type, 0)
                drift = abs(current_score - baseline_score)
                
                metrics_comparison[bias_type.value] = {
                    "current": current_score,
                    "baseline": baseline_score,
                    "drift": drift
                }
                
                if drift > self.alert_threshold:
                    alerts.append({
                        "type": "bias_drift",
                        "bias_type": bias_type.value,
                        "drift": drift,
                        "severity": "high" if drift > 0.2 else "medium"
                    })
        
        # Check for new bias emergence
        if current_assessment.severity in ["high", "critical"]:
            alerts.append({
                "type": "high_bias_detected",
                "severity": current_assessment.severity,
                "affected_groups": current_assessment.affected_groups
            })
        
        # Store monitoring record
        monitoring_record = {
            "timestamp": datetime.now(),
            "assessment": current_assessment,
            "alerts": alerts,
            "metrics_comparison": metrics_comparison
        }
        
        self.monitoring_history.append(monitoring_record)
        
        return {
            "alerts": alerts,
            "metrics_comparison": metrics_comparison,
            "requires_intervention": len(alerts) > 0,
            "suggested_actions": self._suggest_actions(alerts)
        }
    
    def _suggest_actions(self, alerts: List[Dict]) -> List[str]:
        """Suggest actions based on alerts"""
        actions = []
        
        for alert in alerts:
            if alert["type"] == "bias_drift":
                actions.append(
                    f"Investigate {alert['bias_type']} bias drift of {alert['drift']:.2f}"
                )
                if alert["severity"] == "high":
                    actions.append("Consider retraining with updated data")
            
            elif alert["type"] == "high_bias_detected":
                actions.append("Immediate bias mitigation required")
                actions.append("Review recent model or data changes")
        
        return list(set(actions))  # Remove duplicates
    
    def generate_bias_report(self) -> Dict:
        """Generate comprehensive bias monitoring report"""
        if not self.monitoring_history:
            return {"status": "No monitoring data available"}
        
        # Analyze trends
        bias_trends = {}
        for record in self.monitoring_history:
            for bias_type, score in record["assessment"].bias_scores.items():
                if bias_type not in bias_trends:
                    bias_trends[bias_type] = []
                bias_trends[bias_type].append({
                    "timestamp": record["timestamp"],
                    "score": score
                })
        
        # Calculate statistics
        report = {
            "monitoring_period": {
                "start": self.monitoring_history[0]["timestamp"],
                "end": self.monitoring_history[-1]["timestamp"],
                "num_assessments": len(self.monitoring_history)
            },
            "bias_trends": bias_trends,
            "total_alerts": sum(len(r["alerts"]) for r in self.monitoring_history),
            "high_severity_incidents": sum(
                1 for r in self.monitoring_history 
                if r["assessment"].severity in ["high", "critical"]
            ),
            "recommendations": self._generate_report_recommendations()
        }
        
        return report
    
    def _generate_report_recommendations(self) -> List[str]:
        """Generate recommendations based on monitoring history"""
        recommendations = []
        
        # Check for persistent bias
        recent_assessments = self.monitoring_history[-5:] if len(self.monitoring_history) >= 5 else self.monitoring_history
        high_bias_count = sum(
            1 for r in recent_assessments 
            if r["assessment"].severity in ["high", "critical"]
        )
        
        if high_bias_count >= 3:
            recommendations.append(
                "Persistent high bias detected - comprehensive model review recommended"
            )
        
        # Check for increasing trends
        if len(self.monitoring_history) >= 10:
            for bias_type in BiasType:
                scores = [
                    r["assessment"].bias_scores.get(bias_type, 0) 
                    for r in self.monitoring_history[-10:]
                ]
                if len(scores) >= 10:
                    trend = np.polyfit(range(len(scores)), scores, 1)[0]
                    if trend > 0.01:  # Increasing trend
                        recommendations.append(
                            f"Increasing {bias_type.value} bias trend detected"
                        )
        
        if not recommendations:
            recommendations.append("System operating within acceptable bias limits")
        
        return recommendations


# Example usage
def main():
    """Example bias detection and mitigation usage"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create predictions with bias
    predictions = pd.DataFrame({
        'score': np.random.beta(2, 5, n_samples),
        'prediction': np.random.binomial(1, 0.3, n_samples)
    })
    
    # Create sensitive features
    sensitive_features = pd.DataFrame({
        'gender': np.random.choice(['male', 'female', 'non-binary'], n_samples, p=[0.45, 0.45, 0.1]),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], n_samples),
        'income_level': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.3, 0.5, 0.2])
    })
    
    # Add bias to predictions based on sensitive features
    predictions.loc[sensitive_features['gender'] == 'female', 'score'] *= 0.8
    predictions.loc[sensitive_features['age_group'] == '65+', 'score'] *= 0.7
    
    # Create ground truth
    ground_truth = pd.DataFrame({
        'label': np.random.binomial(1, predictions['score'].values)
    })
    
    # Detect bias
    detector = BiasDetector()
    assessment = detector.detect_bias(predictions, ground_truth, sensitive_features)
    
    print("Bias Detection Results:")
    print(f"Severity: {assessment.severity}")
    print(f"Bias Scores:")
    for bias_type, score in assessment.bias_scores.items():
        print(f"  {bias_type.value}: {score:.3f}")
    print(f"Affected Groups: {assessment.affected_groups}")
    print(f"Recommendations:")
    for rec in assessment.recommendations:
        print(f"  - {rec}")
    
    # Apply mitigation
    mitigator = BiasMitigator()
    strategy = mitigator.select_best_strategy(assessment)
    print(f"\nSelected Mitigation Strategy: {strategy.value}")
    
    mitigated_data = mitigator.mitigate(
        predictions, assessment, strategy, sensitive_features
    )
    
    # Re-assess after mitigation
    if 'adjusted_score' in mitigated_data.columns:
        mitigated_predictions = mitigated_data[['adjusted_score']].rename(
            columns={'adjusted_score': 'score'}
        )
        new_assessment = detector.detect_bias(
            mitigated_predictions, ground_truth, sensitive_features
        )
        
        print(f"\nPost-Mitigation Severity: {new_assessment.severity}")
    
    # Set up monitoring
    monitor = BiasMonitor()
    monitor.set_baseline(assessment)
    
    # Simulate monitoring over time
    monitoring_result = monitor.monitor(assessment)
    print(f"\nMonitoring Alerts: {len(monitoring_result['alerts'])}")
    
    # Generate report
    report = monitor.generate_bias_report()
    print(f"\nBias Monitoring Report:")
    print(f"Total Alerts: {report['total_alerts']}")
    print(f"High Severity Incidents: {report['high_severity_incidents']}")


if __name__ == "__main__":
    main()
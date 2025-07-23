"""
Multimodal AI System for Advertising
Integrates visual, audio, text, and contextual data for unified intelligence
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import base64
from PIL import Image
import io
import json
from datetime import datetime


class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CONTEXTUAL = "contextual"
    BEHAVIORAL = "behavioral"
    SENSOR = "sensor"


class FusionStrategy(Enum):
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    HYBRID_FUSION = "hybrid_fusion"
    ATTENTION_FUSION = "attention_fusion"
    HIERARCHICAL_FUSION = "hierarchical_fusion"


@dataclass
class MultimodalInput:
    """Container for multimodal input data"""
    modality: ModalityType
    data: Any
    metadata: Dict
    timestamp: datetime
    confidence: float = 1.0
    
    def preprocess(self) -> Any:
        """Preprocess data based on modality"""
        if self.modality == ModalityType.TEXT:
            return self._preprocess_text()
        elif self.modality == ModalityType.IMAGE:
            return self._preprocess_image()
        elif self.modality == ModalityType.VIDEO:
            return self._preprocess_video()
        elif self.modality == ModalityType.AUDIO:
            return self._preprocess_audio()
        elif self.modality == ModalityType.CONTEXTUAL:
            return self._preprocess_contextual()
        else:
            return self.data
    
    def _preprocess_text(self) -> Dict:
        """Preprocess text data"""
        text = str(self.data)
        return {
            "original": text,
            "tokens": text.split(),
            "length": len(text),
            "language": self.metadata.get("language", "en")
        }
    
    def _preprocess_image(self) -> Dict:
        """Preprocess image data"""
        if isinstance(self.data, str):
            # Base64 encoded image
            image_data = base64.b64decode(self.data)
            image = Image.open(io.BytesIO(image_data))
        else:
            image = self.data
        
        return {
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
            "array": np.array(image)
        }
    
    def _preprocess_video(self) -> Dict:
        """Preprocess video data"""
        return {
            "duration": self.metadata.get("duration", 0),
            "fps": self.metadata.get("fps", 30),
            "resolution": self.metadata.get("resolution", (1920, 1080)),
            "frames": self.metadata.get("key_frames", [])
        }
    
    def _preprocess_audio(self) -> Dict:
        """Preprocess audio data"""
        return {
            "duration": self.metadata.get("duration", 0),
            "sample_rate": self.metadata.get("sample_rate", 44100),
            "channels": self.metadata.get("channels", 2),
            "format": self.metadata.get("format", "wav")
        }
    
    def _preprocess_contextual(self) -> Dict:
        """Preprocess contextual data"""
        return {
            "location": self.metadata.get("location"),
            "weather": self.metadata.get("weather"),
            "time_of_day": self.metadata.get("time_of_day"),
            "device": self.metadata.get("device"),
            "user_context": self.metadata.get("user_context", {})
        }


class ModalityEncoder:
    """Encode different modalities into unified representation"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.encoders = self._initialize_encoders()
    
    def _initialize_encoders(self) -> Dict:
        """Initialize modality-specific encoders"""
        return {
            ModalityType.TEXT: self._create_text_encoder(),
            ModalityType.IMAGE: self._create_image_encoder(),
            ModalityType.VIDEO: self._create_video_encoder(),
            ModalityType.AUDIO: self._create_audio_encoder(),
            ModalityType.CONTEXTUAL: self._create_contextual_encoder()
        }
    
    def encode(self, input_data: MultimodalInput) -> np.ndarray:
        """Encode input data into embedding"""
        preprocessed = input_data.preprocess()
        encoder = self.encoders.get(input_data.modality)
        
        if encoder:
            return encoder(preprocessed)
        else:
            # Fallback to random embedding
            return np.random.randn(self.embedding_dim)
    
    def _create_text_encoder(self):
        """Create text encoder"""
        def encode_text(data: Dict) -> np.ndarray:
            # Simplified text encoding - in production use BERT/GPT
            text_length = data["length"]
            num_tokens = len(data["tokens"])
            
            # Create feature vector
            features = np.zeros(self.embedding_dim)
            features[0] = text_length / 1000  # Normalized length
            features[1] = num_tokens / 100    # Normalized token count
            
            # Simulate semantic embedding
            for i, token in enumerate(data["tokens"][:10]):
                features[i+10] = hash(token) % 100 / 100
            
            return features
        
        return encode_text
    
    def _create_image_encoder(self):
        """Create image encoder"""
        def encode_image(data: Dict) -> np.ndarray:
            # Simplified image encoding - in production use ResNet/ViT
            image_array = data["array"]
            
            # Extract basic features
            features = np.zeros(self.embedding_dim)
            
            # Color histogram features
            if len(image_array.shape) == 3:
                for i in range(3):  # RGB channels
                    hist, _ = np.histogram(image_array[:,:,i], bins=16)
                    features[i*16:(i+1)*16] = hist / hist.sum()
            
            # Texture features (simplified)
            features[48] = np.std(image_array)  # Texture complexity
            features[49] = np.mean(image_array)  # Brightness
            
            # Size features
            features[50] = data["size"][0] / 1920  # Normalized width
            features[51] = data["size"][1] / 1080  # Normalized height
            
            return features
        
        return encode_image
    
    def _create_video_encoder(self):
        """Create video encoder"""
        def encode_video(data: Dict) -> np.ndarray:
            # Simplified video encoding
            features = np.zeros(self.embedding_dim)
            
            features[0] = data["duration"] / 60  # Normalized duration
            features[1] = data["fps"] / 60       # Normalized FPS
            features[2] = data["resolution"][0] / 1920
            features[3] = data["resolution"][1] / 1080
            
            # Simulate motion features
            features[10:20] = np.random.randn(10) * 0.1
            
            return features
        
        return encode_video
    
    def _create_audio_encoder(self):
        """Create audio encoder"""
        def encode_audio(data: Dict) -> np.ndarray:
            # Simplified audio encoding
            features = np.zeros(self.embedding_dim)
            
            features[0] = data["duration"] / 60
            features[1] = data["sample_rate"] / 48000
            features[2] = data["channels"] / 2
            
            # Simulate audio features (rhythm, pitch, etc.)
            features[10:30] = np.random.randn(20) * 0.1
            
            return features
        
        return encode_audio
    
    def _create_contextual_encoder(self):
        """Create contextual encoder"""
        def encode_context(data: Dict) -> np.ndarray:
            features = np.zeros(self.embedding_dim)
            
            # Encode various contextual signals
            if data.get("weather"):
                features[0] = hash(data["weather"]) % 10 / 10
            
            if data.get("time_of_day"):
                features[1] = data["time_of_day"] / 24
            
            if data.get("device"):
                features[2] = hash(data["device"]) % 10 / 10
            
            return features
        
        return encode_context


class MultimodalFusion:
    """Fuse multiple modalities into unified representation"""
    
    def __init__(self, fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION):
        self.fusion_strategy = fusion_strategy
        self.attention_weights = None
    
    def fuse(self, embeddings: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Fuse modality embeddings based on strategy"""
        if self.fusion_strategy == FusionStrategy.EARLY_FUSION:
            return self._early_fusion(embeddings)
        elif self.fusion_strategy == FusionStrategy.LATE_FUSION:
            return self._late_fusion(embeddings)
        elif self.fusion_strategy == FusionStrategy.HYBRID_FUSION:
            return self._hybrid_fusion(embeddings)
        elif self.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            return self._attention_fusion(embeddings)
        elif self.fusion_strategy == FusionStrategy.HIERARCHICAL_FUSION:
            return self._hierarchical_fusion(embeddings)
        else:
            return self._early_fusion(embeddings)
    
    def _early_fusion(self, embeddings: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Concatenate all embeddings"""
        all_embeddings = []
        for modality in sorted(embeddings.keys(), key=lambda x: x.value):
            all_embeddings.append(embeddings[modality])
        
        return np.concatenate(all_embeddings)
    
    def _late_fusion(self, embeddings: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Average embeddings with equal weights"""
        if not embeddings:
            return np.array([])
        
        stacked = np.stack(list(embeddings.values()))
        return np.mean(stacked, axis=0)
    
    def _hybrid_fusion(self, embeddings: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Combine early and late fusion"""
        early = self._early_fusion(embeddings)
        late = self._late_fusion(embeddings)
        
        # Reduce early fusion dimension
        early_reduced = early[:len(late)]
        
        # Combine with weights
        return 0.3 * early_reduced + 0.7 * late
    
    def _attention_fusion(self, embeddings: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Use attention mechanism for fusion"""
        if not embeddings:
            return np.array([])
        
        # Calculate attention weights based on embedding magnitudes
        magnitudes = {k: np.linalg.norm(v) for k, v in embeddings.items()}
        total_magnitude = sum(magnitudes.values())
        
        if total_magnitude > 0:
            attention_weights = {k: v / total_magnitude for k, v in magnitudes.items()}
        else:
            attention_weights = {k: 1.0 / len(embeddings) for k in embeddings.keys()}
        
        self.attention_weights = attention_weights
        
        # Apply attention weights
        weighted_embeddings = []
        for modality, embedding in embeddings.items():
            weighted = embedding * attention_weights[modality]
            weighted_embeddings.append(weighted)
        
        return np.sum(weighted_embeddings, axis=0)
    
    def _hierarchical_fusion(self, embeddings: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Hierarchical fusion based on modality importance"""
        # Define hierarchy
        hierarchy = {
            "primary": [ModalityType.IMAGE, ModalityType.VIDEO],
            "secondary": [ModalityType.TEXT, ModalityType.AUDIO],
            "tertiary": [ModalityType.CONTEXTUAL, ModalityType.BEHAVIORAL]
        }
        
        fused_levels = []
        
        for level, modalities in hierarchy.items():
            level_embeddings = {m: embeddings[m] for m in modalities if m in embeddings}
            if level_embeddings:
                level_fused = self._late_fusion(level_embeddings)
                fused_levels.append(level_fused)
        
        if not fused_levels:
            return np.array([])
        
        # Weight by hierarchy level
        weights = [0.5, 0.3, 0.2][:len(fused_levels)]
        weighted_sum = sum(w * e for w, e in zip(weights, fused_levels))
        
        return weighted_sum / sum(weights)


class MultimodalAdvertisingAnalyzer:
    """Analyze advertising content using multimodal AI"""
    
    def __init__(self):
        self.encoder = ModalityEncoder()
        self.fusion = MultimodalFusion(FusionStrategy.ATTENTION_FUSION)
        self.analysis_history = []
    
    async def analyze_ad_creative(self, 
                                 creative_inputs: List[MultimodalInput]) -> Dict:
        """Analyze advertising creative across modalities"""
        # Encode each modality
        embeddings = {}
        for input_data in creative_inputs:
            embedding = self.encoder.encode(input_data)
            embeddings[input_data.modality] = embedding
        
        # Fuse modalities
        fused_representation = self.fusion.fuse(embeddings)
        
        # Analyze fused representation
        analysis = {
            "emotional_impact": self._analyze_emotional_impact(fused_representation),
            "brand_alignment": self._analyze_brand_alignment(fused_representation),
            "attention_score": self._calculate_attention_score(embeddings),
            "memorability": self._predict_memorability(fused_representation),
            "engagement_prediction": self._predict_engagement(fused_representation)
        }
        
        # Add modality-specific insights
        modality_insights = {}
        for modality, embedding in embeddings.items():
            modality_insights[modality.value] = self._get_modality_insights(
                modality, embedding
            )
        
        analysis["modality_insights"] = modality_insights
        
        # Store analysis
        self.analysis_history.append({
            "timestamp": datetime.now(),
            "num_modalities": len(creative_inputs),
            "fusion_weights": self.fusion.attention_weights,
            "overall_score": np.mean(list(analysis.values())[:5])
        })
        
        return analysis
    
    def _analyze_emotional_impact(self, representation: np.ndarray) -> float:
        """Analyze emotional impact of creative"""
        # Simplified emotion analysis
        emotion_features = representation[100:150]  # Designated emotion features
        emotion_score = np.tanh(np.mean(np.abs(emotion_features)) * 2)
        return float(emotion_score)
    
    def _analyze_brand_alignment(self, representation: np.ndarray) -> float:
        """Analyze brand alignment score"""
        # Simplified brand alignment
        brand_features = representation[150:200]
        alignment_score = 1 / (1 + np.exp(-np.mean(brand_features)))
        return float(alignment_score)
    
    def _calculate_attention_score(self, embeddings: Dict) -> float:
        """Calculate predicted attention score"""
        scores = []
        
        # Visual modalities get higher weight for attention
        visual_weight = 0.6
        audio_weight = 0.3
        other_weight = 0.1
        
        for modality, embedding in embeddings.items():
            if modality in [ModalityType.IMAGE, ModalityType.VIDEO]:
                score = np.mean(np.abs(embedding[:50])) * visual_weight
            elif modality == ModalityType.AUDIO:
                score = np.mean(np.abs(embedding[:30])) * audio_weight
            else:
                score = np.mean(np.abs(embedding[:20])) * other_weight
            
            scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _predict_memorability(self, representation: np.ndarray) -> float:
        """Predict ad memorability"""
        # Factors: uniqueness, repetition, emotional connection
        uniqueness = np.std(representation)
        repetition_pattern = np.mean(np.abs(np.diff(representation)))
        
        memorability = (uniqueness * 0.4 + (1 - repetition_pattern) * 0.6)
        return float(np.clip(memorability, 0, 1))
    
    def _predict_engagement(self, representation: np.ndarray) -> float:
        """Predict engagement rate"""
        # Simplified engagement prediction
        engagement_features = representation[200:250]
        engagement_score = 1 / (1 + np.exp(-np.sum(engagement_features) / 10))
        return float(engagement_score)
    
    def _get_modality_insights(self, 
                              modality: ModalityType,
                              embedding: np.ndarray) -> Dict:
        """Get modality-specific insights"""
        insights = {
            "contribution_score": float(np.linalg.norm(embedding)),
            "complexity": float(np.std(embedding)),
            "dominant_features": self._extract_dominant_features(embedding)
        }
        
        # Modality-specific insights
        if modality == ModalityType.TEXT:
            insights["readability_score"] = float(1 - np.mean(np.abs(embedding[10:20])))
            insights["sentiment"] = "positive" if np.mean(embedding) > 0 else "negative"
        
        elif modality == ModalityType.IMAGE:
            insights["visual_complexity"] = float(np.std(embedding[:48]))
            insights["color_harmony"] = float(1 - np.std(embedding[:48]))
        
        elif modality == ModalityType.VIDEO:
            insights["pacing_score"] = float(np.mean(embedding[10:20]))
            insights["dynamic_range"] = float(np.ptp(embedding))
        
        elif modality == ModalityType.AUDIO:
            insights["audio_clarity"] = float(1 - np.std(embedding[10:30]))
            insights["rhythm_score"] = float(np.mean(np.abs(embedding[10:30])))
        
        return insights
    
    def _extract_dominant_features(self, embedding: np.ndarray) -> List[int]:
        """Extract indices of dominant features"""
        # Get top 5 feature indices
        top_indices = np.argsort(np.abs(embedding))[-5:][::-1]
        return top_indices.tolist()


class MultimodalCampaignOptimizer:
    """Optimize campaigns using multimodal insights"""
    
    def __init__(self):
        self.analyzer = MultimodalAdvertisingAnalyzer()
        self.optimization_history = []
    
    async def optimize_creative_mix(self,
                                   creative_variants: List[List[MultimodalInput]],
                                   target_metrics: Dict) -> Dict:
        """Optimize creative mix based on multimodal analysis"""
        variant_analyses = []
        
        # Analyze each variant
        for i, variant in enumerate(creative_variants):
            analysis = await self.analyzer.analyze_ad_creative(variant)
            variant_analyses.append({
                "variant_id": i,
                "analysis": analysis,
                "predicted_performance": self._calculate_performance_score(
                    analysis, target_metrics
                )
            })
        
        # Rank variants
        ranked_variants = sorted(
            variant_analyses,
            key=lambda x: x["predicted_performance"],
            reverse=True
        )
        
        # Generate optimization recommendations
        recommendations = self._generate_recommendations(
            ranked_variants, target_metrics
        )
        
        return {
            "top_variant": ranked_variants[0]["variant_id"],
            "variant_rankings": ranked_variants,
            "recommendations": recommendations,
            "optimal_mix": self._calculate_optimal_mix(ranked_variants)
        }
    
    def _calculate_performance_score(self, 
                                   analysis: Dict,
                                   target_metrics: Dict) -> float:
        """Calculate performance score based on targets"""
        weights = {
            "emotional_impact": target_metrics.get("emotion_weight", 0.2),
            "brand_alignment": target_metrics.get("brand_weight", 0.2),
            "attention_score": target_metrics.get("attention_weight", 0.3),
            "memorability": target_metrics.get("memory_weight", 0.15),
            "engagement_prediction": target_metrics.get("engagement_weight", 0.15)
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric in analysis:
                score += analysis[metric] * weight
        
        return score
    
    def _generate_recommendations(self,
                                ranked_variants: List[Dict],
                                target_metrics: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        top_variant = ranked_variants[0]["analysis"]
        
        # Check for improvement areas
        if top_variant["emotional_impact"] < 0.7:
            recommendations.append(
                "Increase emotional resonance through storytelling or music"
            )
        
        if top_variant["attention_score"] < 0.6:
            recommendations.append(
                "Enhance visual contrast and movement in first 3 seconds"
            )
        
        if top_variant["memorability"] < 0.5:
            recommendations.append(
                "Add distinctive brand elements or memorable hooks"
            )
        
        # Modality-specific recommendations
        modality_insights = top_variant.get("modality_insights", {})
        
        if "text" in modality_insights:
            if modality_insights["text"]["readability_score"] < 0.7:
                recommendations.append("Simplify text for better readability")
        
        if "image" in modality_insights or "video" in modality_insights:
            visual_complexity = modality_insights.get(
                "image", {}
            ).get("visual_complexity", 0)
            
            if visual_complexity > 0.8:
                recommendations.append("Reduce visual clutter for clearer messaging")
        
        return recommendations
    
    def _calculate_optimal_mix(self, ranked_variants: List[Dict]) -> Dict:
        """Calculate optimal creative mix for testing"""
        # Allocate budget based on predicted performance
        total_score = sum(v["predicted_performance"] for v in ranked_variants)
        
        if total_score > 0:
            allocations = {
                v["variant_id"]: v["predicted_performance"] / total_score
                for v in ranked_variants
            }
        else:
            # Equal allocation if no clear winner
            allocations = {
                v["variant_id"]: 1.0 / len(ranked_variants)
                for v in ranked_variants
            }
        
        # Apply minimum and maximum thresholds
        min_allocation = 0.1
        max_allocation = 0.5
        
        for variant_id in allocations:
            allocations[variant_id] = np.clip(
                allocations[variant_id],
                min_allocation,
                max_allocation
            )
        
        # Normalize
        total = sum(allocations.values())
        allocations = {k: v / total for k, v in allocations.items()}
        
        return allocations


class MultimodalMarketIntelligence:
    """Market intelligence using multimodal data"""
    
    def __init__(self):
        self.analyzer = MultimodalAdvertisingAnalyzer()
        self.market_data = {}
    
    async def analyze_market_trends(self,
                                  market_inputs: List[MultimodalInput],
                                  time_window: str = "30d") -> Dict:
        """Analyze market trends from multimodal data"""
        # Group inputs by source/brand
        grouped_inputs = self._group_by_source(market_inputs)
        
        # Analyze each source
        source_analyses = {}
        for source, inputs in grouped_inputs.items():
            analysis = await self.analyzer.analyze_ad_creative(inputs)
            source_analyses[source] = analysis
        
        # Extract trends
        trends = {
            "dominant_modalities": self._identify_dominant_modalities(source_analyses),
            "creative_patterns": self._extract_creative_patterns(source_analyses),
            "emerging_themes": self._identify_emerging_themes(source_analyses),
            "performance_benchmarks": self._calculate_benchmarks(source_analyses)
        }
        
        return {
            "time_window": time_window,
            "sources_analyzed": len(grouped_inputs),
            "trends": trends,
            "recommendations": self._generate_market_recommendations(trends)
        }
    
    def _group_by_source(self, inputs: List[MultimodalInput]) -> Dict:
        """Group inputs by source/brand"""
        grouped = {}
        for input_data in inputs:
            source = input_data.metadata.get("source", "unknown")
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(input_data)
        return grouped
    
    def _identify_dominant_modalities(self, analyses: Dict) -> List[str]:
        """Identify most effective modalities in market"""
        modality_scores = {}
        
        for source, analysis in analyses.items():
            insights = analysis.get("modality_insights", {})
            for modality, insight in insights.items():
                score = insight.get("contribution_score", 0)
                if modality not in modality_scores:
                    modality_scores[modality] = []
                modality_scores[modality].append(score)
        
        # Calculate average scores
        avg_scores = {
            modality: np.mean(scores)
            for modality, scores in modality_scores.items()
        }
        
        # Sort by effectiveness
        sorted_modalities = sorted(
            avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [m[0] for m in sorted_modalities[:3]]
    
    def _extract_creative_patterns(self, analyses: Dict) -> Dict:
        """Extract common creative patterns"""
        patterns = {
            "high_emotion_brands": [],
            "attention_leaders": [],
            "memory_champions": []
        }
        
        for source, analysis in analyses.items():
            if analysis.get("emotional_impact", 0) > 0.8:
                patterns["high_emotion_brands"].append(source)
            
            if analysis.get("attention_score", 0) > 0.7:
                patterns["attention_leaders"].append(source)
            
            if analysis.get("memorability", 0) > 0.6:
                patterns["memory_champions"].append(source)
        
        return patterns
    
    def _identify_emerging_themes(self, analyses: Dict) -> List[str]:
        """Identify emerging creative themes"""
        # Simplified theme identification
        themes = []
        
        # Check for common patterns
        avg_emotion = np.mean([a.get("emotional_impact", 0) for a in analyses.values()])
        if avg_emotion > 0.7:
            themes.append("Emotional storytelling dominant")
        
        avg_attention = np.mean([a.get("attention_score", 0) for a in analyses.values()])
        if avg_attention > 0.6:
            themes.append("High-impact visuals trending")
        
        # Check modality trends
        text_usage = sum(1 for a in analyses.values() 
                        if "text" in a.get("modality_insights", {}))
        if text_usage > len(analyses) * 0.8:
            themes.append("Text-heavy creatives prevalent")
        
        return themes
    
    def _calculate_benchmarks(self, analyses: Dict) -> Dict:
        """Calculate market benchmarks"""
        metrics = ["emotional_impact", "brand_alignment", "attention_score", 
                  "memorability", "engagement_prediction"]
        
        benchmarks = {}
        for metric in metrics:
            values = [a.get(metric, 0) for a in analyses.values()]
            if values:
                benchmarks[metric] = {
                    "mean": float(np.mean(values)),
                    "top_quartile": float(np.percentile(values, 75)),
                    "median": float(np.median(values))
                }
        
        return benchmarks
    
    def _generate_market_recommendations(self, trends: Dict) -> List[str]:
        """Generate market-based recommendations"""
        recommendations = []
        
        # Based on dominant modalities
        dominant = trends.get("dominant_modalities", [])
        if "video" in dominant[:2]:
            recommendations.append(
                "Prioritize video content - showing strongest market performance"
            )
        
        if "contextual" in dominant:
            recommendations.append(
                "Leverage contextual targeting for competitive advantage"
            )
        
        # Based on creative patterns
        patterns = trends.get("creative_patterns", {})
        if len(patterns.get("high_emotion_brands", [])) > 3:
            recommendations.append(
                "Emotional resonance is key differentiator in current market"
            )
        
        # Based on benchmarks
        benchmarks = trends.get("performance_benchmarks", {})
        if benchmarks.get("attention_score", {}).get("top_quartile", 0) > 0.8:
            recommendations.append(
                "Top performers achieving 80%+ attention scores - optimize for impact"
            )
        
        return recommendations


# Example usage
async def main():
    """Example multimodal AI system usage"""
    # Create multimodal inputs
    inputs = [
        MultimodalInput(
            modality=ModalityType.TEXT,
            data="Discover our revolutionary new product that changes everything",
            metadata={"language": "en"},
            timestamp=datetime.now(),
            confidence=0.95
        ),
        MultimodalInput(
            modality=ModalityType.IMAGE,
            data=Image.new('RGB', (1920, 1080), color='red'),  # Placeholder image
            metadata={"format": "banner"},
            timestamp=datetime.now(),
            confidence=0.90
        ),
        MultimodalInput(
            modality=ModalityType.CONTEXTUAL,
            data={},
            metadata={
                "weather": "sunny",
                "time_of_day": 14,
                "device": "mobile"
            },
            timestamp=datetime.now(),
            confidence=0.85
        )
    ]
    
    # Analyze creative
    analyzer = MultimodalAdvertisingAnalyzer()
    analysis = await analyzer.analyze_ad_creative(inputs)
    
    print("Multimodal Analysis Results:")
    print(f"Emotional Impact: {analysis['emotional_impact']:.2f}")
    print(f"Attention Score: {analysis['attention_score']:.2f}")
    print(f"Memorability: {analysis['memorability']:.2f}")
    print(f"Predicted Engagement: {analysis['engagement_prediction']:.2f}")
    
    # Optimize creative mix
    optimizer = MultimodalCampaignOptimizer()
    variants = [inputs, inputs[:2], inputs[1:]]  # Different combinations
    
    optimization = await optimizer.optimize_creative_mix(
        variants,
        target_metrics={
            "emotion_weight": 0.3,
            "attention_weight": 0.4,
            "engagement_weight": 0.3
        }
    )
    
    print(f"\nOptimal Creative Mix:")
    print(f"Top Variant: {optimization['top_variant']}")
    print(f"Recommendations: {optimization['recommendations']}")


if __name__ == "__main__":
    asyncio.run(main())
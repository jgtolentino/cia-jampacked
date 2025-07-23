"""
Psychographic Profiling System
Analyzes personality traits, communication styles, and behavioral patterns from minimal text input
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter
import json


class PersonalityDimension(Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class CommunicationStyle(Enum):
    ASSERTIVE = "assertive"
    ANALYTICAL = "analytical"
    EXPRESSIVE = "expressive"
    AMIABLE = "amiable"
    DRIVER = "driver"


class ValueSystem(Enum):
    ACHIEVEMENT = "achievement"
    BENEVOLENCE = "benevolence"
    CONFORMITY = "conformity"
    HEDONISM = "hedonism"
    POWER = "power"
    SECURITY = "security"
    SELF_DIRECTION = "self_direction"
    STIMULATION = "stimulation"
    TRADITION = "tradition"
    UNIVERSALISM = "universalism"


@dataclass
class PsychographicProfile:
    """Complete psychographic profile from text analysis"""
    personality_scores: Dict[PersonalityDimension, float]
    communication_style: CommunicationStyle
    primary_values: List[ValueSystem]
    behavioral_traits: Dict[str, float]
    linguistic_markers: Dict[str, Any]
    confidence_score: float
    
    def to_targeting_params(self) -> Dict:
        """Convert profile to advertising targeting parameters"""
        return {
            "personality_targeting": {
                dim.value: score for dim, score in self.personality_scores.items()
                if score > 0.6  # High-scoring dimensions only
            },
            "communication_preference": self.communication_style.value,
            "value_alignment": [v.value for v in self.primary_values[:3]],
            "behavioral_indicators": {
                trait: score for trait, score in self.behavioral_traits.items()
                if score > 0.5
            },
            "confidence": self.confidence_score
        }


class TenWordAnalyzer:
    """Analyze psychographic profile from as few as 10 words"""
    
    def __init__(self):
        self.linguistic_patterns = self._load_linguistic_patterns()
        self.word_associations = self._load_word_associations()
        self.minimum_words = 10
    
    def analyze(self, text: str) -> PsychographicProfile:
        """Analyze text and create psychographic profile"""
        words = self._preprocess_text(text)
        
        if len(words) < self.minimum_words:
            confidence_penalty = len(words) / self.minimum_words
        else:
            confidence_penalty = 1.0
        
        # Extract linguistic features
        linguistic_markers = self._extract_linguistic_markers(words, text)
        
        # Analyze personality dimensions
        personality_scores = self._analyze_personality(words, linguistic_markers)
        
        # Determine communication style
        communication_style = self._determine_communication_style(
            words, linguistic_markers
        )
        
        # Identify primary values
        primary_values = self._identify_values(words, linguistic_markers)
        
        # Extract behavioral traits
        behavioral_traits = self._extract_behavioral_traits(
            words, linguistic_markers
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            len(words), linguistic_markers
        ) * confidence_penalty
        
        return PsychographicProfile(
            personality_scores=personality_scores,
            communication_style=communication_style,
            primary_values=primary_values,
            behavioral_traits=behavioral_traits,
            linguistic_markers=linguistic_markers,
            confidence_score=confidence_score
        )
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for analysis"""
        # Convert to lowercase and extract words
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        return words
    
    def _load_linguistic_patterns(self) -> Dict:
        """Load linguistic patterns for personality analysis"""
        return {
            PersonalityDimension.OPENNESS: {
                "keywords": ["imagine", "creative", "novel", "curious", "explore",
                           "innovative", "unique", "abstract", "possibilities"],
                "patterns": [r'\b(what if|imagine|wonder)\b', r'\b(new|different)\b'],
                "weight": 1.2
            },
            PersonalityDimension.CONSCIENTIOUSNESS: {
                "keywords": ["plan", "organize", "careful", "detail", "responsible",
                           "efficient", "systematic", "thorough", "precise"],
                "patterns": [r'\b(must|should|need to)\b', r'\b(exactly|precisely)\b'],
                "weight": 1.1
            },
            PersonalityDimension.EXTRAVERSION: {
                "keywords": ["exciting", "fun", "social", "party", "friends",
                           "energetic", "enthusiastic", "outgoing", "talkative"],
                "patterns": [r'!+', r'\b(we|us|our)\b', r'\b(together|everyone)\b'],
                "weight": 1.0
            },
            PersonalityDimension.AGREEABLENESS: {
                "keywords": ["help", "kind", "care", "understand", "support",
                           "cooperate", "trust", "warm", "compassionate"],
                "patterns": [r'\b(please|thank you|sorry)\b', r'\b(together|share)\b'],
                "weight": 1.0
            },
            PersonalityDimension.NEUROTICISM: {
                "keywords": ["worry", "stress", "anxious", "nervous", "fear",
                           "upset", "tense", "insecure", "emotional"],
                "patterns": [r'\b(but|however|although)\b', r'\?+', r'\b(maybe|perhaps)\b'],
                "weight": 0.9
            }
        }
    
    def _load_word_associations(self) -> Dict:
        """Load word associations for value and trait analysis"""
        return {
            "achievement": ["success", "accomplish", "achieve", "win", "best", 
                           "excel", "master", "competent"],
            "power": ["control", "influence", "lead", "authority", "dominant",
                     "status", "prestige", "command"],
            "hedonism": ["enjoy", "pleasure", "fun", "exciting", "indulge",
                        "satisfy", "delight", "gratify"],
            "stimulation": ["adventure", "exciting", "new", "thrill", "dare",
                           "bold", "risk", "challenge"],
            "self_direction": ["independent", "free", "choose", "create", "curious",
                              "explore", "decide", "autonomous"],
            "universalism": ["everyone", "world", "fair", "equal", "nature",
                            "peace", "harmony", "justice"],
            "benevolence": ["help", "care", "kind", "loyal", "honest",
                           "genuine", "sincere", "devoted"],
            "tradition": ["traditional", "custom", "respect", "heritage", "family",
                         "culture", "honor", "conventional"],
            "conformity": ["proper", "polite", "obedient", "discipline", "rules",
                          "appropriate", "correct", "comply"],
            "security": ["safe", "secure", "stable", "protect", "certain",
                        "reliable", "consistent", "predictable"]
        }
    
    def _extract_linguistic_markers(self, words: List[str], text: str) -> Dict:
        """Extract linguistic markers from text"""
        markers = {
            "word_count": len(words),
            "unique_words": len(set(words)),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "lexical_diversity": len(set(words)) / len(words) if words else 0,
            "punctuation": {
                "exclamation": text.count('!'),
                "question": text.count('?'),
                "ellipsis": text.count('...'),
                "comma": text.count(',')
            },
            "pronouns": {
                "first_person": sum(1 for w in words if w in ['i', 'me', 'my', 'mine']),
                "second_person": sum(1 for w in words if w in ['you', 'your', 'yours']),
                "third_person": sum(1 for w in words if w in ['he', 'she', 'they', 'them']),
                "first_plural": sum(1 for w in words if w in ['we', 'us', 'our'])
            },
            "sentiment_words": self._count_sentiment_words(words),
            "cognitive_words": self._count_cognitive_words(words),
            "temporal_focus": self._analyze_temporal_focus(words)
        }
        
        return markers
    
    def _count_sentiment_words(self, words: List[str]) -> Dict:
        """Count positive and negative sentiment words"""
        positive_words = ["good", "great", "love", "excellent", "happy", "wonderful",
                         "amazing", "fantastic", "beautiful", "perfect"]
        negative_words = ["bad", "hate", "terrible", "awful", "horrible", "wrong",
                         "poor", "disappointing", "frustrating", "annoying"]
        
        return {
            "positive": sum(1 for w in words if w in positive_words),
            "negative": sum(1 for w in words if w in negative_words),
            "ratio": (sum(1 for w in words if w in positive_words) / 
                     (sum(1 for w in words if w in negative_words) + 1))
        }
    
    def _count_cognitive_words(self, words: List[str]) -> Dict:
        """Count cognitive process words"""
        insight_words = ["think", "know", "consider", "understand", "realize"]
        causation_words = ["because", "since", "therefore", "thus", "hence"]
        discrepancy_words = ["should", "would", "could", "ought", "must"]
        tentative_words = ["maybe", "perhaps", "possibly", "might", "seems"]
        certainty_words = ["always", "never", "definitely", "certainly", "sure"]
        
        return {
            "insight": sum(1 for w in words if w in insight_words),
            "causation": sum(1 for w in words if w in causation_words),
            "discrepancy": sum(1 for w in words if w in discrepancy_words),
            "tentative": sum(1 for w in words if w in tentative_words),
            "certainty": sum(1 for w in words if w in certainty_words)
        }
    
    def _analyze_temporal_focus(self, words: List[str]) -> Dict:
        """Analyze temporal focus (past, present, future)"""
        past_words = ["was", "were", "had", "did", "used", "ago", "yesterday", "before"]
        present_words = ["is", "am", "are", "now", "today", "current", "present"]
        future_words = ["will", "shall", "going", "tomorrow", "soon", "later", "next"]
        
        return {
            "past": sum(1 for w in words if w in past_words),
            "present": sum(1 for w in words if w in present_words),
            "future": sum(1 for w in words if w in future_words)
        }
    
    def _analyze_personality(self, 
                           words: List[str],
                           markers: Dict) -> Dict[PersonalityDimension, float]:
        """Analyze Big Five personality dimensions"""
        scores = {}
        
        for dimension, patterns in self.linguistic_patterns.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for w in words if w in patterns["keywords"])
            score += keyword_matches * 0.1 * patterns["weight"]
            
            # Check patterns
            text = ' '.join(words)
            for pattern in patterns["patterns"]:
                matches = len(re.findall(pattern, text))
                score += matches * 0.05 * patterns["weight"]
            
            # Adjust based on linguistic markers
            if dimension == PersonalityDimension.EXTRAVERSION:
                score += markers["pronouns"]["first_plural"] * 0.05
                score += markers["punctuation"]["exclamation"] * 0.03
            
            elif dimension == PersonalityDimension.OPENNESS:
                score += markers["lexical_diversity"] * 0.2
                score += markers["cognitive_words"]["insight"] * 0.05
            
            elif dimension == PersonalityDimension.CONSCIENTIOUSNESS:
                score += markers["cognitive_words"]["certainty"] * 0.05
                score -= markers["cognitive_words"]["tentative"] * 0.03
            
            elif dimension == PersonalityDimension.AGREEABLENESS:
                score += markers["pronouns"]["second_person"] * 0.04
                score += markers["sentiment_words"]["positive"] * 0.03
            
            elif dimension == PersonalityDimension.NEUROTICISM:
                score += markers["sentiment_words"]["negative"] * 0.05
                score += markers["cognitive_words"]["tentative"] * 0.04
                score += markers["punctuation"]["question"] * 0.03
            
            # Normalize score to 0-1 range
            scores[dimension] = min(1.0, max(0.0, score))
        
        return scores
    
    def _determine_communication_style(self,
                                     words: List[str],
                                     markers: Dict) -> CommunicationStyle:
        """Determine primary communication style"""
        style_scores = {
            CommunicationStyle.ASSERTIVE: 0.0,
            CommunicationStyle.ANALYTICAL: 0.0,
            CommunicationStyle.EXPRESSIVE: 0.0,
            CommunicationStyle.AMIABLE: 0.0,
            CommunicationStyle.DRIVER: 0.0
        }
        
        # Assertive: Direct, confident communication
        assertive_words = ["definitely", "certainly", "must", "will", "know", "fact"]
        style_scores[CommunicationStyle.ASSERTIVE] += sum(
            1 for w in words if w in assertive_words
        ) * 0.2
        style_scores[CommunicationStyle.ASSERTIVE] += markers["cognitive_words"]["certainty"] * 0.1
        
        # Analytical: Detail-oriented, logical
        analytical_words = ["analyze", "data", "research", "evidence", "logic", "reason"]
        style_scores[CommunicationStyle.ANALYTICAL] += sum(
            1 for w in words if w in analytical_words
        ) * 0.2
        style_scores[CommunicationStyle.ANALYTICAL] += markers["cognitive_words"]["causation"] * 0.15
        
        # Expressive: Emotional, enthusiastic
        style_scores[CommunicationStyle.EXPRESSIVE] += markers["punctuation"]["exclamation"] * 0.1
        style_scores[CommunicationStyle.EXPRESSIVE] += markers["sentiment_words"]["positive"] * 0.1
        style_scores[CommunicationStyle.EXPRESSIVE] += (markers["avg_word_length"] < 5) * 0.2
        
        # Amiable: Friendly, cooperative
        amiable_words = ["together", "help", "share", "please", "thank", "appreciate"]
        style_scores[CommunicationStyle.AMIABLE] += sum(
            1 for w in words if w in amiable_words
        ) * 0.2
        style_scores[CommunicationStyle.AMIABLE] += markers["pronouns"]["first_plural"] * 0.1
        
        # Driver: Results-oriented, decisive
        driver_words = ["goal", "achieve", "result", "fast", "now", "immediate"]
        style_scores[CommunicationStyle.DRIVER] += sum(
            1 for w in words if w in driver_words
        ) * 0.2
        style_scores[CommunicationStyle.DRIVER] += markers["temporal_focus"]["future"] * 0.1
        
        # Return style with highest score
        return max(style_scores.items(), key=lambda x: x[1])[0]
    
    def _identify_values(self,
                        words: List[str],
                        markers: Dict) -> List[ValueSystem]:
        """Identify primary values from text"""
        value_scores = {}
        
        for value_name, keywords in self.word_associations.items():
            score = sum(1 for w in words if w in keywords) * 0.2
            
            # Additional scoring based on context
            if value_name == "achievement":
                score += markers["temporal_focus"]["future"] * 0.05
                score += (markers["pronouns"]["first_person"] > 2) * 0.1
            
            elif value_name == "benevolence":
                score += markers["pronouns"]["second_person"] * 0.05
                score += markers["sentiment_words"]["positive"] * 0.03
            
            elif value_name == "universalism":
                score += (markers["pronouns"]["first_plural"] > 1) * 0.1
                score += ("everyone" in words or "all" in words) * 0.2
            
            elif value_name == "security":
                score += markers["cognitive_words"]["certainty"] * 0.05
                score -= markers["cognitive_words"]["tentative"] * 0.03
            
            # Map to ValueSystem enum
            try:
                value_enum = ValueSystem[value_name.upper()]
                value_scores[value_enum] = score
            except KeyError:
                pass
        
        # Sort by score and return top values
        sorted_values = sorted(value_scores.items(), key=lambda x: x[1], reverse=True)
        return [value for value, score in sorted_values if score > 0][:5]
    
    def _extract_behavioral_traits(self,
                                 words: List[str],
                                 markers: Dict) -> Dict[str, float]:
        """Extract behavioral traits from linguistic patterns"""
        traits = {}
        
        # Risk tolerance
        risk_words = ["risk", "chance", "gamble", "venture", "dare", "bold"]
        safety_words = ["safe", "secure", "careful", "cautious", "prudent"]
        risk_score = sum(1 for w in words if w in risk_words)
        safety_score = sum(1 for w in words if w in safety_words)
        traits["risk_tolerance"] = (risk_score - safety_score + 5) / 10
        
        # Innovation tendency
        innovation_words = ["new", "innovative", "creative", "novel", "unique", "different"]
        traditional_words = ["traditional", "classic", "proven", "established", "conventional"]
        traits["innovation_tendency"] = (
            sum(1 for w in words if w in innovation_words) -
            sum(1 for w in words if w in traditional_words) + 5
        ) / 10
        
        # Social orientation
        social_score = (
            markers["pronouns"]["first_plural"] +
            markers["pronouns"]["second_person"] +
            ("together" in words) * 2 +
            ("team" in words) * 2
        )
        traits["social_orientation"] = min(1.0, social_score / 10)
        
        # Decision speed
        immediate_words = ["now", "quick", "fast", "immediate", "urgent", "asap"]
        deliberate_words = ["consider", "think", "analyze", "evaluate", "review"]
        traits["decision_speed"] = (
            sum(1 for w in words if w in immediate_words) -
            sum(1 for w in words if w in deliberate_words) + 5
        ) / 10
        
        # Brand loyalty tendency
        loyalty_words = ["always", "loyal", "trust", "favorite", "best", "only"]
        variety_words = ["try", "new", "different", "change", "switch", "explore"]
        traits["brand_loyalty"] = (
            sum(1 for w in words if w in loyalty_words) -
            sum(1 for w in words if w in variety_words) + 5
        ) / 10
        
        # Normalize all traits to 0-1 range
        for trait in traits:
            traits[trait] = max(0.0, min(1.0, traits[trait]))
        
        return traits
    
    def _calculate_confidence(self, word_count: int, markers: Dict) -> float:
        """Calculate confidence score for the analysis"""
        base_confidence = min(1.0, word_count / 50)  # Max confidence at 50+ words
        
        # Adjust based on lexical diversity
        diversity_factor = markers["lexical_diversity"]
        
        # Adjust based on meaningful content
        meaningful_words = (
            markers["cognitive_words"]["insight"] +
            markers["cognitive_words"]["causation"] +
            markers["sentiment_words"]["positive"] +
            markers["sentiment_words"]["negative"]
        )
        content_factor = min(1.0, meaningful_words / 5)
        
        # Combine factors
        confidence = (base_confidence * 0.5 + 
                     diversity_factor * 0.3 + 
                     content_factor * 0.2)
        
        return confidence


class PsychographicSegmenter:
    """Segment audiences based on psychographic profiles"""
    
    def __init__(self):
        self.segments = self._define_segments()
    
    def _define_segments(self) -> Dict[str, Dict]:
        """Define psychographic segments for targeting"""
        return {
            "ambitious_achievers": {
                "personality": {
                    PersonalityDimension.CONSCIENTIOUSNESS: 0.7,
                    PersonalityDimension.EXTRAVERSION: 0.6
                },
                "values": [ValueSystem.ACHIEVEMENT, ValueSystem.POWER],
                "traits": {"risk_tolerance": 0.6, "innovation_tendency": 0.7},
                "messaging": "Success-oriented, competitive messaging"
            },
            "caring_connectors": {
                "personality": {
                    PersonalityDimension.AGREEABLENESS: 0.8,
                    PersonalityDimension.EXTRAVERSION: 0.6
                },
                "values": [ValueSystem.BENEVOLENCE, ValueSystem.UNIVERSALISM],
                "traits": {"social_orientation": 0.8, "brand_loyalty": 0.7},
                "messaging": "Community-focused, empathetic messaging"
            },
            "analytical_thinkers": {
                "personality": {
                    PersonalityDimension.OPENNESS: 0.7,
                    PersonalityDimension.CONSCIENTIOUSNESS: 0.6
                },
                "values": [ValueSystem.SELF_DIRECTION, ValueSystem.UNIVERSALISM],
                "traits": {"innovation_tendency": 0.6, "decision_speed": 0.3},
                "messaging": "Data-driven, logical messaging"
            },
            "experience_seekers": {
                "personality": {
                    PersonalityDimension.OPENNESS: 0.8,
                    PersonalityDimension.EXTRAVERSION: 0.7
                },
                "values": [ValueSystem.STIMULATION, ValueSystem.HEDONISM],
                "traits": {"risk_tolerance": 0.8, "innovation_tendency": 0.8},
                "messaging": "Adventure-focused, exciting messaging"
            },
            "security_minded": {
                "personality": {
                    PersonalityDimension.CONSCIENTIOUSNESS: 0.7,
                    PersonalityDimension.NEUROTICISM: 0.6
                },
                "values": [ValueSystem.SECURITY, ValueSystem.TRADITION],
                "traits": {"risk_tolerance": 0.2, "brand_loyalty": 0.8},
                "messaging": "Trust-building, reassuring messaging"
            }
        }
    
    def assign_segment(self, profile: PsychographicProfile) -> Tuple[str, float]:
        """Assign profile to best matching segment"""
        best_match = None
        best_score = 0
        
        for segment_name, segment_def in self.segments.items():
            score = self._calculate_match_score(profile, segment_def)
            
            if score > best_score:
                best_score = score
                best_match = segment_name
        
        return best_match, best_score
    
    def _calculate_match_score(self, 
                              profile: PsychographicProfile,
                              segment: Dict) -> float:
        """Calculate how well profile matches segment"""
        score = 0.0
        components = 0
        
        # Match personality dimensions
        for dimension, target_score in segment["personality"].items():
            if dimension in profile.personality_scores:
                diff = abs(profile.personality_scores[dimension] - target_score)
                score += (1 - diff) * 0.3
                components += 0.3
        
        # Match values
        profile_values = set(profile.primary_values[:3])
        segment_values = set(segment["values"])
        value_overlap = len(profile_values & segment_values) / len(segment_values)
        score += value_overlap * 0.4
        components += 0.4
        
        # Match traits
        for trait, target_value in segment["traits"].items():
            if trait in profile.behavioral_traits:
                diff = abs(profile.behavioral_traits[trait] - target_value)
                score += (1 - diff) * 0.15
                components += 0.15
        
        return score / components if components > 0 else 0


class AdCreativePersonalizer:
    """Personalize ad creative based on psychographic profile"""
    
    def __init__(self):
        self.creative_templates = self._load_creative_templates()
    
    def _load_creative_templates(self) -> Dict:
        """Load creative templates for different profiles"""
        return {
            CommunicationStyle.ASSERTIVE: {
                "headline_pattern": "Get {benefit} Now",
                "cta": "Take Action",
                "tone": "direct and confident"
            },
            CommunicationStyle.ANALYTICAL: {
                "headline_pattern": "Data Shows: {statistic} Improvement",
                "cta": "See the Research",
                "tone": "logical and factual"
            },
            CommunicationStyle.EXPRESSIVE: {
                "headline_pattern": "Amazing! {emotional_benefit}",
                "cta": "Join the Fun!",
                "tone": "enthusiastic and emotional"
            },
            CommunicationStyle.AMIABLE: {
                "headline_pattern": "Together We Can {shared_goal}",
                "cta": "Join Our Community",
                "tone": "warm and inclusive"
            },
            CommunicationStyle.DRIVER: {
                "headline_pattern": "Achieve {result} Fast",
                "cta": "Start Now",
                "tone": "results-focused and urgent"
            }
        }
    
    def personalize_creative(self,
                           base_creative: Dict,
                           profile: PsychographicProfile) -> Dict:
        """Personalize creative elements based on profile"""
        personalized = base_creative.copy()
        
        # Get template for communication style
        template = self.creative_templates.get(
            profile.communication_style,
            self.creative_templates[CommunicationStyle.ASSERTIVE]
        )
        
        # Personalize headline
        if "headline" in base_creative:
            personalized["headline"] = self._personalize_headline(
                base_creative["headline"],
                template,
                profile
            )
        
        # Personalize CTA
        personalized["cta"] = template["cta"]
        
        # Adjust tone indicators
        personalized["tone_guidance"] = template["tone"]
        
        # Add value-based messaging
        personalized["value_props"] = self._generate_value_props(
            profile.primary_values[:3]
        )
        
        # Personality-based adjustments
        personalized["visual_style"] = self._recommend_visual_style(
            profile.personality_scores
        )
        
        return personalized
    
    def _personalize_headline(self,
                            original: str,
                            template: Dict,
                            profile: PsychographicProfile) -> str:
        """Personalize headline based on profile"""
        # Extract key benefit from original
        benefit = original.split()[0] if original else "Results"
        
        # Apply template pattern
        pattern = template["headline_pattern"]
        
        # Fill in placeholders
        replacements = {
            "{benefit}": benefit,
            "{statistic}": "73%",  # Would come from data
            "{emotional_benefit}": "Feel Confident",
            "{shared_goal}": "Make a Difference",
            "{result}": "Your Goals"
        }
        
        headline = pattern
        for placeholder, value in replacements.items():
            headline = headline.replace(placeholder, value)
        
        return headline
    
    def _generate_value_props(self, values: List[ValueSystem]) -> List[str]:
        """Generate value propositions based on primary values"""
        value_props = {
            ValueSystem.ACHIEVEMENT: "Reach your full potential",
            ValueSystem.BENEVOLENCE: "Make a positive impact",
            ValueSystem.SECURITY: "Trusted by millions",
            ValueSystem.SELF_DIRECTION: "Take control of your journey",
            ValueSystem.STIMULATION: "Experience something new",
            ValueSystem.HEDONISM: "Enjoy every moment",
            ValueSystem.POWER: "Lead with confidence",
            ValueSystem.UNIVERSALISM: "Good for you, good for all",
            ValueSystem.TRADITION: "Time-tested quality",
            ValueSystem.CONFORMITY: "The right choice"
        }
        
        return [value_props.get(v, "") for v in values if v in value_props]
    
    def _recommend_visual_style(self, personality: Dict) -> Dict:
        """Recommend visual style based on personality"""
        style = {
            "color_intensity": "medium",
            "complexity": "medium",
            "movement": "moderate"
        }
        
        # High openness -> more creative visuals
        if personality.get(PersonalityDimension.OPENNESS, 0) > 0.7:
            style["complexity"] = "high"
            style["color_intensity"] = "vibrant"
        
        # High extraversion -> dynamic visuals
        if personality.get(PersonalityDimension.EXTRAVERSION, 0) > 0.7:
            style["movement"] = "high"
            style["color_intensity"] = "bright"
        
        # High conscientiousness -> clean, organized visuals
        if personality.get(PersonalityDimension.CONSCIENTIOUSNESS, 0) > 0.7:
            style["complexity"] = "low"
            style["layout"] = "structured"
        
        return style


# Example usage
def main():
    """Example psychographic profiling usage"""
    # Create analyzer
    analyzer = TenWordAnalyzer()
    
    # Example 1: Analyze short text (exactly 10 words)
    text1 = "I love exploring new innovative solutions that help everyone succeed"
    profile1 = analyzer.analyze(text1)
    
    print("Profile from 10 words:")
    print(f"Communication Style: {profile1.communication_style.value}")
    print(f"Top Values: {[v.value for v in profile1.primary_values[:3]]}")
    print(f"Confidence: {profile1.confidence_score:.2f}")
    
    # Example 2: Longer text
    text2 = """
    We must carefully analyze the data before making any decisions. 
    It's important to consider all possibilities and ensure we have 
    solid evidence supporting our approach.
    """
    profile2 = analyzer.analyze(text2)
    
    print("\nProfile from longer text:")
    print(f"Communication Style: {profile2.communication_style.value}")
    print("Personality Scores:")
    for dim, score in profile2.personality_scores.items():
        print(f"  {dim.value}: {score:.2f}")
    
    # Segment assignment
    segmenter = PsychographicSegmenter()
    segment, match_score = segmenter.assign_segment(profile2)
    print(f"\nAssigned Segment: {segment} (match: {match_score:.2f})")
    
    # Personalize creative
    personalizer = AdCreativePersonalizer()
    base_creative = {
        "headline": "Discover Amazing Products",
        "description": "Shop our collection today"
    }
    
    personalized = personalizer.personalize_creative(base_creative, profile2)
    print(f"\nPersonalized Creative:")
    print(f"Headline: {personalized['headline']}")
    print(f"CTA: {personalized['cta']}")
    print(f"Tone: {personalized['tone_guidance']}")


if __name__ == "__main__":
    main()
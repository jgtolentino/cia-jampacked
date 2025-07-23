"""
Real-Time Optimization System for Dynamic Prompt Adjustment
Continuously optimizes AI advertising prompts based on performance data
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import json
from collections import deque
import heapq


class OptimizationStrategy(Enum):
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GRADIENT_FREE = "gradient_free"


class PerformanceMetric(Enum):
    CTR = "ctr"
    CONVERSION_RATE = "conversion_rate"
    ENGAGEMENT_RATE = "engagement_rate"
    ROAS = "roas"
    COMPOSITE = "composite"


@dataclass
class PromptVariant:
    """Single prompt variant with performance tracking"""
    id: str
    prompt_text: str
    elements: Dict[str, str]  # Structured elements
    created_at: datetime
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    exploration_bonus: float = 1.0
    
    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions > 0 else 0
    
    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.clicks if self.clicks > 0 else 0
    
    @property
    def roas(self) -> float:
        cost = self.impressions * 0.001  # Assuming $1 CPM
        return self.revenue / cost if cost > 0 else 0
    
    @property
    def performance_score(self) -> float:
        """Composite performance score"""
        if self.impressions < 100:
            # Not enough data, use exploration bonus
            return self.exploration_bonus
        
        # Weighted combination of metrics
        score = (
            self.ctr * 0.3 +
            self.conversion_rate * 0.4 +
            min(self.roas / 10, 1.0) * 0.3  # Normalize ROAS to 0-1
        )
        
        return score


@dataclass
class OptimizationState:
    """Current state of optimization system"""
    active_variants: List[PromptVariant]
    performance_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    best_variant: Optional[PromptVariant] = None
    optimization_rounds: int = 0
    total_impressions: int = 0
    total_conversions: int = 0


class MultiArmedBanditOptimizer:
    """Thompson Sampling for prompt optimization"""
    
    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self.variant_priors = {}  # Beta distribution parameters
    
    def select_variant(self, variants: List[PromptVariant]) -> PromptVariant:
        """Select variant using Thompson Sampling"""
        scores = []
        
        for variant in variants:
            # Get or initialize prior
            if variant.id not in self.variant_priors:
                self.variant_priors[variant.id] = {"alpha": 1, "beta": 1}
            
            prior = self.variant_priors[variant.id]
            
            # Sample from Beta distribution
            sample = np.random.beta(prior["alpha"], prior["beta"])
            scores.append((sample, variant))
        
        # Select variant with highest sampled value
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]
    
    def update_beliefs(self, variant: PromptVariant, reward: bool):
        """Update beliefs based on observed reward"""
        if variant.id not in self.variant_priors:
            self.variant_priors[variant.id] = {"alpha": 1, "beta": 1}
        
        if reward:
            self.variant_priors[variant.id]["alpha"] += 1
        else:
            self.variant_priors[variant.id]["beta"] += 1
    
    def get_variant_confidence(self, variant: PromptVariant) -> Tuple[float, float]:
        """Get confidence interval for variant performance"""
        if variant.id not in self.variant_priors:
            return 0.5, 0.5
        
        prior = self.variant_priors[variant.id]
        mean = prior["alpha"] / (prior["alpha"] + prior["beta"])
        
        # 95% confidence interval
        lower = np.percentile(
            np.random.beta(prior["alpha"], prior["beta"], 1000), 2.5
        )
        upper = np.percentile(
            np.random.beta(prior["alpha"], prior["beta"], 1000), 97.5
        )
        
        return lower, upper


class EvolutionaryOptimizer:
    """Genetic algorithm for prompt evolution"""
    
    def __init__(self, 
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
    
    def evolve_population(self, 
                         current_variants: List[PromptVariant],
                         element_pool: Dict[str, List[str]]) -> List[PromptVariant]:
        """Evolve population of prompts"""
        # Sort by fitness (performance score)
        sorted_variants = sorted(
            current_variants, 
            key=lambda x: x.performance_score, 
            reverse=True
        )
        
        # Keep top performers (elitism)
        elite_count = max(2, len(sorted_variants) // 4)
        new_population = sorted_variants[:elite_count]
        
        # Generate new variants through crossover and mutation
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate and len(sorted_variants) >= 2:
                # Crossover
                parent1 = self._tournament_selection(sorted_variants)
                parent2 = self._tournament_selection(sorted_variants)
                child = self._crossover(parent1, parent2)
            else:
                # Clone and mutate
                parent = self._tournament_selection(sorted_variants)
                child = self._clone_variant(parent)
            
            # Apply mutation
            if np.random.random() < self.mutation_rate:
                child = self._mutate(child, element_pool)
            
            new_population.append(child)
        
        self.generation += 1
        return new_population
    
    def _tournament_selection(self, 
                            variants: List[PromptVariant],
                            tournament_size: int = 3) -> PromptVariant:
        """Select variant using tournament selection"""
        tournament = np.random.choice(variants, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.performance_score)
    
    def _crossover(self, parent1: PromptVariant, parent2: PromptVariant) -> PromptVariant:
        """Crossover two parent variants"""
        child_elements = {}
        
        for key in parent1.elements:
            if np.random.random() < 0.5:
                child_elements[key] = parent1.elements[key]
            else:
                child_elements[key] = parent2.elements.get(key, parent1.elements[key])
        
        return PromptVariant(
            id=f"gen_{self.generation}_{np.random.randint(10000)}",
            prompt_text=self._reconstruct_prompt(child_elements),
            elements=child_elements,
            created_at=datetime.now()
        )
    
    def _mutate(self, 
               variant: PromptVariant,
               element_pool: Dict[str, List[str]]) -> PromptVariant:
        """Mutate a variant"""
        mutated_elements = variant.elements.copy()
        
        # Randomly change one element
        if mutated_elements and element_pool:
            key_to_mutate = np.random.choice(list(mutated_elements.keys()))
            if key_to_mutate in element_pool:
                new_value = np.random.choice(element_pool[key_to_mutate])
                mutated_elements[key_to_mutate] = new_value
        
        return PromptVariant(
            id=f"mut_{variant.id}_{np.random.randint(1000)}",
            prompt_text=self._reconstruct_prompt(mutated_elements),
            elements=mutated_elements,
            created_at=datetime.now()
        )
    
    def _clone_variant(self, variant: PromptVariant) -> PromptVariant:
        """Create a clone of a variant"""
        return PromptVariant(
            id=f"clone_{variant.id}_{np.random.randint(1000)}",
            prompt_text=variant.prompt_text,
            elements=variant.elements.copy(),
            created_at=datetime.now()
        )
    
    def _reconstruct_prompt(self, elements: Dict[str, str]) -> str:
        """Reconstruct prompt from elements"""
        # Simple template reconstruction
        template = "{role} {task_goal} {tone} {audience} {cta}"
        return template.format(**elements)


class RealtimeOptimizationEngine:
    """Main real-time optimization engine"""
    
    def __init__(self,
                 strategy: OptimizationStrategy = OptimizationStrategy.MULTI_ARMED_BANDIT,
                 update_interval: timedelta = timedelta(minutes=5)):
        self.strategy = strategy
        self.update_interval = update_interval
        self.state = OptimizationState(active_variants=[])
        self.element_pool = self._initialize_element_pool()
        
        # Initialize optimizers
        self.bandit_optimizer = MultiArmedBanditOptimizer()
        self.evolutionary_optimizer = EvolutionaryOptimizer()
        
        self.is_running = False
        self.performance_callbacks = []
    
    def _initialize_element_pool(self) -> Dict[str, List[str]]:
        """Initialize pool of prompt elements"""
        return {
            "role": [
                "You are a helpful assistant",
                "You are an expert marketer",
                "You are a creative strategist",
                "You are a conversion specialist"
            ],
            "task_goal": [
                "Create compelling ad copy",
                "Write persuasive marketing messages",
                "Develop engaging content",
                "Craft high-converting copy"
            ],
            "tone": [
                "professional and authoritative",
                "friendly and conversational",
                "urgent and compelling",
                "empathetic and understanding"
            ],
            "audience": [
                "targeting busy professionals",
                "speaking to conscious consumers",
                "addressing tech-savvy users",
                "connecting with value seekers"
            ],
            "cta": [
                "Shop Now",
                "Learn More",
                "Get Started",
                "Discover More"
            ]
        }
    
    def add_variant(self, variant: PromptVariant):
        """Add a new variant to the optimization pool"""
        self.state.active_variants.append(variant)
    
    def create_initial_variants(self, count: int = 10) -> List[PromptVariant]:
        """Create initial set of variants"""
        variants = []
        
        for i in range(count):
            elements = {}
            for key, options in self.element_pool.items():
                elements[key] = np.random.choice(options)
            
            variant = PromptVariant(
                id=f"initial_{i}",
                prompt_text=self._create_prompt_from_elements(elements),
                elements=elements,
                created_at=datetime.now()
            )
            variants.append(variant)
            self.add_variant(variant)
        
        return variants
    
    def _create_prompt_from_elements(self, elements: Dict[str, str]) -> str:
        """Create prompt text from elements"""
        return (
            f"{elements.get('role', '')}. "
            f"{elements.get('task_goal', '')} "
            f"{elements.get('tone', '')} "
            f"{elements.get('audience', '')}. "
            f"Call to action: {elements.get('cta', '')}"
        )
    
    async def start_optimization(self):
        """Start real-time optimization loop"""
        self.is_running = True
        
        while self.is_running:
            # Perform optimization round
            await self._optimization_round()
            
            # Wait for next update
            await asyncio.sleep(self.update_interval.total_seconds())
    
    async def _optimization_round(self):
        """Perform one round of optimization"""
        self.state.optimization_rounds += 1
        
        # Get current performance data
        performance_data = await self._fetch_performance_data()
        
        # Update variant metrics
        self._update_variant_metrics(performance_data)
        
        # Store performance history
        self.state.performance_history.append({
            "round": self.state.optimization_rounds,
            "timestamp": datetime.now(),
            "best_score": max(v.performance_score for v in self.state.active_variants),
            "total_conversions": sum(v.conversions for v in self.state.active_variants)
        })
        
        # Apply optimization strategy
        if self.strategy == OptimizationStrategy.MULTI_ARMED_BANDIT:
            await self._optimize_with_bandit()
        elif self.strategy == OptimizationStrategy.EVOLUTIONARY:
            await self._optimize_with_evolution()
        
        # Update best variant
        self._update_best_variant()
        
        # Trigger callbacks
        for callback in self.performance_callbacks:
            callback(self.state)
    
    async def _fetch_performance_data(self) -> Dict:
        """Fetch latest performance data"""
        # In production, this would connect to analytics APIs
        # Simulated data for example
        data = {}
        
        for variant in self.state.active_variants:
            # Simulate performance based on variant elements
            base_ctr = 0.02
            
            # Adjust based on elements
            if "expert" in variant.elements.get("role", ""):
                base_ctr *= 1.1
            if "urgent" in variant.elements.get("tone", ""):
                base_ctr *= 1.2
            if "Shop Now" in variant.elements.get("cta", ""):
                base_ctr *= 1.15
            
            impressions = np.random.poisson(1000)
            clicks = np.random.binomial(impressions, base_ctr)
            conversions = np.random.binomial(clicks, 0.1)
            revenue = conversions * np.random.uniform(50, 150)
            
            data[variant.id] = {
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "revenue": revenue
            }
        
        return data
    
    def _update_variant_metrics(self, performance_data: Dict):
        """Update variant metrics with new data"""
        for variant in self.state.active_variants:
            if variant.id in performance_data:
                data = performance_data[variant.id]
                variant.impressions += data["impressions"]
                variant.clicks += data["clicks"]
                variant.conversions += data["conversions"]
                variant.revenue += data["revenue"]
                
                # Update bandit beliefs
                if data["impressions"] > 0:
                    reward = data["conversions"] > 0
                    self.bandit_optimizer.update_beliefs(variant, reward)
    
    async def _optimize_with_bandit(self):
        """Optimize using multi-armed bandit"""
        # Remove poorly performing variants
        if len(self.state.active_variants) > 10:
            # Keep only top 80% performers
            sorted_variants = sorted(
                self.state.active_variants,
                key=lambda x: x.performance_score,
                reverse=True
            )
            
            cutoff = int(len(sorted_variants) * 0.8)
            self.state.active_variants = sorted_variants[:cutoff]
        
        # Add new exploratory variants
        if len(self.state.active_variants) < 15:
            new_variant = self._create_exploratory_variant()
            self.add_variant(new_variant)
    
    async def _optimize_with_evolution(self):
        """Optimize using evolutionary algorithm"""
        # Evolve population
        new_variants = self.evolutionary_optimizer.evolve_population(
            self.state.active_variants,
            self.element_pool
        )
        
        # Replace population
        self.state.active_variants = new_variants
    
    def _create_exploratory_variant(self) -> PromptVariant:
        """Create new variant for exploration"""
        if self.state.best_variant:
            # Create variation of best variant
            base_elements = self.state.best_variant.elements.copy()
            
            # Change one random element
            key_to_change = np.random.choice(list(base_elements.keys()))
            base_elements[key_to_change] = np.random.choice(
                self.element_pool[key_to_change]
            )
            
            variant_id = f"explore_{self.state.optimization_rounds}_{np.random.randint(1000)}"
        else:
            # Create random variant
            base_elements = {}
            for key, options in self.element_pool.items():
                base_elements[key] = np.random.choice(options)
            
            variant_id = f"random_{self.state.optimization_rounds}_{np.random.randint(1000)}"
        
        return PromptVariant(
            id=variant_id,
            prompt_text=self._create_prompt_from_elements(base_elements),
            elements=base_elements,
            created_at=datetime.now(),
            exploration_bonus=1.5  # Boost for exploration
        )
    
    def _update_best_variant(self):
        """Update the best performing variant"""
        if self.state.active_variants:
            # Only consider variants with sufficient data
            mature_variants = [
                v for v in self.state.active_variants 
                if v.impressions >= 500
            ]
            
            if mature_variants:
                self.state.best_variant = max(
                    mature_variants,
                    key=lambda x: x.performance_score
                )
    
    def get_champion_variant(self) -> Optional[PromptVariant]:
        """Get current champion variant"""
        return self.state.best_variant
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        if not self.state.active_variants:
            return {"error": "No variants available"}
        
        report = {
            "optimization_rounds": self.state.optimization_rounds,
            "total_variants_tested": len(self.state.active_variants),
            "champion_variant": None,
            "performance_improvement": 0,
            "variant_details": [],
            "optimization_history": list(self.state.performance_history)[-10:]
        }
        
        # Champion details
        if self.state.best_variant:
            report["champion_variant"] = {
                "id": self.state.best_variant.id,
                "elements": self.state.best_variant.elements,
                "performance_score": self.state.best_variant.performance_score,
                "ctr": self.state.best_variant.ctr,
                "conversion_rate": self.state.best_variant.conversion_rate,
                "roas": self.state.best_variant.roas
            }
        
        # All variant details
        for variant in sorted(
            self.state.active_variants,
            key=lambda x: x.performance_score,
            reverse=True
        )[:10]:  # Top 10
            confidence_lower, confidence_upper = self.bandit_optimizer.get_variant_confidence(variant)
            
            report["variant_details"].append({
                "id": variant.id,
                "performance_score": variant.performance_score,
                "impressions": variant.impressions,
                "conversions": variant.conversions,
                "confidence_interval": [confidence_lower, confidence_upper]
            })
        
        # Calculate improvement
        if len(self.state.performance_history) >= 2:
            initial_performance = self.state.performance_history[0]["best_score"]
            current_performance = self.state.performance_history[-1]["best_score"]
            
            if initial_performance > 0:
                report["performance_improvement"] = (
                    (current_performance - initial_performance) / initial_performance * 100
                )
        
        return report
    
    def add_performance_callback(self, callback: Callable):
        """Add callback for performance updates"""
        self.performance_callbacks.append(callback)
    
    async def stop_optimization(self):
        """Stop optimization loop"""
        self.is_running = False


class AdaptiveLearningRate:
    """Adaptive learning rate for optimization"""
    
    def __init__(self, initial_rate: float = 0.1):
        self.initial_rate = initial_rate
        self.current_rate = initial_rate
        self.performance_history = deque(maxlen=10)
    
    def update(self, performance: float):
        """Update learning rate based on performance"""
        self.performance_history.append(performance)
        
        if len(self.performance_history) >= 3:
            # Check if performance is plateauing
            recent_variance = np.var(list(self.performance_history)[-3:])
            
            if recent_variance < 0.001:
                # Plateau detected, increase exploration
                self.current_rate = min(self.current_rate * 1.2, 0.5)
            else:
                # Good progress, reduce exploration
                self.current_rate = max(self.current_rate * 0.95, 0.01)
    
    def get_rate(self) -> float:
        """Get current learning rate"""
        return self.current_rate


# Example usage
async def main():
    """Example real-time optimization usage"""
    # Create optimization engine
    engine = RealtimeOptimizationEngine(
        strategy=OptimizationStrategy.MULTI_ARMED_BANDIT,
        update_interval=timedelta(seconds=10)  # Fast for demo
    )
    
    # Create initial variants
    initial_variants = engine.create_initial_variants(count=5)
    print(f"Created {len(initial_variants)} initial variants")
    
    # Add performance callback
    def performance_update(state: OptimizationState):
        if state.best_variant:
            print(f"\nRound {state.optimization_rounds}:")
            print(f"Best variant: {state.best_variant.id}")
            print(f"Performance: {state.best_variant.performance_score:.3f}")
            print(f"CTR: {state.best_variant.ctr:.2%}")
    
    engine.add_performance_callback(performance_update)
    
    # Run optimization for a short time
    optimization_task = asyncio.create_task(engine.start_optimization())
    
    # Let it run for demo
    await asyncio.sleep(60)
    
    # Stop optimization
    await engine.stop_optimization()
    
    # Get final report
    report = engine.get_performance_report()
    print("\n=== Final Report ===")
    print(f"Rounds completed: {report['optimization_rounds']}")
    print(f"Performance improvement: {report['performance_improvement']:.1f}%")
    
    if report["champion_variant"]:
        print(f"\nChampion Variant:")
        print(f"Elements: {report['champion_variant']['elements']}")
        print(f"CTR: {report['champion_variant']['ctr']:.2%}")
        print(f"ROAS: {report['champion_variant']['roas']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
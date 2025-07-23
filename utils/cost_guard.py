"""
Cost Guard System for JamPacked Autonomous Agents
Prevents infinite loops and runaway costs with budget tokens
"""

import time
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from .structured_logger import get_logger

logger = get_logger(__name__)


class BudgetType(Enum):
    """Types of budget limits"""
    TOKENS = "tokens"
    USD = "usd"
    TIME_SECONDS = "time_seconds"
    ITERATIONS = "iterations"


@dataclass
class Budget:
    """Budget allocation for a goal or operation"""
    budget_type: BudgetType
    initial_amount: float
    remaining_amount: float = field(init=False)
    spent_amount: float = field(default=0.0)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        self.remaining_amount = self.initial_amount - self.spent_amount
    
    def can_spend(self, amount: float) -> bool:
        """Check if we can spend the requested amount"""
        return self.remaining_amount >= amount
    
    def spend(self, amount: float, description: str = "") -> bool:
        """
        Spend from budget
        
        Returns:
            True if spending succeeded, False if insufficient budget
        """
        if not self.can_spend(amount):
            return False
        
        self.spent_amount += amount
        self.remaining_amount -= amount
        self.last_updated = datetime.now()
        
        logger.info(
            "Budget spent",
            budget_type=self.budget_type.value,
            amount_spent=amount,
            remaining=self.remaining_amount,
            description=description
        )
        
        return True
    
    def get_utilization_percent(self) -> float:
        """Get budget utilization as percentage"""
        if self.initial_amount == 0:
            return 0.0
        return (self.spent_amount / self.initial_amount) * 100


class CostGuard:
    """
    Cost guard that prevents runaway spending and infinite loops
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Default budgets
        self.default_budgets = {
            BudgetType.TOKENS: 10000,
            BudgetType.USD: 10.0,
            BudgetType.TIME_SECONDS: 3600,  # 1 hour
            BudgetType.ITERATIONS: 100
        }
        
        # Active budgets per goal/operation
        self.active_budgets: Dict[str, Dict[BudgetType, Budget]] = {}
        
        # Global limits (safety net)
        self.global_limits = {
            BudgetType.TOKENS: self.config.get('global_token_limit', 100000),
            BudgetType.USD: self.config.get('global_usd_limit', 100.0),
            BudgetType.TIME_SECONDS: self.config.get('global_time_limit', 86400),  # 24 hours
            BudgetType.ITERATIONS: self.config.get('global_iteration_limit', 1000)
        }
        
        # Spending history
        self.spending_history: List[Dict[str, Any]] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            BudgetType.TOKENS: 0.8,  # Alert at 80% usage
            BudgetType.USD: 0.8,
            BudgetType.TIME_SECONDS: 0.9,
            BudgetType.ITERATIONS: 0.9
        }
    
    def create_budget(self, 
                     goal_id: str,
                     budget_type: BudgetType,
                     amount: Optional[float] = None) -> Budget:
        """
        Create a budget for a goal
        
        Args:
            goal_id: Unique identifier for the goal
            budget_type: Type of budget to create
            amount: Budget amount (uses default if None)
        """
        if amount is None:
            amount = self.default_budgets.get(budget_type, 1000)
        
        budget = Budget(budget_type=budget_type, initial_amount=amount)
        
        if goal_id not in self.active_budgets:
            self.active_budgets[goal_id] = {}
        
        self.active_budgets[goal_id][budget_type] = budget
        
        logger.info(
            "Budget created",
            goal_id=goal_id,
            budget_type=budget_type.value,
            amount=amount
        )
        
        return budget
    
    def create_goal_budgets(self, 
                          goal_id: str,
                          budgets: Optional[Dict[BudgetType, float]] = None) -> Dict[BudgetType, Budget]:
        """
        Create all budgets for a goal
        
        Args:
            goal_id: Unique identifier for the goal
            budgets: Dict of budget type to amount (uses defaults if None)
        """
        if budgets is None:
            budgets = self.default_budgets
        
        created_budgets = {}
        for budget_type, amount in budgets.items():
            created_budgets[budget_type] = self.create_budget(goal_id, budget_type, amount)
        
        return created_budgets
    
    def check_budget(self, goal_id: str, budget_type: BudgetType, amount: float) -> bool:
        """
        Check if spending is allowed within budget
        
        Returns:
            True if spending is allowed, False otherwise
        """
        if goal_id not in self.active_budgets:
            logger.warning(f"No budgets found for goal {goal_id}")
            return False
        
        if budget_type not in self.active_budgets[goal_id]:
            logger.warning(f"No {budget_type.value} budget found for goal {goal_id}")
            return False
        
        budget = self.active_budgets[goal_id][budget_type]
        return budget.can_spend(amount)
    
    def spend_budget(self, 
                    goal_id: str,
                    budget_type: BudgetType,
                    amount: float,
                    description: str = "",
                    operation: str = "") -> bool:
        """
        Spend from budget with validation
        
        Returns:
            True if spending succeeded, False if over budget
        """
        if not self.check_budget(goal_id, budget_type, amount):
            logger.warning(
                "Budget exceeded",
                goal_id=goal_id,
                budget_type=budget_type.value,
                requested_amount=amount,
                operation=operation
            )
            return False
        
        budget = self.active_budgets[goal_id][budget_type]
        success = budget.spend(amount, description)
        
        if success:
            # Record spending history
            self.spending_history.append({
                "timestamp": datetime.now().isoformat(),
                "goal_id": goal_id,
                "budget_type": budget_type.value,
                "amount": amount,
                "description": description,
                "operation": operation,
                "remaining": budget.remaining_amount
            })
            
            # Check for alerts
            self._check_budget_alerts(goal_id, budget_type, budget)
        
        return success
    
    def get_budget_status(self, goal_id: str) -> Dict[str, Any]:
        """Get budget status for a goal"""
        if goal_id not in self.active_budgets:
            return {"error": f"No budgets found for goal {goal_id}"}
        
        status = {
            "goal_id": goal_id,
            "budgets": {},
            "total_spent_usd": 0.0,
            "warnings": []
        }
        
        for budget_type, budget in self.active_budgets[goal_id].items():
            utilization = budget.get_utilization_percent()
            
            budget_status = {
                "type": budget_type.value,
                "initial": budget.initial_amount,
                "spent": budget.spent_amount,
                "remaining": budget.remaining_amount,
                "utilization_percent": round(utilization, 1),
                "created_at": budget.created_at.isoformat(),
                "last_updated": budget.last_updated.isoformat()
            }
            
            # Add warnings
            if utilization > self.alert_thresholds.get(budget_type, 0.8) * 100:
                status["warnings"].append(f"{budget_type.value} budget at {utilization:.1f}% utilization")
            
            if budget.remaining_amount <= 0:
                status["warnings"].append(f"{budget_type.value} budget exhausted")
            
            status["budgets"][budget_type.value] = budget_status
            
            # Accumulate USD spending
            if budget_type == BudgetType.USD:
                status["total_spent_usd"] = budget.spent_amount
        
        return status
    
    def emergency_stop(self, goal_id: str, reason: str = "Emergency stop"):
        """
        Emergency stop - zero out all budgets for a goal
        """
        if goal_id in self.active_budgets:
            for budget in self.active_budgets[goal_id].values():
                budget.remaining_amount = 0.0
            
            logger.critical(
                "Emergency stop activated",
                goal_id=goal_id,
                reason=reason
            )
    
    def cleanup_expired_budgets(self, max_age_hours: int = 24):
        """Clean up budgets older than max_age_hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        expired_goals = []
        for goal_id, budgets in self.active_budgets.items():
            if all(budget.created_at < cutoff_time for budget in budgets.values()):
                expired_goals.append(goal_id)
        
        for goal_id in expired_goals:
            del self.active_budgets[goal_id]
            logger.info(f"Cleaned up expired budgets for goal {goal_id}")
    
    def _check_budget_alerts(self, goal_id: str, budget_type: BudgetType, budget: Budget):
        """Check if budget alerts should be triggered"""
        utilization = budget.get_utilization_percent()
        threshold = self.alert_thresholds.get(budget_type, 0.8) * 100
        
        if utilization >= threshold:
            logger.warning(
                "Budget alert threshold exceeded",
                goal_id=goal_id,
                budget_type=budget_type.value,
                utilization_percent=round(utilization, 1),
                threshold_percent=round(threshold, 1),
                remaining=budget.remaining_amount
            )
    
    def get_global_spending_summary(self) -> Dict[str, Any]:
        """Get summary of all spending across goals"""
        summary = {
            "total_goals": len(self.active_budgets),
            "total_spent_by_type": {},
            "active_budget_count": 0,
            "exhausted_budget_count": 0,
            "high_utilization_count": 0
        }
        
        for budget_type in BudgetType:
            summary["total_spent_by_type"][budget_type.value] = 0.0
        
        for goal_id, budgets in self.active_budgets.items():
            for budget_type, budget in budgets.items():
                summary["total_spent_by_type"][budget_type.value] += budget.spent_amount
                summary["active_budget_count"] += 1
                
                if budget.remaining_amount <= 0:
                    summary["exhausted_budget_count"] += 1
                
                if budget.get_utilization_percent() > 80:
                    summary["high_utilization_count"] += 1
        
        return summary


# Global cost guard instance
global_cost_guard = CostGuard()


# Convenience functions
def create_goal_budgets(goal_id: str, budgets: Optional[Dict[BudgetType, float]] = None) -> Dict[BudgetType, Budget]:
    """Create budgets for a goal"""
    return global_cost_guard.create_goal_budgets(goal_id, budgets)


def check_budget(goal_id: str, budget_type: BudgetType, amount: float) -> bool:
    """Check if spending is allowed"""
    return global_cost_guard.check_budget(goal_id, budget_type, amount)


def spend_budget(goal_id: str, budget_type: BudgetType, amount: float, 
                description: str = "", operation: str = "") -> bool:
    """Spend from budget"""
    return global_cost_guard.spend_budget(goal_id, budget_type, amount, description, operation)


def get_budget_status(goal_id: str) -> Dict[str, Any]:
    """Get budget status"""
    return global_cost_guard.get_budget_status(goal_id)


def emergency_stop(goal_id: str, reason: str = "Emergency stop"):
    """Emergency stop for a goal"""
    global_cost_guard.emergency_stop(goal_id, reason)


# Context manager for budget-controlled operations
class BudgetGuard:
    """Context manager for budget-controlled operations"""
    
    def __init__(self, goal_id: str, operation: str, max_cost: Dict[BudgetType, float]):
        self.goal_id = goal_id
        self.operation = operation
        self.max_cost = max_cost
        self.spent = {budget_type: 0.0 for budget_type in max_cost.keys()}
    
    def __enter__(self):
        # Check if we have enough budget
        for budget_type, amount in self.max_cost.items():
            if not check_budget(self.goal_id, budget_type, amount):
                raise RuntimeError(f"Insufficient {budget_type.value} budget for operation {self.operation}")
        return self
    
    def spend(self, budget_type: BudgetType, amount: float, description: str = "") -> bool:
        """Spend from budget within this operation"""
        if budget_type not in self.max_cost:
            raise ValueError(f"Budget type {budget_type.value} not allocated for this operation")
        
        if self.spent[budget_type] + amount > self.max_cost[budget_type]:
            logger.warning(f"Operation {self.operation} would exceed allocated {budget_type.value} budget")
            return False
        
        success = spend_budget(self.goal_id, budget_type, amount, description, self.operation)
        if success:
            self.spent[budget_type] += amount
        
        return success
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Convert enum keys to strings for logging
        spent_str = {budget_type.value: amount for budget_type, amount in self.spent.items()}
        max_cost_str = {budget_type.value: amount for budget_type, amount in self.max_cost.items()}
        
        logger.info(
            f"Budget guard completed for {self.operation}",
            goal_id=self.goal_id,
            spent=spent_str,
            max_cost=max_cost_str
        )
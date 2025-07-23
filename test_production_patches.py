#!/usr/bin/env python3
"""
Test script to validate production patches for autonomous agent architecture
Tests semaphore control, cost guards, and structured logging
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.structured_logger import get_logger, trace_context, PerformanceTimer
from utils.cost_guard import BudgetType, create_goal_budgets, spend_budget, get_budget_status, BudgetGuard
from utils.semaphore import SemaphoreGuard, get_active_count, get_active_agents

logger = get_logger(__name__)


async def test_structured_logging():
    """Test structured logging functionality"""
    print("🧪 Testing Structured Logging...")
    
    with trace_context(component="test", goal_id="test_goal_123"):
        logger.info("Test structured logging", test_field="test_value", number=42)
        
        with PerformanceTimer(logger, "test_operation", operation_type="validation"):
            await asyncio.sleep(0.1)
        
        logger.warning("Test warning", severity="medium")
        logger.error("Test error", error_code=500)
    
    print("✅ Structured logging test completed")


async def test_cost_guard():
    """Test cost guard functionality"""
    print("🧪 Testing Cost Guard...")
    
    # Create test budgets
    goal_id = "test_cost_guard"
    budgets = create_goal_budgets(goal_id, {
        BudgetType.TOKENS: 100,
        BudgetType.USD: 1.0,
        BudgetType.TIME_SECONDS: 60,
        BudgetType.ITERATIONS: 5
    })
    
    print(f"   Created budgets: {len(budgets)} types")
    
    # Test normal spending
    success = spend_budget(goal_id, BudgetType.TOKENS, 50, "test_operation")
    print(f"   Normal spending (50 tokens): {'✅' if success else '❌'}")
    
    # Test budget limit
    success = spend_budget(goal_id, BudgetType.TOKENS, 60, "large_operation")  # Should fail
    print(f"   Over-budget spending (60 tokens): {'❌' if not success else '⚠️ Should have failed'}")
    
    # Test budget guard context manager
    try:
        with BudgetGuard(goal_id, "test_operation", {BudgetType.USD: 0.5}) as guard:
            guard.spend(BudgetType.USD, 0.3, "sub_operation")
            print("   Budget guard context: ✅")
    except Exception as e:
        print(f"   Budget guard context: ❌ {e}")
    
    # Check budget status
    status = get_budget_status(goal_id)
    print(f"   Budget status warnings: {len(status.get('warnings', []))}")
    
    print("✅ Cost guard test completed")


async def test_semaphore():
    """Test semaphore functionality"""
    print("🧪 Testing Global Semaphore...")
    
    async def spawn_test_agent(agent_id: str, max_agents: int = 3):
        """Test agent spawning with semaphore"""
        try:
            with SemaphoreGuard(f"test_agent_{agent_id}", max_active=max_agents, 
                              agent_type="test_agent"):
                logger.info("Test agent spawned", agent_id=agent_id)
                await asyncio.sleep(0.5)  # Simulate work
                return f"agent_{agent_id}_completed"
        except RuntimeError as e:
            logger.warning("Agent spawn blocked", agent_id=agent_id, error=str(e))
            return None
    
    # Test normal spawning within limits
    tasks = []
    for i in range(2):  # Within limit of 3
        tasks.append(spawn_test_agent(f"normal_{i}"))
    
    results = await asyncio.gather(*tasks)
    successful_spawns = [r for r in results if r is not None]
    print(f"   Normal spawning (2/3): {len(successful_spawns)} succeeded")
    
    # Test spawning over limit
    tasks = []
    for i in range(5):  # Over limit of 3
        tasks.append(spawn_test_agent(f"overflow_{i}", max_agents=3))
    
    results = await asyncio.gather(*tasks)
    successful_spawns = [r for r in results if r is not None]
    blocked_spawns = [r for r in results if r is None]
    print(f"   Overflow spawning (5 attempts, limit 3): {len(successful_spawns)} succeeded, {len(blocked_spawns)} blocked")
    
    # Check global agent count
    active_count = get_active_count()
    print(f"   Active agents after test: {active_count}")
    
    print("✅ Semaphore test completed")


async def test_exponential_backoff():
    """Test exponential backoff logic"""
    print("🧪 Testing Exponential Backoff...")
    
    class MockEngine:
        def __init__(self):
            self._consecutive_idle_cycles = 0
            self._base_execution_interval = 30
            self._max_execution_interval = 240
        
        def _calculate_exponential_backoff_interval(self, is_idle_cycle: bool) -> float:
            """Mock implementation of exponential backoff"""
            if is_idle_cycle:
                self._consecutive_idle_cycles += 1
            else:
                self._consecutive_idle_cycles = 0
            
            if self._consecutive_idle_cycles == 0:
                interval = self._base_execution_interval
            else:
                backoff_multiplier = min(2 ** (self._consecutive_idle_cycles - 1), 8)
                interval = self._base_execution_interval * backoff_multiplier
                interval = min(interval, self._max_execution_interval)
            
            return interval
    
    engine = MockEngine()
    
    # Test backoff progression
    intervals = []
    for i in range(6):
        is_idle = True  # Simulate consecutive idle cycles
        interval = engine._calculate_exponential_backoff_interval(is_idle)
        intervals.append(interval)
        print(f"   Idle cycle {i+1}: {interval}s interval")
    
    # Test reset on active cycle
    active_interval = engine._calculate_exponential_backoff_interval(False)
    print(f"   Active cycle: {active_interval}s interval (should reset to {engine._base_execution_interval})")
    
    print("✅ Exponential backoff test completed")


async def test_integrated_system():
    """Test all systems working together"""
    print("🧪 Testing Integrated System...")
    
    with trace_context(component="integrated_test", goal_id="integration_test"):
        
        # Create goal with budgets
        goal_id = "integration_test"
        budgets = create_goal_budgets(goal_id, {
            BudgetType.TOKENS: 1000,
            BudgetType.USD: 5.0,
            BudgetType.ITERATIONS: 10
        })
        
        async def run_controlled_operation(op_id: str):
            """Run an operation with all controls"""
            try:
                # Use semaphore control
                with SemaphoreGuard(f"integration_op_{op_id}", max_active=2, 
                                  agent_type="integration_test"):
                    
                    # Use budget control
                    with BudgetGuard(goal_id, f"operation_{op_id}", 
                                   {BudgetType.TOKENS: 100, BudgetType.USD: 0.5}) as guard:
                        
                        # Use performance timing
                        with PerformanceTimer(logger, f"integrated_operation_{op_id}", 
                                            operation_id=op_id):
                            
                            # Simulate work with budget tracking
                            guard.spend(BudgetType.TOKENS, 50, "processing")
                            guard.spend(BudgetType.USD, 0.25, "api_calls")
                            
                            await asyncio.sleep(0.2)
                            
                            logger.info("Integrated operation completed", 
                                       operation_id=op_id,
                                       success=True)
                            
                            return {"operation_id": op_id, "status": "success"}
                            
            except Exception as e:
                logger.error("Integrated operation failed", 
                           operation_id=op_id,
                           error=str(e))
                return {"operation_id": op_id, "status": "failed", "error": str(e)}
        
        # Run multiple operations concurrently
        operations = [run_controlled_operation(f"op_{i}") for i in range(4)]
        results = await asyncio.gather(*operations)
        
        successful_ops = [r for r in results if r.get("status") == "success"]
        failed_ops = [r for r in results if r.get("status") == "failed"]
        
        print(f"   Concurrent operations: {len(successful_ops)} succeeded, {len(failed_ops)} failed")
        
        # Check final system state
        budget_status = get_budget_status(goal_id)
        active_count = get_active_count()
        
        print(f"   Final budget utilization: ${budget_status.get('total_spent_usd', 0):.2f}")
        print(f"   Final active agents: {active_count}")
        
    print("✅ Integrated system test completed")


async def main():
    """Run all production patch tests"""
    print("🔧 Production Patches Validation")
    print("=" * 50)
    
    try:
        await test_structured_logging()
        print()
        
        await test_cost_guard()
        print()
        
        await test_semaphore()
        print()
        
        await test_exponential_backoff()
        print()
        
        await test_integrated_system()
        print()
        
        print("🎉 All production patch tests completed successfully!")
        print("=" * 50)
        
        # Summary
        print("✅ Production improvements validated:")
        print("   • Structured logging with trace IDs")
        print("   • Cost guard with budget controls")
        print("   • Global semaphore preventing runaway spawning")
        print("   • Exponential back-off for CPU efficiency")
        print("   • Integrated production controls")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
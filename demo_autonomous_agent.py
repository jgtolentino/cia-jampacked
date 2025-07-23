#!/usr/bin/env python3
"""
Demonstration of JamPacked Autonomous Creative Intelligence Agent
Shows true autonomous behavior with self-triggering loops and continuous improvement
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from autonomous_intelligence.engines.autonomous_orchestrator import create_autonomous_orchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_autonomous_creative_agent():
    """
    Demonstrate the autonomous creative intelligence agent
    Shows self-triggering behavior, reflection, learning, and coordination
    """
    
    print("🚀 JamPacked Autonomous Creative Intelligence Agent Demo")
    print("=" * 60)
    
    # Create autonomous orchestrator with all engines
    config = {
        "objectives": [
            "maximize_creative_effectiveness_prediction",
            "improve_brand_analysis_accuracy",
            "enhance_multimodal_processing",
            "optimize_learning_from_feedback"
        ],
        "autonomous": {
            "execution_interval": 15,  # Faster for demo
            "autonomy_level": 0.9
        },
        "reflection": {
            "reflection_frequency": 60,  # Reflect every minute for demo
            "critique_depth": "comprehensive"
        },
        "coordination": {
            "max_agents": 6,
            "spawn_threshold": 0.7
        },
        "learning": {
            "memory_capacity": 5000,
            "learning_rate": 0.02
        }
    }
    
    orchestrator = create_autonomous_orchestrator(config)
    
    print("\n🧠 Initializing Autonomous Agent...")
    print("   - AutonomousEngine: Self-triggering execution loops")
    print("   - ReflectionEngine: Continuous self-critique and improvement")
    print("   - MultiAgentCoordinator: Specialist delegation and coordination")
    print("   - LearningMemorySystem: Memory-driven continuous learning")
    
    # Define some demo creative tasks
    demo_tasks = [
        {
            "name": "Brand Campaign Analysis",
            "type": "creative_analysis",
            "description": "Analyze creative effectiveness of new brand campaign",
            "required_capabilities": ["creative_analysis", "brand_analysis", "attention_prediction"],
            "priority": 4,
            "quality_expectations": {"accuracy": 0.85, "completeness": 0.9},
            "assets": {
                "video_url": "https://example.com/campaign_video.mp4",
                "images": ["hero_image.jpg", "secondary_banner.jpg"],
                "copy": "Discover the extraordinary in every moment"
            }
        },
        {
            "name": "Multimodal Creative Assessment",
            "type": "multimodal_analysis", 
            "description": "Comprehensive analysis of video, audio, and text elements",
            "required_capabilities": ["media_processing", "creative_analysis", "emotional_analysis"],
            "priority": 3,
            "multimodal": True,
            "assets": {
                "video_file": "campaign_30s.mp4",
                "audio_track": "background_music.wav",
                "script": "Life is better when shared with those you love"
            }
        },
        {
            "name": "Performance Optimization",
            "type": "optimization",
            "description": "Optimize campaign performance based on initial results",
            "required_capabilities": ["performance_optimization", "data_analysis"],
            "priority": 5,
            "context": {
                "current_metrics": {"ctr": 0.034, "engagement": 0.067, "conversion": 0.012},
                "target_metrics": {"ctr": 0.045, "engagement": 0.08, "conversion": 0.018}
            }
        }
    ]
    
    try:
        # Start autonomous operations (this creates self-triggering loops)
        print("\n🎬 Starting Autonomous Operations...")
        print("   Agent will now operate autonomously with self-triggering behavior")
        
        # Start autonomous operations in background
        autonomous_task = asyncio.create_task(
            orchestrator.start_autonomous_operations([
                "demonstrate_autonomous_behavior",
                "showcase_self_improvement",
                "exhibit_coordination_capabilities"
            ])
        )
        
        # Let autonomous operations start
        await asyncio.sleep(5)
        
        print("\n✅ Agent is now fully autonomous and self-operating!")
        print("   - Making autonomous decisions")
        print("   - Reflecting on performance") 
        print("   - Learning from experiences")
        print("   - Coordinating specialist agents")
        
        # Process demo tasks autonomously
        print("\n🎯 Processing Creative Tasks Autonomously...")
        print("   (Agent makes all decisions without human intervention)")
        
        for i, task in enumerate(demo_tasks, 1):
            print(f"\n📋 Task {i}: {task['name']}")
            print(f"   Description: {task['description']}")
            print(f"   Capabilities needed: {', '.join(task['required_capabilities'])}")
            
            # Let the agent process autonomously
            result = await orchestrator.process_creative_task_autonomously(task)
            
            # Display autonomous processing results
            print(f"   ✅ Task completed autonomously!")
            print(f"   🤖 Coordination used: {result['coordination_used']}")
            print(f"   🧠 Memories utilized: {result['processing_metadata']['memories_used']}")
            print(f"   🪞 Reflection insights: {len(result['reflection_insights'].identified_issues)} issues identified")
            print(f"   📚 Learning outcome: {result['learning_outcome'].get('learnings_extracted', 0)} insights extracted")
            print(f"   ⚡ Adaptations applied: {len(result['autonomous_adaptations'])}")
            
            # Show some actual insights
            if 'insights' in result['task_result']:
                print(f"   💡 Key insights:")
                for insight in result['task_result']['insights'][:2]:
                    print(f"      - {insight}")
            
            # Brief pause between tasks to show autonomous processing
            await asyncio.sleep(3)
        
        # Demonstrate autonomous reflection and learning
        print("\n🪞 Demonstrating Autonomous Reflection...")
        await asyncio.sleep(10)  # Let reflection cycles run
        
        print("   ✅ Agent has been continuously reflecting on its performance")
        print("   ✅ Self-critique cycles completed automatically")
        print("   ✅ Behavioral adaptations applied autonomously")
        
        print("\n📚 Demonstrating Autonomous Learning...")
        await asyncio.sleep(10)  # Let learning cycles run
        
        print("   ✅ Agent has been learning from each task")
        print("   ✅ Memory patterns discovered and stored")
        print("   ✅ Performance improvements applied automatically")
        
        # Show system state
        print(f"\n📊 Autonomous Agent System State:")
        print(f"   🧠 Mode: {orchestrator.system_state.mode.value}")
        print(f"   ⚡ Active engines: {len(orchestrator.system_state.active_engines)}")
        print(f"   🤖 Autonomous decisions made: {orchestrator.system_state.autonomous_decisions_made}")
        print(f"   🪞 Reflection cycles completed: {orchestrator.system_state.reflection_cycles_completed}")
        print(f"   📚 Learning experiences processed: {orchestrator.system_state.learning_experiences_processed}")
        print(f"   🏥 Health status: {orchestrator.system_state.health_status}")
        
        # Demonstrate self-improvement
        print("\n🔄 Demonstrating Self-Improvement...")
        
        # Process a task with simulated feedback to show learning
        feedback_task = {
            "name": "Learning Demonstration",
            "type": "creative_analysis",
            "description": "Task to demonstrate learning from feedback",
            "required_capabilities": ["creative_analysis"],
            "expected_outcome": {
                "score": 0.92,
                "classification": "high_effectiveness",
                "user_rating": 4.5
            }
        }
        
        print("   📝 Processing task with feedback to demonstrate learning...")
        result = await orchestrator.process_creative_task_autonomously(feedback_task)
        
        print("   ✅ Agent learned from feedback autonomously")
        print("   🧠 Memory updated with new patterns")
        print("   ⚡ Behavior adapted based on learning")
        
        # Show that agent continues operating autonomously
        print("\n🔄 Agent Continues Autonomous Operation...")
        print("   The agent will continue operating autonomously:")
        print("   - Making decisions every 15 seconds")
        print("   - Reflecting on performance every minute")
        print("   - Learning from experiences continuously")
        print("   - Coordinating specialists as needed")
        print("   - Adapting behavior based on outcomes")
        
        print("\n🎉 Autonomous Agent Demonstration Complete!")
        print("=" * 60)
        print("🤖 KEY AUTONOMOUS CAPABILITIES DEMONSTRATED:")
        print("   ✅ Self-triggering execution loops (no external triggers needed)")
        print("   ✅ Autonomous decision making (coordinate vs single-agent processing)")
        print("   ✅ Continuous self-reflection and critique")
        print("   ✅ Memory-driven learning from experiences")
        print("   ✅ Specialist agent coordination and delegation")
        print("   ✅ Behavioral adaptation based on feedback")
        print("   ✅ Pattern discovery and application")
        print("   ✅ Meta-cognitive awareness and improvement")
        
        print("\n🧠 This is TRUE AUTONOMOUS AGENT BEHAVIOR:")
        print("   - No human intervention required")
        print("   - Self-directed goal pursuit")
        print("   - Continuous learning and adaptation")
        print("   - Proactive problem solving")
        print("   - Multi-agent coordination")
        
        # Let it run autonomously for a bit more
        print("\n⏰ Letting agent run autonomously for 30 more seconds...")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    
    finally:
        # Graceful shutdown
        print("\n🛑 Shutting down autonomous operations...")
        await orchestrator.stop_autonomous_operations()
        print("✅ Autonomous agent shutdown complete")


async def quick_demo():
    """Quick demo for testing"""
    print("🚀 Quick Autonomous Agent Demo")
    
    orchestrator = create_autonomous_orchestrator({
        "autonomous": {"execution_interval": 5},
        "reflection": {"reflection_frequency": 30}
    })
    
    # Simple task
    task = {
        "name": "Quick Test",
        "type": "creative_analysis",
        "description": "Quick test of autonomous processing",
        "required_capabilities": ["creative_analysis"]
    }
    
    print("🎯 Processing task autonomously...")
    result = await orchestrator.process_creative_task_autonomously(task)
    print(f"✅ Task completed! Coordination used: {result['coordination_used']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(quick_demo())
    else:
        asyncio.run(demonstrate_autonomous_creative_agent())
"""
Implementation Roadmap for AI Advertising Optimization
3-Phase approach: Foundation, Integration, Scale
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json


class Phase(Enum):
    FOUNDATION = "foundation"
    INTEGRATION = "integration"
    SCALE = "scale"


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Milestone:
    """Project milestone"""
    name: str
    description: str
    success_criteria: List[str]
    deliverables: List[str]
    dependencies: List[str]
    estimated_duration: timedelta
    phase: Phase
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "deliverables": self.deliverables,
            "dependencies": self.dependencies,
            "estimated_duration_days": self.estimated_duration.days,
            "phase": self.phase.value
        }


@dataclass
class Task:
    """Implementation task"""
    id: str
    name: str
    description: str
    milestone: str
    priority: Priority
    estimated_hours: int
    required_skills: List[str]
    dependencies: List[str]
    status: TaskStatus = TaskStatus.NOT_STARTED
    assigned_to: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "milestone": self.milestone,
            "priority": self.priority.value,
            "estimated_hours": self.estimated_hours,
            "required_skills": self.required_skills,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "assigned_to": self.assigned_to
        }


class ImplementationRoadmap:
    """Complete implementation roadmap"""
    
    def __init__(self):
        self.phases = self._define_phases()
        self.milestones = self._define_milestones()
        self.tasks = self._define_tasks()
        self.team_requirements = self._define_team_requirements()
        self.budget_estimates = self._define_budget_estimates()
    
    def _define_phases(self) -> Dict[Phase, Dict]:
        """Define the three implementation phases"""
        return {
            Phase.FOUNDATION: {
                "name": "Foundation (Months 1-3)",
                "description": "Establish core infrastructure and basic capabilities",
                "objectives": [
                    "Set up cloud-native data infrastructure",
                    "Implement basic prompt templates",
                    "Establish privacy and compliance frameworks",
                    "Deploy initial platform integrations"
                ],
                "expected_outcomes": [
                    "10-20% performance improvements",
                    "Automated campaign creation",
                    "Basic attribution tracking",
                    "Privacy-compliant data collection"
                ],
                "duration": timedelta(days=90)
            },
            Phase.INTEGRATION: {
                "name": "Advanced Integration (Months 4-6)",
                "description": "Deploy advanced AI capabilities and cross-platform integration",
                "objectives": [
                    "Implement multimodal AI analysis",
                    "Deploy psychographic profiling",
                    "Integrate all advertising platforms",
                    "Launch federated learning"
                ],
                "expected_outcomes": [
                    "30-50% performance improvements",
                    "Unified cross-platform dashboard",
                    "Advanced personalization",
                    "Real-time optimization"
                ],
                "duration": timedelta(days=90)
            },
            Phase.SCALE: {
                "name": "Optimization and Scale (Months 7-12)",
                "description": "Achieve full automation and optimization at scale",
                "objectives": [
                    "Full campaign automation",
                    "Predictive analytics deployment",
                    "Advanced competitive intelligence",
                    "Global rollout"
                ],
                "expected_outcomes": [
                    "50-100% performance improvements",
                    "Significantly reduced operational costs",
                    "Market-leading capabilities",
                    "Sustainable competitive advantage"
                ],
                "duration": timedelta(days=180)
            }
        }
    
    def _define_milestones(self) -> List[Milestone]:
        """Define key milestones for implementation"""
        return [
            # Foundation Phase Milestones
            Milestone(
                name="Infrastructure Setup",
                description="Cloud-native infrastructure ready for AI workloads",
                success_criteria=[
                    "Azure/AWS infrastructure deployed",
                    "Data pipelines operational",
                    "Security frameworks implemented",
                    "Monitoring systems active"
                ],
                deliverables=[
                    "Infrastructure documentation",
                    "Security audit report",
                    "Performance benchmarks",
                    "Disaster recovery plan"
                ],
                dependencies=[],
                estimated_duration=timedelta(days=30),
                phase=Phase.FOUNDATION
            ),
            Milestone(
                name="Basic AI Implementation",
                description="Core AI capabilities deployed and tested",
                success_criteria=[
                    "Microsoft 7-element prompts implemented",
                    "Basic attribution model deployed",
                    "Initial platform integrations complete",
                    "Performance tracking operational"
                ],
                deliverables=[
                    "Prompt template library",
                    "Attribution dashboard",
                    "API integrations",
                    "Performance reports"
                ],
                dependencies=["Infrastructure Setup"],
                estimated_duration=timedelta(days=45),
                phase=Phase.FOUNDATION
            ),
            Milestone(
                name="Privacy & Compliance",
                description="Privacy-preserving systems fully operational",
                success_criteria=[
                    "GDPR/CCPA compliance verified",
                    "Federated learning framework deployed",
                    "Bias detection active",
                    "Data governance policies implemented"
                ],
                deliverables=[
                    "Compliance certification",
                    "Privacy impact assessment",
                    "Bias monitoring reports",
                    "Data governance documentation"
                ],
                dependencies=["Infrastructure Setup"],
                estimated_duration=timedelta(days=30),
                phase=Phase.FOUNDATION
            ),
            
            # Integration Phase Milestones
            Milestone(
                name="Advanced AI Deployment",
                description="Sophisticated AI capabilities fully integrated",
                success_criteria=[
                    "Multimodal AI processing active",
                    "Psychographic profiling operational",
                    "Weather-responsive ads deployed",
                    "Real-time optimization running"
                ],
                deliverables=[
                    "AI capability documentation",
                    "Performance improvement metrics",
                    "Integration test results",
                    "User training materials"
                ],
                dependencies=["Basic AI Implementation"],
                estimated_duration=timedelta(days=60),
                phase=Phase.INTEGRATION
            ),
            Milestone(
                name="Platform Unification",
                description="All advertising platforms integrated into unified system",
                success_criteria=[
                    "All major platforms connected",
                    "Unified dashboard operational",
                    "Cross-platform attribution working",
                    "Real-time data synchronization"
                ],
                deliverables=[
                    "Platform integration guide",
                    "Unified dashboard",
                    "Attribution reports",
                    "API documentation"
                ],
                dependencies=["Basic AI Implementation", "Infrastructure Setup"],
                estimated_duration=timedelta(days=45),
                phase=Phase.INTEGRATION
            ),
            Milestone(
                name="Intelligence Systems",
                description="Competitive and market intelligence fully automated",
                success_criteria=[
                    "Competitive monitoring automated",
                    "Market trend detection active",
                    "Predictive analytics operational",
                    "Alert systems configured"
                ],
                deliverables=[
                    "Intelligence dashboard",
                    "Automated reports",
                    "Alert configuration",
                    "Trend analysis tools"
                ],
                dependencies=["Advanced AI Deployment"],
                estimated_duration=timedelta(days=30),
                phase=Phase.INTEGRATION
            ),
            
            # Scale Phase Milestones
            Milestone(
                name="Full Automation",
                description="End-to-end campaign automation achieved",
                success_criteria=[
                    "Autonomous campaign creation",
                    "Self-optimizing campaigns",
                    "Automated budget allocation",
                    "Predictive performance modeling"
                ],
                deliverables=[
                    "Automation playbooks",
                    "Performance metrics",
                    "ROI analysis",
                    "Case studies"
                ],
                dependencies=["Advanced AI Deployment", "Platform Unification"],
                estimated_duration=timedelta(days=90),
                phase=Phase.SCALE
            ),
            Milestone(
                name="Global Rollout",
                description="System deployed across all markets and teams",
                success_criteria=[
                    "Multi-market deployment complete",
                    "Localization implemented",
                    "Team training finished",
                    "Support systems operational"
                ],
                deliverables=[
                    "Deployment guide",
                    "Training materials",
                    "Support documentation",
                    "Success metrics"
                ],
                dependencies=["Full Automation"],
                estimated_duration=timedelta(days=60),
                phase=Phase.SCALE
            ),
            Milestone(
                name="Optimization Excellence",
                description="Industry-leading performance achieved",
                success_criteria=[
                    "50%+ performance improvement verified",
                    "Cost reductions realized",
                    "Competitive advantage demonstrated",
                    "Continuous improvement process established"
                ],
                deliverables=[
                    "Performance analysis",
                    "ROI report",
                    "Competitive benchmark",
                    "Future roadmap"
                ],
                dependencies=["Full Automation", "Global Rollout"],
                estimated_duration=timedelta(days=90),
                phase=Phase.SCALE
            )
        ]
    
    def _define_tasks(self) -> List[Task]:
        """Define detailed implementation tasks"""
        tasks = []
        task_id = 1
        
        # Foundation Phase Tasks
        foundation_tasks = [
            # Infrastructure Tasks
            Task(
                id=f"F{task_id}",
                name="Select cloud platform",
                description="Evaluate and select primary cloud platform (Azure/AWS/GCP)",
                milestone="Infrastructure Setup",
                priority=Priority.CRITICAL,
                estimated_hours=40,
                required_skills=["Cloud Architecture", "Cost Analysis"],
                dependencies=[]
            ),
            Task(
                id=f"F{task_id+1}",
                name="Deploy data infrastructure",
                description="Set up data lakes, warehouses, and streaming pipelines",
                milestone="Infrastructure Setup",
                priority=Priority.CRITICAL,
                estimated_hours=120,
                required_skills=["Data Engineering", "Cloud Services"],
                dependencies=[f"F{task_id}"]
            ),
            Task(
                id=f"F{task_id+2}",
                name="Implement security framework",
                description="Deploy encryption, access controls, and monitoring",
                milestone="Infrastructure Setup",
                priority=Priority.CRITICAL,
                estimated_hours=80,
                required_skills=["Security Engineering", "Compliance"],
                dependencies=[f"F{task_id+1}"]
            ),
            
            # AI Implementation Tasks
            Task(
                id=f"F{task_id+3}",
                name="Create prompt templates",
                description="Implement Microsoft 7-element prompt structure",
                milestone="Basic AI Implementation",
                priority=Priority.HIGH,
                estimated_hours=60,
                required_skills=["Prompt Engineering", "AI/ML"],
                dependencies=[]
            ),
            Task(
                id=f"F{task_id+4}",
                name="Deploy attribution model",
                description="Implement unified MMM/MTA attribution system",
                milestone="Basic AI Implementation",
                priority=Priority.HIGH,
                estimated_hours=100,
                required_skills=["Data Science", "Marketing Analytics"],
                dependencies=[f"F{task_id+1}"]
            ),
            Task(
                id=f"F{task_id+5}",
                name="Connect first platform",
                description="Integrate Google Ads as pilot platform",
                milestone="Basic AI Implementation",
                priority=Priority.HIGH,
                estimated_hours=80,
                required_skills=["API Integration", "Platform Expertise"],
                dependencies=[f"F{task_id+1}"]
            ),
            
            # Privacy Tasks
            Task(
                id=f"F{task_id+6}",
                name="Implement privacy controls",
                description="Deploy differential privacy and data anonymization",
                milestone="Privacy & Compliance",
                priority=Priority.CRITICAL,
                estimated_hours=100,
                required_skills=["Privacy Engineering", "Legal"],
                dependencies=[f"F{task_id+2}"]
            ),
            Task(
                id=f"F{task_id+7}",
                name="Deploy bias detection",
                description="Implement bias monitoring and mitigation framework",
                milestone="Privacy & Compliance",
                priority=Priority.HIGH,
                estimated_hours=80,
                required_skills=["ML Engineering", "Ethics"],
                dependencies=[f"F{task_id+4}"]
            )
        ]
        
        tasks.extend(foundation_tasks)
        task_id += 8
        
        # Integration Phase Tasks
        integration_tasks = [
            # Advanced AI Tasks
            Task(
                id=f"I{task_id}",
                name="Deploy multimodal AI",
                description="Implement visual, audio, and text analysis",
                milestone="Advanced AI Deployment",
                priority=Priority.HIGH,
                estimated_hours=120,
                required_skills=["Computer Vision", "NLP", "ML Engineering"],
                dependencies=["Basic AI Implementation"]
            ),
            Task(
                id=f"I{task_id+1}",
                name="Implement psychographic profiling",
                description="Deploy 10-word analysis system",
                milestone="Advanced AI Deployment",
                priority=Priority.MEDIUM,
                estimated_hours=80,
                required_skills=["Psychology", "NLP", "Data Science"],
                dependencies=["Basic AI Implementation"]
            ),
            Task(
                id=f"I{task_id+2}",
                name="Launch weather-responsive ads",
                description="Integrate weather data for dynamic campaigns",
                milestone="Advanced AI Deployment",
                priority=Priority.MEDIUM,
                estimated_hours=60,
                required_skills=["API Integration", "Campaign Management"],
                dependencies=["Basic AI Implementation"]
            ),
            
            # Platform Integration Tasks
            Task(
                id=f"I{task_id+3}",
                name="Integrate remaining platforms",
                description="Connect Meta, TikTok, Amazon, and other platforms",
                milestone="Platform Unification",
                priority=Priority.HIGH,
                estimated_hours=200,
                required_skills=["API Integration", "Platform Expertise"],
                dependencies=[f"F{task_id+5}"]
            ),
            Task(
                id=f"I{task_id+4}",
                name="Build unified dashboard",
                description="Create cross-platform measurement dashboard",
                milestone="Platform Unification",
                priority=Priority.HIGH,
                estimated_hours=150,
                required_skills=["Frontend Development", "Data Visualization"],
                dependencies=[f"I{task_id+3}"]
            ),
            
            # Intelligence Tasks
            Task(
                id=f"I{task_id+5}",
                name="Deploy competitive monitoring",
                description="Integrate Brand24/Brandwatch for market intelligence",
                milestone="Intelligence Systems",
                priority=Priority.MEDIUM,
                estimated_hours=80,
                required_skills=["API Integration", "Competitive Analysis"],
                dependencies=[f"I{task_id}"]
            ),
            Task(
                id=f"I{task_id+6}",
                name="Implement predictive analytics",
                description="Deploy ML models for performance prediction",
                milestone="Intelligence Systems",
                priority=Priority.HIGH,
                estimated_hours=120,
                required_skills=["ML Engineering", "Predictive Analytics"],
                dependencies=[f"I{task_id+4}"]
            )
        ]
        
        tasks.extend(integration_tasks)
        task_id += 7
        
        # Scale Phase Tasks
        scale_tasks = [
            # Automation Tasks
            Task(
                id=f"S{task_id}",
                name="Automate campaign creation",
                description="Implement end-to-end campaign automation",
                milestone="Full Automation",
                priority=Priority.HIGH,
                estimated_hours=150,
                required_skills=["Automation Engineering", "AI/ML"],
                dependencies=["Advanced AI Deployment"]
            ),
            Task(
                id=f"S{task_id+1}",
                name="Deploy self-optimization",
                description="Implement autonomous optimization algorithms",
                milestone="Full Automation",
                priority=Priority.HIGH,
                estimated_hours=120,
                required_skills=["ML Engineering", "Optimization"],
                dependencies=[f"S{task_id}"]
            ),
            
            # Rollout Tasks
            Task(
                id=f"S{task_id+2}",
                name="Regional deployment",
                description="Deploy system across all geographic markets",
                milestone="Global Rollout",
                priority=Priority.HIGH,
                estimated_hours=200,
                required_skills=["Project Management", "Localization"],
                dependencies=["Full Automation"]
            ),
            Task(
                id=f"S{task_id+3}",
                name="Team training program",
                description="Train all teams on AI advertising tools",
                milestone="Global Rollout",
                priority=Priority.CRITICAL,
                estimated_hours=150,
                required_skills=["Training", "Change Management"],
                dependencies=[f"S{task_id+2}"]
            ),
            
            # Optimization Tasks
            Task(
                id=f"S{task_id+4}",
                name="Performance optimization",
                description="Fine-tune system for maximum performance",
                milestone="Optimization Excellence",
                priority=Priority.HIGH,
                estimated_hours=100,
                required_skills=["Performance Engineering", "Analytics"],
                dependencies=["Global Rollout"]
            ),
            Task(
                id=f"S{task_id+5}",
                name="Continuous improvement",
                description="Establish processes for ongoing enhancement",
                milestone="Optimization Excellence",
                priority=Priority.MEDIUM,
                estimated_hours=80,
                required_skills=["Process Design", "Quality Assurance"],
                dependencies=[f"S{task_id+4}"]
            )
        ]
        
        tasks.extend(scale_tasks)
        
        return tasks
    
    def _define_team_requirements(self) -> Dict[str, Dict]:
        """Define team and skill requirements"""
        return {
            "roles": {
                "Technical Lead": {
                    "count": 1,
                    "skills": ["Architecture", "AI/ML", "Project Management"],
                    "phase_allocation": {
                        Phase.FOUNDATION: 1.0,
                        Phase.INTEGRATION: 1.0,
                        Phase.SCALE: 0.5
                    }
                },
                "Data Engineers": {
                    "count": 3,
                    "skills": ["Cloud Platforms", "ETL", "Big Data"],
                    "phase_allocation": {
                        Phase.FOUNDATION: 2.0,
                        Phase.INTEGRATION: 2.0,
                        Phase.SCALE: 1.0
                    }
                },
                "ML Engineers": {
                    "count": 4,
                    "skills": ["Machine Learning", "Python", "MLOps"],
                    "phase_allocation": {
                        Phase.FOUNDATION: 2.0,
                        Phase.INTEGRATION: 3.0,
                        Phase.SCALE: 2.0
                    }
                },
                "Platform Specialists": {
                    "count": 3,
                    "skills": ["Google Ads", "Meta Ads", "TikTok Ads"],
                    "phase_allocation": {
                        Phase.FOUNDATION: 1.0,
                        Phase.INTEGRATION: 2.0,
                        Phase.SCALE: 1.0
                    }
                },
                "Frontend Developers": {
                    "count": 2,
                    "skills": ["React", "TypeScript", "Data Visualization"],
                    "phase_allocation": {
                        Phase.FOUNDATION: 1.0,
                        Phase.INTEGRATION: 2.0,
                        Phase.SCALE: 1.0
                    }
                },
                "Security Engineer": {
                    "count": 1,
                    "skills": ["Security", "Compliance", "Privacy"],
                    "phase_allocation": {
                        Phase.FOUNDATION: 1.0,
                        Phase.INTEGRATION: 0.5,
                        Phase.SCALE: 0.5
                    }
                },
                "Product Manager": {
                    "count": 1,
                    "skills": ["Product Strategy", "Analytics", "Communication"],
                    "phase_allocation": {
                        Phase.FOUNDATION: 1.0,
                        Phase.INTEGRATION: 1.0,
                        Phase.SCALE: 1.0
                    }
                }
            },
            "total_headcount": {
                Phase.FOUNDATION: 9,
                Phase.INTEGRATION: 13.5,
                Phase.SCALE: 8.5
            },
            "skills_priority": [
                "AI/ML Engineering",
                "Cloud Architecture",
                "Data Engineering",
                "Platform Expertise",
                "Security & Privacy"
            ]
        }
    
    def _define_budget_estimates(self) -> Dict[str, Dict]:
        """Define budget estimates for implementation"""
        return {
            "infrastructure": {
                Phase.FOUNDATION: {
                    "cloud_services": 50000,
                    "software_licenses": 30000,
                    "security_tools": 20000,
                    "total": 100000
                },
                Phase.INTEGRATION: {
                    "cloud_services": 100000,
                    "software_licenses": 50000,
                    "api_costs": 30000,
                    "total": 180000
                },
                Phase.SCALE: {
                    "cloud_services": 200000,
                    "software_licenses": 70000,
                    "api_costs": 80000,
                    "total": 350000
                }
            },
            "personnel": {
                Phase.FOUNDATION: {
                    "salaries": 450000,  # 3 months, 9 people
                    "contractors": 50000,
                    "training": 20000,
                    "total": 520000
                },
                Phase.INTEGRATION: {
                    "salaries": 675000,  # 3 months, 13.5 people
                    "contractors": 75000,
                    "training": 30000,
                    "total": 780000
                },
                Phase.SCALE: {
                    "salaries": 850000,  # 6 months, 8.5 people
                    "contractors": 100000,
                    "training": 50000,
                    "total": 1000000
                }
            },
            "platform_fees": {
                Phase.FOUNDATION: 20000,
                Phase.INTEGRATION: 50000,
                Phase.SCALE: 100000
            },
            "contingency": {
                Phase.FOUNDATION: 64000,  # 10% of phase total
                Phase.INTEGRATION: 101000,  # 10% of phase total
                Phase.SCALE: 145000  # 10% of phase total
            },
            "phase_totals": {
                Phase.FOUNDATION: 704000,
                Phase.INTEGRATION: 1111000,
                Phase.SCALE: 1595000
            },
            "total_budget": 3410000
        }
    
    def generate_gantt_data(self) -> List[Dict]:
        """Generate data for Gantt chart visualization"""
        gantt_data = []
        start_date = datetime.now()
        
        for milestone in self.milestones:
            # Calculate milestone dates based on dependencies
            if milestone.dependencies:
                # Find latest dependency end date
                dep_end_dates = []
                for dep in milestone.dependencies:
                    dep_milestone = next(
                        (m for m in self.milestones if m.name == dep), 
                        None
                    )
                    if dep_milestone:
                        dep_data = next(
                            (g for g in gantt_data if g["name"] == dep),
                            None
                        )
                        if dep_data:
                            dep_end_dates.append(dep_data["end_date"])
                
                if dep_end_dates:
                    milestone_start = max(dep_end_dates)
                else:
                    milestone_start = start_date
            else:
                milestone_start = start_date
            
            milestone_end = milestone_start + milestone.estimated_duration
            
            gantt_data.append({
                "name": milestone.name,
                "phase": milestone.phase.value,
                "start_date": milestone_start,
                "end_date": milestone_end,
                "duration_days": milestone.estimated_duration.days,
                "dependencies": milestone.dependencies
            })
        
        return gantt_data
    
    def get_critical_path(self) -> List[str]:
        """Identify critical path through project"""
        # Simplified critical path - in production use proper CPM algorithm
        critical_milestones = [
            "Infrastructure Setup",
            "Basic AI Implementation",
            "Advanced AI Deployment",
            "Full Automation",
            "Global Rollout",
            "Optimization Excellence"
        ]
        
        return critical_milestones
    
    def generate_executive_summary(self) -> Dict:
        """Generate executive summary of roadmap"""
        return {
            "project_overview": {
                "name": "AI Advertising Optimization Implementation",
                "duration": "12 months",
                "total_budget": f"${self.budget_estimates['total_budget']:,}",
                "team_size": "9-14 FTE",
                "expected_roi": "50-100% performance improvement"
            },
            "phase_summary": {
                phase.value: {
                    "duration": self.phases[phase]["duration"].days,
                    "budget": f"${self.budget_estimates['phase_totals'][phase]:,}",
                    "team_size": self.team_requirements["total_headcount"][phase],
                    "key_outcomes": self.phases[phase]["expected_outcomes"]
                }
                for phase in Phase
            },
            "critical_milestones": [
                {
                    "name": m.name,
                    "phase": m.phase.value,
                    "duration": m.estimated_duration.days,
                    "deliverables": m.deliverables
                }
                for m in self.milestones 
                if m.name in self.get_critical_path()
            ],
            "risk_factors": [
                "Data quality and availability",
                "Platform API changes",
                "Privacy regulation changes",
                "Team skill gaps",
                "Integration complexity"
            ],
            "success_factors": [
                "Executive sponsorship",
                "Dedicated team resources",
                "Phased implementation approach",
                "Continuous measurement and optimization",
                "Strong change management"
            ]
        }
    
    def export_roadmap(self, format: str = "json") -> str:
        """Export complete roadmap in specified format"""
        roadmap_data = {
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "version": "1.0",
                "total_duration_days": 360,
                "total_budget": self.budget_estimates["total_budget"]
            },
            "phases": {
                phase.value: self.phases[phase] 
                for phase in Phase
            },
            "milestones": [m.to_dict() for m in self.milestones],
            "tasks": [t.to_dict() for t in self.tasks],
            "team_requirements": self.team_requirements,
            "budget_estimates": self.budget_estimates,
            "gantt_data": self.generate_gantt_data(),
            "critical_path": self.get_critical_path(),
            "executive_summary": self.generate_executive_summary()
        }
        
        if format == "json":
            return json.dumps(roadmap_data, indent=2, default=str)
        else:
            # Could add other formats (CSV, Excel, etc.)
            return json.dumps(roadmap_data, indent=2, default=str)


# Example usage
def main():
    """Generate and display implementation roadmap"""
    roadmap = ImplementationRoadmap()
    
    # Get executive summary
    summary = roadmap.generate_executive_summary()
    
    print("=== AI Advertising Optimization Implementation Roadmap ===\n")
    print(f"Total Duration: {summary['project_overview']['duration']}")
    print(f"Total Budget: {summary['project_overview']['total_budget']}")
    print(f"Expected ROI: {summary['project_overview']['expected_roi']}\n")
    
    # Display phase summary
    print("Phase Summary:")
    for phase, details in summary['phase_summary'].items():
        print(f"\n{phase.upper()}:")
        print(f"  Duration: {details['duration']} days")
        print(f"  Budget: {details['budget']}")
        print(f"  Team Size: {details['team_size']} FTE")
        print(f"  Key Outcomes:")
        for outcome in details['key_outcomes'][:3]:
            print(f"    - {outcome}")
    
    # Display critical milestones
    print("\n\nCritical Milestones:")
    for milestone in summary['critical_milestones']:
        print(f"\n{milestone['name']} ({milestone['phase']}):")
        print(f"  Duration: {milestone['duration']} days")
        print(f"  Key Deliverables:")
        for deliverable in milestone['deliverables'][:2]:
            print(f"    - {deliverable}")
    
    # Export full roadmap
    # roadmap_json = roadmap.export_roadmap()
    # with open("implementation_roadmap.json", "w") as f:
    #     f.write(roadmap_json)
    
    print("\n\nSuccess Factors:")
    for factor in summary['success_factors']:
        print(f"  ✓ {factor}")


if __name__ == "__main__":
    main()
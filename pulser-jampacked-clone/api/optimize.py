#!/usr/bin/env python3
"""
Creative Optimization API Endpoint
POST /api/v1/creative/optimize
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import uuid
import numpy as np
from ..agents import jamclone, cesai
from ..database import postgres_client, redis_client
from ..utils import logger, cache

router = APIRouter(prefix="/api/v1/creative", tags=["creative_optimization"])

# Request/Response Models
class PerformanceData(BaseModel):
    impressions: int = Field(..., ge=0)
    clicks: int = Field(..., ge=0)
    conversions: int = Field(..., ge=0)
    engagement_rate: float = Field(..., ge=0, le=1)
    conversion_rate: float = Field(..., ge=0, le=1)
    brand_recall_score: Optional[float] = Field(None, ge=0, le=100)
    consideration_lift: Optional[float] = Field(None, ge=-1, le=1)
    cost_per_acquisition: Optional[float] = None
    
    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions > 0 else 0.0

class OptimizationObjective(BaseModel):
    metric: str = Field(..., pattern="^(engagement|conversion|awareness|consideration|roi)$")
    target_value: Optional[float] = None
    priority: float = Field(default=1.0, ge=0, le=1)

class OptimizationRequest(BaseModel):
    campaign_id: str
    performance_data: PerformanceData
    optimization_objectives: List[OptimizationObjective]
    constraints: Optional[Dict[str, Any]] = None
    time_horizon: Optional[int] = Field(default=7, description="Days to optimize for")
    budget_remaining: Optional[float] = None

class OptimizationTactic(BaseModel):
    tactic_id: str
    category: str = Field(..., pattern="^(creative|targeting|bidding|placement|timing)$")
    description: str
    expected_impact: Dict[str, float]
    implementation_effort: str = Field(..., pattern="^(low|medium|high)$")
    confidence_score: float = Field(..., ge=0, le=1)
    prerequisites: List[str] = []

class A_B_TestRecommendation(BaseModel):
    test_name: str
    hypothesis: str
    variant_a: Dict[str, Any]
    variant_b: Dict[str, Any]
    sample_size_required: int
    expected_duration_days: int
    success_metrics: List[str]

class OptimizationResponse(BaseModel):
    optimization_id: str
    timestamp: datetime
    current_performance_summary: Dict[str, float]
    predicted_performance: Dict[str, float]
    optimization_tactics: List[OptimizationTactic]
    ab_test_recommendations: List[A_B_TestRecommendation]
    expected_roi_improvement: float
    confidence_level: float
    next_review_date: datetime

# Optimization Engine
class CreativeOptimizationEngine:
    def __init__(self):
        self.jamclone_agent = jamclone.JamCloneAgent()
        self.cesai_agent = cesai.CesaiAgent()
        self.pg_client = postgres_client
        self.redis_client = redis_client
        
    async def optimize_campaign(self, request: OptimizationRequest) -> OptimizationResponse:
        """Generate optimization recommendations based on performance data"""
        optimization_id = str(uuid.uuid4())
        
        try:
            # 1. Fetch campaign history
            campaign_history = await self._fetch_campaign_history(request.campaign_id)
            
            # 2. Calculate current performance metrics
            current_metrics = self._calculate_performance_metrics(
                request.performance_data,
                campaign_history
            )
            
            # 3. Run optimization analysis
            optimization_result = await self.cesai_agent.optimize({
                "current_performance": current_metrics,
                "objectives": [obj.dict() for obj in request.optimization_objectives],
                "constraints": request.constraints or {},
                "historical_data": campaign_history,
                "time_horizon": request.time_horizon
            })
            
            # 4. Generate A/B test recommendations
            ab_tests = await self._generate_ab_tests(
                request.campaign_id,
                optimization_result["gaps"],
                request.optimization_objectives
            )
            
            # 5. Calculate expected improvements
            predicted_performance = self._predict_performance(
                current_metrics,
                optimization_result["tactics"]
            )
            
            # 6. Store optimization plan
            await self._store_optimization_plan(
                optimization_id,
                request,
                optimization_result,
                predicted_performance
            )
            
            # 7. Set cache for quick retrieval
            await self._cache_optimization(optimization_id, optimization_result)
            
            return OptimizationResponse(
                optimization_id=optimization_id,
                timestamp=datetime.utcnow(),
                current_performance_summary=current_metrics,
                predicted_performance=predicted_performance,
                optimization_tactics=[
                    OptimizationTactic(**tactic) 
                    for tactic in optimization_result["tactics"]
                ],
                ab_test_recommendations=[
                    A_B_TestRecommendation(**test) 
                    for test in ab_tests
                ],
                expected_roi_improvement=optimization_result["expected_roi_improvement"],
                confidence_level=optimization_result["confidence_level"],
                next_review_date=datetime.utcnow() + timedelta(days=request.time_horizon)
            )
            
        except Exception as e:
            logger.error(f"Optimization failed for {request.campaign_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    async def _fetch_campaign_history(self, campaign_id: str) -> Dict[str, Any]:
        """Fetch historical performance data"""
        query = """
            SELECT 
                date,
                impressions,
                clicks,
                conversions,
                spend,
                effectiveness_score
            FROM campaign_performance
            WHERE campaign_id = $1
            AND date >= NOW() - INTERVAL '30 days'
            ORDER BY date DESC
        """
        
        rows = await self.pg_client.fetch(query, campaign_id)
        
        if not rows:
            return {
                "days_active": 0,
                "total_impressions": 0,
                "avg_ctr": 0,
                "avg_cvr": 0,
                "trend": "new"
            }
        
        # Calculate trends
        history_data = [dict(row) for row in rows]
        recent_performance = history_data[:7]  # Last 7 days
        older_performance = history_data[7:14] if len(history_data) > 7 else []
        
        return {
            "days_active": len(history_data),
            "total_impressions": sum(d["impressions"] for d in history_data),
            "avg_ctr": np.mean([d["clicks"]/d["impressions"] for d in history_data if d["impressions"] > 0]),
            "avg_cvr": np.mean([d["conversions"]/d["clicks"] for d in history_data if d["clicks"] > 0]),
            "trend": self._calculate_trend(recent_performance, older_performance),
            "raw_data": history_data
        }
    
    def _calculate_performance_metrics(self, performance: PerformanceData, history: Dict) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {
            "ctr": performance.ctr,
            "cvr": performance.conversion_rate,
            "engagement_rate": performance.engagement_rate,
            "cpa": performance.cost_per_acquisition or 0,
            "impressions": performance.impressions,
            "conversions": performance.conversions
        }
        
        # Add brand metrics if available
        if performance.brand_recall_score:
            metrics["brand_recall"] = performance.brand_recall_score
        if performance.consideration_lift:
            metrics["consideration_lift"] = performance.consideration_lift
            
        # Add historical context
        if history["days_active"] > 0:
            metrics["performance_trend"] = history["trend"]
            metrics["days_active"] = history["days_active"]
            
        return metrics
    
    def _calculate_trend(self, recent: List[Dict], older: List[Dict]) -> str:
        """Calculate performance trend"""
        if not recent or not older:
            return "stable"
            
        recent_ctr = np.mean([d["clicks"]/d["impressions"] for d in recent if d["impressions"] > 0])
        older_ctr = np.mean([d["clicks"]/d["impressions"] for d in older if d["impressions"] > 0])
        
        if recent_ctr > older_ctr * 1.1:
            return "improving"
        elif recent_ctr < older_ctr * 0.9:
            return "declining"
        else:
            return "stable"
    
    async def _generate_ab_tests(self, campaign_id: str, gaps: Dict, objectives: List[OptimizationObjective]) -> List[Dict]:
        """Generate A/B test recommendations"""
        tests = []
        
        # Prioritize tests based on objectives
        primary_objective = max(objectives, key=lambda x: x.priority)
        
        if primary_objective.metric == "engagement" and gaps.get("engagement_gap", 0) > 0.1:
            tests.append({
                "test_name": "Creative Format Test",
                "hypothesis": "Video creative will increase engagement by 30% vs static images",
                "variant_a": {"creative_type": "static_image", "current": True},
                "variant_b": {"creative_type": "video", "duration": "15s"},
                "sample_size_required": 10000,
                "expected_duration_days": 7,
                "success_metrics": ["engagement_rate", "view_through_rate"]
            })
        
        if primary_objective.metric == "conversion" and gaps.get("conversion_gap", 0) > 0.05:
            tests.append({
                "test_name": "CTA Optimization Test",
                "hypothesis": "Action-oriented CTA will increase conversions by 20%",
                "variant_a": {"cta_text": "Learn More", "current": True},
                "variant_b": {"cta_text": "Get Started Now", "urgency": "high"},
                "sample_size_required": 5000,
                "expected_duration_days": 10,
                "success_metrics": ["conversion_rate", "cost_per_conversion"]
            })
        
        return tests
    
    def _predict_performance(self, current: Dict[str, float], tactics: List[Dict]) -> Dict[str, float]:
        """Predict performance after optimization"""
        predicted = current.copy()
        
        for tactic in tactics:
            for metric, impact in tactic.get("expected_impact", {}).items():
                if metric in predicted:
                    # Apply multiplicative impact
                    predicted[metric] *= (1 + impact)
        
        # Ensure realistic bounds
        if "ctr" in predicted:
            predicted["ctr"] = min(predicted["ctr"], 0.1)  # Cap CTR at 10%
        if "cvr" in predicted:
            predicted["cvr"] = min(predicted["cvr"], 0.2)  # Cap CVR at 20%
            
        return predicted
    
    async def _store_optimization_plan(self, optimization_id: str, request: OptimizationRequest, 
                                      result: Dict, predicted: Dict):
        """Store optimization plan in database"""
        query = """
            INSERT INTO optimization_plans (
                id, campaign_id, current_performance, predicted_performance,
                tactics, objectives, created_at, expires_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        await self.pg_client.execute(
            query,
            optimization_id,
            request.campaign_id,
            request.performance_data.dict(),
            predicted,
            result["tactics"],
            [obj.dict() for obj in request.optimization_objectives],
            datetime.utcnow(),
            datetime.utcnow() + timedelta(days=request.time_horizon)
        )
    
    async def _cache_optimization(self, optimization_id: str, result: Dict):
        """Cache optimization results for quick access"""
        await self.redis_client.setex(
            f"optimization:{optimization_id}",
            3600,  # 1 hour TTL
            result
        )

# Initialize engine
optimization_engine = CreativeOptimizationEngine()

# API Endpoints
@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_campaign(request: OptimizationRequest):
    """
    Generate real-time optimization recommendations
    
    This endpoint:
    - Analyzes current campaign performance
    - Identifies optimization opportunities
    - Generates tactical recommendations
    - Suggests A/B tests
    - Predicts performance improvements
    """
    result = await optimization_engine.optimize_campaign(request)
    
    logger.info(f"Optimization generated: {result.optimization_id} - Expected ROI improvement: {result.expected_roi_improvement:.1%}")
    
    return result

@router.get("/optimize/{optimization_id}")
async def get_optimization_plan(optimization_id: str):
    """Retrieve optimization plan"""
    # Check cache first
    cached = await optimization_engine.redis_client.get(f"optimization:{optimization_id}")
    if cached:
        return cached
    
    # Fallback to database
    query = "SELECT * FROM optimization_plans WHERE id = $1"
    result = await optimization_engine.pg_client.fetchrow(query, optimization_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Optimization plan not found")
    
    return dict(result)

@router.post("/optimize/{optimization_id}/apply")
async def apply_optimization(optimization_id: str, tactic_ids: List[str]):
    """Apply selected optimization tactics"""
    # In production, this would integrate with ad platforms
    query = """
        UPDATE optimization_plans 
        SET applied_tactics = $2, applied_at = $3
        WHERE id = $1
    """
    
    await optimization_engine.pg_client.execute(
        query,
        optimization_id,
        tactic_ids,
        datetime.utcnow()
    )
    
    return {
        "status": "success",
        "optimization_id": optimization_id,
        "applied_tactics": tactic_ids,
        "message": "Optimization tactics queued for implementation"
    }
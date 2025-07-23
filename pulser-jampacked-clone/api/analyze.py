#!/usr/bin/env python3
"""
Creative Analysis API Endpoint
POST /api/v1/creative/analyze
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import asyncio
from ..agents import jamclone
from ..database import postgres_client, chroma_client
from ..utils import logger

router = APIRouter(prefix="/api/v1/creative", tags=["creative_analysis"])

# Request/Response Models
class CreativeAsset(BaseModel):
    asset_type: str = Field(..., pattern="^(video|image|audio|text|interactive)$")
    asset_url: str
    platform_specs: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

class BusinessContext(BaseModel):
    industry: str
    brand_positioning: str
    target_audience: Dict[str, Any]
    competitive_landscape: Optional[List[str]] = []
    market_maturity: Optional[str] = "growth"

class AnalysisRequest(BaseModel):
    campaign_objective: str = Field(..., pattern="^(brand_awareness|consideration|conversion|retention)$")
    creative_assets: List[CreativeAsset]
    business_context: BusinessContext
    success_metrics: List[str] = Field(..., min_items=1)
    analysis_depth: Optional[str] = Field(default="comprehensive", pattern="^(quick|standard|comprehensive)$")

class FeatureScores(BaseModel):
    visual_complexity: float = Field(..., ge=0, le=1)
    message_clarity: float = Field(..., ge=0, le=1)
    brand_prominence: float = Field(..., ge=0, le=1)
    emotional_resonance: float = Field(..., ge=0, le=1)
    cultural_relevance: float = Field(..., ge=0, le=1)
    innovation_level: float = Field(..., ge=0, le=1)

class PerformanceForecast(BaseModel):
    awareness_lift: float
    consideration_increase: float
    engagement_rate: float
    conversion_rate: float
    confidence_level: float

class OptimizationRecommendation(BaseModel):
    recommendation: str
    impact: str = Field(..., pattern="^(high|medium|low)$")
    effort: str = Field(..., pattern="^(high|medium|low)$")
    priority_score: float

class AwardPrediction(BaseModel):
    award_show: str
    probability: float
    category: Optional[str] = None
    confidence: float

class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: datetime
    effectiveness_score: float = Field(..., ge=0, le=100)
    feature_scores: FeatureScores
    performance_forecast: PerformanceForecast
    optimization_recommendations: List[OptimizationRecommendation]
    award_predictions: List[AwardPrediction]
    benchmark_comparison: Dict[str, float]
    processing_time: float

# Analysis Engine
class CreativeAnalysisEngine:
    def __init__(self):
        self.jamclone_agent = jamclone.JamCloneAgent()
        self.pg_client = postgres_client
        self.chroma_client = chroma_client
        
    async def analyze_creative(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute comprehensive creative analysis"""
        start_time = datetime.utcnow()
        analysis_id = str(uuid.uuid4())
        
        try:
            # 1. Validate assets
            await self._validate_assets(request.creative_assets)
            
            # 2. Fetch similar creatives from ChromaDB
            similar_creatives = await self._fetch_similar_creatives(
                request.creative_assets[0],  # Primary asset
                request.business_context.industry
            )
            
            # 3. Run Claude-powered analysis
            analysis_result = await self.jamclone_agent.analyze({
                "creative_asset": request.creative_assets[0].dict(),
                "business_context": request.business_context.dict(),
                "success_metrics": request.success_metrics,
                "similar_creatives": similar_creatives
            })
            
            # 4. Store results
            await self._store_analysis_results(analysis_id, request, analysis_result)
            
            # 5. Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # 6. Format response
            return AnalysisResponse(
                analysis_id=analysis_id,
                timestamp=datetime.utcnow(),
                effectiveness_score=analysis_result["effectiveness_score"],
                feature_scores=FeatureScores(**analysis_result["feature_scores"]),
                performance_forecast=PerformanceForecast(**analysis_result["performance_forecast"]),
                optimization_recommendations=[
                    OptimizationRecommendation(**rec) 
                    for rec in analysis_result["optimization_recommendations"]
                ],
                award_predictions=[
                    AwardPrediction(**pred) 
                    for pred in analysis_result["award_predictions"]
                ],
                benchmark_comparison=analysis_result.get("benchmark_comparison", {}),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for {analysis_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def _validate_assets(self, assets: List[CreativeAsset]):
        """Validate creative assets are accessible"""
        for asset in assets:
            # In production, would validate URLs are accessible
            if not asset.asset_url.startswith(("http://", "https://", "s3://")):
                raise ValueError(f"Invalid asset URL: {asset.asset_url}")
    
    async def _fetch_similar_creatives(self, primary_asset: CreativeAsset, industry: str) -> List[Dict]:
        """Fetch similar creatives from vector database"""
        try:
            results = await self.chroma_client.query(
                collection_name="creative_vectors",
                query_texts=[f"{primary_asset.asset_type} {industry}"],
                n_results=10,
                where={"industry": industry} if industry else None
            )
            return results.get("metadatas", [[]])[0]
        except Exception as e:
            logger.warning(f"Failed to fetch similar creatives: {e}")
            return []
    
    async def _store_analysis_results(self, analysis_id: str, request: AnalysisRequest, results: Dict):
        """Store analysis results in PostgreSQL"""
        query = """
            INSERT INTO creative_analyses (
                id, campaign_objective, industry, brand_positioning,
                effectiveness_score, feature_scores, performance_forecast,
                optimization_recommendations, award_predictions,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        await self.pg_client.execute(
            query,
            analysis_id,
            request.campaign_objective,
            request.business_context.industry,
            request.business_context.brand_positioning,
            results["effectiveness_score"],
            results["feature_scores"],
            results["performance_forecast"],
            results["optimization_recommendations"],
            results["award_predictions"],
            datetime.utcnow()
        )

# Initialize engine
analysis_engine = CreativeAnalysisEngine()

# API Endpoints
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_creative(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze creative effectiveness using AI-powered scoring
    
    This endpoint:
    - Accepts creative assets and business context
    - Runs Claude-powered effectiveness analysis
    - Predicts performance metrics
    - Provides optimization recommendations
    - Estimates award potential
    """
    # Add to background task queue for async processing if needed
    if request.analysis_depth == "comprehensive":
        # For comprehensive analysis, might want to process asynchronously
        pass
    
    # Run analysis
    result = await analysis_engine.analyze_creative(request)
    
    # Log for monitoring
    logger.info(f"Analysis completed: {result.analysis_id} - Score: {result.effectiveness_score}")
    
    return result

@router.get("/analyze/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """Retrieve previous analysis results"""
    query = "SELECT * FROM creative_analyses WHERE id = $1"
    result = await analysis_engine.pg_client.fetchrow(query, analysis_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return dict(result)

@router.get("/benchmarks/{industry}")
async def get_industry_benchmarks(industry: str):
    """Get industry benchmarks for comparison"""
    query = """
        SELECT 
            AVG(effectiveness_score) as avg_score,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY effectiveness_score) as p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY effectiveness_score) as p50,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY effectiveness_score) as p75,
            COUNT(*) as sample_size
        FROM creative_analyses
        WHERE industry = $1
        AND created_at > NOW() - INTERVAL '6 months'
    """
    
    result = await analysis_engine.pg_client.fetchrow(query, industry)
    
    if not result or result['sample_size'] == 0:
        return {
            "industry": industry,
            "benchmarks": {
                "avg_score": 65.0,  # Default benchmarks
                "p25": 55.0,
                "p50": 65.0,
                "p75": 75.0
            },
            "sample_size": 0,
            "note": "Using default benchmarks due to limited data"
        }
    
    return {
        "industry": industry,
        "benchmarks": dict(result),
        "last_updated": datetime.utcnow()
    }
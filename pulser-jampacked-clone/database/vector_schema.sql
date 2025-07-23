-- ChromaDB Vector Schema Configuration
-- This file defines the structure for creative asset embeddings

-- Note: ChromaDB handles schema internally, but this documents our structure
-- Collections and metadata fields for ChromaDB

-- Collection: creative_vectors
-- Primary collection for creative asset embeddings
-- Metadata fields:
-- {
--   "campaign_id": "UUID",
--   "asset_id": "UUID", 
--   "asset_url": "string",
--   "asset_type": "video|image|audio|text|interactive",
--   "brand": "string",
--   "industry": "string",
--   "campaign_objective": "brand_awareness|consideration|conversion|retention",
--   "effectiveness_score": "float",
--   "award_status": "none|shortlisted|winner",
--   "award_show": "string",
--   "creation_date": "ISO date string",
--   "platform": "facebook|instagram|youtube|tiktok|display|other",
--   "country": "string",
--   "language": "string",
--   "has_cta": "boolean",
--   "duration_seconds": "integer",
--   "dominant_colors": ["string"],
--   "visual_complexity": "float",
--   "text_content": "string",
--   "emotional_tone": "string",
--   "target_audience": "string"
-- }

-- Collection: brand_embeddings
-- Brand-specific creative DNA embeddings
-- Metadata fields:
-- {
--   "brand": "string",
--   "industry": "string", 
--   "embedding_type": "visual|message|tone",
--   "date_created": "ISO date string",
--   "campaign_count": "integer",
--   "avg_effectiveness": "float"
-- }

-- Collection: trend_embeddings
-- Creative trend clusters
-- Metadata fields:
-- {
--   "trend_id": "string",
--   "trend_name": "string",
--   "category": "visual|narrative|format|technology",
--   "emergence_date": "ISO date string",
--   "peak_date": "ISO date string",
--   "adoption_rate": "float",
--   "industries": ["string"],
--   "example_campaigns": ["string"]
-- }

-- PostgreSQL helper tables for ChromaDB integration

-- Embedding queue for async processing
CREATE TABLE IF NOT EXISTS embedding_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL,
    asset_url TEXT NOT NULL,
    asset_type VARCHAR(50),
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP
);

-- Embedding metadata cache
CREATE TABLE IF NOT EXISTS embedding_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    collection_name VARCHAR(100) NOT NULL,
    embedding_id VARCHAR(255) NOT NULL,
    metadata JSONB NOT NULL,
    vector_dimensions INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(collection_name, embedding_id)
);

-- Collection statistics
CREATE TABLE IF NOT EXISTS collection_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    collection_name VARCHAR(100) NOT NULL,
    total_embeddings INTEGER DEFAULT 0,
    avg_similarity_score DECIMAL(5, 4),
    last_updated TIMESTAMP DEFAULT NOW(),
    metadata_schema JSONB,
    UNIQUE(collection_name)
);

-- Similarity search cache
CREATE TABLE IF NOT EXISTS similarity_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL,
    collection_name VARCHAR(100) NOT NULL,
    results JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    hit_count INTEGER DEFAULT 0
);

-- Index for performance
CREATE INDEX idx_embedding_queue_status ON embedding_queue(status, created_at);
CREATE INDEX idx_embedding_metadata_collection ON embedding_metadata(collection_name);
CREATE INDEX idx_similarity_cache_query ON similarity_cache(query_hash, collection_name);
CREATE INDEX idx_similarity_cache_expires ON similarity_cache(expires_at);

-- Function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_similarity_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM similarity_cache WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- ChromaDB configuration reference
-- Environment variables needed:
-- CHROMA_HOST=localhost
-- CHROMA_PORT=8001
-- CHROMA_TENANT=default_tenant
-- CHROMA_DATABASE=creative_intelligence
-- EMBEDDING_MODEL=text-embedding-3-large
-- EMBEDDING_DIMENSIONS=1536

-- Sample initialization queries for ChromaDB (Python):
/*
import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.Client(Settings(
    chroma_api_impl="rest",
    chroma_server_host="localhost",
    chroma_server_http_port=8001
))

# Create collections
creative_vectors = client.create_collection(
    name="creative_vectors",
    metadata={"hnsw:space": "cosine"}
)

brand_embeddings = client.create_collection(
    name="brand_embeddings",
    metadata={"hnsw:space": "cosine"}
)

trend_embeddings = client.create_collection(
    name="trend_embeddings", 
    metadata={"hnsw:space": "cosine"}
)
*/
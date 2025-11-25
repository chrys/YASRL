import asyncio
import logging
import uuid
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from .crud import PipelineService
from yasrl.models import QueryResult
from yasrl.exceptions import ConfigurationError, IndexingError, RetrievalError

logger = logging.getLogger(__name__)

# Global service instance
pipeline_service = PipelineService()

class PipelineCreateRequest(BaseModel):
    """Request model for creating a new RAG pipeline."""
    llm: str = Field(..., description="LLM provider name (e.g., 'openai', 'gemini', 'ollama')")
    embed_model: str = Field(..., description="Embedding model provider (e.g., 'openai', 'gemini', 'opensource')")

class PipelineCreateResponse(BaseModel):
    """Response model for pipeline creation."""
    pipeline_id: str = Field(..., description="Unique identifier for the created pipeline")
    message: str = Field(..., description="Success message")

class QueryRequest(BaseModel):
    """Request model for asking questions."""
    query: str = Field(..., description="The question to ask")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="Optional conversation history for context"
    )

class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: str = Field(..., description="The generated answer")
    source_chunks: List[Dict] = Field(..., description="Source chunks used for the answer")

class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    source: str = Field(..., description="Source to index (file path, directory, or URL)")

class IndexResponse(BaseModel):
    """Response model for indexing operations."""
    message: str = Field(..., description="Success message")

class PipelineStatsResponse(BaseModel):
    """Response model for pipeline statistics."""
    indexed_documents: int = Field(..., description="Number of indexed documents")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting YASRL API server...")
    yield
    logger.info("Shutting down YASRL API server...")
    # Cleanup all pipelines
    await pipeline_service.cleanup_all()

app = FastAPI(
    title="YASRL RAG API",
    description="API for creating and using RAG pipelines with YASRL",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/my_chatbot2/api",  # Important for reverse proxy
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.post("/pipelines", response_model=PipelineCreateResponse)
async def create_pipeline(request: PipelineCreateRequest):
    """
    Create a new RAG pipeline.
    
    Creates a new RAG pipeline with the specified LLM and embedding model providers.
    Returns a unique pipeline ID that can be used for subsequent operations.
    """
    try:
        pipeline_id = await pipeline_service.create_pipeline(
            llm=request.llm,
            embed_model=request.embed_model
        )
        
        return PipelineCreateResponse(
            pipeline_id=pipeline_id,
            message=f"Pipeline created successfully with ID: {pipeline_id}"
        )
        
    except ConfigurationError as e:
        logger.error(f"Configuration error creating pipeline: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/pipelines/{pipeline_id}/index", response_model=IndexResponse)
async def index_documents(pipeline_id: str, request: IndexRequest):
    """
    Index documents in a pipeline.
    
    Loads and indexes documents from the specified source into the pipeline's vector store.
    """
    try:
        await pipeline_service.index_source(pipeline_id, request.source)
        
        return IndexResponse(
            message=f"Successfully indexed source: {request.source}"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except IndexingError as e:
        logger.error(f"Indexing error in pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Indexing error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error indexing in pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/pipelines/{pipeline_id}/ask", response_model=QueryResponse)
async def ask_question(pipeline_id: str, request: QueryRequest):
    """
    Ask a question to the RAG pipeline.
    
    Processes the query through the RAG pipeline and returns an answer with source information.
    """
    try:
        result = await pipeline_service.ask(
            pipeline_id=pipeline_id,
            query=request.query,
            conversation_history=request.conversation_history
        )
        
        # Convert SourceChunk objects to dictionaries
        source_chunks = []
        for chunk in result.source_chunks:
            source_chunks.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": getattr(chunk, 'score', None)
            })
        
        return QueryResponse(
            answer=result.answer,
            source_chunks=source_chunks
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RetrievalError as e:
        logger.error(f"Retrieval error in pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Retrieval error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/pipelines/{pipeline_id}/stats", response_model=PipelineStatsResponse)
async def get_pipeline_stats(pipeline_id: str):
    """
    Get statistics for a pipeline.
    
    Returns information about the pipeline such as the number of indexed documents.
    """
    try:
        stats = await pipeline_service.get_stats(pipeline_id)
        return PipelineStatsResponse(**stats)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting stats for pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/pipelines/{pipeline_id}/health")
async def check_pipeline_health(pipeline_id: str):
    """
    Check the health of a pipeline.
    
    Returns the health status of the pipeline's database connection.
    """
    try:
        is_healthy = await pipeline_service.health_check(pipeline_id)
        return {
            "pipeline_id": pipeline_id,
            "healthy": is_healthy,
            "status": "healthy" if is_healthy else "unhealthy"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking health for pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """
    Delete a pipeline and clean up its resources.
    
    Removes the pipeline from memory and cleans up database connections.
    """
    try:
        await pipeline_service.delete_pipeline(pipeline_id)
        return {"message": f"Pipeline {pipeline_id} deleted successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/pipelines")
async def list_pipelines():
    """
    List all active pipelines.
    
    Returns a list of all currently active pipeline IDs.
    """
    pipelines = pipeline_service.list_pipelines()
    return {
        "pipelines": pipelines,
        "count": len(pipelines)
    }

@app.get("/healthz")
async def healthz():
    """Health check endpoint for load balancers."""
    return {"status": "ok"}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    pipelines = pipeline_service.list_pipelines()
    return {
        "message": "YASRL RAG API",
        "version": "1.0.0",
        "docs": "/my_chatbot22/api/docs",
        "active_pipelines": len(pipelines)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
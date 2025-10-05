import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded

from .auth import get_current_user
from .rate_limiter import limiter, _rate_limit_exceeded_handler
from .middleware import ResourceManagementMiddleware
from .database_logger import initialize_db_logger
from .config.manager import ConfigurationManager

from .pipeline_cache import pipeline_cache
from .exceptions import IndexingError, RetrievalError, ConfigurationError

class PipelineCreateRequest(BaseModel):
    """Request model for creating a new RAG pipeline."""
    llm: str = Field(..., description="LLM provider name (e.g., 'openai', 'gemini', 'ollama')")
    embed_model: str = Field(..., description="Embedding model provider (e.g., 'openai', 'gemini', 'opensource')")

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logging.info("Starting YASRL API Integration server...")

    # Initialize Database Logger
    config_manager = ConfigurationManager()
    config = config_manager.load_config()
    if config.database.postgres_uri:
        initialize_db_logger(config.database.postgres_uri)
        logging.info("Database logger initialized.")
    else:
        logging.warning("POSTGRES_URI not found. Database logging is disabled.")

    pipeline_cache.start_cleanup_task()
    yield
    logging.info("Shutting down YASRL API Integration server...")
    await pipeline_cache.stop_cleanup_task()

app = FastAPI(
    title="YASRL API Integration",
    description="API for integrating YASRL with external websites",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(ResourceManagementMiddleware)

@app.post("/pipelines", status_code=201)
@limiter.limit("10/minute")
async def create_pipeline_endpoint(
    request: Request,
    create_request: PipelineCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create or replace a RAG pipeline for the user."""
    website_id = current_user.get("website_id")
    user_id = current_user.get("user_id")
    pipeline_id = f"{website_id}:{user_id}"

    try:
        await pipeline_cache.create_pipeline(pipeline_id, create_request.llm, create_request.embed_model)
        return {"message": "Pipeline created successfully", "pipeline_id": pipeline_id}
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/pipelines/{pipeline_id}/index", response_model=IndexResponse)
@limiter.limit("30/minute")
async def index_documents_endpoint(
    request: Request,
    pipeline_id: str,
    index_request: IndexRequest,
    current_user: dict = Depends(get_current_user)
):
    """Index documents in the user's pipeline."""
    pipeline = await pipeline_cache.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found or has expired.")

    try:
        await pipeline.index(index_request.source, project_id=pipeline_id)
        return IndexResponse(message=f"Successfully indexed source: {index_request.source}")
    except IndexingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/pipelines/{pipeline_id}/ask", response_model=QueryResponse)
@limiter.limit("60/minute")
async def ask_question_endpoint(
    request: Request,
    pipeline_id: str,
    query_request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Ask a question to the user's pipeline."""
    pipeline = await pipeline_cache.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found or has expired.")

    try:
        result = await pipeline.ask(
            query=query_request.query,
            conversation_history=query_request.conversation_history
        )

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
    except RetrievalError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
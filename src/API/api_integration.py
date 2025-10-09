import asyncio
import logging
import os
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from API.auth import get_current_user
from API.auth import create_access_token

from API.rate_limiter import limiter
from yasrl.database_logger import initialize_db_logger
from yasrl.config.manager import ConfigurationManager

from API.pipeline_cache import pipeline_cache
from yasrl.exceptions import IndexingError, RetrievalError, ConfigurationError

from dotenv import load_dotenv
load_dotenv()

class TokenRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

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
app.add_middleware(SlowAPIMiddleware)

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
    project_id = current_user.get("project_id")
    project_name = current_user.get("project_name")
    pipeline_id = f"{website_id}:{user_id}"
    
    print(f"DEBUG: Creating pipeline with ID: {pipeline_id}")
    print(f"DEBUG: Using project: {project_name} ({project_id})")
    print(f"DEBUG: Current user data: {current_user}")  # Add this debug line

    # Load project configuration
    projects = load_projects()
    if project_id not in projects:
        raise HTTPException(status_code=500, detail=f"Project {project_id} not found")
    
    project_config = projects[project_id]
    
    # Use project's LLM and embed model (ignore request values)
    llm = project_config.get("llm", "gemini")
    embed_model = project_config.get("embed_model", "gemini")
    
    print(f"DEBUG: Project config - LLM: {llm}, Embed model: {embed_model}")

    try:
        await pipeline_cache.create_pipeline(
            pipeline_id, 
            llm, 
            embed_model,
            project_id=project_id
        )
        
        print(f"DEBUG: Pipeline created. Cache now contains: {list(pipeline_cache._pipelines.keys())}")
        
        return {
            "message": f"Pipeline created successfully using project '{project_name}'",
            "pipeline_id": pipeline_id,
            "project_name": project_name,
            "project_id": project_id
        }
    except ConfigurationError as e:
        print(f"DEBUG: Configuration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"DEBUG: Error creating pipeline: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
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
    # URL decode the pipeline_id
    decoded_pipeline_id = unquote(pipeline_id)
    
    pipeline = await pipeline_cache.get_pipeline(decoded_pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found or has expired.")

    try:
        await pipeline.index(index_request.source, project_id=decoded_pipeline_id)
        return IndexResponse(message=f"Successfully indexed source: {index_request.source}")
    except IndexingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


import numpy as np
from urllib.parse import unquote

@app.post("/pipelines/{pipeline_id}/ask", response_model=QueryResponse)
@limiter.limit("60/minute")
async def ask_question_endpoint(
    request: Request,
    pipeline_id: str,
    query_request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Ask a question to the user's pipeline."""
    # URL decode the pipeline_id
    decoded_pipeline_id = unquote(pipeline_id)
    print(f"DEBUG: Original pipeline_id: {pipeline_id}")
    print(f"DEBUG: Decoded pipeline_id: {decoded_pipeline_id}")
     # Debug: Check what's in the pipeline cache
    print(f"DEBUG: Available pipelines in cache: {list(pipeline_cache._pipelines.keys())}")
   
    pipeline = await pipeline_cache.get_pipeline(decoded_pipeline_id)
    if not pipeline:
        print(f"DEBUG: Pipeline {decoded_pipeline_id} not found in cache")
        raise HTTPException(status_code=404, detail="Pipeline not found or has expired.")

    try:
        result = await pipeline.ask(
            query=query_request.query,
            conversation_history=query_request.conversation_history
        )

        # Convert numpy types to Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        source_chunks = []
        for chunk in result.source_chunks:
            chunk_data = {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": getattr(chunk, 'score', None)
            }
            # Clean numpy types from chunk data
            chunk_data = convert_numpy_types(chunk_data)
            source_chunks.append(chunk_data)

        return QueryResponse(
            answer=result.answer,
            source_chunks=source_chunks
        )
    except RetrievalError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/token", response_model=TokenResponse)
async def login_for_access_token(token_request: TokenRequest):
    """Authenticate user and return access token with project association."""
    
    # Check if user exists and password is correct
    if (token_request.username not in USER_PROJECT_MAPPING or 
        USER_PROJECT_MAPPING[token_request.username]["password"] != token_request.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    user_config = USER_PROJECT_MAPPING[token_request.username]
    project_id = user_config["project_id"]
    
    # Load and validate project exists
    projects = load_projects()
    if project_id not in projects:
        raise HTTPException(status_code=500, detail=f"Project {project_id} not found in projects.json")
    
    project_config = projects[project_id]
    
    # Create token payload with project information
    payload = {
        "sub": token_request.username,
        "user_id": token_request.username,
        "website_id": user_config["website_id"],
        "project_id": project_id,
        "project_name": project_config.get("name", "Unknown Project")
    }
    
    access_token = create_access_token(payload)
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer"
    )


# User credentials mapped to projects
USER_PROJECT_MAPPING = {
    "happy_user": {
        "password": "happy_pass123",
        "project_id": "1c920a21136444a099fc6450542d405b",
        "website_id": "happy_payments"
    },
    # Add more users/projects here as needed
    # "website2_user": {
    #     "password": "website2_pass456", 
    #     "project_id": "another_project_id",
    #     "website_id": "website2"
    # }
}

def load_projects() -> Dict[str, Dict]:
    """Load projects from projects.json"""
    projects_path = Path(os.getenv("PROJECTS_FILE", "projects.json"))
    if not projects_path.exists():
        return {}
    
    with open(projects_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
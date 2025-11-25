import os
import logging
import uuid
import asyncio
from typing import Optional
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

load_dotenv()

import sys
# Add project root and src to sys.path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
if os.path.join(project_root, "src") not in sys.path:
    sys.path.append(os.path.join(project_root, "src"))

from src.API.crud import PipelineService
from yasrl.database import get_db_connection
from yasrl.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YASRL HTMX UI")

# Setup templates and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Initialize service
pipeline_service = PipelineService()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("base.html", {"request": request, "page": "admin"})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    # Get projects from DB instead of just active pipelines
    projects = pipeline_service.list_projects_from_db()
    return templates.TemplateResponse("admin.html", {"request": request, "projects": projects})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    # Get projects from DB
    projects = pipeline_service.list_projects_from_db()
    return templates.TemplateResponse("chat.html", {"request": request, "projects": projects})

@app.post("/pipelines", response_class=HTMLResponse)
async def create_pipeline(request: Request, name: str = Form(...), llm: str = Form(...), embed_model: str = Form(...)):
    try:
        # Create project in DB
        success = pipeline_service.create_project_in_db(name=name, llm=llm, embed_model=embed_model)
        
        if not success:
             return HTMLResponse(f"<div class='text-red-500'>Error: Failed to create project in database</div>")

        # Return updated list from DB
        projects = pipeline_service.list_projects_from_db()
        return templates.TemplateResponse("partials/project_list.html", {"request": request, "projects": projects})
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        return HTMLResponse(f"<div class='text-red-500'>Error: {e}</div>")

@app.delete("/projects/{project_name}", response_class=HTMLResponse)
async def delete_project(request: Request, project_name: str):
    try:
        from urllib.parse import unquote
        decoded_name = unquote(project_name)
        logger.info(f"Deleting project: {decoded_name}")
        pipeline_service.delete_project_from_db(decoded_name)
        # Return updated list
        projects = pipeline_service.list_projects_from_db()
        return templates.TemplateResponse("partials/project_list.html", {"request": request, "projects": projects})
    except Exception as e:
        logger.error(f"Error deleting project: {e}", exc_info=True)
        return HTMLResponse(f"<div class='text-red-500'>Error: {e}</div>")

@app.get("/projects/{project_name}/sources", response_class=HTMLResponse)
async def get_project_sources(request: Request, project_name: str):
    try:
        from urllib.parse import unquote
        decoded_name = unquote(project_name)
        logger.info(f"Getting sources for project: {decoded_name}")
        sources = pipeline_service.get_project_sources_from_db(decoded_name)
        if sources:
            sources_html = "<ul class='list-disc list-inside'>"
            for source in sources:
                sources_html += f"<li>{source}</li>"
            sources_html += "</ul>"
            return HTMLResponse(sources_html)
        else:
            return HTMLResponse("<span class='text-gray-400'>No sources indexed yet</span>")
    except Exception as e:
        logger.error(f"Error getting sources: {e}", exc_info=True)
        return HTMLResponse("<span class='text-red-400'>Error loading sources</span>")

@app.post("/chat/message", response_class=HTMLResponse)
async def chat_message(request: Request, project_name: str = Form(...), message: str = Form(...)):
    try:
        logger.info(f"Chat message for project: {project_name}, message: {message}")
        
        # Get project details from DB
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            return HTMLResponse("<div class='text-red-500'>Database not configured</div>")
        
        from yasrl.database import get_project_by_name
        from yasrl.vector_store import VectorStoreManager
        
        conn = get_db_connection(postgres_uri)
        try:
            df = get_project_by_name(conn, project_name)
            if df.empty:
                return HTMLResponse(f"<div class='text-red-500'>Project '{project_name}' not found</div>")
            
            project = df.iloc[0]
            llm = project['llm']
            embed_model = project['embed_model']
            project_id = int(project['id'])
        finally:
            conn.close()
        
        # Initialize pipeline for this project (using same table prefix as indexing)
        sanitized_name = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
        table_prefix = f"yasrl_{sanitized_name}"
        
        db_manager = VectorStoreManager(
            postgres_uri=postgres_uri,
            vector_dimensions=768,
            table_prefix=table_prefix
        )
        
        pipeline = await RAGPipeline.create(
            llm=llm,
            embed_model=embed_model,
            db_manager=db_manager
        )
        
        # Query the pipeline
        result = await pipeline.ask(query=message, conversation_history=None)
        
        return templates.TemplateResponse("partials/chat_message.html", {
            "request": request, 
            "message": message, 
            "answer": result.answer,
            "sources": result.source_chunks
        })
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        return HTMLResponse(f"<div class='text-red-500'>Error: {e}</div>")

# Store progress: tracking_id -> int (0-100)
indexing_progress = {}

@app.post("/projects/{project_name}/sources", response_class=HTMLResponse)
async def add_source(request: Request, project_name: str, source: str = Form(None), file: Optional[bytes] = None): # Simplified file handling for now
    # Handle file upload if present
    if not source:
        # TODO: Handle file upload properly, save to disk, get path
        # For now, let's assume source path is provided
        return HTMLResponse("<div class='text-red-500'>Please provide a source path</div>")

    tracking_id = str(uuid.uuid4())
    indexing_progress[tracking_id] = 0
    
    async def run_indexing():
        async def update_progress(p):
            indexing_progress[tracking_id] = p
            
        try:
            await pipeline_service.index_source_with_progress(project_name, source, update_progress)
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            indexing_progress[tracking_id] = -1 # Error state

    # Start indexing in background
    asyncio.create_task(run_indexing())

    return templates.TemplateResponse("partials/indexing_progress.html", {
        "request": request, 
        "project_name": project_name, 
        "tracking_id": tracking_id,
        "source": source
    })

@app.get("/indexing/progress/{tracking_id}")
async def progress_stream(request: Request, tracking_id: str):
    async def event_generator():
        logger.info(f"SSE stream started for tracking_id: {tracking_id}")
        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"Client disconnected for tracking_id: {tracking_id}")
                    break
                
                progress = indexing_progress.get(tracking_id, 0)
                logger.info(f"Sending progress {progress} for tracking_id: {tracking_id}")
                
                if progress == -1:
                    yield {"event": "error", "data": "Indexing failed"}
                    break
                
                # Yield just the data as a string
                yield f"{progress}"
                
                if progress >= 100:
                    logger.info(f"Indexing complete for tracking_id: {tracking_id}")
                    break
                    
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}")

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

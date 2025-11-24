import asyncio
import logging
import os
from pathlib import Path
import importlib.util
from threading import Thread
import uuid
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Dynamic import for generate_qa_pairs module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evals'))



# Import QA generation functions
#from evals.generate_qa_pairs import *

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import database module directly without triggering package __init__
db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'yasrl', 'database.py')
spec = importlib.util.spec_from_file_location("database", db_path)
if spec and spec.loader:
    database_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database_module)
    
    # Get functions from the module
    get_db_connection = database_module.get_db_connection
    setup_projects_table = database_module.setup_projects_table
    setup_project_sources_table = database_module.setup_project_sources_table
    setup_project_qa_pairs_table = database_module.setup_project_qa_pairs_table
    get_projects = database_module.get_projects
    get_project_by_name = database_module.get_project_by_name
    save_single_project = database_module.save_single_project
    delete_project_by_name = database_module.delete_project_by_name
    add_project_sources = database_module.add_project_sources
    get_project_sources = database_module.get_project_sources
    remove_project_sources = database_module.remove_project_sources
    get_qa_pairs_for_source = database_module.get_qa_pairs_for_source
    add_qa_pairs_to_source = database_module.add_qa_pairs_to_source
    delete_qa_pairs_by_ids = database_module.delete_qa_pairs_by_ids
else:
    raise ImportError("Could not load database module")

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database connection
POSTGRES_URI = os.getenv("POSTGRES_URI")
if not POSTGRES_URI:
    logger.warning("POSTGRES_URI not set in environment")
    db_connection = None
else:
    try:
        db_connection = get_db_connection(POSTGRES_URI)
        setup_projects_table(db_connection)
        setup_project_sources_table(db_connection)
        setup_project_qa_pairs_table(db_connection)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        db_connection = None

# ============================================================================
# Indexing Progress Tracking
# ============================================================================
# tracking_id -> {"status": "pending|in_progress|completed|error", "progress": int, "errors": list}
indexing_progress = {}

# ============================================================================
# QA Generation Progress Tracking
# ============================================================================
# tracking_id -> {"status": "pending|in_progress|completed|error", "progress": int, "errors": list, "qa_pairs": list}
qa_generation_progress = {}


def _sanitize_project_name(project_name: str) -> str:
    """
    Sanitize project name for use as table name.
    Converts to lowercase, replaces special characters with underscores.
    Format: happy_payments (no prefix, PGVectorStore adds 'data_' automatically)
    """
    # Convert to lowercase and remove special characters
    sanitized = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
    # Remove consecutive underscores
    sanitized = "_".join(filter(None, sanitized.split("_")))
    # Ensure it doesn't start with number or special char
    if sanitized and not sanitized[0].isalpha():
        sanitized = f"proj_{sanitized}"
    # Return just the project name, PGVectorStore will add 'data_' prefix
    return f"yasrl_{sanitized}"


def start_indexing_async(project_id: int, project_name: str, source: str, tracking_id: str, llm: str, embed_model: str):
    """
    Run the indexing process in a background thread and update progress.
    Uses a dedicated table per project with naming convention: data_yasrl_projectname
    """
    try:
        from yasrl.pipeline import RAGPipeline
        from yasrl.vector_store import VectorStoreManager
        
        indexing_progress[tracking_id] = {"status": "in_progress", "progress": 10, "errors": []}
        logger.info(f"Starting indexing for source: {source} (tracking_id: {tracking_id}, project: {project_name})")
        
        # Create project-specific table prefix
        table_prefix = _sanitize_project_name(project_name)
        logger.info(f"Using table prefix: {table_prefix}")
        
        # Create pipeline instance with project-specific vector store
        try:
            if not POSTGRES_URI:
                raise ValueError("POSTGRES_URI is not configured")
            
            db_manager = VectorStoreManager(
                postgres_uri=POSTGRES_URI,
                vector_dimensions=768,
                table_prefix=table_prefix
            )
            
            pipeline = RAGPipeline(llm=llm, embed_model=embed_model, db_manager=db_manager)
            indexing_progress[tracking_id]["progress"] = 20
            
            # Run indexing async
            async def do_index():
                try:
                    indexing_progress[tracking_id]["progress"] = 30
                    await pipeline.index(source, project_id=str(project_id))
                    indexing_progress[tracking_id]["progress"] = 100
                    indexing_progress[tracking_id]["status"] = "completed"
                    logger.info(f"Indexing completed successfully for source: {source} in project: {project_name}")
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Indexing failed for source {source} in project {project_name}: {error_msg}")
                    indexing_progress[tracking_id]["status"] = "error"
                    indexing_progress[tracking_id]["errors"].append(error_msg)
            
            asyncio.run(do_index())
            
        except Exception as e:
            error_msg = f"Failed to initialize pipeline for project {project_name}: {str(e)}"
            logger.error(error_msg)
            indexing_progress[tracking_id]["status"] = "error"
            indexing_progress[tracking_id]["errors"].append(error_msg)
            
    except Exception as e:
        error_msg = f"Unexpected error during indexing: {str(e)}"
        logger.error(error_msg)
        indexing_progress[tracking_id] = {"status": "error", "progress": 0, "errors": [error_msg]}


def generate_qa_pairs_async(project_id: int, source: str, count: int, tracking_id: str):
    """
    Run QA pair generation in a background thread and update progress.
    Uses subprocess to completely isolate async operations.
    """
    try:
        # Suppress warnings for NLTK lazy loading issues
        import warnings
        warnings.filterwarnings('ignore', message='.*_LazyCorpusLoader.*')
        warnings.filterwarnings('ignore', message='.*WordListCorpusReader.*')
        
        qa_generation_progress[tracking_id] = {"status": "in_progress", "progress": 10, "errors": [], "qa_pairs": []}
        logger.info(f"Starting QA generation for source: {source} (tracking_id: {tracking_id})")
        
        # Step 3: Run QA generation in subprocess to avoid async issues
        qa_generation_progress[tracking_id]["progress"] = 20
        logger.info(f"Running QA generation in subprocess for source: {source}")
        
        import subprocess
        import json
        import tempfile
        
        # Create temp file for results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            result_file = tmp.name
        
        # Build command to run generate_qa_pairs.py
        script_path = os.path.join(os.path.dirname(__file__), '..', '..', 'evals', 'generate_qa_pairs.py')
        cmd = [
            'python',
            script_path,
            '--source', source,
            '--total', str(count),
            '--output', result_file
        ]
        
        qa_generation_progress[tracking_id]["progress"] = 30
        logger.info(f"Executing: {' '.join(cmd)}")
        
        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        qa_generation_progress[tracking_id]["progress"] = 70
        
        if result.returncode != 0:
            raise ValueError(f"QA generation failed: {result.stderr}")
        
        # Read results from temp file
        with open(result_file, 'r') as f:
            results_data = json.load(f)
        
        # Clean up temp file
        os.unlink(result_file)
        
        qa_generation_progress[tracking_id]["progress"] = 80
        logger.info(f"Loaded {len(results_data)} QA pairs from subprocess")
        
        # Convert to format expected by add_qa_pairs_to_source
        qa_pairs = []
        for result in results_data:
            qa_pairs.append({
                'question': result.get('question', 'N/A'),
                'answer': result.get('answer', 'N/A'),
                'context': result.get('context', ''),
                'score': result.get('score', 0.0)
            })
        
        # Step 7: Save QA pairs to database
        if qa_pairs:
            add_qa_pairs_to_source(db_connection, project_id, source, qa_pairs)
            logger.info(f"Successfully saved {len(qa_pairs)} QA pairs to database")
            qa_generation_progress[tracking_id]["qa_pairs"] = qa_pairs
            qa_generation_progress[tracking_id]["progress"] = 100
            qa_generation_progress[tracking_id]["status"] = "completed"
        else:
            raise ValueError("No QA pairs generated")
            
    except Exception as e:
        error_msg = str(e)
        
        # Filter out NLTK lazy loading errors (non-critical, internal library issues)
        if "_LazyCorpusLoader" in error_msg or "WordListCorpusReader" in error_msg:
            logger.warning(f"NLTK lazy loading warning (suppressed from UI): {error_msg}")
            # Don't report NLTK internal errors to the user
            return
        
        logger.error(f"QA generation failed: {error_msg}", exc_info=True)
        qa_generation_progress[tracking_id]["status"] = "error"
        qa_generation_progress[tracking_id]["errors"].append(error_msg)
        qa_generation_progress[tracking_id]["progress"] = 0


# ============================================================================
# Chat Processing
# ============================================================================
# Cache for pipelines: project_id -> RAGPipeline instance
_pipeline_cache = {}


def _get_or_init_pipeline(project_id: int, project_name: str, llm: str, embed_model: str):
    """
    Get cached pipeline or initialize a new one for the project.
    Uses project-specific vector store table.
    """
    cache_key = project_id
    
    # Return cached pipeline if available
    if cache_key in _pipeline_cache:
        logger.info(f"Using cached pipeline for project {project_id}")
        return _pipeline_cache[cache_key]
    
    try:
        from yasrl.pipeline import RAGPipeline
        from yasrl.vector_store import VectorStoreManager
        
        if not POSTGRES_URI:
            raise ValueError("POSTGRES_URI is not configured")
        
        # Create project-specific table prefix
        table_prefix = _sanitize_project_name(project_name)
        logger.info(f"Initializing pipeline for project {project_id} with table prefix: {table_prefix}")
        
        db_manager = VectorStoreManager(
            postgres_uri=POSTGRES_URI,
            vector_dimensions=768,
            table_prefix=table_prefix
        )
        
        pipeline = RAGPipeline(llm=llm, embed_model=embed_model, db_manager=db_manager)
        
        # Cache the pipeline
        _pipeline_cache[cache_key] = pipeline
        logger.info(f"Pipeline initialized and cached for project {project_id}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize pipeline for project {project_id}: {e}")
        raise


@app.route('/api/chat/send', methods=['POST'])
def api_chat_send():
    """
    Process a chat message and return the answer with sources.
    
    Expected request data:
    {
        "project_id": int,
        "message": str,
        "chat_history": [{"role": "user|assistant", "content": str}, ...]
    }
    """
    if not db_connection:
        return jsonify({'success': False, 'error': 'Database not connected'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Request body is required'}), 400
        
        project_id = data.get('project_id')
        message = data.get('message', '').strip()
        chat_history = data.get('chat_history', [])
        
        if not project_id:
            return jsonify({'success': False, 'error': 'Project ID is required'}), 400
        
        if not message:
            return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400
        
        project_id = int(project_id)
        
        # Step 2.1: Lookup Project
        logger.info(f"Chat message received for project {project_id}: {message[:50]}...")
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT llm, embed_model, name FROM projects WHERE id = %s", (project_id,))
            project_row = cursor.fetchone()
            if not project_row:
                return jsonify({'success': False, 'error': 'Project not found'}), 404
            llm, embed_model, project_name = project_row
        
        # Step 2.2: Initialize/Get Pipeline
        try:
            pipeline = _get_or_init_pipeline(project_id, project_name, llm, embed_model)
        except Exception as e:
            error_msg = f"Failed to initialize pipeline: {str(e)}"
            logger.error(error_msg)
            return jsonify({'success': False, 'error': error_msg}), 500
        
        # Step 3: Format Chat History (already in correct format from frontend)
        formatted_history = chat_history if isinstance(chat_history, list) else []
        
        # Step 4: Call pipeline.ask()
        try:
            result = asyncio.run(pipeline.ask(query=message, conversation_history=formatted_history))
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return jsonify({'success': False, 'error': error_msg}), 500
        
        # Step 5: Format Answer + Sources
        answer = getattr(result, 'answer', '') or ''
        source_chunks = getattr(result, 'source_chunks', []) or []
        
        # Extract unique sources from chunks
        sources = []
        seen_sources = set()
        for chunk in source_chunks:
            source = chunk.metadata.get('source', 'Unknown') if hasattr(chunk, 'metadata') else 'Unknown'
            if source not in seen_sources:
                sources.append({
                    'source': source,
                    'text': chunk.text if hasattr(chunk, 'text') else '',
                    'score': float(chunk.score) if hasattr(chunk, 'score') and chunk.score else None
                })
                seen_sources.add(source)
        
        logger.info(f"Answer generated with {len(sources)} unique sources for project {project_id}")
        
        return jsonify({
            'success': True,
            'answer': answer,
            'sources': sources,
            'message': f"Answer generated with {len(sources)} source(s)"
        })
    
    except ValueError as e:
        logger.error(f"Validation error in chat: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        logger.exception(e)
        return jsonify({'success': False, 'error': f"Internal server error: {str(e)}"}), 500


# ============================================================================
# Page Routes
# ============================================================================

@app.route('/')
def index():
    """Main entry point - loads the SPA container."""
    return render_template('index.html')


@app.route('/page/admin')
def admin_page():
    """Admin page content."""
    return render_template('pages/admin.html')


@app.route('/page/chat')
def chat_page():
    """Chat page content."""
    return render_template('pages/chat.html')


@app.route('/page/evaluate')
def evaluate_page():
    """Evaluate page content."""
    return render_template('pages/evaluate.html')


# ============================================================================
# API Routes: Projects
# ============================================================================

@app.route('/api/projects', methods=['GET'])
def api_get_projects():
    """Retrieve all projects and return as HTML list."""
    if not db_connection:
        return render_template('components/projects_list.html', projects=[], error="Database not connected")
    
    try:
        projects_df = get_projects(db_connection)
        projects = projects_df.to_dict('records') if len(projects_df) > 0 else []
        return render_template('components/projects_list.html', projects=projects, error=None)
    except Exception as e:
        logger.error(f"Error retrieving projects: {e}")
        return render_template('components/projects_list.html', projects=[], error=str(e))


@app.route('/api/projects/create', methods=['POST'])
def api_create_project():
    """Create a new project."""
    if not db_connection:
        logger.error("Database not connected")
        return render_template('components/projects_list.html', projects=[], error="Database not connected"), 400
    
    try:
        project_data = {
            'name': request.form.get('name', '').strip(),
            'description': request.form.get('description', '').strip(),
            'llm': request.form.get('llm', 'gemini'),
            'embed_model': request.form.get('embed_model', 'gemini'),
            'sources': []
        }
        
        logger.info(f"Creating project with data: {project_data}")
        
        if not project_data['name']:
            logger.warning("Project name is required")
            return render_template('components/projects_list.html', projects=[], error="Project name is required"), 400
        
        success = save_single_project(db_connection, project_data)
        logger.info(f"save_single_project returned: {success}")
        
        if success:
            projects_df = get_projects(db_connection)
            projects = projects_df.to_dict('records') if len(projects_df) > 0 else []
            logger.info(f"Retrieved {len(projects)} projects after creation")
            return render_template('components/projects_list.html', projects=projects, error=None)
        else:
            logger.error("save_single_project returned False")
            return render_template('components/projects_list.html', projects=[], error="Failed to create project"), 400
    except Exception as e:
        logger.error(f"Error creating project: {e}", exc_info=True)
        return render_template('components/projects_list.html', projects=[], error=str(e)), 400


@app.route('/api/projects/<int:project_id>/details', methods=['GET'])
def api_project_details(project_id):
    """Get project details and sources."""
    if not db_connection:
        return render_template('components/project_details.html', project=None, sources=[], error="Database not connected")
    
    try:
        # Get project info
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
            result = cursor.fetchone()
            if not result:
                return render_template('components/project_details.html', project=None, sources=[], error="Project not found")
            
            project = {
                'id': result[0],
                'name': result[1],
                'description': result[2],
                'embed_model': result[3],
                'llm': result[4],
                'created_at': result[5]
            }
        
        # Get sources
        sources = get_project_sources(db_connection, project_id)
        
        return render_template('components/project_details.html', project=project, sources=sources, error=None)
    except Exception as e:
        logger.error(f"Error retrieving project details: {e}")
        return render_template('components/project_details.html', project=None, sources=[], error=str(e))


@app.route('/api/projects/<int:project_id>/delete', methods=['DELETE'])
def api_delete_project(project_id):
    """Delete a project."""
    if not db_connection:
        return jsonify({'success': False, 'error': 'Database not connected'}), 500
    
    try:
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT name FROM projects WHERE id = %s", (project_id,))
            result = cursor.fetchone()
            if not result:
                return jsonify({'success': False, 'error': 'Project not found'}), 404
            
            project_name = result[0]
        
        delete_project_by_name(db_connection, project_name)
        return jsonify({'success': True, 'message': f'Project "{project_name}" deleted'})
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# API Routes: Sources
# ============================================================================

@app.route('/api/projects/sources/add', methods=['POST'])
def api_add_source():
    """Add a source to a project and start indexing with progress tracking."""
    if not db_connection:
        return jsonify({'success': False, 'error': 'Database not connected'}), 500
    
    try:
        project_id_str = request.form.get('project_id')
        if not project_id_str:
            return jsonify({'success': False, 'error': 'Project ID is required'}), 400
        project_id = int(project_id_str)
        source_type = request.form.get('source_type', 'text')
        
        # Get project info for LLM, embed_model, and name
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT llm, embed_model, name FROM projects WHERE id = %s", (project_id,))
            project_row = cursor.fetchone()
            if not project_row:
                return jsonify({'success': False, 'error': 'Project not found'}), 404
            llm, embed_model, project_name = project_row
        
        if source_type == 'file':
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            source_path = f"uploads/{project_id}/{file.filename}"
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            file.save(source_path)
            sources = [source_path]
            source_for_indexing = source_path
        else:
            source = request.form.get('source', '').strip()
            if not source:
                return jsonify({'success': False, 'error': 'Source path/URL cannot be empty'}), 400
            sources = [source]
            source_for_indexing = source
        
        # Add source to database
        add_project_sources(db_connection, project_id, sources)
        
        # Start indexing in background thread with project-specific table
        tracking_id = str(uuid.uuid4())
        indexing_progress[tracking_id] = {"status": "pending", "progress": 0, "errors": []}
        
        thread = Thread(
            target=start_indexing_async,
            args=(project_id, project_name, source_for_indexing, tracking_id, llm, embed_model),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Source added and indexing started',
            'tracking_ids': [tracking_id]
        })
    except Exception as e:
        logger.error(f"Error adding source: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/projects/sources/remove', methods=['POST'])
def api_remove_source():
    """Remove a source from a project."""
    if not db_connection:
        return jsonify({'success': False, 'error': 'Database not connected'}), 500
    
    try:
        project_id_str = request.form.get('project_id')
        if not project_id_str:
            return jsonify({'success': False, 'error': 'Project ID is required'}), 400
        project_id = int(project_id_str)
        source = request.form.get('source', '').strip()
        
        if not source:
            return jsonify({'success': False, 'error': 'Source cannot be empty'}), 400
        
        remove_project_sources(db_connection, project_id, [source])
        return jsonify({'success': True, 'message': 'Source removed'})
    except Exception as e:
        logger.error(f"Error removing source: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# API Routes: Indexing Progress
# ============================================================================

@app.route('/api/indexing/progress/<tracking_id>', methods=['GET'])
def api_indexing_progress(tracking_id):
    """Return the progress for a given indexing operation."""
    progress = indexing_progress.get(
        tracking_id,
        {"status": "unknown", "progress": 0, "errors": []}
    )
    return jsonify(progress)


# ============================================================================
# API Routes: QA Pairs
# ============================================================================

@app.route('/api/projects/<int:project_id>/sources/<path:source>/qa-pairs', methods=['GET'])
def api_get_qa_pairs(project_id, source):
    """Get QA pairs for a specific project and source."""
    if not db_connection:
        return render_template('components/qa_pairs_list.html', qa_pairs=[], error="Database not connected")
    
    try:
        qa_pairs_df = get_qa_pairs_for_source(db_connection, project_id, source)
        qa_pairs = qa_pairs_df.to_dict('records') if len(qa_pairs_df) > 0 else []
        return render_template('components/qa_pairs_list.html', qa_pairs=qa_pairs, error=None, source=source)
    except Exception as e:
        logger.error(f"Error retrieving QA pairs: {e}")
        return render_template('components/qa_pairs_list.html', qa_pairs=[], error=str(e))


@app.route('/api/projects/qa-pairs/add', methods=['POST'])
def api_add_qa_pair():
    """Add a new QA pair to a project and source."""
    if not db_connection:
        return jsonify({'success': False, 'error': 'Database not connected'}), 500
    
    try:
        project_id_str = request.form.get('project_id')
        if not project_id_str:
            return jsonify({'success': False, 'error': 'Project ID is required'}), 400
        project_id = int(project_id_str)
        source = request.form.get('source', '').strip()
        question = request.form.get('question', '').strip()
        answer = request.form.get('answer', '').strip()
        context = request.form.get('context', '').strip()
        
        if not source or not question or not answer:
            return jsonify({'success': False, 'error': 'Source, question, and answer are required'}), 400
        
        qa_pairs = [{
            'question': question,
            'answer': answer,
            'context': context
        }]
        
        add_qa_pairs_to_source(db_connection, project_id, source, qa_pairs)
        
        return jsonify({'success': True, 'message': 'QA pair added successfully'})
    except Exception as e:
        logger.error(f"Error adding QA pair: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/projects/qa-pairs/<int:qa_pair_id>/delete', methods=['DELETE'])
def api_delete_qa_pair(qa_pair_id):
    """Delete a QA pair."""
    if not db_connection:
        return jsonify({'success': False, 'error': 'Database not connected'}), 500
    
    try:
        delete_qa_pairs_by_ids(db_connection, [qa_pair_id])
        return jsonify({'success': True, 'message': 'QA pair deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting QA pair: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/projects/qa-pairs/update', methods=['POST'])
def api_update_qa_pair():
    """Update an existing QA pair."""
    if not db_connection:
        return jsonify({'success': False, 'error': 'Database not connected'}), 500
    
    try:
        qa_pair_id_str = request.form.get('qa_pair_id')
        project_id_str = request.form.get('project_id')
        if not qa_pair_id_str or not project_id_str:
            return jsonify({'success': False, 'error': 'QA pair ID and project ID are required'}), 400
        qa_pair_id = int(qa_pair_id_str)
        project_id = int(project_id_str)
        source = request.form.get('source', '').strip()
        question = request.form.get('question', '').strip()
        answer = request.form.get('answer', '').strip()
        context = request.form.get('context', '').strip()
        
        if not source or not question or not answer:
            return jsonify({'success': False, 'error': 'Source, question, and answer are required'}), 400
        
        with db_connection.cursor() as cursor:
            cursor.execute("""
                UPDATE project_qa_pairs
                SET question = %s, answer = %s, context = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND project_id = %s
                RETURNING id;
            """, (question, answer, context, qa_pair_id, project_id))
            
            result = cursor.fetchone()
            db_connection.commit()
            
            if result:
                logger.info(f"Updated QA pair {qa_pair_id}")
                return jsonify({'success': True, 'message': 'QA pair updated successfully'})
            else:
                return jsonify({'success': False, 'error': 'QA pair not found or unauthorized'}), 404
    except Exception as e:
        db_connection.rollback()
        logger.error(f"Error updating QA pair: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/projects/qa-pairs/generate', methods=['POST'])
def api_generate_qa_pairs():
    """Generate QA pairs for a source using LlamaIndex DatasetGenerator."""
    if not db_connection:
        return jsonify({'success': False, 'error': 'Database not connected'}), 500
    
    try:
        # Get parameters
        project_id_str = request.form.get('project_id')
        source = request.form.get('source', '').strip()
        count_str = request.form.get('count', '5')
        
        if not project_id_str or not source:
            return jsonify({'success': False, 'error': 'Project ID and source are required'}), 400
        
        project_id = int(project_id_str)
        count = int(count_str)
        
        logger.info(f"Generating {count} QA pairs for source: {source}")
        
        # Start generation in background thread
        tracking_id = str(uuid.uuid4())
        qa_generation_progress[tracking_id] = {"status": "pending", "progress": 0, "errors": [], "qa_pairs": []}
        
        thread = Thread(
            target=generate_qa_pairs_async,
            args=(project_id, source, count, tracking_id),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'tracking_id': tracking_id,
            'message': 'QA generation started'
        })
            
    except ValueError as e:
        return jsonify({'success': False, 'error': 'Invalid project ID or count'}), 400
    except Exception as e:
        logger.error(f"Error starting QA generation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/qa-generation/progress/<tracking_id>', methods=['GET'])
def api_qa_generation_progress(tracking_id):
    """Return the progress for a given QA generation operation."""
    progress = qa_generation_progress.get(
        tracking_id,
        {"status": "unknown", "progress": 0, "errors": [], "qa_pairs": []}
    )
    return jsonify(progress)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
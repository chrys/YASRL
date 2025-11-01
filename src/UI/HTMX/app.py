"""
HTMX SPA for chatbot demonstration with Admin, Chat, and Evaluate pages.
Provides instant page transitions with URL updates.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
import asyncio
import uuid
from pathlib import Path
from threading import Thread
import importlib.util

# Load environment variables first
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

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


def start_indexing_async(project_id: int, source: str, tracking_id: str, llm: str, embed_model: str):
    """
    Run the indexing process in a background thread and update progress.
    """
    try:
        from yasrl.pipeline import RAGPipeline
        
        indexing_progress[tracking_id] = {"status": "in_progress", "progress": 10, "errors": []}
        logger.info(f"Starting indexing for source: {source} (tracking_id: {tracking_id})")
        
        # Create pipeline instance
        try:
            pipeline = RAGPipeline(llm=llm, embed_model=embed_model)
            indexing_progress[tracking_id]["progress"] = 20
            
            # Run indexing async
            async def do_index():
                try:
                    indexing_progress[tracking_id]["progress"] = 30
                    await pipeline.index(source, project_id=str(project_id))
                    indexing_progress[tracking_id]["progress"] = 100
                    indexing_progress[tracking_id]["status"] = "completed"
                    logger.info(f"Indexing completed successfully for source: {source}")
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Indexing failed for source {source}: {error_msg}")
                    indexing_progress[tracking_id]["status"] = "error"
                    indexing_progress[tracking_id]["errors"].append(error_msg)
            
            asyncio.run(do_index())
            
        except Exception as e:
            error_msg = f"Failed to initialize pipeline: {str(e)}"
            logger.error(error_msg)
            indexing_progress[tracking_id]["status"] = "error"
            indexing_progress[tracking_id]["errors"].append(error_msg)
            
    except Exception as e:
        error_msg = f"Unexpected error during indexing: {str(e)}"
        logger.error(error_msg)
        indexing_progress[tracking_id] = {"status": "error", "progress": 0, "errors": [error_msg]}


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
        return render_template('components/projects_list.html', projects=[], error="Database not connected"), 400
    
    try:
        project_data = {
            'name': request.form.get('name', '').strip(),
            'description': request.form.get('description', '').strip(),
            'llm': request.form.get('llm', 'gemini'),
            'embed_model': request.form.get('embed_model', 'gemini'),
            'sources': [] # Sources will be added separately
        }
        
        if not project_data['name']:
            return render_template('components/projects_list.html', projects=[], error="Project name is required"), 400
        
        success = save_single_project(db_connection, project_data)
        
        if success:
            projects_df = get_projects(db_connection)
            projects = projects_df.to_dict('records') if len(projects_df) > 0 else []
            return render_template('components/projects_list.html', projects=projects, error=None)
        else:
            return render_template('components/projects_list.html', projects=[], error="Failed to create project"), 400
    except Exception as e:
        logger.error(f"Error creating project: {e}")
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
        
        # Get project info for LLM and embed_model
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT llm, embed_model FROM projects WHERE id = %s", (project_id,))
            project_row = cursor.fetchone()
            if not project_row:
                return jsonify({'success': False, 'error': 'Project not found'}), 404
            llm, embed_model = project_row
        
        if source_type == 'file':
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            # Store file path (in real app, you'd upload to storage)
            source_path = f"uploads/{project_id}/{file.filename}"
            
            # Create uploads directory if needed
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            
            # Save file
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
        
        # Start indexing in background thread
        tracking_id = str(uuid.uuid4())
        indexing_progress[tracking_id] = {"status": "pending", "progress": 0, "errors": []}
        
        thread = Thread(
            target=start_indexing_async,
            args=(project_id, source_for_indexing, tracking_id, llm, embed_model),
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
        
        # Create QA pair as a list of dicts
        qa_pairs = [{
            'question': question,
            'answer': answer,
            'context': context
        }]
        
        # Add to database
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
        project_id_str = request.args.get('project_id')
        if not project_id_str:
            return jsonify({'success': False, 'error': 'Project ID is required'}), 400
        project_id = int(project_id_str)
        delete_qa_pairs_by_ids(db_connection, project_id, [qa_pair_id])
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
        
        # Update the QA pair using direct SQL since we don't have an update function yet
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


if __name__ == '__main__':
    app.run(debug=True, port=5000)

import os
import gradio as gr
from typing import Tuple, List, Dict, Any
import logging

from yasrl.database import get_db_connection, get_qa_pairs_for_source

logger = logging.getLogger("yasrl.evaluate_tab")

# Type alias for Gradio updates
UIUpdate = Dict[str, Any]

def get_database_connection():
    """Get database connection with error handling."""
    postgres_uri = os.getenv("POSTGRES_URI")
    if not postgres_uri:
        logger.error("POSTGRES_URI environment variable is not set")
        return None
    try:
        return get_db_connection(postgres_uri)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

def check_qa_pairs_for_source(project_id: str, source: str) -> Tuple[bool, int]:
    """
    Check if QA pairs exist for a specific project and source.
    Returns (exists, count).
    """
    if not project_id or not source:
        return False, 0
    
    conn = get_database_connection()
    if not conn:
        return False, 0
    
    try:
        qa_pairs_df = get_qa_pairs_for_source(conn, int(project_id), source)
        exists = not qa_pairs_df.empty
        count = len(qa_pairs_df) if exists else 0
        return exists, count
    except Exception as e:
        logger.error(f"Failed to check QA pairs for project {project_id}, source {source}: {e}")
        return False, 0
    finally:
        conn.close()

def load_qa_pairs_for_source(project_id: str, source: str) -> str:
    """
    Load QA pairs for a specific project and source.
    Returns formatted string of QA pairs.
    """
    if not project_id or not source:
        return "No project or source selected."
    
    conn = get_database_connection()
    if not conn:
        return "Cannot connect to database."
    
    try:
        qa_pairs_df = get_qa_pairs_for_source(conn, int(project_id), source)
        
        if qa_pairs_df.empty:
            return "No QA pairs found for this source."
        
        qa_pairs_text = f"# QA Pairs for Source: {source}\n\n"
        qa_pairs_text += f"**Found {len(qa_pairs_df)} QA pairs:**\n\n"
        
        for i, (_, row) in enumerate(qa_pairs_df.iterrows(), 1):
            qa_pairs_text += f"**Q{i}:** {row['question']}\n\n"
            qa_pairs_text += f"**A{i}:** {row['answer']}\n\n"
            if row['context']:
                qa_pairs_text += f"**Context:** {row['context']}\n\n"
            qa_pairs_text += "---\n\n"
        
        return qa_pairs_text
        
    except Exception as e:
        logger.error(f"Failed to load QA pairs for project {project_id}, source {source}: {e}")
        return f"Error loading QA pairs: {e}"
    finally:
        conn.close()

def on_evaluate_project_change(projects, display_to_pid, selected_display: str) -> Tuple[UIUpdate, str, UIUpdate, str]:
    """
    Handle project selection change in evaluate tab.
    Returns (sources_dropdown_update, sources_info, load_button_update, qa_pairs_display).
    """
    if not selected_display:
        return (
            gr.update(choices=[], value=None, visible=False),
            "No project selected.",
            gr.update(visible=False),
            ""
        )
    
    pid = display_to_pid.get(selected_display)
    if not pid:
        return (
            gr.update(choices=[], value=None, visible=False),
            "Project not found.",
            gr.update(visible=False),
            ""
        )
    
    project_info = projects.get(pid, {})
    sources = project_info.get("sources", [])
    
    if sources:
        sources_info = f"**Project:** {project_info.get('name')}\n\n**Available sources:**\n" + "\n".join(f"- {s}" for s in sources)
        return (
            gr.update(choices=sources, value=None, visible=True),
            sources_info,
            gr.update(visible=False),
            ""
        )
    else:
        sources_info = f"**Project:** {project_info.get('name')}\n\n**No sources found.** You can add sources below."
        return (
            gr.update(choices=[], value=None, visible=False),
            sources_info,
            gr.update(visible=False),
            ""
        )

def on_source_select(projects, display_to_pid, selected_display: str, selected_source: str) -> Tuple[UIUpdate, str]:
    """
    Handle source selection change.
    Returns (load_button_update, qa_status_message).
    """
    if not selected_display or not selected_source:
        return gr.update(visible=False), ""
    
    pid = display_to_pid.get(selected_display)
    if not pid:
        return gr.update(visible=False), "Project not found."
    
    # Check if QA pairs exist for this source
    exists, count = check_qa_pairs_for_source(pid, selected_source)
    
    if exists:
        message = f"✅ Found {count} existing QA pairs for source: {selected_source}"
        return gr.update(visible=True), message
    else:
        message = f"ℹ️ No QA pairs found for source: {selected_source}"
        return gr.update(visible=False), message

def on_add_source_evaluate(projects, display_to_pid, save_single_project_func, selected_display: str, new_source: str) -> Tuple[str, UIUpdate, str, UIUpdate, str]:
    """
    Add a new source to the selected project in evaluate tab.
    Returns (status_message, sources_dropdown_update, sources_info, load_button_update, qa_status).
    """
    if not selected_display:
        return "No project selected.", gr.update(), "No project selected.", gr.update(visible=False), ""
    
    if not new_source or not new_source.strip():
        return "No source provided.", gr.update(), "", gr.update(visible=False), ""
    
    new_source = new_source.strip()
    pid = display_to_pid.get(selected_display)
    if not pid:
        return "Project not found.", gr.update(), "", gr.update(visible=False), ""
    
    project_info = projects.get(pid)
    if not project_info:
        return "Project not found.", gr.update(), "", gr.update(visible=False), ""
    
    # Add source to project
    project_info.setdefault("sources", [])
    if new_source in project_info["sources"]:
        return "Source already exists.", gr.update(), "", gr.update(visible=False), ""
    
    project_info["sources"].append(new_source)
    
    # Save to database using the new single project save function
    try:
        if not save_single_project_func(pid):
            return "Failed to save source to database.", gr.update(), "", gr.update(visible=False), ""
        status = f"Source '{new_source}' added successfully."
    except Exception as e:
        logger.exception(f"Failed to save source {new_source}")
        return f"Failed to save source: {e}", gr.update(), "", gr.update(visible=False), ""
    
    # Update sources dropdown
    sources = project_info.get("sources", [])
    sources_info = f"**Project:** {project_info.get('name')}\n\n**Available sources:**\n" + "\n".join(f"- {s}" for s in sources)
    
    # Check if QA pairs exist for the new source
    exists, count = check_qa_pairs_for_source(pid, new_source)
    if exists:
        qa_status = f"✅ Found {count} existing QA pairs for source: {new_source}"
        load_button_visible = True
    else:
        qa_status = f"ℹ️ No QA pairs found for source: {new_source}"
        load_button_visible = False
    
    return (
        status,
        gr.update(choices=sources, value=new_source),
        sources_info,
        gr.update(visible=load_button_visible),
        qa_status
    )

def on_load_qa_pairs(display_to_pid, selected_display: str, selected_source: str) -> str:
    """
    Load and display QA pairs for the selected source.
    """
    if not selected_display or not selected_source:
        return "No project or source selected."
    
    pid = display_to_pid.get(selected_display)
    if not pid:
        return "Project not found."
    
    return load_qa_pairs_for_source(pid, selected_source)

def create_evaluate_tab(projects, display_to_pid, project_choices_func, save_single_project_func):
    """
    Creates the Evaluate tab for the Gradio interface.
    
    Args:
        projects: Dictionary of projects
        display_to_pid: Mapping from display strings to project IDs
        project_choices_func: Function that returns list of project choices
        save_single_project_func: Function to save a single project to database
    
    Returns:
        Gradio components for the evaluate tab
    """
    
    with gr.TabItem("Evaluate"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("## Evaluation Setup")
                
                # Project selection
                eval_project_dropdown = gr.Dropdown(
                    choices=project_choices_func(),
                    label="Select project for evaluation",
                    value=None
                )
                
                # Project and sources info
                eval_sources_info = gr.Markdown("No project selected.")
                
                # Sources dropdown (initially hidden)
                eval_sources_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select source for evaluation",
                    value=None,
                    visible=False
                )
                
                # Add new source section
                gr.Markdown("### Add New Source")
                eval_new_source = gr.Textbox(
                    label="Add source (URL or path)",
                    placeholder="https://example.com/data.pdf"
                )
                eval_add_source_btn = gr.Button("Add source to project")
                eval_add_source_status = gr.Markdown("")
                
                # QA pairs status and load button
                eval_qa_status = gr.Markdown("")
                eval_load_qa_btn = gr.Button("Load saved QA pairs", visible=False, variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("## QA Pairs")
                gr.Markdown("Select a project and source to view or load existing QA pairs.")
                
                # QA pairs display area
                eval_qa_display = gr.Markdown("")
        
        # Event handlers
        eval_project_dropdown.change(
            fn=lambda x: on_evaluate_project_change(projects, display_to_pid, x),
            inputs=[eval_project_dropdown],
            outputs=[eval_sources_dropdown, eval_sources_info, eval_load_qa_btn, eval_qa_display]
        )
        
        eval_sources_dropdown.change(
            fn=lambda x, y: on_source_select(projects, display_to_pid, x, y),
            inputs=[eval_project_dropdown, eval_sources_dropdown],
            outputs=[eval_load_qa_btn, eval_qa_status]
        )
        
        eval_add_source_btn.click(
            fn=lambda x, y: on_add_source_evaluate(projects, display_to_pid, save_single_project_func, x, y),
            inputs=[eval_project_dropdown, eval_new_source],
            outputs=[eval_add_source_status, eval_sources_dropdown, eval_sources_info, eval_load_qa_btn, eval_qa_status]
        )
        
        eval_load_qa_btn.click(
            fn=lambda x, y: on_load_qa_pairs(display_to_pid, x, y),
            inputs=[eval_project_dropdown, eval_sources_dropdown],
            outputs=[eval_qa_display]
        )
    
    return {
        "project_dropdown": eval_project_dropdown,
        "sources_dropdown": eval_sources_dropdown,
        "qa_display": eval_qa_display
    }
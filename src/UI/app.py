import os
import uuid
import json
import asyncio
import inspect
import logging
from typing import Dict, List, Optional, Any, Coroutine, cast, Tuple
# Type alias for Gradio updates (gr.update(...) returns a dict)
UIUpdate = Dict[str, Any]

import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
import shutil


load_dotenv()

from yasrl.pipeline import RAGPipeline  # used for indexing / pipeline init
from yasrl.vector_store import VectorStoreManager
from yasrl.feedback import FeedbackManager




logger = logging.getLogger("yasrl.app")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# In-memory projects: key = pid (uuid hex), value = dict(name, llm, embed_model, pipeline, sources)
projects: Dict[str, Dict] = {}
# mapping used by UI: display string -> pid
_display_to_pid: Dict[str, str] = {}
# set of (pid, message_index) tuples for which feedback has been given
_feedback_given: set[tuple[str, int]] = set()

PROJECTS_FILE = os.getenv("PROJECTS_FILE", os.path.join(os.getcwd(), "projects.json"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", os.path.join(os.getcwd(), "uploads"))


def load_projects() -> None:
    global projects
    try:
        if os.path.exists(PROJECTS_FILE):
            with open(PROJECTS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            projects = {
                pid: {
                    "name": v.get("name"),
                    "llm": v.get("llm"),
                    "embed_model": v.get("embed_model"),
                    "sources": v.get("sources", []),
                    "pipeline": None,
                }
                for pid, v in data.items()
            }
            logger.info("Loaded %d projects from %s", len(projects), PROJECTS_FILE)
        else:
            projects = {}
            logger.info("No projects file found at %s, starting with empty projects", PROJECTS_FILE)
    except Exception:
        logger.exception("Failed to load projects; starting with empty projects")
        projects = {}


def save_projects() -> None:
    try:
        dirpath = os.path.dirname(PROJECTS_FILE) or "."
        os.makedirs(dirpath, exist_ok=True)
        tmp = PROJECTS_FILE + ".tmp"
        data = {
            pid: {
                "name": v["name"],
                "llm": v["llm"],
                "embed_model": v["embed_model"],
                "sources": v.get("sources", []),
            }
            for pid, v in projects.items()
        }
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, PROJECTS_FILE)
        logger.info("Saved %d projects to %s", len(projects), PROJECTS_FILE)
    except Exception:
        logger.exception("Failed to save projects to %s", PROJECTS_FILE)


# Load on import/run
load_projects()

# Initialize FeedbackManager and setup feedback table
try:
    feedback_manager = FeedbackManager(postgres_uri=os.getenv("POSTGRES_URI") or "")
    feedback_manager.setup_feedback_table()
except Exception as e:
    logger.error("Failed to initialize FeedbackManager or setup table: %s", e)
    # Depending on requirements, you might want to exit or handle this differently
    feedback_manager = None


def _project_choices() -> List[str]:
    """Return list of display labels and populate internal mapping."""
    choices: List[str] = []
    _display_to_pid.clear()
    for pid, info in projects.items():
        display = f"{info['name']} | {pid[:8]}"
        choices.append(display)
        _display_to_pid[display] = pid
    return choices


async def _init_pipeline_async_for_project(pid: str) -> RAGPipeline:
    proj = projects.get(pid)
    if proj is None:
        raise RuntimeError("project not found")
    if proj.get("pipeline") is not None:
        return proj["pipeline"]
    llm = proj.get("llm") or os.getenv("DEMO_LLM", "gemini")
    embed_model = proj.get("embed_model") or os.getenv("DEMO_EMBED_MODEL", "gemini")
    logger.info("Initializing pipeline for project %s (llm=%s embed=%s)", pid, llm, embed_model)
    
    # Use project name instead of project ID for table naming
    project_name = proj.get("name", "").strip()
    # Sanitize project name for use as table name (remove special chars, spaces, etc.)
    sanitized_name = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
    # Remove consecutive underscores and strip leading/trailing underscores
    sanitized_name = "_".join(filter(None, sanitized_name.split("_")))
    # Fallback to pid if name is empty after sanitization
    table_prefix = sanitized_name if sanitized_name else pid

    # Create a new vector store manager for this specific project
    # The table prefix uses the project ID to ensure a unique table per project.
    db_manager = VectorStoreManager(
        postgres_uri=os.getenv("POSTGRES_URI") or "",
        vector_dimensions=768,  # TODO: dimensions to be parameterised
        table_prefix=table_prefix
    )
    pipeline = await RAGPipeline.create(
        llm=llm,
        embed_model=embed_model,
        db_manager=db_manager    
    )
    proj["pipeline"] = pipeline
    return pipeline


def ensure_pipeline_for_project(pid: str) -> Optional[RAGPipeline]:
    try:
        return asyncio.run(_init_pipeline_async_for_project(pid))
    except Exception:
        logger.exception("Pipeline init failed for %s", pid)
        return None


# ---------- Admin actions (used as callbacks) ----------


def create_project(name: str, llm: str, embed_model: str) -> Tuple[UIUpdate, UIUpdate, str]:
    """
    Create project and return updates for admin_dropdown, chat_dropdown and a status string.
    Project name is taken from user input (not replaced by uuid).
    """
    name = (name or "").strip() or f"project-{uuid.uuid4().hex[:6]}"
    llm = (llm or os.getenv("DEMO_LLM", "gemini")).strip()
    embed_model = (embed_model or os.getenv("DEMO_EMBED_MODEL", "gemini")).strip()
    pid = uuid.uuid4().hex
    projects[pid] = {"name": name, "llm": llm, "embed_model": embed_model, "sources": [], "pipeline": None}
    save_projects()
    choices = _project_choices()
    selected = f"{projects[pid]['name']} | {pid[:8]}"
    status = f"Created project '{name}' (llm={llm}, embed={embed_model})"
    logger.info(status)
    # update both dropdowns (admin and chat)
    return gr.update(choices=choices, value=selected), gr.update(choices=choices, value=selected), status


def delete_project(selected_display: str) -> Tuple[UIUpdate, UIUpdate, str]:
    pid = _display_to_pid.get(selected_display)
    if not pid:
        return gr.update(choices=_project_choices()), gr.update(choices=_project_choices()), "No project selected to delete."
    proj = projects.get(pid)
    if not proj:
        return gr.update(choices=_project_choices()), gr.update(choices=_project_choices()), "Project not found."
    # try cleanup if available
    try:
        pipeline_obj = proj.get("pipeline")
        if pipeline_obj is not None:
            cleanup = getattr(pipeline_obj, "cleanup", None)
            if callable(cleanup):
                try:
                    res = cleanup()
                    if inspect.isawaitable(res):
                        asyncio.run(cast(Coroutine[Any, Any, Any], res))
                except Exception:
                    logger.exception("Pipeline cleanup failed for %s", pid)
    except Exception:
        logger.exception("Unexpected cleanup error for %s", pid)

    del projects[pid]
    save_projects()
    choices = _project_choices()
    status = f"Deleted project '{proj.get('name')}' ({pid[:8]})"
    logger.info(status)
    return gr.update(choices=choices, value=None), gr.update(choices=choices, value=None), status


def select_project(display: str) -> Tuple[str, str]:
    """
    Return (info_md, sources_md) about selected project.
    """
    pid = _display_to_pid.get(display)
    if not pid:
        return "No project selected.", "No sources."
    info = projects.get(pid, {})
    sources = info.get("sources", [])
    info_md = (
        f"**Project:** {info.get('name')}\n\n"
        f"- id: `{pid}`\n"
        f"- llm: `{info.get('llm')}`\n"
        f"- embed_model: `{info.get('embed_model')}`\n\n"
        "You can now index sources or add new sources below."
    )
    if sources:
        sources_md = "### Sources\n" + "\n".join(f"- {s}" for s in sources)
    else:
        sources_md = "No sources added yet."
    return info_md, sources_md


def add_source(selected_display: str, source: str) -> Tuple[str, str, str]:
    """
    Add a source (URL/path) to the selected project and persist.
    Returns (status, info_md, sources_md).
    """
    pid = _display_to_pid.get(selected_display)
    if not pid:
        return "No project selected.", "No project selected.", "No sources."
    source = (source or "").strip()
    if not source:
        return "No source provided.", *select_project(selected_display)
    proj = projects.get(pid)
    if proj is None:
        return "Project not found.", *select_project(selected_display)
    proj.setdefault("sources", [])
    if source in proj["sources"]:
        return "Source already added.", *select_project(selected_display)
    proj["sources"].append(source)
    save_projects()
    # attempt to index immediately if pipeline available
    pipeline = ensure_pipeline_for_project(pid)
    if pipeline is None:
        status = f"Source added to project '{proj['name']}', pipeline not initialized yet."
        info_md, sources_md = select_project(selected_display)
        return status, info_md, sources_md
    try:
        res = pipeline.index(source=source, project_id=pid)
        if inspect.isawaitable(res):
            asyncio.run(cast(Coroutine[Any, Any, Any], res))
        status = f"Source added and indexing started for {source}"
        info_md, sources_md = select_project(selected_display)
        return status, info_md, sources_md
    except Exception:
        logger.exception("Indexing failed for %s in project %s", source, pid)
        status = f"Source added but indexing failed for {source}"
        info_md, sources_md = select_project(selected_display)
        return status, info_md, sources_md


def index_source_for_project(selected_display: str, source: str) -> str:
    pid = _display_to_pid.get(selected_display)
    if not pid:
        return "No project selected."
    pipeline = ensure_pipeline_for_project(pid)
    if pipeline is None:
        return "Pipeline init failed; cannot index."
    try:
        res = pipeline.index(source=source, project_id=pid)
        if inspect.isawaitable(res):
            asyncio.run(cast(Coroutine[Any, Any, Any], res))
        return f"Indexing started for {source}"
    except Exception:
        logger.exception("Index failed for %s project %s", source, pid)
        return f"Indexing failed for {source}"


# ---------- Chat actions ----------


def format_history_for_pipeline(history: Optional[List[Tuple[str, str]]]) -> List[Dict[str, str]]:
    if not history:
        return []
    convo: List[Dict[str, str]] = []
    for item in history:
        if not item:
            continue
        user = item[0] if len(item) > 0 else ""
        bot = item[1] if len(item) > 1 else ""
        if user:
            convo.append({"role": "user", "content": user})
        if bot:
            convo.append({"role": "assistant", "content": bot})
    return convo


def respond(selected_display: str, message: str, chat_history: Optional[List[Tuple[str, str]]]):
    pid = _display_to_pid.get(selected_display)
    if not pid:
        return "", (chat_history or []) + [("", "Error: no project selected.")]
    pipeline = ensure_pipeline_for_project(pid)
    if pipeline is None:
        return "", (chat_history or []) + [(message, "Error: pipeline init failed.")]
    conversation_history = format_history_for_pipeline(chat_history)
    try:
        result = asyncio.run(pipeline.ask(query=message, conversation_history=conversation_history))
    except Exception as e:
        logger.exception("Error during pipeline.ask for %s: %s", pid, e)
        new_h = chat_history or []
        new_h.append((message, f"Error during query: {e}"))
        return "", new_h
    sources_text = ""
    source_chunks = getattr(result, "source_chunks", None)
    if source_chunks:
        unique = sorted({c.metadata.get("source", "Unknown") for c in source_chunks})
        if unique:
            sources_text = "\n\n---\nSources:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(unique))
    answer = (getattr(result, "answer", "") or "").strip() + sources_text
    new_history = chat_history or []
    new_history.append((message, answer))
    return "", new_history


def clear_chat():
    """Clear the chat history and feedback state."""
    _feedback_given.clear()
    return "", []


def handle_feedback(selected_display: str, data: gr.LikeData):
    """
    Callback to handle feedback (like/dislike) from the user.
    """
    pid = _display_to_pid.get(selected_display)
    if not pid:
        logger.warning("Feedback received without a project selected.")
        return

    # Create a unique key for the feedback to prevent duplicates
    feedback_key = (pid, data.index[0])
    if feedback_key in _feedback_given:
        logger.warning("Duplicate feedback attempt for project %s, message %d", pid, data.index[0])
        gr.Info("You have already provided feedback for this answer.")
        return

    proj = projects.get(pid)
    if not proj:
        logger.warning("Feedback received for a non-existent project: %s", pid)
        return

    if feedback_manager is None:
        logger.error("FeedbackManager is not initialized, cannot log feedback.")
        return

    try:
        rating = "GOOD" if data.liked else "BAD"
        project_name = proj.get("name", "Unknown")
        answer = data.value
        feedback_manager.add_feedback(project=project_name, chatbot_answer=answer, rating=rating)
        _feedback_given.add(feedback_key)
        gr.Info("Thank you for your feedback!")
        logger.info("Feedback recorded for project '%s': %s", project_name, rating)
    except Exception as e:
        logger.exception("Failed to handle feedback: %s", e)


# ---------- UI (single Gradio app with tabs) ----------


def build_ui(run_mode: str = "local"):
    port = int(os.getenv("CHATBOT_PORT", "7860"))
    host = "127.0.0.1" if run_mode == "local" else "0.0.0.0"
    inbrowser = run_mode == "local"

    logger.info("Starting combined UI (mode=%s host=%s port=%d)", run_mode, host, port)

    with gr.Blocks(title="YASRL") as demo:
        with gr.Tabs():
            # Admin tab
            with gr.TabItem("Admin"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=300):
                        gr.Markdown("## Projects (admin)")
                        admin_dropdown = gr.Dropdown(choices=_project_choices(), label="Select project", value=None)
                        project_info = gr.Markdown("No project selected.")
                        project_sources = gr.Markdown("No sources.")
                        gr.Markdown("### Create new project")
                        name_in = gr.Textbox(label="Project name", placeholder="My Project")
                        llm_in = gr.Textbox(label="LLM", placeholder=os.getenv("DEMO_LLM", "gemini"))
                        embed_in = gr.Textbox(label="Embed model", placeholder=os.getenv("DEMO_EMBED_MODEL", "gemini"))
                        create_btn = gr.Button("Create project")
                        create_status = gr.Markdown("")

                        # Bind later after chat_dropdown exists

                        gr.Markdown("### Delete project")
                        delete_btn = gr.Button("Delete selected project", variant="stop")
                        delete_status = gr.Markdown("")
                        # Bind later after chat_dropdown exists

                        admin_dropdown.change(fn=select_project, inputs=[admin_dropdown], outputs=[project_info, project_sources])

                        gr.Markdown("### Sources")
                        sources_box = gr.Textbox(label="Add source (URL or path)", placeholder="https://example.com/sitemap.xml")
                        add_source_btn = gr.Button("Add source to selected project")
                        add_source_status = gr.Markdown("")
                        add_source_btn.click(fn=add_source, inputs=[admin_dropdown, sources_box], outputs=[add_source_status, project_info, project_sources])
                        
                        # Upload area
                        gr.Markdown("### Upload file to selected project")
                        upload_file = gr.File(label="Upload file", file_count="single", type="filepath")
                        upload_status = gr.Markdown("")
                        upload_file.upload(fn=add_uploaded_file, inputs=[admin_dropdown, upload_file], outputs=[upload_status, project_info, project_sources])
  

                    with gr.Column(scale=4):
                        gr.Markdown("# Admin: Projects")
                        gr.Markdown(
                            "Use this tab to create projects (persisted to projects.json), delete them, and add sources for a selected project."
                        )
                        gr.Markdown("Switch to the 'Chat' tab to ask questions for a selected project.")

            # Chat tab
            with gr.TabItem("Chat"):
                gr.Markdown("# Select Project")
                chat_dropdown = gr.Dropdown(choices=_project_choices(), label="Project", value=None, interactive=True)
                gr.Markdown("Open the Admin tab to create/manage projects and add sources.")
                gr.Markdown("When you create/delete projects in Admin they will appear in this dropdown immediately.")

                chatbot = gr.Chatbot(elem_id="chatbot", label="Chat History", likeable=True)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask your question here...", show_label=False)
                    send = gr.Button("Send")
                    clear = gr.Button("Clear")

                msg.submit(respond, inputs=[chat_dropdown, msg, chatbot], outputs=[msg, chatbot])
                send.click(respond, inputs=[chat_dropdown, msg, chatbot], outputs=[msg, chatbot])
                clear.click(clear_chat, inputs=None, outputs=[msg, chatbot])
                chatbot.like(handle_feedback, inputs=[chat_dropdown], outputs=None)
                chat_dropdown.change(fn=clear_chat, inputs=None, outputs=[msg, chatbot])

        # Bind create/delete now that both dropdowns exist
        create_btn.click(fn=create_project, inputs=[name_in, llm_in, embed_in], outputs=[admin_dropdown, chat_dropdown, create_status])
        delete_btn.click(fn=delete_project, inputs=[admin_dropdown], outputs=[admin_dropdown, chat_dropdown, delete_status])

    demo.launch(server_name=host, server_port=port, share=False, inbrowser=inbrowser)

# Handle uploaded file -> move to uploads/<pid>/ and index via add_source
def add_uploaded_file(selected_display: str, uploaded_file_path: str) -> Tuple[str, str, str]:
    pid = _display_to_pid.get(selected_display)
    if not pid:
        return "No project selected.", "No project selected.", "No sources."

    uploaded_file_path = (uploaded_file_path or "").strip()
    if not uploaded_file_path:
        return "No file uploaded.", *select_project(selected_display)
    if not os.path.isfile(uploaded_file_path):
        return f"Uploaded file not found: {uploaded_file_path}", *select_project(selected_display)

    try:
        os.makedirs(os.path.join(UPLOADS_DIR, pid), exist_ok=True)
        filename = os.path.basename(uploaded_file_path)
        dest_dir = os.path.join(UPLOADS_DIR, pid)
        dest_path = os.path.join(dest_dir, filename)

        # Ensure unique filename if already exists
        if os.path.exists(dest_path):
            stem, ext = os.path.splitext(filename)
            dest_path = os.path.join(dest_dir, f"{stem}_{uuid.uuid4().hex[:8]}{ext}")

        # Move uploaded tmp file into project uploads directory
        shutil.move(uploaded_file_path, dest_path)
        logger.info("Saved uploaded file to %s", dest_path)

        # Reuse existing flow to persist and index
        return add_source(selected_display, dest_path)
    except Exception as e:
        logger.exception("Failed to handle uploaded file: %s", e)
        return f"Failed to save/index uploaded file: {e}", *select_project(selected_display)



if __name__ == "__main__":
    import sys

    mode = os.getenv("CHATBOT_ENV", "").lower()
    if len(sys.argv) > 1 and sys.argv[1] in ("local", "production"):
        mode = sys.argv[1]
    if mode not in ("local", "production"):
        mode = "local"
        
    build_ui(run_mode=mode)
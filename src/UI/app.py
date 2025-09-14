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

load_dotenv()

from yasrl.pipeline import RAGPipeline  # used for indexing / pipeline init

logger = logging.getLogger("yasrl.app")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# In-memory projects: key = pid (uuid hex), value = dict(name, llm, embed_model, pipeline, sources)
projects: Dict[str, Dict] = {}
# mapping used by UI: display string -> pid
_display_to_pid: Dict[str, str] = {}

PROJECTS_FILE = os.getenv("PROJECTS_FILE", os.path.join(os.getcwd(), "projects.json"))


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
    pipeline = await RAGPipeline.create(llm=llm, embed_model=embed_model)
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
        res = pipeline.index(source=source)
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
        res = pipeline.index(source=source)
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

                        gr.Markdown("### Trigger indexing (explicit)")
                        index_box = gr.Textbox(label="Index source (URL or path)", placeholder="https://example.com/sitemap.xml")
                        index_btn = gr.Button("Index for selected project")
                        index_status = gr.Markdown("")
                        index_btn.click(fn=index_source_for_project, inputs=[admin_dropdown, index_box], outputs=[index_status])

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

                chatbot = gr.Chatbot(elem_id="chatbot", label="Chat History")
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask your question here...", show_label=False)
                    send = gr.Button("Send")
                    clear = gr.Button("Clear")

                msg.submit(respond, inputs=[chat_dropdown, msg, chatbot], outputs=[msg, chatbot])
                send.click(respond, inputs=[chat_dropdown, msg, chatbot], outputs=[msg, chatbot])
                clear.click(lambda: ("", []), inputs=None, outputs=[msg, chatbot])

        # Bind create/delete now that both dropdowns exist
        create_btn.click(fn=create_project, inputs=[name_in, llm_in, embed_in], outputs=[admin_dropdown, chat_dropdown, create_status])
        delete_btn.click(fn=delete_project, inputs=[admin_dropdown], outputs=[admin_dropdown, chat_dropdown, delete_status])

    demo.launch(server_name=host, server_port=port, share=False, inbrowser=inbrowser)


if __name__ == "__main__":
    import sys

    mode = os.getenv("CHATBOT_ENV", "").lower()
    if len(sys.argv) > 1 and sys.argv[1] in ("local", "production"):
        mode = sys.argv[1]
    if mode not in ("local", "production"):
        mode = "local"
        
    build_ui(run_mode=mode)
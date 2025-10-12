import os
import uuid
import asyncio
import inspect
import logging
import re
import textwrap
from typing import Dict, List, Optional, Any, Coroutine, cast, Tuple
# Type alias for Gradio updates (gr.update(...) returns a dict)
UIUpdate = Dict[str, Any]

import gradio as gr
from dotenv import load_dotenv
import shutil


load_dotenv()

from UI.project_manager import get_project_manager
from yasrl.pipeline import RAGPipeline  # used for indexing / pipeline init
from yasrl.vector_store import VectorStoreManager
from yasrl.feedback import FeedbackManager




logger = logging.getLogger("yasrl.app")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# In-memory projects: key = pid (db id str), value = dict(name, llm, embed_model, pipeline, sources)
projects: Dict[str, Dict] = {}
# mapping used by UI: display string -> pid
_display_to_pid: Dict[str, str] = {}
# set of (pid, message_index) tuples for which feedback has been given
_feedback_given: set[tuple[str, int]] = set()

QA_COLUMNS = ["question", "answer", "context"]
_PLAIN_TEXT_RE = re.compile(r"<[^>]+>")

UPLOADS_DIR = os.getenv("UPLOADS_DIR", os.path.join(os.getcwd(), "uploads"))

try:
    project_manager = get_project_manager()
except Exception:
    logger.exception("Failed to initialize ProjectManager; project operations will be unavailable.")
    project_manager = None


def load_projects() -> None:
    global projects
    if project_manager is None:
        logger.error("ProjectManager unavailable; cannot load projects from database.")
        projects = {}
        return
    try:
        records = project_manager.list_projects()
        projects = {
            record["id"]: {
                "name": record.get("name"),
                "description": record.get("description"),
                "llm": record.get("llm") or os.getenv("DEMO_LLM", "gemini"),
                "embed_model": record.get("embed_model") or os.getenv("DEMO_EMBED_MODEL", "gemini"),
                "sources": list(record.get("sources", [])),
                "pipeline": None,
            }
            for record in records
        }
        logger.info("Loaded %d projects from database", len(projects))
    except Exception:
        logger.exception("Failed to load projects from database")
        projects = {}


def save_projects() -> None:
    if project_manager is None:
        logger.error("ProjectManager unavailable; cannot persist projects to database.")
        return
    for pid, info in projects.items():
        try:
            project_manager.update_project(
                pid,
                name=info.get("name"),
                llm=info.get("llm"),
                embed_model=info.get("embed_model"),
                sources=info.get("sources", []),
            )
        except Exception:
            logger.exception("Failed to synchronize project %s to database", pid)


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


def _refresh_project_data(pid: str) -> None:
    if project_manager is None:
        return
    try:
        record = project_manager.get_project(pid)
    except Exception:
        logger.exception("Failed to refresh project %s from database", pid)
        return
    if not record:
        return
    existing_pipeline = projects.get(pid, {}).get("pipeline")
    projects[pid] = {
        "name": record.get("name"),
        "description": record.get("description"),
        "llm": record.get("llm") or os.getenv("DEMO_LLM", "gemini"),
        "embed_model": record.get("embed_model") or os.getenv("DEMO_EMBED_MODEL", "gemini"),
        "sources": list(record.get("sources", [])),
        "pipeline": existing_pipeline,
    }


def _project_sources(pid: str) -> List[str]:
    proj = projects.get(pid)
    if not proj:
        return []
    return list(proj.get("sources", []))


def _format_sources_markdown(sources: List[str]) -> str:
    if not sources:
        return "No sources available for this project."
    items = "\n".join(f"- {src}" for src in sources)
    return "### Sources\n" + items


def _source_pair_status(pid: str, source: str) -> Tuple[str, bool]:
    if not source:
        return "Select a source to continue.", False
    if project_manager is None:
        return "ProjectManager unavailable; cannot inspect QA pairs.", False
    try:
        pairs = project_manager.get_project_qa_pairs(pid, source=source)
    except Exception:
        logger.exception("Failed to inspect QA pairs for project %s source %s", pid, source)
        return "Failed to inspect QA pairs for this source.", False
    if pairs:
        return (
            f"Source **{source}** has {len(pairs)} saved QA pair(s). Use **Load saved QA pairs** to review them.",
            True,
        )
    return (f"Source **{source}** has no saved QA pairs yet.", False)


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
    table_prefix = f"yasrl_{sanitized_name or pid}"

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


import pandas as pd

def _normalize_dataframe_rows(rows: Optional[Any]) -> List[Dict[str, str]]:
    """Convert Gradio dataframe payloads into a normalized list of QA dicts."""
    normalized: List[Dict[str, str]] = []
    if rows is None:
        return normalized
    # Handle pandas DataFrame
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return normalized
        rows = rows.values.tolist()
    for row in rows:
        if row is None:
            continue
        if isinstance(row, dict):
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            context = str(row.get("context", "")).strip()
        elif isinstance(row, (list, tuple)):
            question = str(row[0]).strip() if len(row) > 0 and row[0] is not None else ""
            answer = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
            context = str(row[2]).strip() if len(row) > 2 and row[2] is not None else ""
        else:
            continue
        if question or answer or context:
            normalized.append({"question": question, "answer": answer, "context": context})
    return normalized

def _rows_from_pairs(pairs: List[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for pair in pairs:
        rows.append([
            str(pair.get("question", "")),
            str(pair.get("answer", "")),
            str(pair.get("context", "")),
        ])
    return rows


def _clean_excerpt(raw: str, max_words: int = 120) -> str:
    if not raw:
        return ""
    cleaned = _PLAIN_TEXT_RE.sub(" ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]) + " ..."
    return cleaned


def _gather_project_excerpts(pid: str, limit: int, source: Optional[str] = None) -> List[Tuple[str, str]]:
    proj = projects.get(pid)
    if not proj:
        return []
    excerpts: List[Tuple[str, str]] = []
    sources_to_scan = [source] if source else proj.get("sources", [])
    for src in sources_to_scan:
        path = (src or "").strip()
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                raw = handle.read()
        except Exception:
            logger.warning("Failed to read source '%s' for project %s", path, pid)
            continue
        # Split the file into chunks of ~120 words
        words = raw.split()
        chunk_size = 120
        for i in range(0, min(len(words), limit * chunk_size), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                excerpts.append((path, chunk.strip()))
            if len(excerpts) >= limit:
                break
        if len(excerpts) >= limit:
            break
    return excerpts

def _build_pair_from_excerpt(source_path: str, excerpt: str) -> Dict[str, str]:
    filename = os.path.basename(source_path) or "source"
    question = f"What key detail is highlighted in the excerpt from {filename}?"
    answer_words = excerpt.split()
    if len(answer_words) > 60:
        answer = " ".join(answer_words[:60]) + " ..."
    else:
        answer = excerpt
    return {"question": question, "answer": answer, "context": excerpt}


# ---------- Admin actions (used as callbacks) ----------


def create_project(name: str, llm: str, embed_model: str) -> Tuple[UIUpdate, UIUpdate, UIUpdate, str]:
    """
    Create project and return updates for admin_dropdown, chat_dropdown and a status string.
    Project name is taken from user input (not replaced by uuid).
    """
    name = (name or "").strip() or f"project-{uuid.uuid4().hex[:6]}"
    llm = (llm or os.getenv("DEMO_LLM", "gemini")).strip()
    embed_model = (embed_model or os.getenv("DEMO_EMBED_MODEL", "gemini")).strip()
    if project_manager is None:
        status = "ProjectManager unavailable; cannot create project."
        logger.error(status)
        choices = _project_choices()
        return (
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=choices),
            status,
        )

    try:
        record = project_manager.create_project(
            name=name,
            llm=llm,
            embed_model=embed_model,
            description=None,
            sources=[],
        )
    except Exception as exc:
        logger.exception("Failed to create project '%s'", name)
        status = f"Failed to create project '{name}': {exc}"
        choices = _project_choices()
        return (
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=choices),
            status,
        )

    pid = record["id"]
    projects[pid] = {
        "name": record.get("name"),
        "description": record.get("description"),
        "llm": record.get("llm"),
        "embed_model": record.get("embed_model"),
        "sources": list(record.get("sources", [])),
        "pipeline": None,
    }
    choices = _project_choices()
    selected = f"{projects[pid]['name']} | {pid[:8]}"
    status = f"Created project '{name}' (llm={llm}, embed={embed_model})"
    logger.info(status)
    return (
        gr.update(choices=choices, value=selected),
        gr.update(choices=choices, value=selected),
        gr.update(choices=choices, value=selected),
        status,
    )


def delete_project(selected_display: str) -> Tuple[UIUpdate, UIUpdate, UIUpdate, str]:
    pid = _display_to_pid.get(selected_display)
    if not pid:
        choices = _project_choices()
        return (
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=choices),
            "No project selected to delete.",
        )
    proj = projects.get(pid)
    if not proj:
        choices = _project_choices()
        return (
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=choices),
            "Project not found.",
        )
    if project_manager is None:
        status = "ProjectManager unavailable; cannot delete project."
        logger.error(status)
        choices = _project_choices()
        return (
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=choices),
            status,
        )
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

    try:
        project_manager.delete_project(pid)
    except Exception:
        logger.exception("Failed to delete project %s from database", pid)
        choices = _project_choices()
        status = f"Failed to delete project '{proj.get('name')}'."
        return (
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=choices),
            status,
        )

    del projects[pid]
    choices = _project_choices()
    status = f"Deleted project '{proj.get('name')}' ({pid[:8]})"
    logger.info(status)
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        status,
    )


def select_project(display: str) -> Tuple[str, str]:
    """
    Return (info_md, sources_md) about selected project.
    """
    pid = _display_to_pid.get(display)
    if not pid:
        return "No project selected.", "No sources."
    if project_manager is None:
        logger.error("ProjectManager unavailable; cannot fetch project details.")
        return "Project manager unavailable.", "No sources."

    record = project_manager.get_project(pid)
    if record is None:
        logger.warning("Project %s not found in database during selection", pid)
        return "Project not found.", "No sources."

    info = projects.setdefault(
        pid,
        {
            "name": record.get("name"),
            "description": record.get("description"),
            "llm": record.get("llm"),
            "embed_model": record.get("embed_model"),
            "sources": list(record.get("sources", [])),
            "pipeline": None,
        },
    )
    info.update(
        {
            "name": record.get("name"),
            "description": record.get("description"),
            "llm": record.get("llm"),
            "embed_model": record.get("embed_model"),
            "sources": list(record.get("sources", [])),
        }
    )

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
    if project_manager is None:
        status = "ProjectManager unavailable; cannot add source."
        logger.error(status)
        return status, *select_project(selected_display)

    try:
        record = project_manager.add_source(pid, source)
        proj["sources"] = list(record.get("sources", []))
    except Exception:
        logger.exception("Failed to persist source '%s' for project %s", source, pid)
        return "Failed to add source in database.", *select_project(selected_display)

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


# ---------- Evaluation actions ----------


def load_project_qa_pairs(selected_display: str, source_value: str) -> Tuple[UIUpdate, str, str, UIUpdate]:
    pid = _display_to_pid.get(selected_display)
    if not pid:
        return gr.update(value=[], headers=QA_COLUMNS), "Select a project before loading QA pairs.", "No project selected.", gr.update(interactive=False)
    if not source_value:
        return gr.update(value=[], headers=QA_COLUMNS), "Select a source before loading QA pairs.", "Select a source to continue.", gr.update(interactive=False)
    if project_manager is None:
        logger.error("ProjectManager unavailable; cannot load QA pairs.")
        return gr.update(value=[], headers=QA_COLUMNS), "ProjectManager unavailable.", "ProjectManager unavailable; cannot load QA pairs.", gr.update(interactive=False)
    try:
        pairs = project_manager.get_project_qa_pairs(pid, source=source_value)
    except Exception:
        logger.exception("Failed to load QA pairs for project %s source %s", pid, source_value)
        return gr.update(value=[], headers=QA_COLUMNS), "Failed to load QA pairs from database.", "Failed to load QA pairs.", gr.update(interactive=False)
    rows = _rows_from_pairs(pairs)
    status = (
        f"Loaded {len(rows)} QA pair(s) for source '{source_value}'."
        if rows
        else "No saved QA pairs found for this source."
    )
    source_message, has_pairs = _source_pair_status(pid, source_value)
    return gr.update(value=rows, headers=QA_COLUMNS), status, source_message, gr.update(interactive=has_pairs)


def save_project_qa_pairs(selected_display: str, source_value: str, dataframe_rows: Any) -> Tuple[str, str, UIUpdate]:
    pid = _display_to_pid.get(selected_display)
    if not pid:
        return "Select a project before saving.", "Select a project before saving.", gr.update(interactive=False)
    if not source_value:
        return "Select a source before saving QA pairs.", "Select a source before saving QA pairs.", gr.update(interactive=False)
    if project_manager is None:
        logger.error("ProjectManager unavailable; cannot save QA pairs.")
        return "ProjectManager unavailable.", "ProjectManager unavailable; cannot save QA pairs.", gr.update(interactive=False)
    normalized = _normalize_dataframe_rows(dataframe_rows)
    try:
        saved = project_manager.replace_project_qa_pairs(pid, source_value, normalized)
    except Exception:
        logger.exception("Failed to save QA pairs for project %s source %s", pid, source_value)
        return "Failed to save QA pairs to database.", "Failed to save QA pairs.", gr.update(interactive=False)
    status = (
        f"Saved {saved} QA pair(s) for source '{source_value}'."
        if saved
        else "No QA pairs saved (need question and answer)."
    )
    source_message, has_pairs = _source_pair_status(pid, source_value)
    return status, source_message, gr.update(interactive=has_pairs)


from evals.deepeval_evaluation import synthesize_evaluation_dataset
from evals.deepeval_gemini import create_gemini_synthesizer

def _extract_context_chunks(pid: str, source: str, pair_count: int) -> list[str]:
    """Extract up to pair_count context chunks from the given source file."""
    proj = projects.get(pid)
    if not proj or not source or not os.path.isfile(source):
        return []
    
    with open(source, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Split into meaningful chunks (paragraphs or sentences)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    if not paragraphs:
        # Fallback to sentence-based chunking
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        # Group sentences into chunks of 3-5 sentences
        chunk_size = 4
        paragraphs = [
            '. '.join(sentences[i:i+chunk_size]) + '.'
            for i in range(0, len(sentences), chunk_size)
            if sentences[i:i+chunk_size]
        ]
    
    return paragraphs[:pair_count]

def generate_dataset_for_project(
    selected_display: str,
    source_value: str,
    pair_count: int,
    current_rows: Any,
) -> Tuple[UIUpdate, str]:
    pid = _display_to_pid.get(selected_display)
    if not pid:
        return gr.update(value=[], headers=QA_COLUMNS), "Select a project before generating QA pairs."
    if not source_value:
        existing_rows = _rows_from_pairs(_normalize_dataframe_rows(current_rows))
        return gr.update(value=existing_rows, headers=QA_COLUMNS), "Select a source before generating QA pairs."
    
    pair_count = max(1, min(int(pair_count or 1), 20))  # Limit to reasonable number
    existing_pairs = _normalize_dataframe_rows(current_rows)
    context_chunks = _extract_context_chunks(pid, source_value, pair_count)
    
    if not context_chunks:
        return gr.update(value=_rows_from_pairs(existing_pairs), headers=QA_COLUMNS), "No readable content found for this source."
    
    try:
        logger.info(f"Generating {pair_count} QA pairs using Gemini synthesizer")
        
        # Use the Gemini wrapper for synthesis
        synthesizer = create_gemini_synthesizer()
        qa_pairs = synthesizer.generate_qa_pairs(
            contexts=context_chunks,
            num_pairs_per_context=1,  # One QA pair per context chunk
            question_types=["factual", "explanatory", "definitional"]
        )
        
        logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")
        
    except Exception as e:
        logger.exception("Gemini QA generation failed: %s", e)
        return gr.update(value=_rows_from_pairs(existing_pairs), headers=QA_COLUMNS), f"Failed to generate QA pairs: {e}"
    
    # Convert to format expected by the UI
    new_rows = [
        [pair.get("question", ""), pair.get("answer", ""), pair.get("context", "")]
        for pair in qa_pairs 
        if pair.get("question") and pair.get("answer")
    ]
    
    # Combine with existing pairs
    existing_rows = _rows_from_pairs(existing_pairs)
    combined_rows = existing_rows + new_rows
    
    status = f"Generated {len(new_rows)} QA pairs for source '{source_value}'. Total: {len(combined_rows)} pairs."
    return gr.update(value=combined_rows, headers=QA_COLUMNS), status

def clear_qa_pairs() -> Tuple[UIUpdate, str]:
    return gr.update(value=[], headers=QA_COLUMNS), "Cleared QA pairs table (unsaved changes lost)."


def sync_evaluation_dropdown(display: str) -> UIUpdate:
    choices = _project_choices()
    value = display if display in choices else None
    return gr.update(choices=choices, value=value)


def on_select_evaluation_project(selected_display: str) -> Tuple[UIUpdate, str, UIUpdate, UIUpdate, str, str, UIUpdate]:
    if not selected_display or selected_display not in _display_to_pid:
        return (
            gr.update(choices=[], value=None, interactive=False),
            "No sources available for this project.",
            gr.update(interactive=False),
            gr.update(value=[], headers=QA_COLUMNS),
            "",
            "Select a project to view its sources.",
            gr.update(value=""),
        )
    pid = _display_to_pid[selected_display]
    _refresh_project_data(pid)
    sources = _project_sources(pid)
    sources_md = _format_sources_markdown(sources)
    dropdown_update = gr.update(choices=sources, value=None, interactive=bool(sources))
    return (
        dropdown_update,
        sources_md,
        gr.update(interactive=False),
        gr.update(value=[], headers=QA_COLUMNS),
        "",
        f"Project **{projects[pid]['name']}** selected. Choose a source or add a new one.",
        gr.update(value=""),
    )


def on_select_evaluation_source(selected_display: str, source_value: str) -> Tuple[str, UIUpdate]:
    pid = _display_to_pid.get(selected_display)
    if not pid:
        return "Select a project first.", gr.update(interactive=False)
    if not source_value:
        return "Select a source to continue.", gr.update(interactive=False)
    message, has_pairs = _source_pair_status(pid, source_value)
    return message, gr.update(interactive=has_pairs)


def add_source_for_evaluation(selected_display: str, new_source: str) -> Tuple[str, UIUpdate, str, UIUpdate, UIUpdate]:
    pid = _display_to_pid.get(selected_display)
    new_source = (new_source or "").strip()
    if not pid:
        return "Select a project before adding a source.", gr.update(), "No project selected.", gr.update(interactive=False), gr.update(value=new_source)
    if not new_source:
        return "Enter a source (URL or path) before adding.", gr.update(), "No project selected.", gr.update(interactive=False), gr.update(value="")
    if project_manager is None:
        return "ProjectManager unavailable; cannot add source.", gr.update(), "ProjectManager unavailable.", gr.update(interactive=False), gr.update(value=new_source)

    _refresh_project_data(pid)
    sources = _project_sources(pid)
    if new_source not in sources:
        try:
            record = project_manager.add_source(pid, new_source)
            _refresh_project_data(pid)
            sources = list(record.get("sources", []))
        except Exception:
            logger.exception("Failed to add source '%s' for project %s via evaluation tab", new_source, pid)
            return "Failed to add source in database.", gr.update(), _format_sources_markdown(sources), gr.update(interactive=False), gr.update(value=new_source)
    message, has_pairs = _source_pair_status(pid, new_source)
    dropdown_update = gr.update(choices=sources, value=new_source, interactive=True)
    sources_md = _format_sources_markdown(sources)
    return message, dropdown_update, sources_md, gr.update(interactive=has_pairs), gr.update(value="")


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
    # Determine the correct index value
    if isinstance(data.index, int):
        feedback_index = data.index
    else:
        feedback_index = data.index[0]
        
    feedback_key = (pid, feedback_index)
    if feedback_key in _feedback_given:
        logger.warning("Duplicate feedback attempt for project %s, message %d", pid, feedback_index)
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
                            "Use this tab to create projects (stored in PostgreSQL), delete them, and add sources for a selected project."
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
                clear.click(clear_chat, inputs=None, outputs=[msg, chatbot])
                chatbot.like(handle_feedback, inputs=[chat_dropdown], outputs=None)
                chat_dropdown.change(fn=clear_chat, inputs=None, outputs=[msg, chatbot])

            # Evaluation tab
            with gr.TabItem("Evaluation"):
                gr.Markdown("# Evaluation")
                evaluation_dropdown = gr.Dropdown(
                    choices=_project_choices(),
                    label="Select project",
                    value=None,
                    interactive=True,
                )
                sources_md = gr.Markdown("Select a project to view its sources.")
                source_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select source",
                    value=None,
                    interactive=False,
                )
                with gr.Row():
                    new_source_box = gr.Textbox(label="Add source (URL or path)", placeholder="https://example.com/feed.xml")
                    add_source_eval_btn = gr.Button("Add source", variant="secondary")
                source_status = gr.Markdown("")
                pair_count_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of QA pairs to generate",
                )
                with gr.Row():
                    load_pairs_btn = gr.Button("Load saved QA pairs", interactive=False)
                    generate_pairs_btn = gr.Button("Generate QA pairs", variant="primary")
                    clear_pairs_btn = gr.Button("Clear table", variant="secondary")
                qa_dataframe = gr.Dataframe(
                    headers=QA_COLUMNS,
                    datatype=["str", "str", "str"],
                    row_count=(0, "dynamic"),
                    col_count=(len(QA_COLUMNS), "fixed"),
                    interactive=True,
                    label="QA pairs",
                )
                save_pairs_btn = gr.Button("Save QA pairs", variant="primary")
                eval_status = gr.Markdown("")

                load_pairs_btn.click(
                    fn=load_project_qa_pairs,
                    inputs=[evaluation_dropdown, source_dropdown],
                    outputs=[qa_dataframe, eval_status, source_status, load_pairs_btn],
                )
                generate_pairs_btn.click(
                    fn=generate_dataset_for_project,
                    inputs=[evaluation_dropdown, source_dropdown, pair_count_slider, qa_dataframe],
                    outputs=[qa_dataframe, eval_status],
                )
                clear_pairs_btn.click(
                    fn=clear_qa_pairs,
                    inputs=None,
                    outputs=[qa_dataframe, eval_status],
                )
                save_pairs_btn.click(
                    fn=save_project_qa_pairs,
                    inputs=[evaluation_dropdown, source_dropdown, qa_dataframe],
                    outputs=[eval_status, source_status, load_pairs_btn],
                )

                evaluation_dropdown.change(
                    fn=on_select_evaluation_project,
                    inputs=[evaluation_dropdown],
                    outputs=[source_dropdown, sources_md, load_pairs_btn, qa_dataframe, eval_status, source_status, new_source_box],
                )
                source_dropdown.change(
                    fn=on_select_evaluation_source,
                    inputs=[evaluation_dropdown, source_dropdown],
                    outputs=[source_status, load_pairs_btn],
                )
                add_source_eval_btn.click(
                    fn=add_source_for_evaluation,
                    inputs=[evaluation_dropdown, new_source_box],
                    outputs=[source_status, source_dropdown, sources_md, load_pairs_btn, new_source_box],
                )

        admin_dropdown.change(
            fn=sync_evaluation_dropdown,
            inputs=[admin_dropdown],
            outputs=[evaluation_dropdown],
        )

        # Bind create/delete now that both dropdowns exist
        create_btn.click(
            fn=create_project,
            inputs=[name_in, llm_in, embed_in],
            outputs=[admin_dropdown, chat_dropdown, evaluation_dropdown, create_status],
        )
        delete_btn.click(
            fn=delete_project,
            inputs=[admin_dropdown],
            outputs=[admin_dropdown, chat_dropdown, evaluation_dropdown, delete_status],
        )

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
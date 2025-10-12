import gradio as gr
import logging
import os
import pandas as pd
from typing import Dict, List, Any

from yasrl.database import get_db_connection, setup_project_qa_pairs_table, get_qa_pairs_for_source, save_qa_pairs
from evals.deepeval_evaluation import generate_evaluation_dataset
from llama_index.core import Document

logger = logging.getLogger("yasrl.evaluation_ui")

# Ensure the project_qa_pairs table exists
try:
    conn = get_db_connection(os.getenv("POSTGRES_URI"))
    setup_project_qa_pairs_table(conn)
    conn.close()
except Exception as e:
    logger.exception("Failed to setup project_qa_pairs table on startup.")

def _get_project_id(display_to_pid, selected_display):
    pid_hex = display_to_pid.get(selected_display)
    if not pid_hex:
        return None
    # Convert hex PID to integer for database
    try:
        return int(pid_hex, 16)
    except (ValueError, TypeError):
        logger.error(f"Could not convert project PID '{pid_hex}' to an integer.")
        return None

def update_project_dropdown(projects: Dict):
    choices = list(projects.keys())
    return gr.update(choices=choices)

def on_project_select(selected_display: str, projects: Dict, display_to_pid: Dict):
    pid = display_to_pid.get(selected_display)
    if not pid:
        return gr.update(choices=[], value=None), gr.update(visible=False, value=""), gr.update(value=None)

    project_info = projects.get(pid)
    if not project_info:
        return gr.update(choices=[], value=None), gr.update(visible=False, value=""), gr.update(value=None)

    sources = project_info.get("sources", [])

    return gr.update(choices=sources, value=None), gr.update(visible=False, value=""), gr.update(value=None)

def on_source_select(selected_display: str, selected_source: str, display_to_pid: Dict):
    if not selected_display or not selected_source:
        return gr.update(visible=False, value=""), gr.update(visible=False)

    project_id = _get_project_id(display_to_pid, selected_display)
    if project_id is None:
        return gr.update(visible=True, value="Error: Invalid Project ID."), gr.update(visible=False)

    conn = get_db_connection(os.getenv("POSTGRES_URI"))
    try:
        qa_pairs = get_qa_pairs_for_source(conn, project_id, selected_source)
        if not qa_pairs.empty:
            return gr.update(visible=True, value=f"{len(qa_pairs)} QA pairs found for this source."), gr.update(visible=True)
        else:
            return gr.update(visible=True, value="No QA pairs found for this source."), gr.update(visible=False)
    finally:
        conn.close()

async def generate_and_append_pairs(selected_display: str, selected_source: str, num_questions: int, existing_pairs_df: pd.DataFrame, display_to_pid: Dict):
    project_id = _get_project_id(display_to_pid, selected_display)
    if not project_id:
        return existing_pairs_df, "Error: Project not found."

    try:
        from yasrl.loaders import DocumentLoader
        loader = DocumentLoader()
        documents = loader.load_documents(selected_source)

        dataset = await generate_evaluation_dataset(
            documents=documents,
            max_questions=int(num_questions)
        )

        new_pairs = []
        for tc in dataset.test_cases:
            new_pairs.append({
                "Question": tc.input,
                "Answer": tc.expected_output,
                "Context": tc.context[0] if tc.context else ""
            })

        new_pairs_df = pd.DataFrame(new_pairs)

        if existing_pairs_df is not None:
            combined_df = pd.concat([existing_pairs_df, new_pairs_df]).drop_duplicates(subset=['Question'])
        else:
            combined_df = new_pairs_df

        return combined_df, f"Generated {len(new_pairs)} new QA pairs."

    except Exception as e:
        logger.exception("Failed to generate QA pairs.")
        return existing_pairs_df, f"Error generating pairs: {e}"


def load_qa_pairs(selected_display: str, selected_source: str, display_to_pid: Dict):
    project_id = _get_project_id(display_to_pid, selected_display)
    if not project_id:
        return pd.DataFrame(), "Error: Project not found."

    conn = get_db_connection(os.getenv("POSTGRES_URI"))
    try:
        df = get_qa_pairs_for_source(conn, project_id, selected_source)
        return df, f"Loaded {len(df)} QA pairs."
    finally:
        conn.close()

def save_qa_pairs_to_db(selected_display: str, selected_source: str, qa_pairs_df: pd.DataFrame, display_to_pid: Dict):
    project_id = _get_project_id(display_to_pid, selected_display)
    if not project_id:
        return "Error: Project not found."

    conn = get_db_connection(os.getenv("POSTGRES_URI"))
    try:
        save_qa_pairs(conn, project_id, selected_source, qa_pairs_df)
        return f"Successfully saved {len(qa_pairs_df)} QA pairs."
    except Exception as e:
        return f"Error saving QA pairs: {e}"
    finally:
        conn.close()


def build_evaluation_tab(projects: Dict, display_to_pid: Dict):
    with gr.TabItem("Evaluation"):
        gr.Markdown("## Evaluation Workflow")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Select Project & Source")
                # We need to get the display names for the dropdown
                project_choices = list(display_to_pid.keys())
                eval_project_dropdown = gr.Dropdown(choices=project_choices, label="Select Project")
                eval_sources_dropdown = gr.Dropdown(label="Select Source")

                gr.Markdown("### 2. Generate QA Pairs")
                qa_status_message = gr.Markdown("Select a source to see existing QA pairs.")
                load_qa_button = gr.Button("Load Saved QA Pairs", visible=False)

                num_qa_pairs = gr.Number(label="Number of Questions to Generate", value=5, minimum=1, step=1)
                generate_qa_button = gr.Button("Generate New QA Pairs")

            with gr.Column(scale=2):
                gr.Markdown("### 3. Edit and Save QA Pairs")
                qa_pairs_editor = gr.DataFrame(
                    headers=["Question", "Answer", "Context"],
                    col_count=(3, "fixed"),
                    interactive=True,
                    label="Generated/Loaded QA Pairs",
                    wrap=True,
                    max_rows=20
                )
                save_qa_button = gr.Button("Save QA Pairs to Database")
                save_status_message = gr.Markdown()

        # Event Handlers
        eval_project_dropdown.change(
            fn=on_project_select,
            inputs=[eval_project_dropdown, gr.State(projects), gr.State(display_to_pid)],
            outputs=[eval_sources_dropdown, qa_status_message, qa_pairs_editor]
        )

        eval_sources_dropdown.change(
            fn=on_source_select,
            inputs=[eval_project_dropdown, eval_sources_dropdown, gr.State(display_to_pid)],
            outputs=[qa_status_message, load_qa_button]
        )

        load_qa_button.click(
            fn=load_qa_pairs,
            inputs=[eval_project_dropdown, eval_sources_dropdown],
            outputs=[qa_pairs_editor, save_status_message],
            fn_kwargs={"display_to_pid": display_to_pid}
        )

        generate_qa_button.click(
            fn=generate_and_append_pairs,
            inputs=[eval_project_dropdown, eval_sources_dropdown, num_qa_pairs, qa_pairs_editor, gr.State(display_to_pid)],
            outputs=[qa_pairs_editor, save_status_message],
        )

        save_qa_button.click(
            fn=save_qa_pairs_to_db,
            inputs=[eval_project_dropdown, eval_sources_dropdown, qa_pairs_editor],
            outputs=[save_status_message],
            fn_kwargs={"display_to_pid": display_to_pid}
        )

    return (
        eval_project_dropdown, eval_sources_dropdown, qa_status_message,
        load_qa_button, num_qa_pairs, generate_qa_button,
        qa_pairs_editor, save_qa_button, save_status_message
    )
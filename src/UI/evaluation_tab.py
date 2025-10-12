import gradio as gr
from gradio.components import Component
from typing import Dict, List


def build_evaluation_tab(initial_choices: List[str]) -> Dict[str, Component]:
    with gr.TabItem("Evaluation"):
        gr.Markdown("## Evaluation Dataset Editor")
        dropdown = gr.Dropdown(
            choices=initial_choices,
            label="Project",
            value=None if not initial_choices else initial_choices[0],
            interactive=True,
        )
        load_button = gr.Button("Load QA Pairs from Database")
        qa_table = gr.Dataframe(
            headers=["question", "answer", "context"],
            datatype=["str", "str", "str"],
            row_count=(1, "dynamic"),
            col_count=3,
            wrap=True,
            label="Question / Answer Pairs",
        )
        save_button = gr.Button("Save to Database", variant="primary")
        status = gr.Markdown("")
    return {
        "dropdown": dropdown,
        "load_button": load_button,
        "dataframe": qa_table,
        "save_button": save_button,
        "status": status,
    }

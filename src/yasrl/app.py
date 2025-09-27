import os
import asyncio
import logging
from typing import Optional, List, Tuple

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from yasrl.pipeline import RAGPipeline

logger = logging.getLogger("yasrl.app")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


async def _init_pipeline_async() -> None:
    """Async pipeline initialization."""
    global pipeline
    if pipeline is not None:
        return
    llm = os.getenv("DEMO_LLM", "gemini")
    embed_model = os.getenv("DEMO_EMBED_MODEL", "gemini")
    logger.info("Initializing pipeline (llm=%s embed=%s)...", llm, embed_model)
    pipeline = await RAGPipeline.create(llm=llm, embed_model=embed_model)
    logger.info("Pipeline initialized.")


def ensure_pipeline() -> None:
    """Synchronously ensure the pipeline is initialized (best-effort)."""
    global pipeline
    if pipeline is None:
        try:
            asyncio.run(_init_pipeline_async())
        except Exception:
            logger.exception("Pipeline init failed")
            pipeline = None


def format_history_for_pipeline(history: Optional[List[Tuple[str, str]]]) -> List[dict]:
    """
    Convert Gradio chat history (list of (user, bot) tuples) to pipeline conversation list.
    Returns a list like [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}].
    """
    if not history:
        return []
    convo: List[dict] = []
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


def respond(message: str, chat_history: Optional[List[Tuple[str, str]]]):
    """
    Gradio sync wrapper. Returns ("", new_chat_history).
    """
    ensure_pipeline()
    if pipeline is None:
        logger.error("Pipeline not available when answering")
        return "", (chat_history or []) + [(message, "Error: chatbot not available. Check logs.")]

    conversation_history = format_history_for_pipeline(chat_history)

    try:
        # call pipeline.ask asynchronously from sync context
        result = asyncio.run(pipeline.ask(query=message, conversation_history=conversation_history))
    except Exception as e:
        logger.exception("Error while asking pipeline: %s", e)
        new_h = chat_history or []
        new_h.append((message, f"Error during query: {e}"))
        return "", new_h

    # Build sources list for display
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


def build_ui(run_mode: str = "local"):
    """
    Build and launch the Gradio UI.
      run_mode: "local" -> bind 127.0.0.1 and open browser
                "production" -> bind 0.0.0.0 and don't open browser
    """
    port = int(os.getenv("CHATBOT_PORT", "7860"))
    host = "127.0.0.1" if run_mode == "local" else "0.0.0.0"
    inbrowser = run_mode == "local"

    logger.info("Starting Gradio UI (mode=%s host=%s port=%d)", run_mode, host, port)
    with gr.Blocks(title="YASRL Chatbot") as demo:
        gr.Markdown("# 🤖 YASRL Chatbot\nAsk questions based on the pre-indexed content.")
        chatbot = gr.Chatbot(elem_id="chatbot", label="Chat History")
        with gr.Row():
            msg = gr.Textbox(placeholder="Ask your question here...", show_label=False)
            submit = gr.Button("Send")
            clear = gr.Button("Clear")
        gr.Examples(
            examples=[
                "What is Vasilias?",
                "What services are offered?",
                "Tell me about the blog posts."
            ],
            inputs=msg
        )

        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        submit.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear.click(lambda: ("", []), inputs=None, outputs=[msg, chatbot])

    demo.launch(server_name=host, server_port=port, share=False, inbrowser=inbrowser)


if __name__ == "__main__":
    # choose mode: env CHATBOT_ENV or CLI arg "local"/"production"
    mode = os.getenv("CHATBOT_ENV", "").lower()
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("local", "production"):
        mode = sys.argv[1]

    if mode not in ("local", "production"):
        # default: local when developer runs directly
        mode = "local"

    # Try to initialize pipeline before UI (best-effort)
    try:
        if mode == "production":
            # in production we attempt init but won't fail startup if it errors
            try:
                asyncio.run(_init_pipeline_async())
            except Exception:
                logger.exception("Pre-init failed; will try on first request")
        else:
            # in local dev, init to catch errors early
            asyncio.run(_init_pipeline_async())
    except Exception:
        logger.exception("Pipeline init attempt failed")

    build_ui(run_mode=mode)
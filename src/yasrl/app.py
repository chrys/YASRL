import gradio as gr
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from yasrl.pipeline import RAGPipeline
from yasrl.database import log_feedback

# --- Global Pipeline Instance ---
pipeline: RAGPipeline | None = None

async def initialize_pipeline():
    """Initializes the RAG pipeline if it hasn't been already."""
    global pipeline
    if pipeline is None:
        print("🚀 Initializing RAG pipeline for the UI...")
        pipeline = await RAGPipeline.create(
            llm="gemini",
            embed_model="gemini"
        )
        print("✅ Pipeline initialized successfully.")

def format_history_for_pipeline(history):
    """Converts Gradio's chat history format to the pipeline's expected format."""
    if not history:
        return None
    
    formatted_history = []
    for user_msg, bot_msg in history:
        cleaned_bot_msg = bot_msg.split("\n\n---")[0] if bot_msg else ""
        formatted_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            formatted_history.append({"role": "assistant", "content": cleaned_bot_msg})
    return formatted_history

def chat_function_streaming(message, history):
    """
    Handles the streaming response from the RAG pipeline and yields
    the chatbot's answer token by token.
    """
    if pipeline is None:
        yield "Error: Chatbot is not available. Initialization failed."
        return

    conversation_history = format_history_for_pipeline(history)

    print(f"🤔 Processing query: '{message}'")
    result = asyncio.run(
        pipeline.ask(query=message, conversation_history=conversation_history)
    )
    answer = result.answer

    # Append sources to the final answer
    sources_text = ""
    if result.source_chunks:
        sources_text = "\n\n---\n**Sources:**\n"
        unique_sources = sorted(list(set(
            chunk.metadata.get('source', 'Unknown') for chunk in result.source_chunks
        )))
        for i, source in enumerate(unique_sources, 1):
            sources_text += f"*{i}. {source}*\n"
    
    # Yield the complete answer with sources
    yield answer + sources_text

def handle_feedback(feedback: gr.LikeData):
    """
    Handles user feedback (like/dislike) on chatbot responses.
    Logs the feedback to the database.
    """
    rating = "GOOD" if feedback.liked else "BAD"
    # The message is the chatbot's answer that was liked/disliked
    log_feedback(chatbot_answer=feedback.value, rating=rating)
    print(f"Received feedback: {'👍' if feedback.liked else '👎'} for answer: '{feedback.value}'")

def build_ui():
    """Builds the Gradio chatbot interface using Blocks for more control."""
    print("🎨 Building Gradio UI...")

    with gr.Blocks(theme=gr.themes.Soft(), title="YASRL Chatbot") as demo:
        gr.Markdown(
            """
            # 🤖 YASRL Chatbot
            This chatbot uses a RAG pipeline to answer questions based on pre-indexed web content.
            """
        )

        chatbot = gr.Chatbot(
            height=500,
            label="Chat History",
            bubble_full_width=False,
            likeable=True  # Enable feedback icons
        )

        with gr.Row():
            txt = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Ask your question here...",
                container=False,
            )
            btn = gr.Button("Submit", variant="primary")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            user_message = history[-1][0]
            history[-1][1] = ""
            # Stream the response
            for character in chat_function_streaming(user_message, history[:-1]):
                history[-1][1] = character
                yield history

        # Wire up the components
        txt.submit(user, [txt, chatbot], [txt, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        btn.click(user, [txt, chatbot], [txt, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        # Register the feedback handler
        chatbot.like(handle_feedback, None, None)

    print("🌐 Launching UI... Access it at http://127.0.0.1:7860 or your public URL.")
    # Ensure the app starts on the correct port if specified
    server_port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=server_port)

if __name__ == "__main__":
    # Run pipeline initialization in an event loop
    try:
        asyncio.run(initialize_pipeline())
    except Exception as e:
        print(f"❌ Fatal error during pipeline initialization: {e}")
        # Exit if the pipeline can't be created, as the app is useless without it.
        exit(1)

    build_ui()
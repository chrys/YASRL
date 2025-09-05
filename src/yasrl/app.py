import gradio as gr
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from yasrl.pipeline import RAGPipeline

# --- Global Pipeline Instance ---
# We initialize the pipeline once to avoid reloading it on every message.
# This reduces latency and resource usage.
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
        # The bot message might contain markdown for sources, so we clean it.
        cleaned_bot_msg = bot_msg.split("\n\n---")[0]
        formatted_history.append({"role": "user", "content": user_msg})
        formatted_history.append({"role": "assistant", "content": cleaned_bot_msg})
    return formatted_history

# --- The Core Chat Function for Gradio ---
def chat_function(message, history):
    """
    This function is called by Gradio for each user message.
    It handles the async logic of the RAG pipeline.
    """
    # This is a simple way to handle async startup in a sync context.
    if pipeline is None:
        try:
            asyncio.run(initialize_pipeline())
        except Exception as e:
            print(f"❌ Error initializing pipeline: {e}")
            return "Error: The chatbot pipeline could not be initialized. Please check the server logs."

    if pipeline is None:
        # This case should ideally not be reached if initialization is successful.
        return "Error: Chatbot is not available. Initialization failed."
    
    # Convert Gradio history to the format our pipeline expects
    conversation_history = format_history_for_pipeline(history)

    # Run the async 'ask' method
    print(f"🤔 Processing query: '{message}'")
    result = asyncio.run(
        pipeline.ask(query=message, conversation_history=conversation_history)
    )
    print(f"💡 Got answer: '{result.answer}'")

    # Format the sources for display in the chatbot UI
    sources_text = ""
    if result.source_chunks:
        sources_text = "\n\n---\n**Sources:**\n"
        # Use a set to show only unique source URLs
        unique_sources = sorted(list(set(
            chunk.metadata.get('source', 'Unknown') for chunk in result.source_chunks
        )))
        for i, source in enumerate(unique_sources, 1):
            sources_text += f"*{i}. {source}*\n"
    
    return result.answer + sources_text

# --- Build and Launch the Gradio UI ---
def build_ui():
    """Builds the Gradio chatbot interface."""
    print("🎨 Building Gradio UI...")
    
    with gr.Blocks(theme=gr.themes.Soft(), title="YASRL Chatbot") as demo:
        gr.Markdown(
            """
            # 🤖 YASRL Chatbot
            This chatbot uses a RAG pipeline to answer questions based on pre-indexed web content.
            """
        )
        gr.ChatInterface(
            fn=chat_function,
            examples=[
                "What is Vasilias?",
                "What services are offered?",
                "Tell me about the blog posts."
            ],
            title="YASRL Chatbot",
            chatbot=gr.Chatbot(height=500, label="Chat History"),
            textbox=gr.Textbox(placeholder="Ask your question here...", container=False, scale=7),
        )
    
    print("🌐 Launching UI... Access it at http://127.0.0.1:7860")
    demo.launch()

if __name__ == "__main__":
    build_ui()
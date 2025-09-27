import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import gradio as gr

from yasrl.app import (
    format_history_for_pipeline,
    chat_function_streaming,
    handle_feedback,
)
from yasrl.pipeline import RAGPipeline
from yasrl.models import QueryResult, SourceChunk

# --- Tests for format_history_for_pipeline ---

def test_format_history_for_pipeline_empty():
    """Test with an empty history."""
    assert format_history_for_pipeline([]) is None

def test_format_history_for_pipeline_single_turn():
    """Test with a single user-bot exchange."""
    history = [("Hello", "Hi there!")]
    expected = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert format_history_for_pipeline(history) == expected

def test_format_history_for_pipeline_with_none_bot_message():
    """Test history where the last bot message is None."""
    history = [("Hello", "Hi there!"), ("How are you?", None)]
    expected = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    assert format_history_for_pipeline(history) == expected

def test_format_history_for_pipeline_with_source():
    """Test with a bot message containing a source."""
    history = [("What is YASRL?", "It is a RAG pipeline.\n\n---\n**Sources:**\n*1. source.com*\n")]
    expected = [
        {"role": "user", "content": "What is YASRL?"},
        {"role": "assistant", "content": "It is a RAG pipeline."},
    ]
    assert format_history_for_pipeline(history) == expected


# --- Tests for chat_function_streaming ---

@pytest.fixture(autouse=True)
def set_pipeline_and_reset():
    """Set a mock pipeline for the tests and reset it after."""
    from yasrl import app

    # Create a mock pipeline for testing
    mock_pipeline_instance = AsyncMock(spec=RAGPipeline)
    app.pipeline = mock_pipeline_instance

    yield mock_pipeline_instance # Provide the mock to the tests

    # Reset the global pipeline object after the test
    app.pipeline = None

def test_chat_function_streaming_initialization_error():
    """Test streaming function when the pipeline is not initialized."""
    from yasrl import app
    app.pipeline = None # Ensure pipeline is None

    # The function is a generator, so we need to iterate it
    result_generator = chat_function_streaming("Hello", [])
    response = next(result_generator)

    assert "Error: Chatbot is not available" in response

def test_chat_function_streaming_success_no_sources(set_pipeline_and_reset):
    """Test a successful streaming interaction with no sources."""
    mock_pipeline_instance = set_pipeline_and_reset
    mock_pipeline_instance.ask.return_value = QueryResult(
        answer="This is a test answer.",
        source_chunks=[]
    )

    result_generator = chat_function_streaming("What is this?", [])
    response = next(result_generator)

    assert response == "This is a test answer."
    mock_pipeline_instance.ask.assert_awaited_once_with(
        query="What is this?",
        conversation_history=None
    )

def test_chat_function_streaming_success_with_sources(set_pipeline_and_reset):
    """Test a successful streaming interaction with history and sources."""
    mock_pipeline_instance = set_pipeline_and_reset
    mock_pipeline_instance.ask.return_value = QueryResult(
        answer="This is another test answer.",
        source_chunks=[
            SourceChunk(text="chunk1", metadata={"source": "url1"}),
            SourceChunk(text="chunk2", metadata={"source": "url2"}),
        ]
    )

    history = [("Previous question", "Previous answer")]
    result_generator = chat_function_streaming("Another question", history)
    response = next(result_generator)

    assert "This is another test answer." in response
    assert "**Sources:**" in response
    assert "*1. url1*" in response
    assert "*2. url2*" in response

    expected_history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]
    mock_pipeline_instance.ask.assert_awaited_once_with(
        query="Another question",
        conversation_history=expected_history
    )


# --- Tests for handle_feedback ---

@patch('yasrl.app.log_feedback')
def test_handle_feedback_like(mock_log_feedback):
    """Test the feedback handler for a 'like' event."""
    # Mock the LikeData object as its constructor is not meant for direct use
    feedback_data = MagicMock()
    feedback_data.value = "This was a great answer!"
    feedback_data.liked = True

    handle_feedback(feedback_data)

    mock_log_feedback.assert_called_once_with(
        chatbot_answer="This was a great answer!",
        rating="GOOD"
    )

@patch('yasrl.app.log_feedback')
def test_handle_feedback_dislike(mock_log_feedback):
    """Test the feedback handler for a 'dislike' event."""
    # Mock the LikeData object
    feedback_data = MagicMock()
    feedback_data.value = "This answer was unhelpful."
    feedback_data.liked = False

    handle_feedback(feedback_data)

    mock_log_feedback.assert_called_once_with(
        chatbot_answer="This answer was unhelpful.",
        rating="BAD"
    )
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from src.yasrl.app import format_history_for_pipeline, chat_function
from src.yasrl.pipeline import RAGPipeline
from src.yasrl.models import QueryResult, SourceChunk

# --- Tests for format_history_for_pipeline ---

def test_format_history_for_pipeline_empty():
    """Test with an empty history."""
    history = []
    assert format_history_for_pipeline(history) is None

def test_format_history_for_pipeline_single_turn():
    """Test with a single user-bot exchange."""
    history = [("Hello", "Hi there!")]
    expected = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    assert format_history_for_pipeline(history) == expected

def test_format_history_for_pipeline_multiple_turns():
    """Test with multiple user-bot exchanges."""
    history = [
        ("Hello", "Hi there!"),
        ("How are you?", "I'm good, thanks!")
    ]
    expected = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm good, thanks!"}
    ]
    assert format_history_for_pipeline(history) == expected

def test_format_history_for_pipeline_with_source():
    """Test with a bot message containing a source."""
    history = [("What is YASRL?", "It is a RAG pipeline.\n\n---\n**Sources:**\n*1. some-source.com*\n")]
    expected = [
        {"role": "user", "content": "What is YASRL?"},
        {"role": "assistant", "content": "It is a RAG pipeline."}
    ]
    assert format_history_for_pipeline(history) == expected

# --- Tests for chat_function ---

@pytest.fixture(autouse=True)
def reset_pipeline():
    """Reset the global pipeline object before each test."""
    from src.yasrl import app
    app.pipeline = None
    yield
    app.pipeline = None


@patch('src.yasrl.app.initialize_pipeline', new_callable=AsyncMock)
def test_chat_function_initialization_error(mock_init_pipeline):
    """Test chat_function when pipeline initialization fails."""
    mock_init_pipeline.side_effect = Exception("Initialization failed!")

    response = chat_function("Hello", [])

    assert "Error: The chatbot pipeline could not be initialized." in response
    mock_init_pipeline.assert_awaited_once()

@patch('src.yasrl.app.RAGPipeline.create', new_callable=AsyncMock)
def test_chat_function_success_no_history_no_sources(mock_create_pipeline):
    """Test a successful chat interaction with no history and no sources."""
    # Mock the pipeline and its 'ask' method
    mock_pipeline_instance = AsyncMock(spec=RAGPipeline)
    mock_pipeline_instance.ask.return_value = QueryResult(
        answer="This is a test answer.",
        source_chunks=[]
    )
    mock_create_pipeline.return_value = mock_pipeline_instance

    # Set the global pipeline variable
    from src.yasrl import app
    app.pipeline = mock_pipeline_instance

    response = chat_function("What is this?", [])

    assert response == "This is a test answer."
    mock_pipeline_instance.ask.assert_awaited_once_with(
        query="What is this?",
        conversation_history=None
    )

@patch('src.yasrl.app.RAGPipeline.create', new_callable=AsyncMock)
def test_chat_function_success_with_history_and_sources(mock_create_pipeline):
    """Test a successful chat interaction with history and sources."""
    # Mock the pipeline and its 'ask' method
    mock_pipeline_instance = AsyncMock(spec=RAGPipeline)
    mock_pipeline_instance.ask.return_value = QueryResult(
        answer="This is another test answer.",
        source_chunks=[
            SourceChunk(text="chunk1", metadata={"source": "url1"}),
            SourceChunk(text="chunk2", metadata={"source": "url2"}),
            SourceChunk(text="chunk3", metadata={"source": "url1"}) # Duplicate source
        ]
    )
    mock_create_pipeline.return_value = mock_pipeline_instance

    # Set the global pipeline variable
    from src.yasrl import app
    app.pipeline = mock_pipeline_instance

    history = [("Previous question", "Previous answer")]
    response = chat_function("Another question", history)

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

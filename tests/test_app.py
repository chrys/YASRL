from urllib import response
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from yasrl.app import format_history_for_pipeline, respond
from yasrl.pipeline import RAGPipeline
from yasrl.models import QueryResult, SourceChunk

# --- Tests for format_history_for_pipeline ---


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


# --- Tests for chat_function ---

@pytest.fixture(autouse=True)
def reset_pipeline():
    """Reset the global pipeline object before each test."""
    from yasrl import app
    app.pipeline = None
    yield
    app.pipeline = None




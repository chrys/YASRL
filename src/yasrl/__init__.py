"""
YasRL: A lightweight, extensible, and scalable RAG framework.
"""

__version__ = "0.1.0"

from .pipeline import RAGPipeline
from .models import QueryResult, SourceChunk
from .exceptions import (
    yasrlError,
    ConfigurationError,
    IndexingError,
    RetrievalError,
)

__all__ = [
    "__version__",
    "RAGPipeline",
    "QueryResult",
    "SourceChunk",
    "yasrlError",
    "ConfigurationError",
    "IndexingError",
    "RetrievalError",
]
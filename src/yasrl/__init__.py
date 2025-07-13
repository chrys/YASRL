__version__ = "0.1.0"

# Public API imports (update these as you implement the modules)
# from .pipeline import RAGPipeline
from .models import QueryResult, SourceChunk
# from .exceptions import yasrlError, ConfigurationError, IndexingError, RetrievalError, EvaluationError

__all__ = [
    "__version__",
    # "RAGPipeline",
    "QueryResult",
    "SourceChunk",
    # "yasrlError",
    # "ConfigurationError",
    # "IndexingError",
]
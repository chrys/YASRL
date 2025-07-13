class yasrlError(Exception):
    """
    Base exception for all yasrl errors.
    Raised for general errors in the yasrl library.
    """
    def __init__(self, message=None):
        if message is None:
            message = "An error occurred in the yasrl library."
        super().__init__(message)

class ConfigurationError(yasrlError):
    """
    Raised when a required environment variable or configuration is missing or invalid.
    """
    def __init__(self, message=None):
        if message is None:
            message = "Missing or invalid configuration for yasrl."
        super().__init__(message)

class IndexingError(yasrlError):
    """
    Raised when a failure occurs during document indexing.
    """
    def __init__(self, message=None):
        if message is None:
            message = "Error during document indexing in yasrl."
        super().__init__(message)

class RetrievalError(yasrlError):
    """
    Raised when a failure occurs during query retrieval.
    """
    def __init__(self, message=None):
        if message is None:
            message = "Error during query retrieval in yasrl."
        super().__init__(message)

class EvaluationError(yasrlError):
    """
    Raised when a failure occurs during evaluation.
    """
    def __init__(self, message=None):
        if message is None:
            message = "Error during evaluation in yasrl."
        super().__init__(message)

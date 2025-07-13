import pytest
from yasrl.exceptions import yasrlError, ConfigurationError, IndexingError, RetrievalError, EvaluationError

def test_yasrlerror():
    with pytest.raises(yasrlError) as exc:
        raise yasrlError()
    assert "yasrl" in str(exc.value)

def test_configurationerror():
    with pytest.raises(ConfigurationError) as exc:
        raise ConfigurationError()
    assert "configuration" in str(exc.value).lower()

def test_indexingerror():
    with pytest.raises(IndexingError) as exc:
        raise IndexingError()
    assert "indexing" in str(exc.value).lower()

def test_retrievalerror():
    with pytest.raises(RetrievalError) as exc:
        raise RetrievalError()
    assert "retrieval" in str(exc.value).lower()

def test_evaluationerror():
    with pytest.raises(EvaluationError) as exc:
        raise EvaluationError()
    assert "evaluation" in str(exc.value).lower()

def test_inheritance():
    assert issubclass(ConfigurationError, yasrlError)
    assert issubclass(IndexingError, yasrlError)
    assert issubclass(RetrievalError, yasrlError)
    assert issubclass(EvaluationError, yasrlError)

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from yasrl.evaluation.ragas_evaluator import RagasEvaluator
from yasrl.providers.llm import LLMProvider
from yasrl.providers.embeddings import EmbeddingProvider
from yasrl.models import QueryResult, SourceChunk
from datasets import Dataset

@pytest.fixture
def mock_llm_provider():
    return MagicMock(spec=LLMProvider)

@pytest.fixture
def mock_embedding_provider():
    return MagicMock(spec=EmbeddingProvider)

@pytest.fixture
def ragas_evaluator(mock_llm_provider, mock_embedding_provider):
    return RagasEvaluator(
        llm_provider=mock_llm_provider,
        embedding_provider=mock_embedding_provider,
    )

def test_ragas_evaluator_init(ragas_evaluator):
    assert ragas_evaluator.llm_provider is not None
    assert ragas_evaluator.embedding_provider is not None
    assert "faithfulness" in ragas_evaluator.supported_metrics

@patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate")
def test_evaluate_single(mock_ragas_evaluate, ragas_evaluator):
    mock_ragas_evaluate.return_value = {"faithfulness": 1.0}

    async def run_test():
        result = await ragas_evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG is a technique.",
            context=["RAG stands for Retrieval-Augmented Generation."],
            ground_truth="RAG is a technique for improving LLMs.",
        )
        assert result["faithfulness"] == 1.0

    asyncio.run(run_test())

@patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate")
def test_evaluate(mock_ragas_evaluate, ragas_evaluator):
    mock_ragas_evaluate.return_value = {
        "faithfulness": 0.8,
        "answer_relevancy": 0.9,
    }

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = QueryResult(
        answer="RAG is a technique.",
        source_chunks=[SourceChunk(text="RAG stands for Retrieval-Augmented Generation.", score=0.9)]
    )

    async def run_test():
        results = await ragas_evaluator.evaluate(
            pipeline=mock_pipeline,
            questions=["What is RAG?"],
            ground_truths=["RAG is a technique for improving LLMs."],
        )
        assert results["overall_scores"]["faithfulness"] == 0.8
        assert results["overall_scores"]["answer_relevancy"] == 0.9

    asyncio.run(run_test())

def test_prepare_dataset(ragas_evaluator):
    questions = ["q1"]
    answers = ["a1"]
    contexts = [["c1"]]
    ground_truths = ["g1"]

    dataset = ragas_evaluator._prepare_dataset(questions, answers, contexts, ground_truths)

    assert isinstance(dataset, Dataset)
    assert dataset[0]["question"] == "q1"

@patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate", new_callable=AsyncMock)
async def test_evaluate_with_caching(mock_ragas_evaluate, ragas_evaluator):
    mock_ragas_evaluate.return_value = {
        "faithfulness": 1.0
    }

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = QueryResult(
        answer="RAG is a technique.",
        source_chunks=[SourceChunk(text="RAG stands for Retrieval-Augmented Generation.", score=0.9)]
    )

    questions = ["What is RAG?"]
    ground_truths = ["RAG is a technique for improving LLMs."]

    # First call
    await ragas_evaluator.evaluate(
        pipeline=mock_pipeline,
        questions=questions,
        ground_truths=ground_truths,
    )

    # Second call
    await ragas_evaluator.evaluate(
        pipeline=mock_pipeline,
        questions=questions,
        ground_truths=ground_truths,
    )

    assert mock_ragas_evaluate.call_count == 2

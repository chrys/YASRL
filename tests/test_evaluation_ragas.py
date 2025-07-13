import pytest
import asyncio
<<<<<<< HEAD
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
=======
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from datasets import Dataset

from yasrl.evaluation.ragas_evaluator import RagasEvaluator
from yasrl.evaluation.base import EvaluationError
from yasrl.models import QueryResult, SourceChunk
from yasrl.providers.llm import LLMProvider
from yasrl.providers.embeddings import EmbeddingProvider


@pytest.fixture
def mock_llm_provider():
    mock = MagicMock(spec=LLMProvider)
    mock.get_llm = MagicMock(return_value="mock_llm")
    return mock


@pytest.fixture
def mock_embedding_provider():
    mock = MagicMock(spec=EmbeddingProvider)
    mock.get_embedding_model = MagicMock(return_value="mock_embeddings")
    return mock


@pytest.fixture
def ragas_evaluator(mock_llm_provider, mock_embedding_provider):
    return RagasEvaluator(mock_llm_provider, mock_embedding_provider)

def make_metric_mock(mean_value):
    metric_mock = MagicMock()
    metric_mock.dropna.return_value.mean.side_effect = lambda *args, **kwargs: mean_value
    metric_mock.dropna.return_value.__len__.side_effect = lambda *args, **kwargs: 1
    return metric_mock
    
class TestRagasEvaluator:
    def test_initialization_default_metrics(self, mock_llm_provider, mock_embedding_provider):
        """Test evaluator initialization with default metrics."""
        evaluator = RagasEvaluator(mock_llm_provider, mock_embedding_provider)
        expected_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        assert evaluator.metric_names == expected_metrics
        assert evaluator.supported_metrics == expected_metrics
        assert len(evaluator.metrics) == 4

    def test_initialization_custom_metrics(self, mock_llm_provider, mock_embedding_provider):
        """Test evaluator initialization with custom metrics."""
        custom_metrics = ["faithfulness", "answer_relevancy"]
        evaluator = RagasEvaluator(mock_llm_provider, mock_embedding_provider, custom_metrics)
        assert evaluator.metric_names == custom_metrics
        assert len(evaluator.metrics) == 2

    def test_initialization_invalid_metrics(self, mock_llm_provider, mock_embedding_provider):
        """Test evaluator initialization with invalid metrics raises error."""
        invalid_metrics = ["faithfulness", "invalid_metric"]
        with pytest.raises(ValueError, match="Unsupported metrics"):
            RagasEvaluator(mock_llm_provider, mock_embedding_provider, invalid_metrics)

    def test_prepare_dataset_with_ground_truths(self, ragas_evaluator):
        """Test dataset preparation with ground truths."""
        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]
        contexts = [["C1"], ["C2"]]
        ground_truths = ["GT1", "GT2"]

        dataset = ragas_evaluator._prepare_dataset(questions, answers, contexts, ground_truths)
        
        assert isinstance(dataset, Dataset)
        assert dataset["question"] == questions
        assert dataset["answer"] == answers
        assert dataset["contexts"] == contexts
        assert dataset["ground_truth"] == ground_truths

    def test_prepare_dataset_no_ground_truth(self, ragas_evaluator):
        """Test dataset preparation without ground truths."""
        questions = ["Q1"]
        answers = ["A1"]
        contexts = [["C1"]]
        ground_truths = [None]

        dataset = ragas_evaluator._prepare_dataset(questions, answers, contexts, ground_truths)
        
        assert "ground_truth" not in dataset.column_names

    def test_prepare_dataset_partial_ground_truths(self, ragas_evaluator):
        """Test dataset preparation with partial ground truths."""
        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]
        contexts = [["C1"], ["C2"]]
        ground_truths = ["GT1", None]  # Partial ground truths

        dataset = ragas_evaluator._prepare_dataset(questions, answers, contexts, ground_truths)
        
        # Should not include ground_truth field when not all are provided
        assert "ground_truth" not in dataset.column_names

    @patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate")
    def test_evaluate_single_success(self, mock_ragas_evaluate, ragas_evaluator):
        """Test successful single evaluation."""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = QueryResult(
            answer="RAG is a technique.",
            source_chunks=[SourceChunk(text="RAG stands for Retrieval-Augmented Generation.", score=0.9)]
        )

        # Mock RAGAS evaluation result with to_pandas method
        mock_result = MagicMock()
        mock_result.to_pandas.return_value.to_dict.return_value = [
            {"faithfulness": 1.0, "answer_relevancy": 0.9}
        ]
        # Make it work with iteration and attribute access
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_ragas_evaluate.return_value = mock_result

        result = ragas_evaluator.evaluate_single(
            question="What is RAG?",
            expected_answer="RAG is a technique.",
            pipeline=mock_pipeline,
            ground_truth="RAG is a technique for improving LLMs.",
        )

        # Verify calls
        mock_pipeline.assert_called_once_with("What is RAG?")
        mock_ragas_evaluate.assert_called_once()
        assert isinstance(result, dict)

    @patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate")
    def test_evaluate_single_error_handling(self, mock_ragas_evaluate, ragas_evaluator):
        """Test error handling in single evaluation."""
        mock_pipeline = MagicMock()
        mock_ragas_evaluate.side_effect = Exception("RAGAS failed")

        with pytest.raises(EvaluationError, match="Failed to evaluate with RAGAS"):
            ragas_evaluator.evaluate_single(
                question="Test question",
                expected_answer="Test answer",
                pipeline=mock_pipeline
            )



    def test_format_results_evaluation_result(self, ragas_evaluator):
        """Test formatting RAGAS EvaluationResult objects."""
        # Mock EvaluationResult with to_pandas method
        mock_results = MagicMock()
        mock_df = MagicMock()
        mock_df.columns = ["faithfulness", "answer_relevancy", "extra_metric"]
        mock_df.to_dict.return_value = [
            {"faithfulness": 0.8, "answer_relevancy": 0.9, "extra_metric": 0.5}
        ]
        mock_df.__getitem__.side_effect = lambda col: make_metric_mock(0.90)
        mock_results.to_pandas.return_value = mock_df

        formatted = ragas_evaluator._format_results(mock_results)
        
        assert "overall_scores" in formatted
        assert "per_question_results" in formatted
        assert len(formatted["per_question_results"]) == 1

    def test_format_results_dict_fallback(self, ragas_evaluator):
        """Test formatting dict-like results."""
        mock_results = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
            "extra_metric": 0.5  # This should be filtered out
        }

        formatted = ragas_evaluator._format_results(mock_results)
        
        assert "overall_scores" in formatted
        assert "per_question_results" in formatted
        assert "extra_metric" not in formatted["overall_scores"]
        assert "faithfulness" in formatted["overall_scores"]
        assert "answer_relevancy" in formatted["overall_scores"]


    def test_evaluate_mismatched_lengths(self, ragas_evaluator):
        """Test evaluation with mismatched question and ground truth lengths."""
        mock_pipeline = MagicMock()

        async def run_test():
            with pytest.raises(ValueError, match="Number of questions must match"):
                await ragas_evaluator.evaluate(
                    questions=["Q1", "Q2"],
                    expected_answers=["A1", "A2"],
                    pipeline=mock_pipeline,
                    ground_truths=["GT1"]  # Mismatched length
                )

        asyncio.run(run_test())

    @patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate")
    def test_evaluate_error_handling(self, mock_ragas_evaluate, ragas_evaluator):
        """Test error handling in multi-question evaluation."""
        mock_pipeline = MagicMock()
        mock_pipeline.ask = AsyncMock(return_value=QueryResult(
            answer="Test answer",
            source_chunks=[SourceChunk(text="Test context", score=0.9)]
        ))
        mock_ragas_evaluate.side_effect = Exception("RAGAS evaluation failed")

        async def run_test():
            with pytest.raises(EvaluationError, match="Failed to evaluate with RAGAS"):
                await ragas_evaluator.evaluate(
                    questions=["What is RAG?"],
                    expected_answers=["Expected answer"],
                    pipeline=mock_pipeline
                )

        asyncio.run(run_test())

    @patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate")
    def test_evaluate_pipeline_batch(self, mock_ragas_evaluate, ragas_evaluator):
        """Test batch evaluation functionality."""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.ask = AsyncMock(return_value=QueryResult(
            answer="Test answer",
            source_chunks=[SourceChunk(text="Test context", score=0.9)]
        ))

        # Mock RAGAS results for batches
        mock_results = []
        for i in range(2):  # 2 batches
            mock_result = MagicMock()
            mock_df = MagicMock()
            mock_df.columns = ["faithfulness"]
            mock_df.to_dict.return_value = [{"faithfulness": 0.8 + i * 0.1}]
            mock_df.__getitem__.side_effect = lambda col: MagicMock(
                dropna=lambda: MagicMock(mean=lambda: 0.8 + i * 0.1, __len__=lambda: 1)
            )
            mock_result.to_pandas.return_value = mock_df
            mock_results.append(mock_result)
        
        mock_ragas_evaluate.side_effect = mock_results

        async def run_test():
            results = await ragas_evaluator.evaluate_pipeline_batch(
                pipeline=mock_pipeline,
                questions=["Q1", "Q2", "Q3"],  # Will create 2 batches with batch_size=2
                batch_size=2
            )
            
            assert "overall_scores" in results
            assert "per_question_results" in results
            assert "metadata" in results
            assert results["metadata"]["batch_count"] == 2

        asyncio.run(run_test())

    def test_aggregate_batch_results(self, ragas_evaluator):
        """Test aggregation of batch results."""
        batch_results = [
            {
                "overall_scores": {"faithfulness": 0.8},
                "per_question_results": [{"faithfulness": 0.8, "answer_relevancy": 0.9}],
                "metadata": {"total_questions": 1}
            },
            {
                "overall_scores": {"faithfulness": 0.9},
                "per_question_results": [{"faithfulness": 0.9, "answer_relevancy": 0.8}],
                "metadata": {"total_questions": 1}
            }
        ]

        aggregated = ragas_evaluator._aggregate_batch_results(batch_results)
        
        assert aggregated["overall_scores"]["faithfulness"] == pytest.approx(0.85) # Average of 0.8 and 0.9
        assert aggregated["overall_scores"]["answer_relevancy"] == 0.85  # Average of 0.9 and 0.8
        assert len(aggregated["per_question_results"]) == 2
        assert aggregated["metadata"]["batch_count"] == 2
        assert aggregated["metadata"]["total_questions"] == 2

    def test_aggregate_batch_results_missing_metrics(self, ragas_evaluator):
        """Test aggregation with missing metrics in some results."""
        batch_results = [
            {
                "overall_scores": {"faithfulness": 0.8},
                "per_question_results": [{"faithfulness": 0.8}],
                "metadata": {"total_questions": 1}
            },
            {
                "overall_scores": {"faithfulness": 0.9},
                "per_question_results": [{"answer_relevancy": 0.8}],  # Missing faithfulness
                "metadata": {"total_questions": 1}
            }
        ]

        aggregated = ragas_evaluator._aggregate_batch_results(batch_results)
        
        # Should handle missing metrics gracefully
        assert "faithfulness" in aggregated["overall_scores"]
        assert "answer_relevancy" in aggregated["overall_scores"]

    @patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate")
    def test_evaluate_pipeline_batch_all_batches_fail(self, mock_ragas_evaluate, ragas_evaluator):
        """Test batch evaluation when all batches fail."""
        mock_pipeline = MagicMock()
        mock_ragas_evaluate.side_effect = Exception("All batches failed")

        async def run_test():
            with pytest.raises(EvaluationError, match="All batches failed during evaluation"):
                await ragas_evaluator.evaluate_pipeline_batch(
                    pipeline=mock_pipeline,
                    questions=["Q1", "Q2"],
                    batch_size=1
                )

        asyncio.run(run_test())

    def test_supported_metrics_property(self, ragas_evaluator):
        """Test the supported_metrics property."""
        expected_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        assert ragas_evaluator.supported_metrics == expected_metrics

    @patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate")
    def test_evaluate_without_ground_truths(self, mock_ragas_evaluate, ragas_evaluator):
        """Test evaluation without ground truths."""
        mock_pipeline = MagicMock()
        mock_pipeline.ask = AsyncMock(return_value=QueryResult(
            answer="Test answer",
            source_chunks=[SourceChunk(text="Test context", score=0.9)]
        ))

        mock_result = MagicMock()
        mock_df = MagicMock()
        mock_df.columns = ["faithfulness", "answer_relevancy"]
        mock_df.to_dict.return_value = [{"faithfulness": 0.8, "answer_relevancy": 0.9}]
        mock_df.__getitem__.side_effect = lambda col: MagicMock(
            dropna=lambda: MagicMock(mean=lambda: 0.85, __len__=lambda: 1)
        )
        mock_result.to_pandas.return_value = mock_df
        mock_ragas_evaluate.return_value = mock_result

        async def run_test():
            results = await ragas_evaluator.evaluate(
                questions=["What is RAG?"],
                expected_answers=["Expected answer"],
                pipeline=mock_pipeline,
                ground_truths=None  # No ground truths
            )
            
            assert "overall_scores" in results
            assert "metadata" in results

        asyncio.run(run_test())

    def test_metric_name_access(self, ragas_evaluator):
        """Test access to metric names and objects."""
        # Test that metric names are accessible
        assert hasattr(ragas_evaluator, 'metric_names')
        assert hasattr(ragas_evaluator, 'metrics')
        assert len(ragas_evaluator.metric_names) == len(ragas_evaluator.metrics)
        
        # Test that metrics have the expected name attribute
        for metric in ragas_evaluator.metrics:
            assert hasattr(metric, 'name') or hasattr(metric, '__name__')


class TestRagasEvaluatorIntegration:
    """Integration tests that test the evaluator with more realistic scenarios."""
    
    @patch("yasrl.evaluation.ragas_evaluator.ragas_evaluate")
    def test_end_to_end_evaluation_flow(self, mock_ragas_evaluate, mock_llm_provider, mock_embedding_provider):
        """Test complete evaluation flow from start to finish."""
        evaluator = RagasEvaluator(
            mock_llm_provider, 
            mock_embedding_provider, 
            metrics=["faithfulness", "answer_relevancy"]
        )
        
        # Mock pipeline with realistic data
        mock_pipeline = MagicMock()
        mock_pipeline.ask = AsyncMock(return_value=QueryResult(
            answer="Retrieval-Augmented Generation (RAG) is a framework that combines retrieval and generation.",
            source_chunks=[
                SourceChunk(text="RAG combines retrieval with generation for better AI responses.", score=0.95),
                SourceChunk(text="The framework retrieves relevant documents before generating answers.", score=0.87)
            ]
        ))

        # Mock realistic RAGAS results
        mock_result = MagicMock()
        mock_df = MagicMock()
        mock_df.columns = ["faithfulness", "answer_relevancy"]
        mock_df.to_dict.return_value = [
            {"faithfulness": 0.92, "answer_relevancy": 0.88}
        ]
        mock_df.__getitem__.side_effect = lambda col: MagicMock(
            dropna=lambda: MagicMock(mean=lambda: 0.90, __len__=lambda: 1)
        )
        mock_result.to_pandas.return_value = mock_df
        mock_ragas_evaluate.return_value = mock_result

        async def run_test():
            results = await evaluator.evaluate(
                questions=["What is RAG and how does it work?"],
                expected_answers=["Expected comprehensive answer"],
                pipeline=mock_pipeline,
                ground_truths=["RAG is a technique that combines retrieval and generation for improved AI responses."]
            )
            
            # Verify complete result structure
            assert "overall_scores" in results
            assert "per_question_results" in results
            assert "metadata" in results
            
            # Verify metadata completeness
            metadata = results["metadata"]
            assert metadata["total_questions"] == 1
            assert metadata["evaluator"] == "RAGAS"
            assert metadata["metrics_used"] == ["faithfulness", "answer_relevancy"]
            
            # Verify that pipeline was called correctly
            mock_pipeline.ask.assert_called_once_with("What is RAG and how does it work?")
            
            # Verify RAGAS was called with correct data structure
            mock_ragas_evaluate.assert_called_once()
            call_args = mock_ragas_evaluate.call_args
            dataset = call_args[1]["dataset"]  # dataset is passed as keyword arg
            assert "question" in dataset.column_names
            assert "answer" in dataset.column_names
            assert "contexts" in dataset.column_names

        asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__])
>>>>>>> 774589e (ragas)

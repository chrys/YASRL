from typing import Any, List, Dict, Optional
from yasrl.evaluation.base import BaseEvaluator, RAGPipeline
from yasrl.models import QueryResult
import logging
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from yasrl.providers.llm import LLMProvider
from yasrl.providers.embeddings import EmbeddingProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RagasEvaluator(BaseEvaluator):
    """
    Evaluator using the RAGAS framework.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        metrics: Optional[List[str]] = None,
        batch_size: int = 10,
        cache: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RAGAS evaluator.
        """
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self._supported_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        self.metrics = self._validate_metrics(metrics)
        self.batch_size = batch_size
        self.cache = cache if cache is not None else {}

    @property
    def supported_metrics(self) -> List[str]:
        """Return a list of metrics this evaluator supports."""
        return list(self._supported_metrics.keys())

    def _validate_metrics(self, metrics: Optional[List[str]]) -> List[Any]:
        """Validate and return the metric functions."""
        if metrics is None:
            return list(self._supported_metrics.values())

        validated_metrics = []
        for metric in metrics:
            if metric not in self._supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}. Supported metrics are: {self.supported_metrics}")
            validated_metrics.append(self._supported_metrics[metric])
        return validated_metrics

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    async def evaluate(
        self,
        pipeline: RAGPipeline,
        questions: List[str],
        ground_truths: List[str],
    ) -> Dict[str, Any]:
        cache_key = f"{questions}-{ground_truths}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        """
        Run RAGAS evaluation on a question set.
        """

        pipeline_results = [pipeline.query(q) for q in questions]
        answers = [res.answer for res in pipeline_results]
        contexts = [[chunk.text for chunk in res.source_chunks] for res in pipeline_results]

        dataset = self._prepare_dataset(questions, answers, contexts, ground_truths)

        results = await asyncio.to_thread(
            ragas_evaluate,
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm_provider.get_llm(),
            embeddings=self.embedding_provider.get_embeddings(),
        )

        formatted_results = self._format_results(results)
        self.cache[cache_key] = formatted_results
        return formatted_results

    def _prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Dataset:
        """
        Prepare the dataset for RAGAS evaluation.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        return Dataset.from_dict(data)

    def _format_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the RAGAS results into a standardized format.
        """
        return {
            "overall_scores": {k: v for k, v in results.items() if k in self.supported_metrics},
            "per_question_results": results.to_pandas().to_dict("records"),
        }

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single question-answer pair.
        """
        ground_truth_list = [ground_truth] if ground_truth else [None]
        dataset = self._prepare_dataset(
            [question], [answer], [context], ground_truth_list
        )

        results = ragas_evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm_provider.get_llm(),
            embeddings=self.embedding_provider.get_embeddings(),
        )

        return {metric.name: results[metric.name] for metric in self.metrics}

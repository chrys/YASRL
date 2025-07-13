<<<<<<< HEAD
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

=======
from typing import Dict, Any, List, Optional, cast
import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from .base import BaseEvaluator, EvaluationError
from ..models import QueryResult, SourceChunk
from ..providers.llm import LLMProvider
from ..providers.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class RagasEvaluator(BaseEvaluator):
    """
    RAGAS-based evaluator for RAG pipeline performance assessment.
    
    This evaluator integrates with the RAGAS library to provide standard RAG metrics
    including faithfulness, answer relevancy, context precision, and context recall.
    
    Args:
        llm_provider: LLM provider for evaluation
        embedding_provider: Embedding provider for evaluation
        metrics: List of RAGAS metrics to use. Defaults to all supported metrics.
        
    Example:
        >>> evaluator = RagasEvaluator(llm_provider, embedding_provider)
        >>> results = await evaluator.evaluate(questions, ground_truths, pipeline)
        >>> print(f"Faithfulness: {results['overall_scores']['faithfulness']}")
    """
    
>>>>>>> 774589e (ragas)
    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
<<<<<<< HEAD
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
=======
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            llm_provider: LLM provider for evaluation
            embedding_provider: Embedding provider for evaluation
            metrics: List of metric names to use. If None, uses all supported metrics.
        """
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        
        # Map metric names to RAGAS metric objects
        self._metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall
        }
        
        # Set up metrics to use
        if metrics is None:
            self.metric_names = list(self._metric_map.keys())
        else:
            invalid_metrics = set(metrics) - set(self._metric_map.keys())
            if invalid_metrics:
                raise ValueError(f"Unsupported metrics: {invalid_metrics}. "
                               f"Supported: {list(self._metric_map.keys())}")
            self.metric_names = metrics
            
        self.metrics = [self._metric_map[name] for name in self.metric_names]
        
        logger.info(f"Initialized RAGAS evaluator with metrics: {self.metric_names}")

    @property
    def supported_metrics(self) -> List[str]:
        """Return list of supported RAGAS metrics."""
        return list(self._metric_map.keys())
>>>>>>> 774589e (ragas)

    def _prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
<<<<<<< HEAD
        ground_truths: List[str],
    ) -> Dataset:
        """
        Prepare the dataset for RAGAS evaluation.
        """
=======
        ground_truths: List[Optional[str]]
    ) -> Dataset:
        """
        Prepare data in RAGAS dataset format.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists for each question
            ground_truths: List of ground truth answers (can contain None)
            
        Returns:
            Dataset formatted for RAGAS evaluation
        """
        # Filter out None ground truths for metrics that require them
>>>>>>> 774589e (ragas)
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
<<<<<<< HEAD
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
=======
        }
        
        # Only include ground truth if all are provided
        if all(gt is not None for gt in ground_truths):
            data["ground_truth"] = ground_truths
            
        return Dataset.from_dict(data)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def evaluate_single(
        self,
        question: str,
        expected_answer: str,
        pipeline,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair using RAGAS metrics.
        
        Args:
            question: The question to evaluate
            answer: The generated answer
            context: List of context strings used for generation
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary with metric scores
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            # Use pipeline to get context for the question
            query_result = pipeline(question)
            context = [chunk.text for chunk in query_result.source_chunks]
            
            dataset = self._prepare_dataset(
                [question], [expected_answer], [context], [ground_truth]
            )
            
            results = ragas_evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm_provider.get_llm(),
                embeddings=self.embedding_provider.get_embedding_model()
            )
            
            return {metric.name: results[metric.name] for metric in self.metrics}
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed for question: {question[:50]}... Error: {e}")
            raise EvaluationError(f"Failed to evaluate with RAGAS: {e}") from e

    
    def _format_results(self, results) -> Dict[str, Any]:
        """
        Format the RAGAS results into a standardized format.
        """
        # Check if results is an EvaluationResult object (from RAGAS)
        if hasattr(results, 'to_pandas'):
            # RAGAS EvaluationResult object
            df = results.to_pandas()
            per_question_results = df.to_dict("records")
            
            # Calculate overall scores by averaging
            overall_scores = {}
            for metric_name in self.metric_names:
                if metric_name in df.columns:
                    metric_values = df[metric_name].dropna()
                    if len(metric_values) > 0:
                        overall_scores[metric_name] = float(metric_values.mean())
                        
            return {
                "overall_scores": overall_scores,
                "per_question_results": per_question_results,
            }
        else:
            # Fallback for dict-like results
            overall_scores = {k: v for k, v in results.items() if k in self.metric_names}
            return {
                "overall_scores": overall_scores,
                "per_question_results": [overall_scores],
            }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def evaluate(
        self,
        questions: List[str],
        expected_answers: List[str],
        pipeline,
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a list of questions using the pipeline and RAGAS metrics.
        
        Args:
            pipeline: RAG pipeline to evaluate
            questions: List of questions to evaluate
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dictionary containing evaluation results in standardized format
            
        Raises:
            EvaluationError: If evaluation fails
        """
        if ground_truths and len(questions) != len(ground_truths):
            raise ValueError("Number of questions must match number of ground truths")
            
        try:
            logger.info(f"Starting RAGAS evaluation for {len(questions)} questions")
            
            # Generate answers and collect contexts
            answers = []
            contexts = []
            
            for question in questions:
                # Use pipeline to get context for the question
                query_result = await pipeline.ask(question)
                context = [chunk.text for chunk in query_result.source_chunks]
                
                answers.append(query_result.answer)
                contexts.append(context)
            
            # Prepare dataset for RAGAS
            ground_truth_list = cast(List[Optional[str]], ground_truths if ground_truths else [None] * len(questions))
            dataset = self._prepare_dataset(questions, answers, contexts, ground_truth_list)
            
            # Run RAGAS evaluation
            results = ragas_evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm_provider.get_llm(),
                embeddings=self.embedding_provider.get_embedding_model(),
            )
            
            # Format results
            formatted_results = self._format_results(results)
            
            # Add metadata
            formatted_results["metadata"] = {
                "total_questions": len(questions),
                "metrics_used": self.metric_names,
                "evaluator": "RAGAS"
            }
            
            logger.info("RAGAS evaluation completed successfully")
            return formatted_results
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            raise EvaluationError(f"Failed to evaluate with RAGAS: {e}") from e

    async def evaluate_pipeline_batch(
        self,
        pipeline,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate questions in batches for better performance.
        
        Args:
            pipeline: RAG pipeline to evaluate
            questions: List of questions to evaluate
            ground_truths: Optional list of ground truth answers
            batch_size: Number of questions to process in each batch
            
        Returns:
            Dictionary containing evaluation results
        """
        if ground_truths and len(questions) != len(ground_truths):
            raise ValueError("Number of questions must match number of ground truths")
            
        logger.info(f"Starting batch RAGAS evaluation for {len(questions)} questions "
                   f"with batch size {batch_size}")
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_ground_truths = None
            if ground_truths:
                batch_ground_truths = ground_truths[i:i + batch_size]
                
            try:
                batch_results = await self.evaluate(pipeline, batch_questions, batch_ground_truths)
                all_results.append(batch_results)
                logger.info(f"Completed batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1} failed: {e}")
                # Continue with next batch
                continue
        
        if not all_results:
            raise EvaluationError("All batches failed during evaluation")
        
        # Aggregate results from all batches
        return self._aggregate_batch_results(all_results)

    def _aggregate_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple batches.
        
        Args:
            batch_results: List of results from each batch
            
        Returns:
            Aggregated results in standardized format
        """
        # Combine per-question results
        all_per_question = []
        for batch in batch_results:
            all_per_question.extend(batch.get("per_question_results", []))
        
        # Calculate overall scores by averaging across all questions
        overall_scores = {}
        for metric in self.metric_names:
            metric_values = [
                result[metric] for result in all_per_question 
                if metric in result and isinstance(result[metric], (int, float))
            ]
            if metric_values:
                overall_scores[metric] = sum(metric_values) / len(metric_values)
        
        return {
            "overall_scores": overall_scores,
            "per_question_results": all_per_question,
            "metadata": {
                "total_questions": len(all_per_question),
                "metrics_used": self.metric_names,
                "evaluator": "RAGAS",
                "batch_count": len(batch_results)
            }
        }
>>>>>>> 774589e (ragas)

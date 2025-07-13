from abc import ABC, abstractmethod
from typing import Any, List, Dict
import time
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EvaluationError(Exception):
    """Custom exception for evaluation-specific failures."""
    pass

class RAGPipeline(ABC):
    """Abstract base class for a RAG pipeline."""
    @abstractmethod
    def __call__(self, question: str) -> Any:
        """Execute the RAG pipeline for a given question."""
        pass

class BaseEvaluator(ABC):
    """
    Abstract base class for an evaluation framework.

    This class defines the interface for evaluators that assess the performance of RAG pipelines.
    It supports multiple metrics, asynchronous evaluation, and standardized reporting.

    Example of implementing a custom evaluator:

    class MyEvaluator(BaseEvaluator):
        @property
        def supported_metrics(self) -> List[str]:
            return ["accuracy", "latency"]

        def evaluate_single(self, question: str, expected_answer: str, pipeline: RAGPipeline) -> Dict[str, Any]:
            start_time = time.time()
            try:
                actual_answer = pipeline(question)
                latency = time.time() - start_time
                accuracy = 1.0 if actual_answer == expected_answer else 0.0
                return {"accuracy": accuracy, "latency": latency}
            except Exception as e:
                logging.error(f"Error evaluating question '{question}': {e}")
                raise EvaluationError(f"Failed to evaluate question: {question}") from e
    """

    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """Return a list of metrics this evaluator supports."""
        pass

    @abstractmethod
    def evaluate_single(self, question: str, expected_answer: str, pipeline: RAGPipeline) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair against a RAG pipeline.

        Args:
            question: The question to be evaluated.
            expected_answer: The ground truth answer.
            pipeline: The RAG pipeline to evaluate.

        Returns:
            A dictionary containing the evaluation results for the supported metrics.
        """
        pass

    async def _evaluate_single_async(self, question: str, expected_answer: str, pipeline: RAGPipeline) -> Dict[str, Any]:
        """Asynchronous wrapper for evaluate_single."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate_single, question, expected_answer, pipeline)

    async def evaluate(self, questions: List[str], expected_answers: List[str], pipeline: RAGPipeline) -> Dict[str, Any]:
        """
        Evaluate a list of questions against a RAG pipeline.

        Args:
            questions: A list of questions to be evaluated.
            expected_answers: A list of ground truth answers.
            pipeline: The RAG pipeline to evaluate.

        Returns:
            A dictionary containing the aggregated evaluation results.
        """
        if len(questions) != len(expected_answers):
            raise ValueError("The number of questions must match the number of expected answers.")

        start_time = time.time()

        tasks = [self._evaluate_single_async(q, a, pipeline) for q, a in zip(questions, expected_answers)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time

        per_question_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Evaluation failed for question '{questions[i]}': {result}")
                per_question_results.append({"error": str(result)})
            else:
                per_question_results.append(result)

        return self._aggregate_results(per_question_results, total_time)

    def _aggregate_results(self, per_question_results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """
        Aggregate results from per-question evaluations.

        Args:
            per_question_results: A list of evaluation results for each question.
            total_time: The total time taken for the evaluation.

        Returns:
            A dictionary with aggregated results.
        """
        overall_scores = {metric: 0.0 for metric in self.supported_metrics}
        valid_results_count = {metric: 0 for metric in self.supported_metrics}

        for result in per_question_results:
            if "error" not in result:
                for metric in self.supported_metrics:
                    if metric in result:
                        overall_scores[metric] += result[metric]
                        valid_results_count[metric] += 1

        for metric in self.supported_metrics:
            if valid_results_count[metric] > 0:
                overall_scores[metric] /= valid_results_count[metric]

        return {
            "overall_scores": overall_scores,
            "per_question_results": per_question_results,
            "metadata": {
                "total_questions": len(per_question_results),
                "total_time": total_time,
                "supported_metrics": self.supported_metrics,
            }
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report from the results dictionary.

        Args:
            results: The evaluation results dictionary.

        Returns:
            A string containing the human-readable report.
        """
        report = []
        report.append("="*20 + " Evaluation Report " + "="*20)

        # Metadata
        metadata = results.get("metadata", {})
        report.append("\n[Metadata]")
        report.append(f"  Total Questions: {metadata.get('total_questions', 'N/A')}")
        report.append(f"  Total Time: {metadata.get('total_time', 'N/A'):.4f} seconds")
        report.append(f"  Supported Metrics: {', '.join(metadata.get('supported_metrics', []))}")

        # Overall Scores
        overall_scores = results.get("overall_scores", {})
        report.append("\n[Overall Scores]")
        for metric, score in overall_scores.items():
            report.append(f"  - {metric.capitalize()}: {score:.4f}")

        # Per-Question Breakdown
        per_question_results = results.get("per_question_results", [])
        report.append("\n[Per-Question Results]")
        for i, result in enumerate(per_question_results):
            report.append(f"  Question {i+1}:")
            if "error" in result:
                report.append(f"    - Error: {result['error']}")
            else:
                for metric, value in result.items():
                    report.append(f"    - {metric.capitalize()}: {value}")

        report.append("\n" + "="*59)
        return "\n".join(report)

import unittest
import asyncio
from unittest.mock import MagicMock, patch
from abc import ABC

from yasrl.evaluation.base import BaseEvaluator, EvaluationError, RAGPipeline

class MockRAGPipeline(RAGPipeline):
    def __init__(self, answer_map: dict):
        self._answer_map = answer_map

    def __call__(self, question: str):
        if question in self._answer_map:
            return self._answer_map[question]
        raise ValueError("Question not in answer map")

class MockEvaluator(BaseEvaluator):
    @property
    def supported_metrics(self) -> list[str]:
        return ["metric1", "metric2"]

    def evaluate_single(self, question: str, expected_answer: str, pipeline: RAGPipeline) -> dict:
        if "error" in question:
            raise EvaluationError("This is a drill!")

        answer = pipeline(question)
        return {
            "metric1": 1 if answer == expected_answer else 0,
            "metric2": len(answer),
        }

class TestBaseEvaluator(unittest.TestCase):
    def test_abstract_class_instantiation(self):
        with self.assertRaises(TypeError):
            BaseEvaluator()

    def test_mock_evaluator_instantiation(self):
        try:
            MockEvaluator()
        except TypeError:
            self.fail("MockEvaluator should be instantiable")

    def test_supported_metrics(self):
        evaluator = MockEvaluator()
        self.assertEqual(evaluator.supported_metrics, ["metric1", "metric2"])

    def test_evaluate_single(self):
        pipeline = MockRAGPipeline({"q1": "a1"})
        evaluator = MockEvaluator()
        result = evaluator.evaluate_single("q1", "a1", pipeline)
        self.assertEqual(result, {"metric1": 1, "metric2": 2})

    def test_evaluate_single_with_error(self):
        pipeline = MockRAGPipeline({})
        evaluator = MockEvaluator()
        with self.assertRaises(EvaluationError):
            evaluator.evaluate_single("error_question", "a1", pipeline)

    def test_generate_report(self):
        evaluator = MockEvaluator()
        results = {
            "overall_scores": {"metric1": 0.5, "metric2": 2.5},
            "per_question_results": [
                {"metric1": 1, "metric2": 2},
                {"metric1": 0, "metric2": 3},
                {"error": "Something went wrong"},
            ],
            "metadata": {
                "total_questions": 3,
                "total_time": 1.23,
                "supported_metrics": ["metric1", "metric2"],
            },
        }
        report = evaluator.generate_report(results)
        self.assertIn("Evaluation Report", report)
        self.assertIn("Total Questions: 3", report)
        self.assertIn("Metric1: 0.5000", report)

    def test_aggregate_results(self):
        evaluator = MockEvaluator()
        per_question_results = [
            {"metric1": 1, "metric2": 10},
            {"metric1": 0, "metric2": 20},
            {"error": "failed"},
        ]
        aggregated = evaluator._aggregate_results(per_question_results, 1.0)
        self.assertAlmostEqual(aggregated["overall_scores"]["metric1"], 0.5)
        self.assertAlmostEqual(aggregated["overall_scores"]["metric2"], 15.0)
        self.assertEqual(len(aggregated["per_question_results"]), 3)

    def test_async_evaluation(self):
        pipeline = MockRAGPipeline({"q1": "a1", "q2": "a2"})
        evaluator = MockEvaluator()

        questions = ["q1", "q2"]
        expected_answers = ["a1", "a3"] # one wrong answer

        results = asyncio.run(evaluator.evaluate(questions, expected_answers, pipeline))

        self.assertAlmostEqual(results["overall_scores"]["metric1"], 0.5)
        self.assertAlmostEqual(results["overall_scores"]["metric2"], 2.0)
        self.assertEqual(len(results["per_question_results"]), 2)

    def test_evaluation_with_partial_failures(self):
        pipeline = MockRAGPipeline({"q1": "a1"})
        evaluator = MockEvaluator()

        questions = ["q1", "error_question"]
        expected_answers = ["a1", "a2"]

        results = asyncio.run(evaluator.evaluate(questions, expected_answers, pipeline))

        self.assertIn("error", results["per_question_results"][1])
        self.assertAlmostEqual(results["overall_scores"]["metric1"], 1.0)
        self.assertAlmostEqual(results["overall_scores"]["metric2"], 2.0)

if __name__ == "__main__":
    unittest.main()

"""Analyse DeepEval metric failures and map them to RAG scenarios.

This module inspects a DeepEval results JSON file and highlights which
metrics fell below acceptable thresholds. Each metric is mapped to a
high-level RAG scenario so we can quickly diagnose whether we are dealing
with a retrieval or generation issue.

Usage (from repo root):
	python -m src.evals.analysing_failures path/to/results.json

If no path is provided, the script attempts to inspect the
``deepeval_evaluation_results_happy_payments.json`` artifact in
``src/evals/results``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from typing import Any, Iterable, Mapping


DEFAULT_RESULTS_PATH = Path(__file__).with_name("results") / (
	"deepeval_evaluation_results_happy_payments.json"
)

# Metric-specific thresholds. Scores are between 0 and 1.
DEFAULT_THRESHOLDS: Mapping[str, float] = {
	"Answer Relevancy": 0.75,
	"Faithfulness": 0.80,
	"Contextual Precision": 0.70,
	"Contextual Recall": 0.70,
}


@dataclass(frozen=True)
class ScenarioDefinition:
	"""Represents a failure scenario in the RAG pipeline."""

	code: str
	name: str
	stage: str
	description: str


SCENARIOS: Mapping[str, ScenarioDefinition] = {
	"A": ScenarioDefinition(
		code="A",
		name="Missing Supporting Context",
		stage="retrieval",
		description="Retriever could not surface the evidence required to answer the question.",
	),
	"B": ScenarioDefinition(
		code="B",
		name="Irrelevant Context Returned",
		stage="retrieval",
		description="Retriever pulled context that does not pertain to the query.",
	),
	"C": ScenarioDefinition(
		code="C",
		name="Hallucinated or Unfaithful Answer",
		stage="generation",
		description="LLM response contradicts the retrieved evidence.",
	),
	"D": ScenarioDefinition(
		code="D",
		name="Answer Not Relevant to Query",
		stage="generation",
		description="LLM response fails to address the question despite sufficient evidence.",
	),
}


METRIC_TO_SCENARIO: Mapping[str, str] = {
	"Contextual Recall": "A",
	"Contextual Precision": "B",
	"Faithfulness": "C",
	"Answer Relevancy": "D",
}


@dataclass
class MetricFailure:
	metric: str
	score: float
	threshold: float
	reason: str
	scenario: ScenarioDefinition


@dataclass
class TestCaseSummary:
	query: str
	status: str
	failures: list[MetricFailure]

	@property
	def scenario_codes(self) -> list[str]:
		return sorted({failure.scenario.code for failure in self.failures})


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Analyse DeepEval results and highlight failing scenarios."
	)
	parser.add_argument(
		"results_path",
		nargs="?",
		default=str(DEFAULT_RESULTS_PATH),
		help="Path to a DeepEval results JSON file.",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=None,
		help="Override minimum acceptable score for all metrics.",
	)
	return parser.parse_args()


def load_results(path: Path) -> Mapping[str, Any]:
	if not path.exists():
		raise SystemExit(f"Results file not found: {path}")

	with path.open("r", encoding="utf-8") as fp:
		return json.load(fp)


def determine_threshold(metric_name: str, override: float | None) -> float:
	if override is not None:
		return override
	return DEFAULT_THRESHOLDS.get(metric_name, 0.75)


def evaluate_metrics(
	metrics: Iterable[Mapping[str, Any]], threshold_override: float | None = None
) -> list[MetricFailure]:
	failures: list[MetricFailure] = []

	for metric in metrics:
		name = metric.get("name")
		score = float(metric.get("score", 0.0))
		reason = metric.get("reason", "")
		if name is None:
			continue

		threshold = determine_threshold(name, threshold_override)
		if score >= threshold:
			continue

		scenario_code = METRIC_TO_SCENARIO.get(name)
		if scenario_code is None:
			# Unknown metric; flag it generically so we do not miss issues.
			scenario = ScenarioDefinition(
				code="?",
				name=f"Unclassified metric '{name}'",
				stage="unknown",
				description="No scenario mapping defined for this metric.",
			)
		else:
			scenario = SCENARIOS[scenario_code]

		failures.append(
			MetricFailure(
				metric=name,
				score=score,
				threshold=threshold,
				reason=reason,
				scenario=scenario,
			)
		)

	return failures


def summarise_test_case(
	test_case: Mapping[str, Any], threshold_override: float | None = None
) -> TestCaseSummary:
	query = test_case.get("input", "<unknown input>")
	metrics = test_case.get("metrics", [])
	failures = evaluate_metrics(metrics, threshold_override)
	status = "FAIL" if failures else "PASS"

	return TestCaseSummary(query=query, status=status, failures=failures)


def analyse(results_blob: Mapping[str, Any], threshold_override: float | None) -> list[TestCaseSummary]:
	test_cases = results_blob.get("results", [])
	return [summarise_test_case(case, threshold_override) for case in test_cases]


def render_summary(test_summaries: list[TestCaseSummary]) -> None:
	if not test_summaries:
		print("No test cases found in results.")
		return

	scenario_totals: dict[str, int] = {}

	for idx, summary in enumerate(test_summaries, start=1):
		heading = f"{idx}. {summary.status} — {summary.query}"
		print(heading)

		if not summary.failures:
			print("   ✓ All metrics meet the configured thresholds.\n")
			continue

		for failure in summary.failures:
			scenario = failure.scenario
			scenario_totals[scenario.code] = scenario_totals.get(scenario.code, 0) + 1

			print(
				"   - Scenario {code} ({name}) [{stage}]".format(
					code=scenario.code,
					name=scenario.name,
					stage=scenario.stage,
				)
			)
			print(
				"     Metric: {metric} → {score:.2f} (threshold {threshold:.2f})".format(
					metric=failure.metric,
					score=failure.score,
					threshold=failure.threshold,
				)
			)
			if failure.reason:
				print(f"     Reason: {failure.reason.strip()}")
			if scenario.description:
				print(f"     Diagnosis: {scenario.description}")
			print()

	print("Scenario breakdown:")
	if scenario_totals:
		for code in sorted(scenario_totals):
			scenario = SCENARIOS.get(code)
			label = f"Scenario {code}"
			if scenario is not None:
				label = f"Scenario {code} — {scenario.name}"
			print(f"  · {label}: {scenario_totals[code]} occurrence(s)")
	else:
		print("  · No failures detected.")


def main() -> None:
	args = parse_args()
	results_path = Path(args.results_path)
	results_blob = load_results(results_path)
	summaries = analyse(results_blob, args.threshold)
	render_summary(summaries)


if __name__ == "__main__":
	main()

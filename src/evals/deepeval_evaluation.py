"""
DeepEval RAG Evaluation Demo

This script demonstrates how to evaluate a RAG pipeline using the DeepEval framework
with a Gemini model for evaluation.
"""

import asyncio
import os
import json
import logging
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from yasrl.pipeline import RAGPipeline
from UI.project_manager import get_project_manager

# Import our custom Gemini wrapper instead of DeepEval's built-in models
from evals.deepeval_gemini import create_gemini_synthesizer, create_gemini_model_for_deepeval

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)

async def synthesize_evaluation_dataset(
    context_chunks: list[str],
    num_pairs: int = 5,
    llm_model: str = "gemini-2.5-flash",
) -> list[dict]:
    """
    Generate QA pairs using our custom Gemini wrapper.
    This replaces the DeepEval synthesizer since DeepEval doesn't natively support Gemini.
    """
    logger.info(f"Generating {num_pairs} QA pairs using Gemini synthesizer (model={llm_model})")

    if not context_chunks:
        logger.warning("No context chunks provided for dataset synthesis.")
        return []

    try:
        # Use our custom Gemini synthesizer
        synthesizer = create_gemini_synthesizer(model_name=llm_model)
        
        # Generate QA pairs
        qa_pairs = synthesizer.generate_qa_pairs(
            contexts=context_chunks,
            num_pairs_per_context=max(1, num_pairs // len(context_chunks)),
            question_types=["factual", "explanatory", "analytical", "definitional"]
        )
        
        logger.info(f"Gemini synthesizer generated {len(qa_pairs)} QA pairs.")
        
        # Limit to requested number
        return qa_pairs[:num_pairs]
        
    except Exception as e:
        logger.exception(f"Failed to generate QA pairs: {e}")
        return []

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print("DeepEval evaluation module loaded successfully")

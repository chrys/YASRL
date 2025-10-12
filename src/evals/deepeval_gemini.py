"""
Gemini wrapper for DeepEval compatibility.

This module provides a wrapper around Google's Gemini API that implements
the interface expected by DeepEval metrics, allowing Gemini to be used
for evaluation tasks.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
import google.generativeai as genai
from deepeval.models.base_model import DeepEvalBaseLLM

logger = logging.getLogger(__name__)


class DeepEvalGeminiModel(DeepEvalBaseLLM):
    """
    Gemini model wrapper that implements the DeepEval base model interface.
    
    This allows Gemini to be used with DeepEval metrics for RAG evaluation.
    """
    
    def __init__(
        self, 
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        """
        Initialize the Gemini model wrapper.
        
        Args:
            model_name: The Gemini model to use (e.g., "gemini-2.5-flash")
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure the API key
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be provided either as parameter or environment variable")
        
        genai.configure(api_key=api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model {model_name}: {e}")
            raise
    
    def load_model(self, *args, **kwargs) -> "DeepEvalBaseLLM":
        """
        Load the model. This is called by DeepEval but our model is already loaded in __init__.
        Returns self to satisfy the interface.
        """
        return self  # Model is already loaded in __init__
    
    def generate(self, prompt: str, *args, **kwargs) -> str:
        """
        Generate text using the Gemini model.
        
        Args:
            prompt: The input prompt
            *args: Additional positional arguments
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            # Configure generation parameters
            generation_config = genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini model")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating with Gemini: {e}")
            # Return a fallback response to prevent evaluation from failing
            return "Error: Unable to generate response"
    
    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        """
        Async version of generate. For now, we'll use the sync version.
        Gemini's Python SDK doesn't have full async support yet.
        """
        return self.generate(prompt, *args, **kwargs)
    
    def get_model_name(self, *args, **kwargs) -> str:
        """
        Return the model name for DeepEval reporting.
        """
        return self.model_name
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error in batch generation for prompt: {e}")
                responses.append("Error: Unable to generate response")
        
        return responses
    
    def __str__(self) -> str:
        return f"DeepEvalGeminiModel(model={self.model_name}, temp={self.temperature})"
    
    def __repr__(self) -> str:
        return self.__str__()


class GeminiSynthesizer:
    """
    Gemini-based synthesizer for generating QA pairs from context.
    
    This provides an alternative to DeepEval's built-in synthesizer,
    using Gemini for question and answer generation.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize the Gemini synthesizer.
        
        Args:
            model_name: The Gemini model to use
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        """
        self.gemini_model = DeepEvalGeminiModel(model_name=model_name, api_key=api_key)
        logger.info("Initialized GeminiSynthesizer")
    
    def generate_qa_pairs(
        self, 
        contexts: List[str], 
        num_pairs_per_context: int = 1,
        question_types: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from context chunks.
        
        Args:
            contexts: List of context strings
            num_pairs_per_context: Number of QA pairs to generate per context
            question_types: Types of questions to generate (optional)
            
        Returns:
            List of QA pairs with 'question', 'answer', and 'context' keys
        """
        if not contexts:
            logger.warning("No contexts provided for QA generation")
            return []
        
        qa_pairs = []
        
        for i, context in enumerate(contexts, 1):
            if not context.strip():
                continue
                
            logger.info(f"Generating QA pairs for context {i}/{len(contexts)}")
            
            for j in range(num_pairs_per_context):
                try:
                    # Generate question
                    question_prompt = self._build_question_prompt(context, question_types)
                    question = self.gemini_model.generate(question_prompt)
                    
                    if not question.strip():
                        logger.warning(f"Empty question generated for context {i}")
                        continue
                    
                    # Generate answer
                    answer_prompt = self._build_answer_prompt(context, question)
                    answer = self.gemini_model.generate(answer_prompt)
                    
                    if not answer.strip():
                        logger.warning(f"Empty answer generated for context {i}")
                        continue
                    
                    qa_pairs.append({
                        "question": question.strip(),
                        "answer": answer.strip(),
                        "context": context.strip()
                    })
                    
                except Exception as e:
                    logger.error(f"Error generating QA pair for context {i}: {e}")
                    continue
        
        logger.info(f"Generated {len(qa_pairs)} QA pairs from {len(contexts)} contexts")
        return qa_pairs
    
    def _build_question_prompt(self, context: str, question_types: Optional[List[str]] = None) -> str:
        """Build prompt for question generation."""
        base_prompt = """
Based on the following context, generate a clear, specific question that can be answered using the information provided.

The question should:
- Be directly answerable from the context
- Be clear and well-formed
- Focus on key information in the context
- Not require external knowledge

Context:
{context}

Generate only the question, without any additional text or explanation.

Question:"""
        
        if question_types:
            types_text = ", ".join(question_types)
            base_prompt += f"\n\nPrefer these question types: {types_text}"
        
        return base_prompt.format(context=context)
    
    def _build_answer_prompt(self, context: str, question: str) -> str:
        """Build prompt for answer generation."""
        return """
Based on the following context, provide a comprehensive and accurate answer to the question.

The answer should:
- Be directly based on the provided context
- Be complete and informative
- Not include information not present in the context
- Be concise but thorough

Context:
{context}

Question:
{question}

Provide only the answer, without any additional text or explanation.

Answer:""".format(context=context, question=question)
    
    async def a_generate_qa_pairs(
        self, 
        contexts: List[str], 
        num_pairs_per_context: int = 1,
        question_types: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Async version of generate_qa_pairs.
        For now, delegates to the sync version.
        """
        return self.generate_qa_pairs(contexts, num_pairs_per_context, question_types)


def create_gemini_model_for_deepeval(
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0
) -> DeepEvalGeminiModel:
    """
    Convenience function to create a Gemini model for DeepEval usage.
    
    Args:
        model_name: The Gemini model to use
        api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        temperature: Sampling temperature
        
    Returns:
        Configured DeepEvalGeminiModel instance
    """
    return DeepEvalGeminiModel(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature
    )


def create_gemini_synthesizer(
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None
) -> GeminiSynthesizer:
    """
    Convenience function to create a Gemini synthesizer.
    
    Args:
        model_name: The Gemini model to use
        api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        
    Returns:
        Configured GeminiSynthesizer instance
    """
    return GeminiSynthesizer(model_name=model_name, api_key=api_key)


if __name__ == "__main__":
    # Test the wrapper
    import asyncio
    
    async def test_gemini_wrapper():
        """Test the Gemini wrapper functionality."""
        print("Testing Gemini wrapper for DeepEval...")
        
        try:
            # Test model creation
            model = create_gemini_model_for_deepeval()
            print(f"✅ Created model: {model}")
            
            # Test basic generation
            test_prompt = "What is the capital of France?"
            response = model.generate(test_prompt)
            print(f"✅ Generated response: {response}")
            
            # Test synthesizer
            synthesizer = create_gemini_synthesizer()
            test_contexts = [
                "Paris is the capital and largest city of France. It is located in the north-central part of the country."
            ]
            
            qa_pairs = synthesizer.generate_qa_pairs(test_contexts, num_pairs_per_context=1)
            print(f"✅ Generated {len(qa_pairs)} QA pairs")
            
            for pair in qa_pairs:
                print(f"Q: {pair['question']}")
                print(f"A: {pair['answer']}")
                print("---")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    asyncio.run(test_gemini_wrapper())

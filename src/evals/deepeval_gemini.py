import asyncio
from typing import Optional, List, Dict, Any, Union
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.utils import trim_text
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

class Gemini(DeepEvalBaseLLM):
    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        self.model = model
        # Configure the generative AI client
        if api_key:
            genai.configure(api_key=api_key)

    def load_model(self):
        return genai.GenerativeModel(self.model)

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        # Gemini does not support `max_tokens` in the same way as OpenAI,
        # but we can simulate it by truncating the output if needed.
        # For now, we will rely on the model's default behavior.
        response = model.generate_content(prompt)
        return trim_text(response.text)

    async def a_generate(self, prompt: str) -> str:
        model = self.load_model()
        try:
            response = await asyncio.to_thread(model.generate_content, prompt)
            return trim_text(response.text)
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {e}")
            # Depending on the desired behavior, you might want to return an empty string,
            # raise the exception, or handle it in another way.
            return ""

    def get_model_name(self):
        return self.model
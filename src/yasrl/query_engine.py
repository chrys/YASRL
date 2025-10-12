import logging
from typing import List, Optional

from llama_index.core.llms import ChatMessage

from .models import QueryResult
from .providers.llm import LLMProvider
from .query_processor import QueryProcessor
from .reranker import ReRanker

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Handles the query process, including retrieving relevant documents,
    reranking, and generating a final answer using an LLM.
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        reranker: ReRanker,
        llm_provider: LLMProvider,
    ):
        """
        Initializes the QueryEngine.

        Args:
            query_processor: The query processor for retrieving documents.
            reranker: The reranker for re-scoring retrieved documents.
            llm_provider: The LLM provider for generating the final answer.
        """
        self.query_processor = query_processor
        self.reranker = reranker
        self.llm_provider = llm_provider

    async def query(
        self, query: str, conversation_history: Optional[List[dict]] = None
    ) -> QueryResult:
        """
        Asks a question and returns the answer with source information.

        Args:
            query: The question to ask.
            conversation_history: Optional conversation history for context.

        Returns:
            A QueryResult object containing the answer and source chunks.
        """
        logger.info(f"Processing query: {query}")

        try:
            # 1. Get relevant chunks from the query processor
            retrieved_chunks = await self.query_processor.process_query(query, top_k=3)

            # 2. Re-rank the retrieved chunks
            reranked_chunks = self.reranker.rerank(query, retrieved_chunks)
            if not reranked_chunks:
                logger.warning("No relevant chunks found after re-ranking.")
                return QueryResult(
                    answer="I'm sorry, I couldn't find any relevant information to answer your question.",
                    source_chunks=[],
                )

            # 3. Prepare context from source chunks
            context = "\n\n".join([chunk.text for chunk in reranked_chunks])

            # Build the prompt
            system_prompt = (
                "You are a helpful assistant. Use the provided context to answer questions accurately. "
                "If the context doesn't contain enough information to answer the question, say so clearly."
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

            # Get response from LLM
            llm = self.llm_provider.get_llm()

            # Handle different LLM provider interfaces
            try:
                if hasattr(llm, "achat"):
                    messages = [
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt),
                    ]
                    response = await llm.achat(messages)
                    answer = response.message.content
                elif hasattr(llm, "agenerate"):
                    response = await llm.agenerate([user_prompt])
                    answer = response.generations[0][0].text
                elif hasattr(llm, "generate"):
                    response = llm.generate([user_prompt])
                    answer = response.generations[0][0].text
                elif hasattr(llm, "complete"):
                    response = (
                        await llm.acomplete(user_prompt)
                        if hasattr(llm, "acomplete")
                        else llm.complete(user_prompt)
                    )
                    answer = str(response)
                else:
                    logger.warning(
                        f"Unknown LLM interface for {type(llm)}. Using fallback."
                    )
                    answer = f"Found {len(reranked_chunks)} relevant sources but cannot generate a response."

            except Exception as llm_error:
                logger.error(f"LLM call failed: {llm_error}")
                answer = f"Found relevant information but encountered an error generating the response: {llm_error}"

            # Create and return the result
            result = QueryResult(answer=answer.strip(), source_chunks=reranked_chunks)

            logger.info(
                f"Query processed successfully. Found {len(reranked_chunks)} source chunks."
            )
            return result

        except Exception as e:
            logger.error(f"Failed to process query '{query}': {e}")
            return QueryResult(
                answer=f"Sorry, I encountered an error while processing your question: {e}",
                source_chunks=[],
            )
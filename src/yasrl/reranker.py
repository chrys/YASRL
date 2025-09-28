import logging
from typing import List
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from .models import SourceChunk

logger = logging.getLogger(__name__)

class ReRanker:
    """
    A wrapper around a cross-encoder model to re-rank and filter source chunks.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 2):
        """
        Initializes the ReRanker.

        Args:
            model_name: The name of the cross-encoder model to use.
            top_n: The number of top chunks to keep after re-ranking.
        """
        logger.info(f"Initializing ReRanker with model: {model_name} and top_n: {top_n}")
        self._reranker = SentenceTransformerRerank(top_n=top_n, model=model_name)

    def rerank(self, query: str, chunks: List[SourceChunk]) -> List[SourceChunk]:
        """
        Re-ranks a list of SourceChunk objects based on a query.

        Args:
            query: The user's query string.
            chunks: The list of source chunks retrieved from the vector store.

        Returns:
            A new, smaller list of re-ranked and filtered source chunks.
        """
        if not chunks:
            return []

        # Convert our SourceChunk objects into the format LlamaIndex's reranker expects
        nodes_to_rerank = [
            NodeWithScore(
                node=TextNode(text=chunk.text, metadata=chunk.metadata), 
                score=chunk.score
            ) for chunk in chunks
        ]

        # Perform the re-ranking
        reranked_nodes = self._reranker.postprocess_nodes(
            nodes_to_rerank, query_bundle=QueryBundle(query_str=query)
        )
        
        # Convert the re-ranked nodes back to our SourceChunk format
        reranked_chunks = []
        for node_with_score in reranked_nodes:
            node = node_with_score.node
            # Ensure the node is a TextNode before accessing its attributes
            if isinstance(node, TextNode):
                reranked_chunks.append(
                    SourceChunk(
                        text=node.text,
                        metadata=node.metadata or {},
                        score=node_with_score.score or 0.0
                    )
                )
        
        logger.info(f"Re-ranked {len(chunks)} chunks down to {len(reranked_chunks)}.")
        return reranked_chunks
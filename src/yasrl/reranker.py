import logging
from typing import List, Optional
from .models import SourceChunk

# Try to import reranker dependencies - make them optional
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sentence-transformers not available - reranking will be skipped")

logger = logging.getLogger(__name__)

class ReRanker:
    """
    A wrapper around a cross-encoder model to re-rank and filter source chunks.
    Falls back to returning top_n chunks without reranking if dependencies unavailable.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 2):
        """
        Initializes the ReRanker.

        Args:
            model_name: The name of the cross-encoder model to use.
            top_n: The number of top chunks to keep after re-ranking.
        """
        self.top_n = top_n
        self._reranker: Optional[SentenceTransformerRerank] = None
        
        if RERANKER_AVAILABLE:
            logger.info(f"Initializing ReRanker with model: {model_name} and top_n: {top_n}")
            self._reranker = SentenceTransformerRerank(top_n=top_n, model=model_name)
        else:
            logger.warning(f"ReRanker initialized in fallback mode (no reranking) - will return top {top_n} chunks by score")

    def rerank(self, query: str, chunks: List[SourceChunk]) -> List[SourceChunk]:
        """
        Re-ranks a list of SourceChunk objects based on a query.
        Falls back to simple score-based filtering if reranker unavailable.

        Args:
            query: The user's query string.
            chunks: The list of source chunks retrieved from the vector store.

        Returns:
            A new, smaller list of re-ranked and filtered source chunks.
        """
        if not chunks:
            return []

        # Fallback mode: just return top_n chunks by score
        if not RERANKER_AVAILABLE or self._reranker is None:
            sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            result = sorted_chunks[:self.top_n]
            logger.info(f"Fallback mode: returning top {len(result)} of {len(chunks)} chunks by score")
            return result

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
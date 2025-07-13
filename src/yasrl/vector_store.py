from typing import List
from yasrl.models import SourceChunk

class VectorStoreManager:
    """
    A mock vector store manager that simulates retrieving source chunks.
    """

    def retrieve(self, query_embedding: List[float], top_k: int) -> List[SourceChunk]:
        """
        Retrieves a list of source chunks based on a query embedding.

        Args:
            query_embedding: The embedding of the query.
            top_k: The number of chunks to retrieve.

        Returns:
            A list of source chunks.
        """
        # In a real implementation, this would perform a similarity search
        # in a vector store. For this mock, we return a fixed list of chunks.
        mock_chunks = [
            SourceChunk(text="This is the first mock chunk.", metadata={"source": "doc1.txt"}, score=0.9),
            SourceChunk(text="This is the second mock chunk.", metadata={"source": "doc2.txt"}, score=0.8),
            SourceChunk(text="This is the third mock chunk.", metadata={"source": "doc3.txt"}, score=0.7),
            SourceChunk(text="This is the fourth mock chunk.", metadata={"source": "doc4.txt"}, score=0.6),
            SourceChunk(text="This is the fifth mock chunk.", metadata={"source": "doc5.txt"}, score=0.5),
            SourceChunk(text="This is the sixth mock chunk.", metadata={"source": "doc6.txt"}, score=0.4),
            SourceChunk(text="This is the seventh mock chunk.", metadata={"source": "doc7.txt"}, score=0.3),
            SourceChunk(text="This is the eighth mock chunk.", metadata={"source": "doc8.txt"}, score=0.2),
            SourceChunk(text="This is the ninth mock chunk.", metadata={"source": "doc9.txt"}, score=0.1),
            SourceChunk(text="This is the tenth mock chunk.", metadata={"source": "doc10.txt"}, score=0.05),
        ]
        return mock_chunks[:top_k]

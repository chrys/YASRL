import logging
from typing import List, Optional
from urllib.parse import urlparse
import uuid
import psycopg2
import numpy as np
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import VectorStoreQuery 
from psycopg2.extensions import connection
from psycopg2.pool import SimpleConnectionPool
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.schema import TextNode
from typing import List, Dict, Any

from yasrl.exceptions import IndexingError, RetrievalError

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages a PostgreSQL vector store with upsert capabilities.
    """

    def __init__(
        self,
        postgres_uri: str,
        vector_dimensions: int,
        table_prefix: str = "yasrl",
    ):
        """
        Initializes the VectorStoreManager.

        Args:
            postgres_uri: The connection URI for the PostgreSQL database.
            vector_dimensions: The dimensionality of the embedding vectors.
            table_prefix: The prefix for the database table names.
        """
        self.postgres_uri = postgres_uri
        self.vector_dimensions = vector_dimensions
        self.table_name = f"{table_prefix}_data"  
        self._vector_store: Optional[PGVectorStore] = None
        self._pool: Optional[SimpleConnectionPool] = None

    async def ainit(self):
        """Asynchronously initializes the connection pool."""
        logger.info("Initializing connection pool...")
        try:
            self._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.postgres_uri,
            )
            logger.info("Connection pool initialized.")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise IndexingError(f"Failed to initialize connection pool: {e}") from e

    async def close(self):
        """Closes the connection pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Connection pool closed.")

    @property
    def vector_store(self) -> PGVectorStore:
        """
        Returns the LlamaIndex PGVectorStore instance.
        """
        if self._vector_store is None:
            try:
                self._vector_store = PGVectorStore.from_params(
                    host=self._parsed_uri.hostname,
                    port=self._parsed_uri.port or 5432,
                    database=self._parsed_uri.path.lstrip("/"),
                    user=self._parsed_uri.username,
                    password=self._parsed_uri.password,
                    table_name= self.table_name,
                    embed_dim=self.vector_dimensions,
                    hybrid_search=False,
                    text_search_config="english",
                )
            except Exception as e:
                logger.error(f"Failed to initialize PGVectorStore: {e}")
                raise IndexingError(f"Failed to initialize PGVectorStore: {e}") from e
        return self._vector_store

    @property
    def _parsed_uri(self):
        """
        Parses the PostgreSQL connection URI.
        """
        return urlparse(self.postgres_uri)

    def _get_connection(self) -> connection:
        """
        Gets a connection from the pool.
        """
        if not self._pool:
            # This is a synchronous method, so we can't await ainit.
            # For testing purposes, if the pool is not initialized, we raise an error.
            # In a real app, ainit() should be called at startup.
            raise IndexingError("Connection pool is not initialized.")
        return self._pool.getconn()

    def _release_connection(self, conn: connection):
        """
        Releases a connection back to the pool.
        """
        if self._pool:
            self._pool.putconn(conn)
    

    def upsert_documents(self, document_id: str, chunks: list):
        """
        Deletes existing chunks for a document and inserts new ones using LlamaIndex.
        """
        try:
            # Delete existing chunks using LlamaIndex's delete capability
            self.vector_store.delete(
                ref_doc_id=document_id  
            )

            # Insert new chunks using LlamaIndex
            for chunk in chunks:
                if not hasattr(chunk, 'id_') or not chunk.id_:
                    chunk.id_ = str(uuid.uuid4())
                if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata['document_id'] = document_id
                
            self.vector_store.add(chunks)
            logger.info(f"Upsert for document_id: {document_id} completed successfully.")
        except Exception as e:
            logger.error(f"Failed to upsert document: {e}")
            raise IndexingError(f"Failed to upsert document: {e}") from e
 
 
    def retrieve_chunks(self, query_embedding: List[float], top_k: int = 10) -> list:
        """
        Retrieves the most similar chunks from the vector store.
        """
        try:
            query_vector = np.array(query_embedding, dtype=np.float32)
            vector_store_query = VectorStoreQuery(
                query_embedding=query_vector.tolist(),
                similarity_top_k=top_k,
                mode=VectorStoreQueryMode.DEFAULT
            )
            result = self.vector_store.query(vector_store_query)
            return getattr(result, "nodes", [])
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            raise RetrievalError(f"Failed to retrieve chunks: {e}") from e

    async def check_connection(self) -> bool:
        """
        Checks if a connection to the database can be established.
        Returns True if successful, False otherwise.
        """
        try:
            conn = self._get_connection()
            self._release_connection(conn)
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
        
    async def get_document_count(self, project_id: str | None = None) -> int:
        """
        Returns the number of indexed documents, optionally filtered by project_id.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                if project_id is not None:
                    cursor.execute(
                        f"SELECT COUNT(DISTINCT document_id) FROM {self.table_name} WHERE metadata->>'project_id' = %s",
                        (project_id,),
                    )
                else:
                    cursor.execute(
                        f"SELECT COUNT(DISTINCT document_id) FROM {self.table_name}"
                    )
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
        finally:
            self._release_connection(conn)

    async def get_all_chunks(self, project_id: str) -> List[TextNode]:
        """
        Retrieves all chunks for a given project_id.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Assuming the table has 'id', 'text', 'metadata', and 'embedding' columns
                cursor.execute(
                    f"SELECT node_id, text, metadata, embedding FROM {self.table_name} WHERE metadata ->> 'project_id' = %s",
                    (project_id,),
                )
                results = cursor.fetchall()
                nodes = []
                for row in results:
                    node_id, text, metadata, embedding = row
                    # The metadata is stored as a JSONB, so it should be a dict
                    nodes.append(TextNode(id_=node_id, text=text, metadata=metadata, embedding=embedding))
                return nodes
        except Exception as e:
            logger.error(f"Failed to get all chunks for project {project_id}: {e}")
            return []
        finally:
            self._release_connection(conn)

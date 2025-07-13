import logging
from typing import List, Optional
from urllib.parse import urlparse

import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore
from psycopg2.extensions import connection

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
        self.table_name = f"{table_prefix}_chunks"
        self._vector_store: Optional[PGVectorStore] = None
        self._connection: Optional[connection] = None

    @property
    def vector_store(self) -> PGVectorStore:
        """
        Returns the LlamaIndex PGVectorStore instance.
        """
        if self._vector_store is None:
            try:
                self._vector_store = PGVectorStore.from_params(
                    host=self._parsed_uri.hostname,
                    port=5432,
                    database=self._parsed_uri.path.lstrip("/"),
                    user=self._parsed_uri.username,
                    password=self._parsed_uri.password,
                    table_name=self.table_name,
                    embed_dim=self.vector_dimensions,
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
        Establishes a connection to the PostgreSQL database.
        """
        try:
            self._connection = psycopg2.connect(self.postgres_uri)
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise IndexingError(f"Failed to connect to PostgreSQL: {e}") from e
        return self._connection

    def setup_schema(self):
        """
        Creates the required tables and indexes in the database.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                logger.info(f"Creating table: {self.table_name}")
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id UUID PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        chunk_text TEXT NOT NULL,
                        embedding VECTOR({self.vector_dimensions}) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                logger.info(f"Creating index on document_id for table: {self.table_name}")
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_document_id ON {self.table_name} (document_id);
                """)
                logger.info(f"Creating vector index on embedding for table: {self.table_name}")
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding ON {self.table_name}
                    USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
                """)
            conn.commit()
            logger.info("Schema setup completed successfully.")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Failed to set up schema: {e}")
            raise IndexingError(f"Failed to set up schema: {e}") from e
        finally:
            conn.close()  # Close the connection we got, not self._connection

    def upsert_documents(self, document_id: str, chunks: list):
        """
        Deletes existing chunks for a document and inserts new ones.

        Args:
            document_id: The ID of the document to upsert.
            chunks: A list of document chunks to insert.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                logger.info(f"Deleting existing chunks for document_id: {document_id}")
                self.delete_document(document_id)

                logger.info(f"Inserting {len(chunks)} new chunks for document_id: {document_id}")
                for chunk in chunks:
                    self.vector_store.add([chunk])

            conn.commit()
            logger.info(f"Upsert for document_id: {document_id} completed successfully.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to upsert document: {e}")
            raise IndexingError(f"Failed to upsert document: {e}") from e

    def retrieve_chunks(self, query_embedding: List[float], top_k: int = 10) -> list:
        """
        Retrieves the most similar chunks from the vector store.

        Args:
            query_embedding: The embedding of the query.
            top_k: The number of chunks to retrieve.

        Returns:
            A list of the most similar chunks.
        """
        try:
            return self.vector_store.query(query_embedding, similarity_top_k=top_k)
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            raise RetrievalError(f"Failed to retrieve chunks: {e}") from e

    def delete_document(self, document_id: str):
        """
        Removes all chunks for a document from the vector store.

        Args:
            document_id: The ID of the document to delete.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                logger.info(f"Deleting chunks for document_id: {document_id}")
                cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE metadata->>'document_id' = %s",
                    (document_id,),
                )
            conn.commit()
            logger.info(f"Deletion for document_id: {document_id} completed successfully.")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Failed to delete document: {e}")
            raise IndexingError(f"Failed to delete document: {e}") from e

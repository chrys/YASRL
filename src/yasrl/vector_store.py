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
        self.table_name = table_prefix
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

<<<<<<< HEAD
    def setup_schema(self):
        """
        Creates the required tables and indexes in the database.
=======
    async def check_connection(self) -> bool:
        """
        Checks if a connection can be established to the database.
        """
        conn = None
        try:
            conn = self._get_connection()
            return conn is not None
        except Exception:
            return False
        finally:
            if conn:
                self._release_connection(conn)

    async def get_document_count(self, project_id: str | None = None) -> int:
        """
        Returns the number of documents (chunks) in the vector store.
        If project_id is provided, counts only chunks for that project.
>>>>>>> b82e0dc (fixed indexing)
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
<<<<<<< HEAD
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
            self._release_connection(conn)
=======
                if project_id is not None:
                    # Count distinct documents for a specific project
                    query = f'SELECT COUNT(DISTINCT metadata->>\'document_id\') FROM "{self.table_name}" WHERE metadata->>\'project_id\' = %s'
                    params = (project_id,)
                else:
                    # Count all distinct documents in the table
                    query = f'SELECT COUNT(DISTINCT metadata->>\'document_id\') FROM "{self.table_name}"'
                    params = None
                
                cursor.execute(query, params)
                row = cursor.fetchone()
                return row[0] if row is not None else 0
        except psycopg2.Error as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
        finally:
            self._release_connection(conn)
        
        
>>>>>>> b82e0dc (fixed indexing)

    def upsert_documents(self, document_id: str, chunks: list):
        """
        Deletes existing chunks for a document and inserts new ones in a single transaction.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                logger.info(f"Deleting existing chunks for document_id: {document_id}")
                cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE metadata->>'document_id' = %s",
                    (document_id,),
                )

                logger.info(f"Inserting {len(chunks)} new chunks for document_id: {document_id}")
                for chunk in chunks:
                    if not hasattr(chunk, 'id_') or not chunk.id_:
                        chunk.id_ = str(uuid.uuid4())
                    if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata['document_id'] = document_id
                    
                    self.vector_store.add([chunk])

            conn.commit()
            logger.info(f"Upsert for document_id: {document_id} completed successfully.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to upsert document: {e}")
            raise IndexingError(f"Failed to upsert document: {e}") from e
        finally:
            self._release_connection(conn)

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
<<<<<<< HEAD

=======
        
>>>>>>> b82e0dc (fixed indexing)
    def delete_document(self, document_id: str):
        """
        Removes all chunks for a document from the vector store.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Check if table exists before attempting delete
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (self.table_name,))
                
                result = cursor.fetchone()
                table_exists = result[0] if result is not None else False
                if not table_exists:
                    logger.info(f"Table {self.table_name} doesn't exist, skipping delete for document_id: {document_id}")
                    return
                    
                logger.info(f"Deleting chunks for document_id: {document_id}")
                cursor.execute(
                    f'DELETE FROM "{self.table_name}" WHERE metadata->>\'document_id\' = %s',
                    (document_id,),
                )
            conn.commit()
            logger.info(f"Deletion for document_id: {document_id} completed successfully.")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Failed to delete document: {e}")
            raise IndexingError(f"Failed to delete document: {e}") from e
        finally:
<<<<<<< HEAD
            self._release_connection(conn)
=======
            self._release_connection(conn)
        
                
    async def ensure_table_exists(self) -> bool:
        """
        Ensures the table exists for this project, creates it if it doesn't.
        Returns True if table exists/was created successfully.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (self.table_name,))
                
                row = cursor.fetchone()
                table_exists = row[0] if row is not None else False

                if not table_exists:
                    logger.info(f"Table {self.table_name} doesn't exist, creating it...")
                    # Let LlamaIndex create the table by accessing vector_store property
                    # This triggers table creation in PGVectorStore
                    _ = self.vector_store
                    logger.info(f"Table {self.table_name} created successfully")
                else:
                    logger.info(f"Table {self.table_name} already exists")

                return True
            
        except Exception as e:
            logger.error(f"Failed to ensure table exists for {self.table_name}: {e}")
            raise IndexingError(f"Failed to ensure table exists: {e}") from e
        finally:
            self._release_connection(conn)
>>>>>>> b82e0dc (fixed indexing)

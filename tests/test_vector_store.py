import unittest
from unittest.mock import MagicMock, patch
import psycopg2
from llama_index.core.schema import TextNode

from yasrl.vector_store import VectorStoreManager
from yasrl.exceptions import IndexingError, RetrievalError


class TestVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.postgres_uri = "postgresql://user:pass@localhost:5432/testdb"
        self.vector_dimensions = 768
        self.table_prefix = "test"
        self.manager = VectorStoreManager(
            postgres_uri=self.postgres_uri,
            vector_dimensions=self.vector_dimensions,
            table_prefix=self.table_prefix,
        )

    def test_init(self):
        """Test VectorStoreManager initialization."""
        self.assertEqual(self.manager.postgres_uri, self.postgres_uri)
        self.assertEqual(self.manager.vector_dimensions, self.vector_dimensions)
        self.assertEqual(self.manager.table_name, "test_data")
        self.assertIsNone(self.manager._vector_store)
        with self.assertRaises(IndexingError):
            self.manager._get_connection()

    def test_parsed_uri(self):
        """Test URI parsing."""
        parsed = self.manager._parsed_uri
        self.assertEqual(parsed.hostname, "localhost")
        self.assertEqual(parsed.port, 5432)
        self.assertEqual(parsed.username, "user")
        self.assertEqual(parsed.password, "pass")
        self.assertEqual(parsed.path, "/testdb")

    @patch("yasrl.vector_store.PGVectorStore")
    def test_vector_store_property(self, mock_pg_vector_store):
        """Test vector_store property creates PGVectorStore instance."""
        mock_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_store

        # First call should create the store
        result = self.manager.vector_store
        self.assertEqual(result, mock_store)
        mock_pg_vector_store.from_params.assert_called_once_with(
            host="localhost",
            port=5432,
            database="testdb",
            user="user",
            password="pass",
            table_name="test_data",
            embed_dim=768,
            hybrid_search=False,
            text_search_config='english'
        )

        # Second call should return cached instance
        result2 = self.manager.vector_store
        self.assertEqual(result2, mock_store)
        # Should still only be called once (cached)
        self.assertEqual(mock_pg_vector_store.from_params.call_count, 1)

    @patch("yasrl.vector_store.PGVectorStore")
    def test_vector_store_property_error(self, mock_pg_vector_store):
        """Test vector_store property handles initialization errors."""
        mock_pg_vector_store.from_params.side_effect = Exception("Connection failed")

        with self.assertRaises(IndexingError) as context:
            _ = self.manager.vector_store

        self.assertIn("Failed to initialize PGVectorStore", str(context.exception))

    @patch("yasrl.vector_store.SimpleConnectionPool")
    def test_get_connection_success(self, mock_pool):
        """Test successful database connection."""
        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        self.manager._pool = mock_pool.return_value

        result = self.manager._get_connection()

        self.manager._pool.getconn.assert_called_once()
        self.assertEqual(result, mock_conn)

    @patch("yasrl.vector_store.SimpleConnectionPool")
    def test_get_connection_error(self, mock_pool):
        """Test database connection error handling."""
        mock_pool.return_value.getconn.side_effect = psycopg2.Error("Connection failed")
        self.manager._pool = mock_pool.return_value
        with self.assertRaises(psycopg2.Error):
            self.manager._get_connection()

    @patch("yasrl.vector_store.PGVectorStore")
    def test_upsert_documents_success(self, mock_pg_vector_store):
        """Test successful document upsert."""
        mock_vector_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_vector_store
        self.manager._vector_store = mock_vector_store
        
        document_id = "test_doc"
        chunks = [
            TextNode(text="chunk1"),
            TextNode(text="chunk2"),
        ]
        self.manager.upsert_documents(document_id, chunks)

        # Should be called once for the single transaction
        mock_vector_store.add.assert_called_once_with(chunks)
        mock_vector_store.delete.assert_called_once_with(ref_doc_id=document_id)

    @patch("yasrl.vector_store.PGVectorStore")
    def test_upsert_documents_error(self, mock_pg_vector_store):
        """Test upsert documents error handling."""
        mock_vector_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_vector_store
        self.manager._vector_store = mock_vector_store
        mock_vector_store.add.side_effect = Exception("Add failed")

        document_id = "test_doc"
        chunks = [TextNode(text="chunk1")]
        with self.assertRaises(IndexingError) as context:
            self.manager.upsert_documents(document_id, chunks)

        self.assertIn("Failed to upsert document", str(context.exception))


    @patch("yasrl.vector_store.PGVectorStore")
    def test_retrieve_chunks_error(self, mock_pg_vector_store):
        """Test retrieve chunks error handling."""
        mock_vector_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_vector_store
        mock_vector_store.query.side_effect = Exception("Query failed")

        query_embedding = [0.1, 0.2, 0.3]

        with self.assertRaises(RetrievalError) as context:
            self.manager.retrieve_chunks(query_embedding)

        self.assertIn("Failed to retrieve chunks", str(context.exception))


if __name__ == "__main__":
    unittest.main()
import unittest
from unittest.mock import MagicMock, patch
import psycopg2
from llama_index.core.schema import TextNode

from yasrl.vector_store import VectorStoreManager
from yasrl.exceptions import IndexingError, RetrievalError


class TestVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.postgres_uri = "postgresql://user:pass@localhost:5432/testdb"
        self.vector_dimensions = 1536
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
        self.assertEqual(self.manager.table_name, "test_chunks")
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
            table_name="test_chunks",
            embed_dim=1536,
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

    @patch("yasrl.vector_store.psycopg2.connect")
    def test_get_connection_success(self, mock_connect):
        """Test successful database connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # Mock the connection pool
        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        self.manager._pool = mock_pool

        result = self.manager._get_connection()

        mock_pool.getconn.assert_called_once()
        self.assertEqual(result, mock_conn)

        @patch("yasrl.vector_store.psycopg2.connect")
        def test_get_connection_error(self, mock_connect):
            """Test database connection error handling."""
            mock_connect.side_effect = psycopg2.Error("Connection failed")

            with self.assertRaises(IndexingError) as context:
                self.manager._get_connection()

            self.assertIn("Connection pool is not initialized.", str(context.exception))

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    def test_setup_schema_success(self, mock_get_connection):
        """Test successful schema setup."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        self.manager.setup_schema()

        mock_get_connection.assert_called_once()
        self.assertEqual(mock_cursor.execute.call_count, 3)  # CREATE TABLE + 2 CREATE INDEX
        mock_conn.commit.assert_called_once()
        # With connection pooling, we release the connection, not close it.

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    def test_setup_schema_error(self, mock_get_connection):
        """Test schema setup error handling."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg2.Error("Schema error")

        with self.assertRaises(IndexingError) as context:
            self.manager.setup_schema()

        self.assertIn("Failed to set up schema", str(context.exception))
        mock_conn.rollback.assert_called_once()
        # With connection pooling, we release the connection, not close it.

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    @patch("yasrl.vector_store.PGVectorStore")
    def test_upsert_documents_success(self, mock_pg_vector_store, mock_get_connection):
        """Test successful document upsert."""
        mock_vector_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_vector_store
        self.manager._vector_store = mock_vector_store
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        document_id = "test_doc"
        chunks = [
            TextNode(text="chunk1"),
            TextNode(text="chunk2"),
        ]

        self.manager.upsert_documents(document_id, chunks)

        # Should be called once for the single transaction
        mock_get_connection.assert_called_once()
        mock_cursor.execute.assert_called_once_with(
            f"DELETE FROM {self.manager.table_name} WHERE metadata->>'document_id' = %s",
            (document_id,),
        )
        self.assertEqual(mock_vector_store.add.call_count, 2)  # One per chunk
        mock_conn.commit.assert_called_once()

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    @patch("yasrl.vector_store.PGVectorStore")
    def test_upsert_documents_error(self, mock_pg_vector_store, mock_get_connection):
        """Test upsert documents error handling."""
        mock_vector_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_vector_store
        self.manager._vector_store = mock_vector_store
        mock_vector_store.add.side_effect = Exception("Add failed")

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        document_id = "test_doc"
        chunks = [TextNode(text="chunk1")]

        with self.assertRaises(IndexingError) as context:
            self.manager.upsert_documents(document_id, chunks)

        self.assertIn("Failed to upsert document", str(context.exception))
        mock_conn.rollback.assert_called()

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    def test_delete_document_success(self, mock_get_connection):
        """Test successful document deletion."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        document_id = "test_doc"
        self.manager.delete_document(document_id)

        mock_get_connection.assert_called_once()
        mock_cursor.execute.assert_called_once_with(
            f"DELETE FROM {self.manager.table_name} WHERE metadata->>'document_id' = %s",
            (document_id,),
        )
        mock_conn.commit.assert_called_once()

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    def test_delete_document_error(self, mock_get_connection):
        """Test delete document error handling."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg2.Error("Delete failed")

        document_id = "test_doc"

        with self.assertRaises(IndexingError) as context:
            self.manager.delete_document(document_id)

        self.assertIn("Failed to delete document", str(context.exception))
        mock_conn.rollback.assert_called_once()

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
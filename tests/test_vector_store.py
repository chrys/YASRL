import sys
import unittest
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

sys.path.append('/home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages')
sys.path.append('src')
from llama_index.core.schema import TextNode

from yasrl.exceptions import IndexingError, RetrievalError
from yasrl.vector_store import VectorStoreManager


class TestVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.postgres_uri = "postgresql://user:password@localhost:5432/database"
        self.vector_dimensions = 1536
        self.table_prefix = "test"
        self.manager = VectorStoreManager(
            postgres_uri=self.postgres_uri,
            vector_dimensions=self.vector_dimensions,
            table_prefix=self.table_prefix,
        )

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    def test_setup_schema(self, mock_get_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        self.manager.setup_schema()

        mock_get_connection.assert_called_once()
        self.assertEqual(mock_cursor.execute.call_count, 3)
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    def test_setup_schema_error(self, mock_get_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg2.Error("Test error")

        with pytest.raises(IndexingError):
            self.manager.setup_schema()

        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    @patch("llama_index.vector_stores.postgres.PGVectorStore")
    def test_upsert_documents(self, mock_pg_vector_store, mock_get_connection):
        mock_vector_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_vector_store
        self.manager._vector_store = mock_vector_store
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        document_id = "test_doc"
        chunks = [
            TextNode(text="chunk1"),
            TextNode(text="chunk2"),
        ]

        self.manager.upsert_documents(document_id, chunks)

        mock_get_connection.assert_called_once()
        mock_cursor.execute.assert_called_once_with(
            f"DELETE FROM {self.manager.table_name} WHERE metadata->>'document_id' = %s",
            (document_id,),
        )
        self.assertEqual(mock_vector_store.add.call_count, 2)
        mock_conn.commit.assert_called_once()

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    @patch("llama_index.vector_stores.postgres.PGVectorStore")
    def test_upsert_documents_error(self, mock_pg_vector_store, mock_get_connection):
        mock_vector_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_vector_store
        self.manager._vector_store = mock_vector_store
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_vector_store.add.side_effect = psycopg2.Error("Test error")

        document_id = "test_doc"
        chunks = [
            TextNode(text="chunk1")
        ]

        with pytest.raises(IndexingError):
            self.manager.upsert_documents(document_id, chunks)

        mock_conn.rollback.assert_called_once()

    @patch("llama_index.vector_stores.postgres.PGVectorStore")
    def test_retrieve_chunks(self, mock_pg_vector_store):
        mock_vector_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_vector_store
        self.manager._vector_store = mock_vector_store
        mock_vector_store.query.return_value = ["chunk1", "chunk2"]

        query_embedding = [0.1] * self.vector_dimensions
        top_k = 2
        chunks = self.manager.retrieve_chunks(query_embedding, top_k)

        mock_vector_store.query.assert_called_once_with(
            query_embedding, similarity_top_k=top_k
        )
        self.assertEqual(chunks, ["chunk1", "chunk2"])

    @patch("llama_index.vector_stores.postgres.PGVectorStore")
    def test_retrieve_chunks_error(self, mock_pg_vector_store):
        mock_vector_store = MagicMock()
        mock_pg_vector_store.from_params.return_value = mock_vector_store
        self.manager._vector_store = mock_vector_store
        mock_vector_store.query.side_effect = Exception("Test error")

        query_embedding = [0.1] * self.vector_dimensions

        with pytest.raises(RetrievalError):
            self.manager.retrieve_chunks(query_embedding)

    @patch("yasrl.vector_store.VectorStoreManager._get_connection")
    def test_delete_document(self, mock_get_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

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
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg2.Error("Test error")

        document_id = "test_doc"

        with pytest.raises(IndexingError):
            self.manager.delete_document(document_id)

        mock_conn.rollback.assert_called_once()


if __name__ == "__main__":
    unittest.main()

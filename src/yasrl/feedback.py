import logging
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extensions import connection
from typing import Optional

logger = logging.getLogger(__name__)

class FeedbackManager:
    """Manages feedback operations in the database."""

    def __init__(self, postgres_uri: str):
        """
        Initializes the FeedbackManager.

        Args:
            postgres_uri: The connection URI for the PostgreSQL database.
        """
        self.postgres_uri = postgres_uri
        self._pool: Optional[SimpleConnectionPool] = None

    def init_pool(self):
        """Initializes the connection pool."""
        if self._pool is None:
            try:
                logger.info("Initializing feedback connection pool...")
                self._pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=5,
                    dsn=self.postgres_uri,
                )
                logger.info("Feedback connection pool initialized.")
            except psycopg2.Error as e:
                logger.error(f"Failed to initialize feedback connection pool: {e}")
                raise

    def close_pool(self):
        """Closes the connection pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Feedback connection pool closed.")

    def _get_connection(self) -> connection:
        """Gets a connection from the pool."""
        if not self._pool:
            self.init_pool()
        if not self._pool:
            raise ConnectionError("Feedback connection pool is not initialized.")
        return self._pool.getconn()

    def _release_connection(self, conn: connection):
        """Releases a connection back to the pool."""
        if self._pool:
            self._pool.putconn(conn)

    def setup_feedback_table(self):
        """Creates the chatbot_feedback table if it doesn't exist."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                logger.info("Creating chatbot_feedback table if it doesn't exist...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chatbot_feedback (
                        id SERIAL PRIMARY KEY,
                        project TEXT NOT NULL,
                        user_id TEXT,
                        chatbot_answer TEXT NOT NULL,
                        feedback_text TEXT,
                        rating VARCHAR(4) NOT NULL CHECK (rating IN ('GOOD', 'BAD')),
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        is_resolved BOOLEAN
                    );
                """)
                conn.commit()
                logger.info("chatbot_feedback table setup complete.")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Failed to create chatbot_feedback table: {e}")
            raise
        finally:
            self._release_connection(conn)

    def add_feedback(self, project: str, chatbot_answer: str, rating: str):
        """
        Adds a new feedback entry to the chatbot_feedback table.

        Args:
            project: The name of the project.
            chatbot_answer: The answer provided by the chatbot.
            rating: The rating ('GOOD' or 'BAD').
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chatbot_feedback (project, chatbot_answer, rating)
                    VALUES (%s, %s, %s)
                    """,
                    (project, chatbot_answer, rating),
                )
                conn.commit()
                logger.info(f"Added feedback for project '{project}' with rating '{rating}'.")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Failed to add feedback: {e}")
            raise
        finally:
            self._release_connection(conn)
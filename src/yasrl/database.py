import logging
import psycopg2
from psycopg2.extensions import connection
from typing import Optional, List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def get_db_connection(postgres_uri: str) -> connection:
    """
    Establishes a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(postgres_uri)
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise

def setup_project_qa_pairs_table(conn: connection):
    """
    Creates the project_qa_pairs table if it doesn't exist.
    """
    try:
        with conn.cursor() as cursor:
            logger.info("Creating project_qa_pairs table if it doesn't exist...")
            cursor.execute("""
                CREATE SEQUENCE IF NOT EXISTS project_qa_pairs_id_seq;
                CREATE TABLE IF NOT EXISTS project_qa_pairs (
                    id bigint NOT NULL DEFAULT nextval('project_qa_pairs_id_seq'),
                    project_id bigint,
                    metadata jsonb,
                    created_at timestamp with time zone,
                    updated_at timestamp with time zone,
                    source text,
                    question text,
                    answer text,
                    context text,
                    CONSTRAINT project_qa_pairs_pkey PRIMARY KEY (project_id, source, question)
                );
                ALTER SEQUENCE project_qa_pairs_id_seq OWNED BY project_qa_pairs.id;
            """)
            conn.commit()
            logger.info("project_qa_pairs table setup complete.")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to create project_qa_pairs table: {e}")
        raise

def get_qa_pairs_for_source(conn: connection, project_id: int, source: str) -> pd.DataFrame:
    """
    Retrieves QA pairs for a specific project and source from the database.
    """
    query = """
        SELECT question, answer, context
        FROM project_qa_pairs
        WHERE project_id = %s AND source = %s;
    """
    try:
        df = pd.read_sql_query(query, conn, params=(project_id, source))
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve QA pairs for project {project_id} and source {source}: {e}")
        return pd.DataFrame()

def save_qa_pairs(conn: connection, project_id: int, source: str, qa_pairs: pd.DataFrame):
    """
    Saves QA pairs to the database using an upsert approach.
    """
    upsert_query = """
        INSERT INTO project_qa_pairs (id, project_id, source, question, answer, context, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (project_id, source, question) DO UPDATE SET
            answer = EXCLUDED.answer,
            context = EXCLUDED.context,
            updated_at = CURRENT_TIMESTAMP;
    """

    try:
        with conn.cursor() as cursor:
            for _, row in qa_pairs.iterrows():
                # Generate a deterministic ID based on the question and source
                qa_id = hash((project_id, source, row['Question']))
                cursor.execute(upsert_query, (
                    qa_id,
                    project_id,
                    source,
                    row['Question'],
                    row['Answer'],
                    row['Context']
                ))

            conn.commit()
            logger.info(f"Successfully saved {len(qa_pairs)} QA pairs for project {project_id} and source {source}.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save QA pairs for project {project_id} and source {source}: {e}")
        raise
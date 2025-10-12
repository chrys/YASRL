"""Project management utilities backed by PostgreSQL."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Ensure environment variables from .env are loaded before accessing POSTGRES_URI
load_dotenv()


class ProjectManager:
    """Handles CRUD operations for projects stored in PostgreSQL."""

    def __init__(self, postgres_uri: str):
        if not postgres_uri:
            raise ValueError("POSTGRES_URI must be set to use ProjectManager.")

        self.postgres_uri = postgres_uri
        self._pool: Optional[SimpleConnectionPool] = None
        self._init_pool()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _init_pool(self) -> None:
        if self._pool is None:
            self._pool = SimpleConnectionPool(minconn=1, maxconn=5, dsn=self.postgres_uri)
            logger.info("ProjectManager connection pool initialized.")

    def _ensure_schema(self) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS projects (
                        id BIGSERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL UNIQUE,
                        description TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                # Extensions for RAG metadata
                cur.execute(
                    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS llm VARCHAR(128) DEFAULT 'gemini';"
                )
                cur.execute(
                    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS embed_model VARCHAR(128) DEFAULT 'gemini';"
                )
                cur.execute(
                    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS sources JSONB DEFAULT '[]'::jsonb;"
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS project_qa_pairs (
                        id BIGSERIAL PRIMARY KEY,
                        project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        context TEXT,
                        source TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                conn.commit()
        except Exception:  # pragma: no cover - logging path
            conn.rollback()
            logger.exception("Failed to ensure projects table schema.")
            raise
        finally:
            self._release_connection(conn)

    def _get_connection(self) -> psycopg2.extensions.connection:
        if self._pool is None:
            raise RuntimeError("ProjectManager connection pool is not initialized.")
        return self._pool.getconn()

    def _release_connection(self, conn: psycopg2.extensions.connection) -> None:
        if self._pool and conn:
            self._pool.putconn(conn)

    @staticmethod
    def _coerce_project_id(project_id: Any) -> int:
        if isinstance(project_id, int):
            return project_id
        if isinstance(project_id, str) and project_id.strip().isdigit():
            return int(project_id.strip())
        raise ValueError(f"Invalid project_id: {project_id}")

    @staticmethod
    def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        row["id"] = str(row["id"])
        row["sources"] = row.get("sources") or []
        return row

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def list_projects(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, name, description, llm, embed_model,
                           COALESCE(sources, '[]'::jsonb) AS sources,
                           created_at
                    FROM projects
                    ORDER BY name;
                    """
                )
                rows = cur.fetchall()
                return [self._normalize_row(dict(row)) for row in rows]
        finally:
            self._release_connection(conn)

    def get_project(self, project_id: Any) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, name, description, llm, embed_model,
                           COALESCE(sources, '[]'::jsonb) AS sources,
                           created_at
                    FROM projects
                    WHERE id = %s;
                    """,
                    (self._coerce_project_id(project_id),),
                )
                row = cur.fetchone()
                return self._normalize_row(dict(row)) if row else None
        finally:
            self._release_connection(conn)

    def get_project_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, name, description, llm, embed_model,
                           COALESCE(sources, '[]'::jsonb) AS sources,
                           created_at
                    FROM projects
                    WHERE LOWER(name) = LOWER(%s);
                    """,
                    (name.strip(),),
                )
                row = cur.fetchone()
                return self._normalize_row(dict(row)) if row else None
        finally:
            self._release_connection(conn)

    def create_project(
        self,
        name: str,
        llm: str,
        embed_model: str,
        description: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO projects (name, description, llm, embed_model, sources)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, name, description, llm, embed_model,
                              COALESCE(sources, '[]'::jsonb) AS sources,
                              created_at;
                    """,
                    (name.strip(), description, llm.strip(), embed_model.strip(), Json(sources or [])),
                )
                conn.commit()
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("Failed to fetch record for newly created project.")
                logger.info("Created project '%s' (id=%s)", name, row["id"])
                return self._normalize_row(dict(row))
        except Exception:
            conn.rollback()
            logger.exception("Failed to create project '%s'", name)
            raise
        finally:
            self._release_connection(conn)

    def update_project(
        self,
        project_id: Any,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        llm: Optional[str] = None,
        embed_model: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        fields: List[str] = []
        values: List[Any] = []

        if name is not None:
            fields.append("name = %s")
            values.append(name.strip())
        if description is not None:
            fields.append("description = %s")
            values.append(description)
        if llm is not None:
            fields.append("llm = %s")
            values.append(llm.strip())
        if embed_model is not None:
            fields.append("embed_model = %s")
            values.append(embed_model.strip())
        if sources is not None:
            fields.append("sources = %s")
            values.append(Json(sources))

        if not fields:
            return self.get_project(project_id)

        project_id_int = self._coerce_project_id(project_id)
        values.append(project_id_int)

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    UPDATE projects
                    SET {', '.join(fields)}
                    WHERE id = %s
                    RETURNING id, name, description, llm, embed_model,
                              COALESCE(sources, '[]'::jsonb) AS sources,
                              created_at;
                    """,
                    tuple(values),
                )
                conn.commit()
                row = cur.fetchone()
                return self._normalize_row(dict(row)) if row else None
        except Exception:
            conn.rollback()
            logger.exception("Failed to update project %s", project_id_int)
            raise
        finally:
            self._release_connection(conn)

    def add_source(self, project_id: Any, source: str) -> Dict[str, Any]:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError(f"Project {project_id} not found.")

        sources = project.get("sources", [])
        if source not in sources:
            sources.append(source)
            project = self.update_project(project_id, sources=sources)
        return project  # type: ignore[return-value]

    def get_project_qa_pairs(self, project_id: Any, source: Optional[str] = None) -> List[Dict[str, Any]]:
        project_id_int = self._coerce_project_id(project_id)
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if source:
                    cur.execute(
                        """
                        SELECT id, question, answer, context, source, metadata, created_at, updated_at
                        FROM project_qa_pairs
                        WHERE project_id = %s AND source = %s
                        ORDER BY id;
                        """,
                        (project_id_int, source),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, question, answer, context, source, metadata, created_at, updated_at
                        FROM project_qa_pairs
                        WHERE project_id = %s
                        ORDER BY id;
                        """,
                        (project_id_int,),
                    )
                rows = cur.fetchall()
                result: List[Dict[str, Any]] = []
                for row in rows:
                    payload = dict(row)
                    payload["id"] = str(payload["id"])
                    result.append(payload)
                return result
        finally:
            self._release_connection(conn)

    def replace_project_qa_pairs(self, project_id: Any, source: str, qa_pairs: List[Dict[str, Any]]) -> int:
        if not source:
            raise ValueError("Source is required when saving QA pairs.")
        project_id_int = self._coerce_project_id(project_id)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM project_qa_pairs WHERE project_id = %s AND source = %s;",
                    (project_id_int, source),
                )
                inserted = 0
                if qa_pairs:
                    insert_values = [
                        (
                            project_id_int,
                            (pair.get("question") or "").strip(),
                            (pair.get("answer") or "").strip(),
                            (pair.get("context") or "").strip() or None,
                            source,
                            Json(pair.get("metadata") or {}),
                        )
                        for pair in qa_pairs
                        if (pair.get("question") or "").strip()
                        and (pair.get("answer") or "").strip()
                    ]
                    if insert_values:
                        args_str = ",".join(["(%s, %s, %s, %s, %s, %s, NOW())"] * len(insert_values))
                        flat_values: List[Any] = []
                        for values in insert_values:
                            flat_values.extend(values)
                        cur.execute(
                            f"""
                            INSERT INTO project_qa_pairs (project_id, question, answer, context, source, metadata, updated_at)
                            VALUES {args_str}
                            """,
                            tuple(flat_values),
                        )
                        inserted = len(insert_values)
                conn.commit()
                return inserted
        except Exception:
            conn.rollback()
            logger.exception("Failed to replace QA pairs for project %s source %s", project_id_int, source)
            raise
        finally:
            self._release_connection(conn)

    def delete_project(self, project_id: Any) -> None:
        project_id_int = self._coerce_project_id(project_id)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM projects WHERE id = %s;", (project_id_int,))
                conn.commit()
                logger.info("Deleted project %s", project_id_int)
        except Exception:
            conn.rollback()
            logger.exception("Failed to delete project %s", project_id_int)
            raise
        finally:
            self._release_connection(conn)

    def close(self) -> None:
        if self._pool:
            self._pool.closeall()
            logger.info("ProjectManager connection pool closed.")


_project_manager: Optional[ProjectManager] = None


def get_project_manager() -> ProjectManager:
    global _project_manager
    if _project_manager is None:
        postgres_uri = os.getenv("POSTGRES_URI", "")
        if not postgres_uri:
            raise RuntimeError("POSTGRES_URI must be set to manage projects.")
        _project_manager = ProjectManager(postgres_uri)
    return _project_manager

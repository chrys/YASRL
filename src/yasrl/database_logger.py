import logging
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from fastapi import Request, Response
import json

logger = logging.getLogger(__name__)

class DatabaseLogger:
    def __init__(self, postgres_uri: str):
        self._pool = SimpleConnectionPool(minconn=1, maxconn=5, dsn=postgres_uri)
        self._create_log_table()

    def _create_log_table(self):
        """Creates the http_log table if it doesn't exist."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS http_log (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        method VARCHAR(10),
                        path VARCHAR(255),
                        status_code INTEGER,
                        processing_time_ms INTEGER,
                        request_size_bytes INTEGER,
                        response_size_bytes INTEGER,
                        client_ip VARCHAR(50),
                        website_id VARCHAR(255),
                        request_headers TEXT,
                        response_headers TEXT,
                        request_body TEXT,
                        response_body TEXT
                    );
                """)
                conn.commit()
                logger.info("http_log table checked/created successfully.")
        except psycopg2.Error as e:
            logger.error(f"Failed to create http_log table: {e}")
            conn.rollback()
        finally:
            self._pool.putconn(conn)

    async def log_request(self, request: Request, response: Response, process_time: float):
        """Logs an HTTP request and response to the database."""
        conn = self._pool.getconn()
        try:
            request_body = await request.body()

            # It's tricky to get the response body in middleware without consuming it.
            # We'll log what we can. For full response logging, a different approach is needed.
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            # Get website_id
            website_id = None
            if "x-api-key" in request.headers:
                from .auth import VALID_API_KEYS
                api_key = request.headers["x-api-key"]
                website_id = VALID_API_KEYS.get(api_key)

            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO http_log (
                        method, path, status_code, processing_time_ms,
                        request_size_bytes, response_size_bytes, client_ip, website_id,
                        request_headers, response_headers, request_body, response_body
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    request.method,
                    request.url.path,
                    response.status_code,
                    int(process_time * 1000),
                    len(request_body),
                    len(response_body),
                    request.client.host,
                    website_id,
                    json.dumps(dict(request.headers)),
                    json.dumps(dict(response.headers)),
                    request_body.decode('utf-8', errors='ignore'),
                    response_body.decode('utf-8', errors='ignore')
                ))
                conn.commit()
        except psycopg2.Error as e:
            logger.error(f"Failed to log request to database: {e}")
            conn.rollback()
        except Exception as e:
            logger.error(f"An unexpected error occurred in database logger: {e}")
        finally:
            self._pool.putconn(conn)
            # Return a new response with the consumed body, so the client still gets it
            return Response(content=response_body, status_code=response.status_code, headers=dict(response.headers))

# This will be initialized in the main application file
db_logger: DatabaseLogger = None

def initialize_db_logger(postgres_uri: str):
    global db_logger
    db_logger = DatabaseLogger(postgres_uri)
    return db_logger
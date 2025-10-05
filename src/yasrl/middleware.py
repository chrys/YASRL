import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from .database_logger import db_logger

logger = logging.getLogger(__name__)

class ResourceManagementMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time

        # The response body is consumed by the db_logger, so we need to get a new response
        if db_logger:
            new_response = await db_logger.log_request(request, response, process_time)
            return new_response

        return response
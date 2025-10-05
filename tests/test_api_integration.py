import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
import os

# Set environment variables for testing before importing the app
os.environ["JWT_SECRET_KEY"] = "test-secret"
os.environ["POSTGRES_URI"] = "postgresql://test:test@localhost:5432/testdb"
os.environ["OPENAI_API_KEY"] = "test-key"

# Mock objects that will be used across tests
mock_pipeline = AsyncMock()
mock_pipeline.ask.return_value = MagicMock(answer="mock answer", source_chunks=[])
mock_pipeline.index.return_value = None

async def override_get_current_user():
    """A mock user session for testing authenticated endpoints."""
    return {"user_id": "test_user", "website_id": "website1"}

@pytest.fixture
def patched_app():
    """
    Provides a FastAPI app instance with external dependencies (DB, RAG pipeline) patched.
    This fixture should be used by other client fixtures.
    """
    # This mock needs to behave like the real log_request and return a Response
    async def mock_log_request_func(request, response, process_time):
        from fastapi import Response
        response_body = b''
        async for chunk in response.body_iterator:
            response_body += chunk
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    mock_db_logger = MagicMock()
    mock_db_logger.log_request = AsyncMock(side_effect=mock_log_request_func)

    with patch("src.yasrl.pipeline_cache.RAGPipeline.create", new=AsyncMock(return_value=mock_pipeline)), \
         patch("src.yasrl.api_integration.initialize_db_logger"), \
         patch("src.yasrl.middleware.db_logger", mock_db_logger):

        from src.yasrl.api_integration import app
        # Attach the mock to the app's state so we can access it in tests
        app.state.mock_db_logger = mock_db_logger
        yield app

@pytest.fixture
def client(patched_app):
    """
    Provides a test client for an authenticated user.
    This is the most common client for tests.
    """
    from src.yasrl.auth import get_current_user
    patched_app.dependency_overrides[get_current_user] = override_get_current_user

    with TestClient(patched_app) as c:
        yield c

    patched_app.dependency_overrides.clear()

@pytest.fixture
def unauth_client(patched_app):
    """Provides a test client that is not authenticated."""
    with TestClient(patched_app) as c:
        yield c

# Helper to create auth tokens, since we're not importing the app at the top level
def create_test_access_token(user_id="test_user"):
    from src.yasrl.auth import create_access_token
    return create_access_token(data={"sub": user_id})

def get_auth_headers(website_api_key="website1_key", user_id="test_user"):
    token = create_test_access_token(user_id=user_id)
    return {"X-API-Key": website_api_key, "Authorization": f"Bearer {token}"}

def test_create_pipeline_success(client):
    """Test successful creation of a pipeline."""
    headers = get_auth_headers()
    response = client.post(
        "/pipelines",
        headers=headers,
        json={"llm": "openai", "embed_model": "openai"}
    )
    assert response.status_code == 201
    json_response = response.json()
    assert json_response["message"] == "Pipeline created successfully"
    assert json_response["pipeline_id"] == "website1:test_user"

def test_unauthenticated_request(unauth_client):
    """Test that requests without authentication fail."""
    response = unauth_client.post(
        "/pipelines",
        headers={"X-API-Key": "website1_key"}, # Provide API key but no JWT
        json={"llm": "openai", "embed_model": "openai"}
    )
    assert response.status_code == 401

def test_invalid_api_key(unauth_client):
    """Test that requests with an invalid API key fail."""
    headers = get_auth_headers(website_api_key="invalid_key")
    response = unauth_client.post(
        "/pipelines",
        headers=headers,
        json={"llm": "openai", "embed_model": "openai"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API Key"

def test_index_documents_success(client):
    """Test successful document indexing."""
    client.post("/pipelines", headers=get_auth_headers(), json={"llm": "openai", "embed_model": "openai"})

    headers = get_auth_headers()
    pipeline_id = "website1:test_user"
    response = client.post(
        f"/pipelines/{pipeline_id}/index",
        headers=headers,
        json={"source": "/path/to/docs"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successfully indexed source: /path/to/docs"
    mock_pipeline.index.assert_called_once()

def test_ask_question_success(client):
    """Test successful question asking."""
    client.post("/pipelines", headers=get_auth_headers(), json={"llm": "openai", "embed_model": "openai"})

    headers = get_auth_headers()
    pipeline_id = "website1:test_user"
    response = client.post(
        f"/pipelines/{pipeline_id}/ask",
        headers=headers,
        json={"query": "What is YASRL?"}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "mock answer"
    mock_pipeline.ask.assert_called_once()

def test_ask_nonexistent_pipeline(client):
    """Test asking a question to a pipeline that doesn't exist."""
    headers = get_auth_headers(user_id="nonexistent_user")
    pipeline_id = "website1:nonexistent_user"
    response = client.post(
        f"/pipelines/{pipeline_id}/ask",
        headers=headers,
        json={"query": "What is YASRL?"}
    )
    assert response.status_code == 404

def test_rate_limiting(client):
    """Test that rate limiting is enforced."""
    from src.yasrl.rate_limiter import limiter
    limiter._storage.storage.clear()

    headers = get_auth_headers()
    # The create endpoint has a limit of 10/minute
    for i in range(10):
        response = client.post("/pipelines", headers=headers, json={"llm": "openai", "embed_model": "openai"})
        assert response.status_code == 201

    # The 11th request should be blocked
    response = client.post("/pipelines", headers=headers, json={"llm": "openai", "embed_model": "openai"})
    assert response.status_code == 429
    limiter._storage.storage.clear()

def test_middleware_logs_request(client):
    """Test that the middleware calls the database logger."""
    headers = get_auth_headers()
    client.post("/pipelines", headers=headers, json={"llm": "openai", "embed_model": "openai"})
    assert client.app.state.mock_db_logger.log_request.called
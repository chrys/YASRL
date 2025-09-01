# Key Features:
Pipeline Management: Create, list, and delete pipelines
Document Indexing: Index documents from various sources
Question Answering: Ask questions with conversation history support
Health Checks: Monitor pipeline health
Statistics: Get pipeline statistics
Auto Documentation: FastAPI automatically generates API docs at docs
Resource Management: Proper cleanup of resources on shutdown
Error Handling: Comprehensive error handling with appropriate HTTP status codes
The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs.

# Create a pipeline:
curl -X POST "http://localhost:8000/pipelines" \
     -H "Content-Type: application/json" \
     -d '{"llm": "gemini", "embed_model": "gemini"}'

Response:
{
  "pipeline_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "Pipeline created successfully with ID: 123e4567-e89b-12d3-a456-426614174000"
}

# Index documents
curl -X POST "http://localhost:8000/pipelines/123e4567-e89b-12d3-a456-426614174000/index" \
     -H "Content-Type: application/json" \
     -d '{"source": "https://example.com"}'

# Ask a question
curl -X POST "http://localhost:8000/pipelines/123e4567-e89b-12d3-a456-426614174000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is YASRL?"}'
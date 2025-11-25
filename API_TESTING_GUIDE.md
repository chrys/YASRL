# Testing the YASRL API

This guide shows how to test the RAG API endpoints using curl and Python.

## Prerequisites

Make sure the API server is running:
```bash
.venv/bin/python -m uvicorn src.API.api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/healthz
```

Expected response:
```json
{"status": "ok"}
```

### 2. Create a Pipeline

```bash
curl -X POST http://localhost:8000/pipelines \
  -H "Content-Type: application/json" \
  -d '{
    "llm": "gemini",
    "embed_model": "gemini"
  }'
```

Expected response:
```json
{
  "pipeline_id": "abc123-def456-...",
  "message": "Pipeline created successfully with ID: abc123-def456-..."
}
```

**Save the `pipeline_id` for subsequent requests!**

### 3. Index Documents

```bash
# Replace PIPELINE_ID with the ID from step 2
curl -X POST http://localhost:8000/pipelines/PIPELINE_ID/index \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/Users/chrys/Projects/YASRL/data/VW_dataMar25.txt"
  }'
```

Expected response:
```json
{
  "message": "Successfully indexed source: /Users/chrys/Projects/YASRL/data/VW_dataMar25.txt"
}
```

### 4. Ask Questions (Chat)

```bash
# Replace PIPELINE_ID with your pipeline ID
curl -X POST http://localhost:8000/pipelines/PIPELINE_ID/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "conversation_history": []
  }'
```

Expected response:
```json
{
  "answer": "This document is about...",
  "source_chunks": [
    {
      "text": "...",
      "metadata": {
        "source": "/Users/chrys/Projects/YASRL/data/VW_dataMar25.txt",
        "chunk_id": "..."
      },
      "score": 0.85
    }
  ],
  "metadata": {
    "model": "gemini",
    "timestamp": "2025-11-25T11:33:00Z"
  }
}
```

### 5. Get Pipeline Statistics

```bash
curl http://localhost:8000/pipelines/PIPELINE_ID/stats
```

### 6. List All Pipelines

```bash
curl http://localhost:8000/pipelines
```

### 7. Delete a Pipeline

```bash
curl -X DELETE http://localhost:8000/pipelines/PIPELINE_ID
```

## Python Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Create a pipeline
response = requests.post(f"{BASE_URL}/pipelines", json={
    "llm": "gemini",
    "embed_model": "gemini"
})
pipeline_data = response.json()
pipeline_id = pipeline_data["pipeline_id"]
print(f"Created pipeline: {pipeline_id}")

# 2. Index a document
response = requests.post(
    f"{BASE_URL}/pipelines/{pipeline_id}/index",
    json={"source": "/Users/chrys/Projects/YASRL/data/VW_dataMar25.txt"}
)
print(f"Indexing response: {response.json()}")

# 3. Ask a question
response = requests.post(
    f"{BASE_URL}/pipelines/{pipeline_id}/ask",
    json={
        "query": "What is this document about?",
        "conversation_history": []
    }
)
result = response.json()
print(f"\nQuestion: What is this document about?")
print(f"Answer: {result['answer']}")
print(f"\nSources ({len(result['source_chunks'])} chunks):")
for i, chunk in enumerate(result['source_chunks'], 1):
    print(f"  {i}. Score: {chunk['score']:.2f} - {chunk['text'][:100]}...")

# 4. Follow-up question with conversation history
conversation_history = [
    {"role": "user", "content": "What is this document about?"},
    {"role": "assistant", "content": result['answer']}
]

response = requests.post(
    f"{BASE_URL}/pipelines/{pipeline_id}/ask",
    json={
        "query": "Can you provide more details?",
        "conversation_history": conversation_history
    }
)
result = response.json()
print(f"\nFollow-up Question: Can you provide more details?")
print(f"Answer: {result['answer']}")

# 5. Get statistics
response = requests.get(f"{BASE_URL}/pipelines/{pipeline_id}/stats")
stats = response.json()
print(f"\nPipeline Statistics:")
print(json.dumps(stats, indent=2))

# 6. Cleanup
response = requests.delete(f"{BASE_URL}/pipelines/{pipeline_id}")
print(f"\nDeleted pipeline: {response.json()}")
```

## Complete Test Script

Save this as `test_api.sh`:

```bash
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Testing YASRL API ===${NC}\n"

# 1. Health check
echo -e "${GREEN}1. Health Check${NC}"
curl -s http://localhost:8000/healthz | jq
echo -e "\n"

# 2. Create pipeline
echo -e "${GREEN}2. Creating Pipeline${NC}"
RESPONSE=$(curl -s -X POST http://localhost:8000/pipelines \
  -H "Content-Type: application/json" \
  -d '{"llm": "gemini", "embed_model": "gemini"}')
echo $RESPONSE | jq
PIPELINE_ID=$(echo $RESPONSE | jq -r '.pipeline_id')
echo -e "Pipeline ID: $PIPELINE_ID\n"

# 3. Index document
echo -e "${GREEN}3. Indexing Document${NC}"
curl -s -X POST http://localhost:8000/pipelines/$PIPELINE_ID/index \
  -H "Content-Type: application/json" \
  -d '{"source": "/Users/chrys/Projects/YASRL/data/VW_dataMar25.txt"}' | jq
echo -e "\n"

# 4. Ask question
echo -e "${GREEN}4. Asking Question${NC}"
curl -s -X POST http://localhost:8000/pipelines/$PIPELINE_ID/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "conversation_history": []}' | jq
echo -e "\n"

# 5. Get stats
echo -e "${GREEN}5. Getting Statistics${NC}"
curl -s http://localhost:8000/pipelines/$PIPELINE_ID/stats | jq
echo -e "\n"

# 6. List pipelines
echo -e "${GREEN}6. Listing All Pipelines${NC}"
curl -s http://localhost:8000/pipelines | jq
echo -e "\n"

# 7. Delete pipeline
echo -e "${GREEN}7. Deleting Pipeline${NC}"
curl -s -X DELETE http://localhost:8000/pipelines/$PIPELINE_ID | jq
echo -e "\n"

echo -e "${BLUE}=== Test Complete ===${NC}"
```

Make it executable and run:
```bash
chmod +x test_api.sh
./test_api.sh
```

## Testing with HTTPie (Alternative)

If you have HTTPie installed (`pip install httpie`):

```bash
# Create pipeline
http POST localhost:8000/pipelines llm=gemini embed_model=gemini

# Index document
http POST localhost:8000/pipelines/PIPELINE_ID/index source="/path/to/doc.txt"

# Ask question
http POST localhost:8000/pipelines/PIPELINE_ID/ask \
  query="What is this about?" \
  conversation_history:='[]'
```

## Common Issues

1. **Pipeline not found**: Make sure you're using the correct `pipeline_id` from the create response
2. **Indexing fails**: Verify the file path exists and is readable
3. **Empty responses**: Ensure documents are indexed before asking questions
4. **Connection refused**: Check that the API server is running on port 8000

#Create a pipeline:
curl -X POST "http://localhost:8000/pipelines" \
     -H "Content-Type: application/json" \
     -d '{"llm": "gemini", "embed_model": "gemini"}'
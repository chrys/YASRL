# yasrl: Yet Another RAG Library

## Installation

yasrl is a Python library for building Retrieval-Augmented Generation (RAG) pipelines with opinionated defaults and simple configuration.

### Basic Installation

```bash
pip install yasrl
```

### Optional Provider Dependencies

To use specific providers or features, install with extras:

- **OpenAI support:**
  ```bash
  pip install yasrl[openai]
  ```
- **Gemini support:**
  ```bash
  pip install yasrl[gemini]
  ```
- **Ollama support:**
  ```bash
  pip install yasrl[ollama]
  ```
- **PostgreSQL vector store:**
  ```bash
  pip install yasrl[postgres]
  ```
- **Development tools:**
  ```bash
  pip install yasrl[dev]
  ```
- **Testing tools:**
  ```bash
  pip install yasrl[test]
  ```
- **Documentation tools:**
  ```bash
  pip install yasrl[docs]
  ```

## Requirements
- Python 3.8+
- For provider features, set required environment variables (see documentation)

## Getting Started
See the documentation for usage examples and API reference.

---
For more details, visit the docs or see the project specification in `docs/project_specifications.md`.

## Functionality 
This Python code defines a web API using the FastAPI framework to expose the functionality of your YASRL (Yet Another Simple RAG Library) project. It acts as a server that allows external systems to create and interact with chatbot pipelines over the internet.

The script begins by setting up a global, in-memory dictionary named pipelines. This dictionary will hold all the active RAGPipeline instances that are created, using a unique ID as the key for each one. This allows the API to manage multiple, independent chatbot sessions simultaneously. It then defines several data structures using Pydantic's BaseModel. These models, such as PipelineCreateRequest and QueryRequest, define the expected JSON format for incoming requests and outgoing responses, enabling automatic data validation and clear API documentation.

The core of the application is the FastAPI instance, configured with a lifespan manager and a root_path. The lifespan function ensures that resources are managed correctly; it logs a startup message and, upon server shutdown, iterates through all active pipelines to call their cleanup methods, preventing resource leaks. The root_path is set to /my_chatbot2/api, which is crucial for production, as it tells FastAPI that it's operating behind a reverse proxy at a specific sub-path, allowing it to generate correct URLs for the interactive API documentation.

The API exposes several endpoints, each corresponding to a specific function. For example, a POST request to /pipelines triggers the create_pipeline function, which initializes a new RAGPipeline instance, stores it in the global dictionary with a unique UUID, and returns this ID to the client. Other endpoints, like /pipelines/{pipeline_id}/ask and /pipelines/{pipeline_id}/index, use this ID to retrieve the correct pipeline instance and then call its corresponding methods (ask() or index()). This is handled elegantly by a FastAPI dependency function, get_pipeline, which centralizes the logic for looking up a pipeline and automatically returns a "404 Not Found" error if the ID is invalid. The API also includes utility endpoints for health checks, statistics, and listing or deleting active pipelines, providing a complete management interface.
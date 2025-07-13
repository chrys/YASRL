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

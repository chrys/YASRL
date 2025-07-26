# Environment Configuration

YASRL uses environment variables for configuration, with automatic loading from `.env` files for convenience and security.

## .env File Setup

YASRL automatically looks for and loads a `.env` file from your project's root directory. This is the recommended way to manage configuration, especially for sensitive information like API keys.

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your actual values:
   ```env
   OPENAI_API_KEY=your-actual-openai-key
   GOOGLE_API_KEY=your-actual-google-key
   POSTGRES_URI=postgresql://user:pass@localhost:5432/db
   OLLAMA_HOST=localhost:11434
   ```

3. **Important**: Add `.env` to your `.gitignore` to keep secrets safe:
   ```gitignore
   .env
   ```

## Required Environment Variables

Depending on which providers you use, you'll need different environment variables:

### For OpenAI Provider
- `OPENAI_API_KEY` - Your OpenAI API key

### For Gemini Provider  
- `GOOGLE_API_KEY` - Your Google AI API key

### For Ollama Provider
- `OLLAMA_HOST` - Ollama server host (default: localhost:11434)

### Database
- `POSTGRES_URI` - PostgreSQL connection string with pgvector extension

## Configuration Override with Environment Variables

You can override any configuration setting using environment variables with the `YASRL_` prefix:

```env
# Override LLM settings
YASRL_LLM_PROVIDER=gemini
YASRL_LLM_TEMPERATURE=0.8
YASRL_LLM_MAX_TOKENS=2048

# Override embedding settings  
YASRL_EMBEDDING_PROVIDER=openai
YASRL_CHUNK_SIZE=800

# Override retrieval settings
YASRL_RETRIEVAL_TOP_K=15
YASRL_RERANK_TOP_K=3

# Override database settings
YASRL_TABLE_PREFIX=my_app
YASRL_CONNECTION_POOL_SIZE=5

# Override logging
YASRL_LOG_LEVEL=DEBUG
```

## Environment Variable Priority

Configuration is loaded in the following priority order (highest to lowest):

1. **Environment variables** (including those from `.env` file)
2. **Local config file** (`yasrl.yaml` or `yasrl.json`)
3. **Global config file** (`~/.yasrl/config.yaml`)
4. **Default values**

## Using Different Environment Prefixes

You can customize the environment variable prefix when initializing the configuration manager:

```python
from yasrl.config import ConfigurationManager

# Use custom prefix (e.g., MYAPP_LLM_PROVIDER instead of YASRL_LLM_PROVIDER)
config_manager = ConfigurationManager(env_prefix="MYAPP")
```

## Development vs Production

For development, use the `.env` file approach. For production deployment, you can:

1. Set environment variables directly in your deployment environment
2. Use container orchestration secrets (Docker, Kubernetes)
3. Use cloud provider secret management services
4. Use environment variable management tools

The environment loader will automatically pick up variables from any of these sources.

## Troubleshooting

### Common Issues

1. **Missing API Key Error**: Ensure your `.env` file is in the project root and contains the required keys
2. **Configuration Not Loading**: Check that your `.env` file uses the correct variable names (see `.env.example`)
3. **Database Connection Error**: Verify your `POSTGRES_URI` format and database accessibility

### Debugging

Enable debug logging to see configuration loading details:

```env
YASRL_LOG_LEVEL=DEBUG
```

This will show which configuration sources are being loaded and in what order.

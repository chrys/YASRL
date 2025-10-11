import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict

from UI.project_manager import get_project_manager
from src.yasrl.pipeline import RAGPipeline
from src.yasrl.exceptions import ConfigurationError
from src.yasrl.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

try:
    project_manager = get_project_manager()
except Exception:
    logger.exception("Failed to initialize ProjectManager for pipeline cache.")
    project_manager = None

class PipelineCache:
    """
    Manages the lifecycle of RAGPipeline instances with a TTL for automatic cleanup.
    """
    def __init__(self, ttl_minutes: int = 60):
        self._pipelines: Dict[str, RAGPipeline] = {}
        self._last_accessed: Dict[str, datetime] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        from typing import Optional
        self._cleanup_task: Optional[asyncio.Task] = None

    async def get_pipeline(self, pipeline_id: str) -> RAGPipeline | None:
        """
        Retrieves a pipeline from the cache, updating its last accessed time.
        """
        if pipeline_id in self._pipelines:
            logger.info(f"Accessing pipeline {pipeline_id} from cache.")
            self._last_accessed[pipeline_id] = datetime.now(timezone.utc)
            return self._pipelines[pipeline_id]
        return None

    async def create_pipeline(self, pipeline_id: str, llm: str, embed_model: str, project_id: str | None = None) -> RAGPipeline:
        """
        Creates a new RAG pipeline and adds it to the cache.
        If a pipeline with the same ID already exists, it will be replaced.
        
        Args:
            pipeline_id: Unique identifier for the pipeline
            llm: LLM model to use
            embed_model: Embedding model to use
            project_id: Optional project ID for project-specific database table naming
        """
        logger.info(f"Creating new pipeline {pipeline_id} with llm={llm}, embed_model={embed_model}, project_id={project_id}")
        
        try:
            # If a pipeline with the same ID exists, clean it up first
            if pipeline_id in self._pipelines:
                logger.warning(f"Pipeline {pipeline_id} already exists. Replacing it.")
                await self.delete_pipeline(pipeline_id)
    
            # Create pipeline with project-specific configuration if project_id is provided
            if project_id and project_manager is not None:
                project_record = project_manager.get_project(project_id)
                if project_record is None:
                    logger.warning("Project %s not found in database; creating pipeline without project-specific config", project_id)
                    pipeline = await RAGPipeline.create(llm=llm, embed_model=embed_model)
                else:
                    project_name = (project_record.get("name") or "").strip()
                    sanitized_name = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
                    sanitized_name = "_".join(filter(None, sanitized_name.split("_")))
                    table_prefix = f"yasrl_{sanitized_name or project_record['id']}"

                    logger.info("Using project-specific table prefix: %s", table_prefix)

                    db_manager = VectorStoreManager(
                        postgres_uri=os.getenv("POSTGRES_URI") or "",
                        vector_dimensions=768,
                        table_prefix=table_prefix,
                    )

                    pipeline = await RAGPipeline.create(
                        llm=llm,
                        embed_model=embed_model,
                        db_manager=db_manager,
                    )
            else:
                # Default pipeline creation without project-specific config
                pipeline = await RAGPipeline.create(llm=llm, embed_model=embed_model)
    
            # Add to cache
            self._pipelines[pipeline_id] = pipeline
            self._last_accessed[pipeline_id] = datetime.now(timezone.utc)
            logger.info(f"Successfully created and cached pipeline {pipeline_id}")
            return pipeline
            
        except ConfigurationError as e:
            logger.error(f"Configuration error creating pipeline {pipeline_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating pipeline {pipeline_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
  
  
    async def delete_pipeline(self, pipeline_id: str):
        """
        Deletes a pipeline and its resources.
        """
        if pipeline_id in self._pipelines:
            logger.info(f"Deleting pipeline {pipeline_id}")
            pipeline = self._pipelines.pop(pipeline_id)
            self._last_accessed.pop(pipeline_id, None)
            if hasattr(pipeline, 'cleanup'):
                await pipeline.cleanup()
            logger.info(f"Successfully deleted pipeline {pipeline_id}")

    async def _cleanup_idle_pipelines(self):
        """
        Periodically checks for and removes idle pipelines.
        """
        while True:
            await asyncio.sleep(60)  # Check every minute
            now = datetime.now(timezone.utc)
            idle_pipelines = []
            for pipeline_id, last_access_time in self._last_accessed.items():
                if now - last_access_time > self.ttl:
                    idle_pipelines.append(pipeline_id)

            for pipeline_id in idle_pipelines:
                logger.info(f"Pipeline {pipeline_id} has been idle for too long. Cleaning up.")
                await self.delete_pipeline(pipeline_id)

    def start_cleanup_task(self):
        """
        Starts the background task for cleaning up idle pipelines.
        """
        if self._cleanup_task is None or self._cleanup_task.done():
            logger.info("Starting idle pipeline cleanup task.")
            self._cleanup_task = asyncio.create_task(self._cleanup_idle_pipelines())

    async def stop_cleanup_task(self):
        """
        Stops the background cleanup task.
        """
        if self._cleanup_task and not self._cleanup_task.done():
            logger.info("Stopping idle pipeline cleanup task.")
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled successfully.")

        # Clean up any remaining pipelines
        logger.info("Cleaning up all remaining pipelines on shutdown.")
        for pipeline_id in list(self._pipelines.keys()):
            await self.delete_pipeline(pipeline_id)

pipeline_cache = PipelineCache(ttl_minutes=30)
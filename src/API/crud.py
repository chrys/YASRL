import uuid
import logging
import os
from typing import Dict, List, Optional, Any
from yasrl.pipeline import RAGPipeline
from yasrl.exceptions import ConfigurationError, IndexingError, RetrievalError
from yasrl.models import QueryResult
from yasrl.database import get_db_connection, get_projects

logger = logging.getLogger(__name__)

class PipelineService:
    def __init__(self):
        self.pipelines: Dict[str, RAGPipeline] = {}

    async def create_pipeline(self, llm: str, embed_model: str) -> str:
        """
        Creates a new RAG pipeline and returns its ID.
        """
        pipeline_id = str(uuid.uuid4())
        logger.info(f"Creating pipeline {pipeline_id} with llm={llm}, embed_model={embed_model}")
        
        try:
            pipeline = await RAGPipeline.create(
                llm=llm,
                embed_model=embed_model
            )
            self.pipelines[pipeline_id] = pipeline
            logger.info(f"Successfully created pipeline {pipeline_id}")
            return pipeline_id
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            raise ConfigurationError(f"Failed to create pipeline: {e}")

    def create_project_in_db(self, name: str, llm: str, embed_model: str, description: str = "") -> bool:
        """
        Creates a new project in the database.
        """
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            logger.warning("POSTGRES_URI not set, cannot create project in DB")
            return False
        
        try:
            conn = get_db_connection(postgres_uri)
            try:
                from yasrl.database import save_single_project
                project_data = {
                    "name": name,
                    "llm": llm,
                    "embed_model": embed_model,
                    "description": description,
                    "sources": []
                }
                return save_single_project(conn, project_data)
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error creating project in DB: {e}")
            return False

    def get_pipeline(self, pipeline_id: str) -> RAGPipeline:
        """
        Retrieves a pipeline by ID.
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        return self.pipelines[pipeline_id]

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """
        Deletes a pipeline and cleans up resources.
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.pipelines[pipeline_id]
        try:
            if hasattr(pipeline, 'cleanup'):
                await pipeline.cleanup()
            del self.pipelines[pipeline_id]
            logger.info(f"Successfully deleted pipeline {pipeline_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting pipeline {pipeline_id}: {e}")
            raise

    def list_pipelines(self) -> List[str]:
        """
        Returns a list of active pipeline IDs.
        """
        return list(self.pipelines.keys())

    def list_projects_from_db(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all projects stored in the database.
        """
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            logger.warning("POSTGRES_URI not set, cannot list projects from DB")
            return []
        
        try:
            conn = get_db_connection(postgres_uri)
            try:
                df = get_projects(conn)
                # Convert DataFrame to list of dicts, handling potential empty/NaN values if needed
                return df.to_dict('records')
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error listing projects from DB: {e}")
            return []

    def delete_project_from_db(self, project_name: str) -> bool:
        """
        Deletes a project from the database by name.
        Also drops the associated embedding and metadata tables.
        """
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            logger.warning("POSTGRES_URI not set, cannot delete project from DB")
            return False
        
        try:
            conn = get_db_connection(postgres_uri)
            try:
                from yasrl.database import delete_project_by_name
                
                # Drop the embedding and metadata tables for this project
                sanitized_name = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
                embeddings_table = f"yasrl_{sanitized_name}_embeddings"
                metadata_table = f"yasrl_{sanitized_name}_metadata"
                
                cursor = conn.cursor()
                try:
                    # Drop tables if they exist
                    cursor.execute(f"DROP TABLE IF EXISTS {embeddings_table} CASCADE")
                    cursor.execute(f"DROP TABLE IF EXISTS {metadata_table} CASCADE")
                    conn.commit()
                    logger.info(f"Dropped tables {embeddings_table} and {metadata_table}")
                except Exception as e:
                    logger.warning(f"Error dropping tables for project {project_name}: {e}")
                    conn.rollback()
                finally:
                    cursor.close()
                
                # Delete the project record
                return delete_project_by_name(conn, project_name)
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error deleting project from DB: {e}")
            return False

    def get_project_sources_from_db(self, project_name: str) -> List[str]:
        """
        Gets all sources for a project from the database.
        """
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            logger.warning("POSTGRES_URI not set, cannot get project sources from DB")
            return []
        
        try:
            conn = get_db_connection(postgres_uri)
            try:
                from yasrl.database import get_project_by_name, get_project_sources
                df = get_project_by_name(conn, project_name)
                if df.empty:
                    return []
                
                project_id = int(df.iloc[0]['id'])
                return get_project_sources(conn, project_id)
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error getting project sources from DB: {e}")
            return []

    async def index_source(self, pipeline_id: str, source: str) -> None:
        """
        Indexes a source into the specified pipeline.
        """
        pipeline = self.get_pipeline(pipeline_id)
        try:
            logger.info(f"Indexing source '{source}' in pipeline {pipeline_id}")
            await pipeline.index(source, project_id=None)
        except Exception as e:
            logger.error(f"Indexing error in pipeline {pipeline_id}: {e}")
            raise IndexingError(f"Failed to index source: {e}")

    async def index_source_with_progress(self, project_name: str, source: str, progress_callback=None) -> None:
        """
        Indexes a source for a specific project with progress updates.
        """
        logger.info(f"Starting index_source_with_progress for project={project_name}, source={source}")
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ConfigurationError("POSTGRES_URI not set")

        try:
            if progress_callback:
                logger.info("Calling progress_callback(5) - Starting")
                await progress_callback(5)
            
            # Get project details to know LLM/Embed model
            conn = get_db_connection(postgres_uri)
            try:
                from yasrl.database import get_project_by_name, add_project_sources
                df = get_project_by_name(conn, project_name)
                if df.empty:
                    raise ValueError(f"Project {project_name} not found")
                
                project = df.iloc[0]
                llm = project['llm']
                embed_model = project['embed_model']
                project_id = int(project['id'])
                
                if progress_callback:
                    logger.info("Calling progress_callback(15) - Project loaded")
                    await progress_callback(15)
                
                # Add source to DB first
                add_project_sources(conn, project_id, [source])
                
            finally:
                conn.close()

            if progress_callback:
                logger.info("Calling progress_callback(25) - Source added to DB")
                await progress_callback(25)

            # Initialize pipeline for this project
            from yasrl.vector_store import VectorStoreManager
            
            # Sanitize project name for table prefix
            sanitized_name = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
            table_prefix = f"yasrl_{sanitized_name}"
            
            db_manager = VectorStoreManager(
                postgres_uri=postgres_uri,
                vector_dimensions=768,
                table_prefix=table_prefix
            )
            
            if progress_callback:
                logger.info("Calling progress_callback(40) - DB manager created")
                await progress_callback(40)
            
            pipeline = await RAGPipeline.create(
                llm=llm,
                embed_model=embed_model,
                db_manager=db_manager
            )
            
            if progress_callback:
                logger.info("Calling progress_callback(60) - Pipeline created")
                await progress_callback(60)
            
            # Indexing
            logger.info("Starting indexing...")
            await pipeline.index(source, project_id=str(project_id))
            logger.info("Indexing completed")
            
            if progress_callback:
                logger.info("Calling progress_callback(100) - Finished")
                await progress_callback(100)
                
        except Exception as e:
            logger.error(f"Error indexing source for project {project_name}: {e}")
            raise

    async def ask(self, pipeline_id: str, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> QueryResult:
        """
        Asks a question to the specified pipeline.
        """
        pipeline = self.get_pipeline(pipeline_id)
        try:
            logger.info(f"Processing query in pipeline {pipeline_id}: {query}")
            return await pipeline.ask(
                query=query,
                conversation_history=conversation_history
            )
        except Exception as e:
            logger.error(f"Retrieval error in pipeline {pipeline_id}: {e}")
            raise RetrievalError(f"Failed to process query: {e}")

    async def get_stats(self, pipeline_id: str) -> Dict:
        """
        Gets statistics for a pipeline.
        """
        pipeline = self.get_pipeline(pipeline_id)
        return await pipeline.get_statistics()

    async def health_check(self, pipeline_id: str) -> bool:
        """
        Checks the health of a pipeline.
        """
        pipeline = self.get_pipeline(pipeline_id)
        return await pipeline.health_check()

    async def cleanup_all(self):
        """
        Cleans up all pipelines.
        """
        for pipeline_id in list(self.pipelines.keys()):
            await self.delete_pipeline(pipeline_id)

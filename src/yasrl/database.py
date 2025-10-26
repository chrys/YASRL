import logging
import psycopg2
from psycopg2.extensions import connection
from typing import Optional, List, Dict, Any
import pandas as pd
import json

logger = logging.getLogger(__name__)

def get_db_connection(postgres_uri: str) -> connection:
    """
    Establishes a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(postgres_uri)
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise

def setup_projects_table(conn: connection):
    """
    Creates the projects table if it doesn't exist.
    """
    try:
        with conn.cursor() as cursor:
            logger.info("Creating projects table if it doesn't exist...")
            
            # Create table only if it doesn't exist 
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id BIGSERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT,
                    sources JSONB DEFAULT '[]'::jsonb,
                    embed_model VARCHAR(100),
                    llm VARCHAR(100),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            
            conn.commit()
            logger.info("projects table setup complete.")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to create projects table: {e}")
        raise

def setup_project_qa_pairs_table(conn: connection):
    """
    Creates the project_qa_pairs table if it doesn't exist.
    """
    try:
        with conn.cursor() as cursor:
            logger.info("Creating project_qa_pairs table if it doesn't exist...")
            
            # Create table only if it doesn't exist 
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_qa_pairs (
                    id BIGSERIAL PRIMARY KEY,
                    project_id BIGINT NOT NULL,
                    source TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT,
                    context TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT project_qa_pairs_project_id_fkey 
                        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    CONSTRAINT project_qa_pairs_unique 
                        UNIQUE (project_id, source, question)
                );
            """)
            
            conn.commit()
            logger.info("project_qa_pairs table setup complete.")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to create project_qa_pairs table: {e}")
        raise
def get_qa_pairs_for_source(conn: connection, project_id: int, source: str) -> pd.DataFrame:
    """
    Retrieves QA pairs for a specific project and source from the database.
    """
    query = """
        SELECT question, answer, context
        FROM project_qa_pairs
        WHERE project_id = %s AND source = %s;
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (project_id, source))
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=["question", "answer", "context"])
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve QA pairs for project {project_id} and source {source}: {e}")
        return pd.DataFrame()

def get_projects(conn: connection) -> pd.DataFrame:
    """
    Retrieves all projects from the database.
    """
    query = """
        SELECT id, name, description, sources, embed_model, llm, created_at
        FROM projects
        ORDER BY created_at DESC;
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=["id", "name", "description", "sources", "embed_model", "llm", "created_at"])
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve projects: {e}")
        return pd.DataFrame()

def get_project_by_name(conn: connection, name: str) -> pd.DataFrame:
    """
    Retrieves a specific project by name from the database.
    """
    query = """
        SELECT id, name, description, sources, embed_model, llm, created_at
        FROM projects
        WHERE name = %s;
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (name,))
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=["id", "name", "description", "sources", "embed_model", "llm", "created_at"])
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve project {name}: {e}")
        return pd.DataFrame()

def save_qa_pairs(conn: connection, project_id: int, source: str, qa_pairs: pd.DataFrame):
    """
    Saves QA pairs to the database using an upsert approach.
    """
    upsert_query = """
        INSERT INTO project_qa_pairs (project_id, source, question, answer, context, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (project_id, source, question) DO UPDATE SET
            answer = EXCLUDED.answer,
            context = EXCLUDED.context,
            updated_at = CURRENT_TIMESTAMP;
    """
    
    try:
        with conn.cursor() as cursor:
            for _, row in qa_pairs.iterrows():
                cursor.execute(upsert_query, (
                    int(project_id),  # Convert numpy.int64 to Python int
                    str(source),      # Ensure it's a Python string
                    str(row['Question']),
                    str(row['Answer']),
                    str(row['Context'])
                ))
            
            conn.commit()
            logger.info(f"Successfully saved {len(qa_pairs)} QA pairs for project {project_id} and source {source}.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save QA pairs for project {project_id} and source {source}: {e}")
        raise

def save_projects(conn: connection, projects: pd.DataFrame):
    """
    Saves projects to the database using an upsert approach.
    """
    upsert_query = """
        INSERT INTO projects (name, description, sources, embed_model, llm, created_at)
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (name) DO UPDATE SET
            description = EXCLUDED.description,
            sources = EXCLUDED.sources,
            embed_model = EXCLUDED.embed_model,
            llm = EXCLUDED.llm;
    """
    
    try:
        with conn.cursor() as cursor:
            for _, row in projects.iterrows():
                # Convert sources to JSON string if it's not already
                sources_json = row['sources'] if isinstance(row['sources'], str) else json.dumps(row['sources'])
                
                cursor.execute(upsert_query, (
                    str(row['name']),        # Ensure it's a Python string
                    str(row['description']), # Ensure it's a Python string
                    sources_json,
                    str(row['embed_model']), # Ensure it's a Python string
                    str(row['llm'])          # Ensure it's a Python string
                ))
            
            conn.commit()
            logger.info(f"Successfully saved {len(projects)} projects.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save projects: {e}")
        raise

def save_single_project(conn: connection, project_data: dict) -> bool:
    """
    Save a single project to the database using upsert.
    Returns True if successful, False otherwise.
    """
    upsert_query = """
        INSERT INTO projects (name, description, sources, embed_model, llm, created_at)
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (name) DO UPDATE SET
            description = EXCLUDED.description,
            sources = EXCLUDED.sources,
            embed_model = EXCLUDED.embed_model,
            llm = EXCLUDED.llm,
            created_at = CURRENT_TIMESTAMP
        RETURNING id;
    """
    
    try:
        with conn.cursor() as cursor:
            # Convert sources to JSON string if it's not already
            sources_json = json.dumps(project_data['sources']) if isinstance(project_data['sources'], list) else project_data['sources']
            
            cursor.execute(upsert_query, (
                str(project_data['name']),
                str(project_data.get('description', '')),
                sources_json,
                str(project_data['embed_model']),
                str(project_data['llm'])
            ))
            
            result = cursor.fetchone()
            conn.commit()
            
            if result:
                project_id = result[0]
                logger.info(f"Successfully saved project '{project_data['name']}' with ID {project_id}")
                return True
            else:
                logger.error(f"Failed to get project ID after saving '{project_data['name']}'")
                return False
                
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save project '{project_data['name']}': {e}")
        return False

def delete_project_by_name(conn: connection, name: str) -> bool:
    """
    Delete a project from the database by name.
    Returns True if a project was deleted, False otherwise.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM projects WHERE name = %s", (name,))
            conn.commit()
            deleted_count = cursor.rowcount
            logger.info(f"Deleted {deleted_count} project(s) with name '{name}'")
            return deleted_count > 0
    except Exception as e:
        logger.exception(f"Failed to delete project '{name}' from database")
        conn.rollback()
        return False

# Also update the main() function to handle the conversion properly:
def main():
    """
    Demonstrates the functionality of the database functions.
    """
    import os
    import pandas as pd
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get configuration from environment
    POSTGRES_URI = os.getenv("POSTGRES_URI")
    if not POSTGRES_URI:
        raise ValueError("POSTGRES_URI environment variable is not set in .env file")
    
    PROJECT_ID = 1
    SOURCE = "example_source.txt"

    print("🔧 Database Functions Demo")
    print("=" * 50)

    conn = None
    try:
        # 1. Connect to the database
        print("1. Connecting to PostgreSQL database...")
        conn = get_db_connection(POSTGRES_URI)
        print("✅ Database connection established")

        # 2. Set up the tables
        print("\n2. Setting up database tables...")
        setup_projects_table(conn)
        setup_project_qa_pairs_table(conn)
        print("✅ Tables setup complete")

        # 3. Create sample projects DataFrame
        print("\n3. Creating sample projects...")
        projects_data = [
            {
                "name": "YASRL Documentation Project",
                "description": "A project for testing YASRL functionality with documentation",
                "sources": ["docs/api.md", "docs/tutorial.md", "README.md"],
                "embed_model": "gemini",
                "llm": "openai"
            },
            {
                "name": "Research Papers Analysis",
                "description": "RAG system for analyzing research papers",
                "sources": ["papers/paper1.pdf", "papers/paper2.pdf"],
                "embed_model": "openai",
                "llm": "gemini"
            },
            {
                "name": "Customer Support KB",
                "description": "Knowledge base for customer support queries",
                "sources": ["kb/faq.txt", "kb/troubleshooting.md"],
                "embed_model": "huggingface",
                "llm": "openai"
            }
        ]
        projects_df = pd.DataFrame(projects_data)
        print(f"✅ Created {len(projects_df)} sample projects")
        print(projects_df[['name', 'description', 'llm']].to_string(index=False))

        # 4. Save projects to the database
        print(f"\n4. Saving projects to database...")
        save_projects(conn, projects_df)
        print("✅ Projects saved successfully")

        # 5. Retrieve all projects from the database
        print(f"\n5. Retrieving all projects from database...")
        retrieved_projects = get_projects(conn)
        print(f"✅ Retrieved {len(retrieved_projects)} projects:")
        print(retrieved_projects[['id', 'name', 'llm', 'embed_model']].to_string(index=False))

        # 6. Get a specific project by name
        print(f"\n6. Retrieving specific project by name...")
        specific_project = get_project_by_name(conn, "YASRL Documentation Project")
        if not specific_project.empty:
            project_id = int(specific_project.iloc[0]['id'])  # Convert to Python int
            print(f"✅ Found project with ID: {project_id}")
            print(specific_project[['name', 'description']].to_string(index=False))
        else:
            print("❌ Project not found")
            return

        # 7. Create sample QA pairs DataFrame for the retrieved project
        print(f"\n7. Creating sample QA pairs for project ID {project_id}...")
        qa_data = [
            {
                "Question": "What is YASRL?", 
                "Answer": "YASRL is a Simple RAG Library built in Python.", 
                "Context": "YASRL is a Python library that provides a simple interface for Retrieval-Augmented Generation."
            },
            {
                "Question": "What database does YASRL use?", 
                "Answer": "YASRL uses PostgreSQL with pgvector for vector storage.", 
                "Context": "YASRL stores data in PostgreSQL and uses pgvector extension for similarity search."
            },
            {
                "Question": "What is the purpose of QA pairs?", 
                "Answer": "QA pairs are used for evaluation and testing of the RAG system.", 
                "Context": "Question-Answer pairs serve as ground truth data for evaluating RAG performance."
            }
        ]
        qa_df = pd.DataFrame(qa_data)
        print(f"✅ Created {len(qa_df)} sample QA pairs")

        # 8. Save QA pairs to the database
        print(f"\n8. Saving QA pairs to database (project_id={project_id}, source='{SOURCE}')...")
        save_qa_pairs(conn, project_id, SOURCE, qa_df)
        print("✅ QA pairs saved successfully")

        # 9. Retrieve QA pairs from the database
        print(f"\n9. Retrieving QA pairs from database...")
        retrieved_qa = get_qa_pairs_for_source(conn, project_id, SOURCE)
        print(f"✅ Retrieved {len(retrieved_qa)} QA pairs:")
        print(retrieved_qa.to_string(index=False))

        # 10. Test project upsert functionality
        print(f"\n10. Testing project upsert functionality...")
        updated_projects_data = [
            {
                "name": "YASRL Documentation Project",  # Same name - should update
                "description": "Updated: A comprehensive project for testing YASRL functionality with documentation and examples",
                "sources": ["docs/api.md", "docs/tutorial.md", "README.md", "examples/"],
                "embed_model": "gemini",
                "llm": "gemini"  # Changed from openai to gemini
            }
        ]
        updated_projects_df = pd.DataFrame(updated_projects_data)
        save_projects(conn, updated_projects_df)
        print("✅ Project upsert operation completed")

        # 11. Verify the project updates
        print(f"\n11. Verifying project updates...")
        final_projects = get_projects(conn)
        print(f"✅ Final projects in database:")
        print(final_projects[['id', 'name', 'llm', 'embed_model']].to_string(index=False))

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 12. Clean up
        if conn is not None:
            conn.close()
            print("\n12. Database connection closed")

    print("\n🎉 Database functions demo completed successfully!")

if __name__ == "__main__":
    main()
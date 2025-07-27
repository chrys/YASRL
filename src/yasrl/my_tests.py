import asyncio
import os
from pathlib import Path
import psycopg2

# Load environment variables explicitly
from dotenv import load_dotenv
load_dotenv()

def reset_database_table():
    """Drop and recreate the yasrl_chunks table with correct dimensions."""
    conn = psycopg2.connect(os.getenv("POSTGRES_URI"))
    try:
        with conn.cursor() as cursor:
            print("🔄 Resetting database table...")
            cursor.execute("DROP TABLE IF EXISTS yasrl_chunks;")
            print("✅ Old table dropped successfully!")
        conn.commit()
    except Exception as e:
        print(f"❌ Failed to reset database table: {e}")
    finally:
        conn.close()

async def demonstrate_yasrl_system():
    """
    Simple demonstration of the YASRL RAG pipeline system.
    Shows indexing documents and asking questions.
    """
    try:
        # Reset the database table first
        reset_database_table()
        # Import your pipeline
        from yasrl.pipeline import RAGPipeline
        
        # Verify required environment variables are loaded
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ GOOGLE_API_KEY not found in environment")
            print("Make sure your .env file is in the project root directory")
            return
            
        if not os.getenv("POSTGRES_URI"):
            print("❌ POSTGRES_URI not found in environment")  
            print("Make sure your .env file is in the project root directory")
            return
        
        print("🚀 YASRL System Demonstration")
        print("=" * 40)
        
         # Debug: Check what's in the config
        from yasrl.config.manager import ConfigurationManager
        config_manager = ConfigurationManager()
        config = config_manager.load_config()
        print(f"Debug - config.google_api_key: {getattr(config, 'google_api_key', 'NOT FOUND')}")
        print(f"Debug - Environment GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY', 'NOT FOUND')}")

        
        # Initialize the pipeline
        print("\n1. Initializing RAG Pipeline...")
        pipeline = await RAGPipeline.create(
            llm="gemini",           # Using Gemini for LLM
            embed_model="gemini"    # Using Gemini for embeddings
        )
        print("✅ Pipeline initialized successfully!")
        
        import logging
        logging.info("This log should appear in yasrl.log if file logging is enabled.")
        
        # Create a sample document for demonstration
        demo_docs_path = Path("demo_docs")
        demo_docs_path.mkdir(exist_ok=True)
        
        sample_doc = demo_docs_path / "sample.txt"
        with open(sample_doc, "w") as f:
            f.write("""
            YASRL (Yet Another Simple RAG Library) is a Python library designed 
            for building Retrieval-Augmented Generation pipelines. It provides 
            a streamlined, developer-first experience by acting as a high-level 
            orchestration layer over the LlamaIndex framework.
            
            Key features include:
            - Simple API with opinionated defaults
            - Support for multiple LLM providers (OpenAI, Gemini, Ollama)
            - PostgreSQL vector storage with PGVector
            - Built-in evaluation framework
            - Comprehensive error handling
            """)
        
        # Index the document
        print(f"\n2. Indexing document: {sample_doc}")
        await pipeline.index(str(demo_docs_path))
        print("✅ Documents indexed successfully!")
        
        # Get pipeline statistics
        stats = await pipeline.get_statistics()
        print(f"📊 Pipeline Statistics: {stats}")
        
        # Ask questions
        questions = [
            "What is YASRL?",
            "What are the key features of YASRL?",
            "Which LLM providers does YASRL support?"
        ]
        
        print("\n3. Asking questions...")
        for i, question in enumerate(questions, 1):
            print(f"\n❓ Question {i}: {question}")
            
            # Get answer from the pipeline
            result = await pipeline.ask(question)
            
            print(f"💡 Answer: {result.answer}")
            print(f"📚 Sources found: {len(result.source_chunks)}")
            
            # Show source information
            for j, chunk in enumerate(result.source_chunks[:2], 1):  # Show first 2 sources
                source = chunk.metadata.get('source', 'Unknown')
                score = chunk.score or 0.0
                print(f"   Source {j}: {Path(source).name} (relevance: {score:.3f})")
        
        print("\n🎉 YASRL demonstration completed successfully!")
        
        # Cleanup
        sample_doc.unlink()
        demo_docs_path.rmdir()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure YASRL is properly installed and all dependencies are available.")
    
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")

def run_demo():
    """Synchronous wrapper to run the async demonstration."""
    asyncio.run(demonstrate_yasrl_system())

if __name__ == "__main__":
    run_demo()
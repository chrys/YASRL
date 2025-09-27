import psycopg2
import os

def get_db_connection():
    """Establishes a connection to the PostgreSQL database using DATABASE_URL."""
    try:
        # It's better to use a single DATABASE_URL environment variable
        # as it's a common practice and supported by many services.
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            print("❌ DATABASE_URL environment variable is not set.")
            return None
        conn = psycopg2.connect(db_url)
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Error connecting to the database: {e}")
        return None

def log_feedback(chatbot_answer: str, rating: str):
    """
    Logs feedback to the chatbot_feedback table.

    Args:
        chatbot_answer (str): The answer provided by the chatbot.
        rating (str): The feedback rating, either 'GOOD' or 'BAD'.
    """
    conn = get_db_connection()
    if conn is None:
        print("Skipping feedback logging due to database connection issue.")
        return

    try:
        with conn.cursor() as cur:
            project_name = os.environ.get("PROJECT_NAME", "yasrl")
            # As per schema, feedback_text is NOT NULL. We'll use the rating value.
            feedback_text = rating.upper()

            sql_query = """
                INSERT INTO chatbot_feedback (project, chatbot_answer, feedback_text, rating)
                VALUES (%s, %s, %s, %s);
            """
            cur.execute(sql_query, (project_name, chatbot_answer, feedback_text, rating.upper()))

        conn.commit()
        print(f"✅ Feedback logged successfully: '{rating}'")

    except Exception as e:
        print(f"❌ Error logging feedback: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
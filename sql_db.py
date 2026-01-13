import sqlite3
import os

DB_NAME = "students.db"

def init_db():
    """Initializes the database with dummy data if it doesn't exist."""
    if os.path.exists(DB_NAME):
        # Optional: Remove to reset on restart, or just keep it.
        # For this demo, let's keep it if it exists, or forcing regeneration ensures data integrity.
        os.remove(DB_NAME)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            course TEXT NOT NULL,
            fees REAL,
            enrollment_date TEXT,
            gpa REAL
        )
    ''')

    # Dummy Data
    students = [
        ("Vignesh Ladar", "Master of AI", 25000.0, "2025-01-15", 3.8),
        ("Sarah Jones", "Master of Data Science", 22000.0, "2025-02-01", 3.9),
        ("Mike Ross", "Bachelor of Law", 18000.0, "2024-07-01", 3.5),
        ("Rachel Green", "Master of AI", 25000.0, "2025-01-15", 3.2),
        ("Harvey Specter", "Master of Business", 30000.0, "2024-03-01", 4.0),
        ("Louis Litt", "Master of Finance", 28000.0, "2024-03-01", 3.7),
        ("Jessica Pearson", "PhD Computer Science", 15000.0, "2023-01-01", 4.0),
        ("Donna Paulsen", "Master of Arts", 12000.0, "2025-02-20", 3.9),
    ]

    cursor.executemany('''
        INSERT INTO students (name, course, fees, enrollment_date, gpa)
        VALUES (?, ?, ?, ?, ?)
    ''', students)

    conn.commit()
    conn.close()
    print(f"Initialized {DB_NAME} with dummy data.")

def query_database(query: str):
    """
    Executes a read-only SQL query against the students database.
    WARNING: This is valid for a demo. In production, use parameterized queries/ORM to prevent injection.
    """
    # Safety Check: only allow SELECT
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        conn.close()

        if not results:
            return "No results found."

        # Format as list of dicts for LLM readability
        formatted_results = []
        for row in results:
            formatted_results.append(dict(zip(columns, row)))
        
        return str(formatted_results)

    except Exception as e:
        return f"SQL Error: {str(e)}"

# Run init on import (or manually)
if __name__ == "__main__":
    init_db()
    print(query_database("SELECT * FROM students"))
else:
    # Ensure DB exists when imported by the app
    if not os.path.exists(DB_NAME):
        init_db()

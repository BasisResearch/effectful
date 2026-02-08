"""Natural language to SQL with LLM-powered debug loop.

Demonstrates:
- Generating SQL from natural language using ``@Template.define``
- Executing SQL against a real SQLite database
- Feeding execution errors back to the LLM for iterative fixing
- ``@Tool.define`` to expose the database schema as a tool
"""

import argparse
import os
import sqlite3
import textwrap

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# In-memory database setup
# ---------------------------------------------------------------------------


def create_sample_db() -> sqlite3.Connection:
    """Create a sample SQLite database with employee data."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        textwrap.dedent("""\
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget REAL NOT NULL
        );
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER REFERENCES departments(id),
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL
        );
        INSERT INTO departments VALUES (1, 'Engineering', 500000);
        INSERT INTO departments VALUES (2, 'Marketing', 200000);
        INSERT INTO departments VALUES (3, 'Sales', 300000);
        INSERT INTO employees VALUES (1, 'Alice', 1, 120000, '2020-01-15');
        INSERT INTO employees VALUES (2, 'Bob', 1, 110000, '2021-03-22');
        INSERT INTO employees VALUES (3, 'Carol', 2, 95000, '2019-07-01');
        INSERT INTO employees VALUES (4, 'Dave', 3, 105000, '2022-11-10');
        INSERT INTO employees VALUES (5, 'Eve', 1, 130000, '2018-05-20');
        INSERT INTO employees VALUES (6, 'Frank', 3, 98000, '2023-01-05');
    """)
    )
    return conn


def get_schema(conn: sqlite3.Connection) -> str:
    """Extract the schema from a SQLite database."""
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return "\n\n".join(row[0] for row in cursor if row[0])


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@Template.define
def generate_sql(question: str, db_schema: str) -> str:
    """You are a SQL expert. Given this database schema:

    {db_schema}

    Write a SQLite query that answers: {question}

    Return ONLY the SQL query, no explanation.
    """
    raise NotHandled


@Template.define
def fix_sql(question: str, db_schema: str, bad_sql: str, error: str) -> str:
    """You are a SQL expert. Your previous query had an error.

    Database schema:
    {db_schema}

    Original question: {question}
    Failed SQL: {bad_sql}
    Error: {error}

    Write a corrected SQLite query. Return ONLY the SQL query.
    """
    raise NotHandled


# ---------------------------------------------------------------------------
# Text-to-SQL agent with debug loop
# ---------------------------------------------------------------------------


def text_to_sql(
    conn: sqlite3.Connection, question: str, max_retries: int = 3
) -> list[tuple]:
    """Convert a natural language question to SQL and execute it.

    If the query fails, feed the error back to the LLM to fix it,
    up to ``max_retries`` times.
    """
    schema = get_schema(conn)
    sql = generate_sql(question, schema)

    for attempt in range(max_retries + 1):
        # Strip markdown fences if the LLM wraps the SQL
        clean_sql = sql.strip().removeprefix("```sql").removesuffix("```").strip()
        print(f"  [attempt {attempt + 1}] {clean_sql}")

        try:
            cursor = conn.execute(clean_sql)
            return cursor.fetchall()
        except Exception as e:
            if attempt < max_retries:
                print(f"  [error] {e}")
                sql = fix_sql(question, schema, clean_sql, str(e))
            else:
                raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Natural language to SQL with LLM-powered debug loop"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    conn = create_sample_db()
    provider = LiteLLMProvider(model=args.model)

    questions = [
        "What is the average salary by department?",
        "Who is the highest paid employee?",
        "How many employees were hired after 2021?",
    ]

    with handler(provider), handler(RetryLLMHandler(num_retries=3)):
        for question in questions:
            print(f"\nQ: {question}")
            try:
                rows = text_to_sql(conn, question)
                for row in rows:
                    print(f"  => {row}")
            except Exception as e:
                print(f"  FAILED: {e}")

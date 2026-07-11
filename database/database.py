"""
database/database.py
=====================
Low-level SQLite engine: where the database file lives, how to open a
connection, and how to create it from schema.sql.

This module is deliberately self-contained — it does NOT import
config/settings.py or anything else from the existing pipeline, so
this whole database/ package stays fully decoupled until Phase 2
explicitly wires it in. Nothing here is imported anywhere else in the
project yet.

database/db_manager.py builds the actual insert_*/get_*/search_*
business logic on top of get_connection() below.
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

# ---------------------------------------------------------------------
# Paths — computed independently of config/settings.py on purpose (see
# module docstring). Layout mirrors the existing project's data/
# folder convention without touching it.
# ---------------------------------------------------------------------
_DATABASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DATABASE_DIR)

DB_PATH = os.path.join(_PROJECT_ROOT, "data", "frouros.db")
SCHEMA_PATH = os.path.join(_DATABASE_DIR, "schema.sql")


def create_database(db_path: str = DB_PATH) -> None:
    """Ensure the database file's parent directory exists and the
    SQLite file itself is created. Safe to call repeatedly."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    # Opening a connection is enough to create the file if missing.
    conn = sqlite3.connect(db_path)
    conn.close()


def create_tables(db_path: str = DB_PATH, schema_path: str = SCHEMA_PATH) -> None:
    """Execute schema.sql against the database, creating every table /
    index if it doesn't already exist. Safe to call repeatedly —
    every statement in schema.sql uses IF NOT EXISTS."""
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def initialize_database(db_path: str = DB_PATH, schema_path: str = SCHEMA_PATH) -> None:
    """Convenience one-shot setup: create_database() + create_tables()."""
    create_database(db_path)
    create_tables(db_path, schema_path)


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open a new SQLite connection configured for this project:
      - row_factory = sqlite3.Row, so rows behave like read-only dicts
        (row["event_id"]) instead of positional tuples.
      - PRAGMA foreign_keys = ON, since SQLite disables FK enforcement
        by default per-connection — required for ON DELETE CASCADE /
        ON DELETE SET NULL in schema.sql to actually take effect.

    Callers are responsible for closing the connection (db_manager.py
    does this via the connection_scope() context manager below).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection_scope(db_path: str = DB_PATH) -> Iterator[sqlite3.Connection]:
    """
    Context manager that opens a connection, commits on success, rolls
    back on any exception, and always closes the connection:

        with connection_scope() as conn:
            conn.execute(...)
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

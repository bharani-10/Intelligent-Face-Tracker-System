import os
from uuid import uuid4

import numpy as np
import psycopg
import pytest

from app.database import DatabaseManager


pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="Set DATABASE_URL to run PostgreSQL integration tests",
)


def database_or_skip():
    try:
        return DatabaseManager()
    except psycopg.Error as error:
        pytest.skip(f"PostgreSQL is unavailable for integration tests: {error}")


def test_database_creation_and_tables(tmp_path):
    database = database_or_skip()

    with database._connect() as connection:
        tables = {
            row["table_name"]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
        }
    assert {"visitors", "events"}.issubset(tables)


def test_insert_retrieve_visitor_and_embedding(tmp_path):
    database = database_or_skip()
    embedding = np.full(512, 0.1, dtype=np.float32)
    visitor_id = f"TEST_VISITOR_{uuid4().hex}"
    count_before = database.get_unique_visitor_count()

    database.register_visitor(visitor_id, embedding, "2026-10-04T10:00:00")
    visitor = database.get_visitor(visitor_id)

    assert visitor["visitor_id"] == visitor_id
    np.testing.assert_allclose(visitor["embedding"], embedding)
    assert database.get_unique_visitor_count() == count_before + 1


def test_insert_event_and_history(tmp_path):
    database = database_or_skip()
    visitor_id = f"TEST_VISITOR_{uuid4().hex}"
    database.register_visitor(visitor_id, np.ones(512, dtype=np.float32), "2026-10-04T10:00:00")
    entries_before = database.get_event_count("ENTRY")

    event_id = database.log_event(
        visitor_id, "ENTRY", "2026-10-04T10:01:00", "entries/entry.jpg", 7
    )
    history = database.get_event_history(visitor_id)

    assert event_id > 0
    assert history[0]["event_type"] == "ENTRY"
    assert history[0]["track_id"] == 7
    assert database.get_event_count("ENTRY") == entries_before + 1
    assert visitor_id in database.get_currently_present_visitors()
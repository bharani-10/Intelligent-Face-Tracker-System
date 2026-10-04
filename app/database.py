import logging
import os
from typing import Any, Optional

import numpy as np
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)
load_dotenv()


class DatabaseConfigurationError(RuntimeError):
    """Raised when PostgreSQL configuration is missing or invalid."""


class DatabaseManager:
    def __init__(self, database_url: Optional[str] = None, env_var: str = "DATABASE_URL"):
        self.database_url = database_url or os.getenv(env_var)
        if not self.database_url:
            raise DatabaseConfigurationError(
                f"PostgreSQL connection is required. Set {env_var} to a PostgreSQL URL."
            )
        self.initialize_schema()

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self.database_url, row_factory=dict_row)
        except psycopg.Error as error:
            logger.error("Database connection failed: %s", error)
            raise

    def initialize_schema(self) -> None:
        try:
            with self._connect() as connection:
                with connection.transaction():
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS visitors (
                            id BIGSERIAL PRIMARY KEY,
                            visitor_id TEXT UNIQUE NOT NULL,
                            embedding BYTEA NOT NULL,
                            first_seen TIMESTAMPTZ NOT NULL,
                            last_seen TIMESTAMPTZ NOT NULL
                        )
                        """
                    )
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS events (
                            event_id BIGSERIAL PRIMARY KEY,
                            visitor_id TEXT NOT NULL REFERENCES visitors(visitor_id),
                            event_type TEXT NOT NULL CHECK (event_type IN ('ENTRY', 'EXIT')),
                            timestamp TIMESTAMPTZ NOT NULL,
                            image_path TEXT NOT NULL DEFAULT '',
                            track_id BIGINT
                        )
                        """
                    )
                    connection.execute(
                        "CREATE INDEX IF NOT EXISTS idx_events_visitor_id ON events(visitor_id)"
                    )
                    connection.execute(
                        "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)"
                    )
        except psycopg.Error:
            logger.exception("Database schema initialization failed")
            raise

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(embedding_bytes: bytes) -> np.ndarray:
        return np.frombuffer(bytes(embedding_bytes), dtype=np.float32)

    def register_visitor(self, visitor_id: str, embedding: np.ndarray, timestamp: str) -> None:
        try:
            with self._connect() as connection:
                with connection.transaction():
                    connection.execute(
                        """
                        INSERT INTO visitors (visitor_id, embedding, first_seen, last_seen)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (visitor_id, self.serialize_embedding(embedding), timestamp, timestamp),
                    )
        except psycopg.Error:
            logger.exception("Database error registering visitor %s", visitor_id)
            raise

    def update_last_seen(self, visitor_id: str, timestamp: str) -> None:
        try:
            with self._connect() as connection:
                with connection.transaction():
                    connection.execute(
                        "UPDATE visitors SET last_seen = %s WHERE visitor_id = %s",
                        (timestamp, visitor_id),
                    )
        except psycopg.Error:
            logger.exception("Database error updating visitor %s", visitor_id)
            raise

    def log_event(
        self,
        visitor_id: str,
        event_type: str,
        timestamp: str,
        image_path: str = "",
        track_id: Optional[int] = None,
    ) -> int:
        try:
            with self._connect() as connection:
                with connection.transaction():
                    row = connection.execute(
                        """
                        INSERT INTO events (visitor_id, event_type, timestamp, image_path, track_id)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING event_id
                        """,
                        (visitor_id, event_type, timestamp, image_path, track_id),
                    ).fetchone()
                    return int(row["event_id"])
        except psycopg.Error:
            logger.exception("Database error logging %s for %s", event_type, visitor_id)
            raise

    insert_event = log_event

    def retrieve_visitors(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT id, visitor_id, first_seen, last_seen FROM visitors ORDER BY id"
                ).fetchall()
            )

    def get_visitor(self, visitor_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, visitor_id, embedding, first_seen, last_seen FROM visitors WHERE visitor_id = %s",
                (visitor_id,),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["embedding"] = self.deserialize_embedding(result["embedding"])
            return result

    def get_visitor_embeddings(self) -> list[tuple[str, np.ndarray]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT visitor_id, embedding FROM visitors ORDER BY id"
            ).fetchall()
            return [
                (row["visitor_id"], self.deserialize_embedding(row["embedding"]))
                for row in rows
            ]

    retrieve_visitor_embeddings = get_visitor_embeddings

    def get_event_history(self, visitor_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if visitor_id is None:
                query = "SELECT * FROM events ORDER BY timestamp, event_id"
                rows = connection.execute(query).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM events WHERE visitor_id = %s ORDER BY timestamp, event_id",
                    (visitor_id,),
                ).fetchall()
            return list(rows)

    get_visitor_history = get_event_history

    def get_unique_visitor_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) AS count FROM visitors").fetchone()["count"])

    def get_event_count(self, event_type: Optional[str] = None) -> int:
        with self._connect() as connection:
            if event_type is None:
                row = connection.execute("SELECT COUNT(*) AS count FROM events").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM events WHERE event_type = %s",
                    (event_type,),
                ).fetchone()
            return int(row["count"])

    def get_total_entry_events(self) -> int:
        return self.get_event_count("ENTRY")

    def get_total_exit_events(self) -> int:
        return self.get_event_count("EXIT")

    def get_currently_present_visitors(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT visitor_id
                FROM (
                    SELECT DISTINCT ON (visitor_id) visitor_id, event_type
                    FROM events
                    ORDER BY visitor_id, event_id DESC
                ) latest
                WHERE event_type = 'ENTRY'
                ORDER BY visitor_id
                """
            ).fetchall()
            return [row["visitor_id"] for row in rows]
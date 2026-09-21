"""Neo4j connection management for Veritas.

This module wraps the official ``neo4j`` Python driver in a small, reusable
:class:`Neo4jConnection` helper that:

* validates connectivity eagerly (fail fast with a clear message),
* exposes convenience methods for read/write Cypher execution,
* supports use as a context manager so sessions/drivers are always closed.

All database access in the ingestion and retrieval layers goes through this
class so connection handling and error catching live in exactly one place.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

from veritas.config import Neo4jSettings, get_settings

logger = logging.getLogger(__name__)


class Neo4jConnectionError(RuntimeError):
    """Raised when Veritas cannot establish or use a Neo4j connection."""


class Neo4jConnection:
    """Thin, safe wrapper around the Neo4j driver.

    Parameters
    ----------
    settings:
        Optional :class:`~veritas.config.Neo4jSettings`. When omitted, the
        global application settings are used.

    Examples
    --------
    >>> with Neo4jConnection() as conn:
    ...     conn.execute_write("CREATE (:Ping {ts: timestamp()})")
    """

    def __init__(self, settings: Neo4jSettings | None = None) -> None:
        self._settings = settings or get_settings().neo4j
        self._driver: Driver | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> "Neo4jConnection":
        """Create the driver and verify connectivity.

        Raises
        ------
        Neo4jConnectionError
            If the server is unreachable or authentication fails.
        """
        if self._driver is not None:
            return self

        try:
            self._driver = GraphDatabase.driver(
                self._settings.uri,
                auth=(
                    self._settings.username,
                    self._settings.password.get_secret_value(),
                ),
            )
            # Eagerly confirm the server is reachable and credentials are valid.
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self._settings.uri)
        except AuthError as exc:
            raise Neo4jConnectionError(
                f"Authentication failed for user '{self._settings.username}'. "
                "Check NEO4J_USERNAME / NEO4J_PASSWORD."
            ) from exc
        except ServiceUnavailable as exc:
            raise Neo4jConnectionError(
                f"Neo4j is not reachable at '{self._settings.uri}'. "
                "Is the database running and the URI correct?"
            ) from exc
        except Neo4jError as exc:  # pragma: no cover - defensive catch-all
            raise Neo4jConnectionError(f"Unexpected Neo4j error: {exc}") from exc

        return self

    @property
    def driver(self) -> Driver:
        """Return the live driver, connecting lazily if needed."""
        if self._driver is None:
            self.connect()
        assert self._driver is not None  # for type-checkers
        return self._driver

    def close(self) -> None:
        """Close the underlying driver and release all pooled connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed.")

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #
    def execute_read(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a read query and return a list of plain-dict records."""
        return self._run(query, parameters, write=False)

    def execute_write(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a write query inside a managed transaction."""
        return self._run(query, parameters, write=True)

    def _run(
        self, query: str, parameters: dict[str, Any] | None, *, write: bool
    ) -> list[dict[str, Any]]:
        """Execute Cypher using a managed transaction and normalise results."""
        params = parameters or {}
        try:
            with self.driver.session(database=self._settings.database) as session:
                def _work(tx: Any) -> list[dict[str, Any]]:
                    result = tx.run(query, **params)
                    return [record.data() for record in result]

                if write:
                    return session.execute_write(_work)
                return session.execute_read(_work)
        except Neo4jError as exc:
            logger.error("Cypher execution failed: %s\nQuery: %s", exc, query)
            raise Neo4jConnectionError(f"Cypher execution failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Context manager protocol
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "Neo4jConnection":
        return self.connect()

    def __exit__(self, *_exc: object) -> None:
        self.close()

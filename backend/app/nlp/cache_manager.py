"""SQLite-backed cache manager — avoid re-computing expensive operations.

CacheManager provides a thin key→value store backed by SQLite (via the Python
standard library ``sqlite3`` module — no extra dependencies).

Cached items:
- Search results from external APIs
- Extracted summaries
- Embedding vectors (JSON-serialised numpy arrays)
- Any other JSON-serialisable object

Features:
- Optional TTL (time-to-live in seconds)
- Namespaced keys (prefix + user-supplied key)
- Thread-safe via Python's sqlite3 check_same_thread=False + a module-level
  lock for write operations
- Automatic schema creation on first use
- ``clear()`` and ``clear_namespace()`` helpers for housekeeping
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "nlp_cache.db"
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cache_entries (
    namespace TEXT NOT NULL,
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    PRIMARY KEY (namespace, key)
)
"""
_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_expires ON cache_entries(expires_at)"

_WRITE_LOCK = threading.Lock()


class CacheManager:
    """SQLite-backed key-value cache.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to
        ``<backend root>/nlp_cache.db``.
    default_ttl:
        Default TTL in seconds.  ``None`` means entries never expire.
    namespace:
        A string prefix applied to all keys (allows logical separation of
        different cache domains, e.g. ``"search"``, ``"summary"``, ``"embed"``).
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        default_ttl: Optional[float] = None,
        namespace: str = "default",
    ) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.default_ttl = default_ttl
        self.namespace = namespace
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for *key*, or ``None`` if missing/expired."""
        conn = self._get_conn()
        hashed = self._hash(key)
        now = time.time()
        try:
            cursor = conn.execute(
                "SELECT value, expires_at FROM cache_entries "
                "WHERE namespace = ? AND key = ?",
                (self.namespace, hashed),
            )
            row = cursor.fetchone()
        except sqlite3.Error as exc:
            logger.debug("Cache get error: %s", exc)
            return None

        if row is None:
            return None

        value_str, expires_at = row
        if expires_at is not None and expires_at < now:
            self.delete(key)
            return None

        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            return value_str

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Store *value* under *key*.

        Parameters
        ----------
        key:
            Cache key (will be hashed internally).
        value:
            Any JSON-serialisable object.
        ttl:
            TTL override in seconds.  Uses ``default_ttl`` if ``None``.
        """
        hashed = self._hash(key)
        now = time.time()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None

        try:
            value_str = json.dumps(value)
        except (TypeError, ValueError):
            logger.debug("CacheManager.set: value for key %r is not JSON-serialisable; skipping.", key)
            return

        with _WRITE_LOCK:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache_entries "
                    "(namespace, key, value, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                    (self.namespace, hashed, value_str, now, expires_at),
                )
                conn.commit()
            except sqlite3.Error as exc:
                logger.debug("Cache set error: %s", exc)

    def exists(self, key: str) -> bool:
        """Return ``True`` if *key* exists and has not expired."""
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        """Remove *key* from the cache."""
        hashed = self._hash(key)
        with _WRITE_LOCK:
            conn = self._get_conn()
            try:
                conn.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND key = ?",
                    (self.namespace, hashed),
                )
                conn.commit()
            except sqlite3.Error as exc:
                logger.debug("Cache delete error: %s", exc)

    def clear(self) -> None:
        """Remove ALL entries (all namespaces)."""
        with _WRITE_LOCK:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
            except sqlite3.Error as exc:
                logger.debug("Cache clear error: %s", exc)

    def clear_namespace(self) -> None:
        """Remove all entries in the current namespace."""
        with _WRITE_LOCK:
            conn = self._get_conn()
            try:
                conn.execute(
                    "DELETE FROM cache_entries WHERE namespace = ?",
                    (self.namespace,),
                )
                conn.commit()
            except sqlite3.Error as exc:
                logger.debug("Cache clear_namespace error: %s", exc)

    def purge_expired(self) -> int:
        """Delete all expired entries.  Returns the number of rows removed."""
        now = time.time()
        with _WRITE_LOCK:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                )
                conn.commit()
                return cursor.rowcount
            except sqlite3.Error:
                return 0

    def cache_key(self, *parts: str) -> str:
        """Build a canonical cache key from multiple string parts."""
        return "|".join(str(p) for p in parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Ensure the cache table and index exist."""
        try:
            conn = self._get_conn()
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            conn.commit()
        except sqlite3.Error as exc:
            logger.warning("CacheManager could not initialise DB at %s: %s", self._db_path, exc)

    def _get_conn(self) -> sqlite3.Connection:
        """Return (or lazily open) the SQLite connection."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        return self._conn

    @staticmethod
    def _hash(key: str) -> str:
        """Return a 32-character (128-bit) hex digest of *key*.

        Using the first 32 hex characters (128 bits) of SHA-256 gives a
        negligible collision probability for typical cache sizes (< 10 million
        entries).  The shortened key keeps storage compact without meaningful
        loss of collision resistance for this use-case.
        """
        return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:32]

    def __del__(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

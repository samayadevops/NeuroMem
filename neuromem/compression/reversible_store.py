"""Local reversible storage layer for compressed memory artifacts.

The :class:`ReversibleStore` persists the **original, uncompressed**
content associated with a compressed memory so it can be recovered
later.  It is the "undo" backbone of the compression pipeline: while
:class:`~neuromem.compression.models.MemorySnapshot` stores a compact
summary plus a ``raw_reference`` string, this module is what that
reference resolves to.

Design rationale
----------------
- **File-based, sharded key-value layout.**  Content is addressed by a
  caller-supplied ``memory_id`` and laid out on disk under a two-character
  shard directory derived from the id.  Sharding keeps any single
  directory small even at scale (hundreds of thousands of entries).
- **Atomic writes.**  Each entry is written to a temporary file in the
  same directory and then moved into place with :func:`os.replace`, which
  is atomic on POSIX and Windows.  A crash mid-write therefore never
  produces a truncated or half-written entry.
- **Integrity sidecar.**  Alongside every ``.raw`` payload a small ``.meta``
  JSON file records the original ``memory_id``, byte size, SHA-256 digest,
  and creation timestamp.  :meth:`ReversibleStore.retrieve_original`
  re-computes the digest and raises on mismatch, detecting on-disk
  corruption or tampering.
- **Explicit UTF-8 everywhere.**  Content is always encoded/decoded as
  UTF-8 regardless of the host platform's default encoding, avoiding the
  classic Windows ``cp1252`` pitfalls.
- **Decoupled from the backends.**  The store knows nothing about ChromaDB
  or Kuzu.  Its only contract is ``memory_id -> raw_content``.  Graph
  nodes and vector records carry the ``memory_id`` in their
  properties/metadata; the compressed ``MemorySnapshot.raw_reference``
  field holds the very same id, forming the link between the compact
  representation (in ChromaDB/Kuzu) and the full fidelity original (here).

On-disk layout
--------------
::

    <storage_path>/
    └── objects/
        ├── sn/
        │   ├── snap_abc123.raw      <- raw UTF-8 text
        │   └── snap_abc123.meta     <- JSON integrity sidecar
        └── be/
            ├── belief_xyz.raw
            └── belief_xyz.meta

Exception model
---------------
All errors derive from :class:`~neuromem.core.exceptions.NeuroMemError`,
so a blanket ``except NeuroMemError`` still catches them.  The two
domain-specific exceptions are :class:`MemoryNotFoundError` (the common
"key missing" case) and :class:`InvalidMemoryIdError` (rejected ids).

Future extension
----------------
The public methods form a natural contract.  If alternative backends are
needed (SQLite, S3, etc.) they can be extracted into an abstract
``BaseReversibleStore`` and this file becomes the file-based reference
implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from loguru import logger

from neuromem.core.exceptions import NeuroMemError


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_OBJECTS_DIR = "objects"
"""Name of the directory under the storage root holding all entries."""

_RAW_SUFFIX = ".raw"
"""Filename suffix for the raw content payload."""

_META_SUFFIX = ".meta"
"""Filename suffix for the JSON integrity sidecar."""

_SHARD_WIDTH = 2
"""Number of leading characters of the sanitised id used as a shard dir."""

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]")
"""Characters permitted in a memory_id without replacement."""

_SAFE_ID_MIN_LEN = 1
"""Minimum length of a sanitised memory_id."""

_SAFE_ID_MAX_LEN = 512
"""Maximum length of a memory_id (post-sanitisation) — filesystem guard."""


# ═══════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════

class ReversibleStoreError(NeuroMemError):
    """Base exception for all :class:`ReversibleStore` failures.

    All store-specific errors derive from this class (and transitively
    from :class:`~neuromem.core.exceptions.NeuroMemError`), so callers
    can either target the store layer precisely or catch
    ``NeuroMemError`` for blanket handling.
    """


class MemoryNotFoundError(ReversibleStoreError):
    """No original content is stored under the requested ``memory_id``.

    Parameters
    ----------
    memory_id:
        The identifier that was looked up but not found.
    """

    def __init__(self, memory_id: str) -> None:
        self.memory_id: str = memory_id
        super().__init__(
            f"No original content stored for memory_id={memory_id!r}",
            context={"memory_id": memory_id},
        )


class InvalidMemoryIdError(ReversibleStoreError):
    """A ``memory_id`` failed validation and was rejected.

    Parameters
    ----------
    memory_id:
        The offending identifier value.
    reason:
        Human-readable explanation of why it was rejected.
    """

    def __init__(self, memory_id: object, reason: str) -> None:
        self.memory_id: object = memory_id
        self.reason: str = reason
        super().__init__(
            f"Invalid memory_id {memory_id!r}: {reason}",
            context={"memory_id": memory_id, "reason": reason},
        )


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _utcnow() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _validate_and_sanitize(memory_id: str) -> str:
    """Validate *memory_id* and return a filesystem-safe sanitised form.

    Non-permitted characters (anything outside ``[A-Za-z0-9._-]``) are
    replaced with ``_``.  The original id is preserved verbatim inside
    the ``.meta`` sidecar for traceability, so sanitisation only affects
    the on-disk filename — it never loses information.

    Parameters
    ----------
    memory_id:
        The caller-supplied identifier.

    Returns
    -------
    str
        A sanitised, filesystem-safe filename stem.

    Raises
    ------
    InvalidMemoryIdError
        If the id is empty, not a string, or sanitises to an empty or
        over-long string.
    """
    if not isinstance(memory_id, str):
        raise InvalidMemoryIdError(memory_id, "memory_id must be a string")
    if not memory_id or not memory_id.strip():
        raise InvalidMemoryIdError(memory_id, "memory_id must not be empty")

    sanitized = _SAFE_ID_RE.sub("_", memory_id.strip())
    if len(sanitized) < _SAFE_ID_MIN_LEN:
        raise InvalidMemoryIdError(memory_id, "memory_id is empty after sanitisation")
    if len(sanitized) > _SAFE_ID_MAX_LEN:
        raise InvalidMemoryIdError(
            memory_id,
            f"memory_id exceeds maximum length of {_SAFE_ID_MAX_LEN} characters",
        )
    return sanitized


def _entry_paths(root: Path, sanitized: str) -> tuple[Path, Path, Path]:
    """Return ``(shard_dir, raw_path, meta_path)`` for a sanitised id.

    The shard directory is the first :data:`_SHARD_WIDTH` characters of
    the sanitised id, falling back to ``"zz"`` if the id is shorter than
    the shard width.
    """
    shard = sanitized[:_SHARD_WIDTH] if len(sanitized) >= _SHARD_WIDTH else "zz"
    shard_dir = root / _OBJECTS_DIR / shard
    return shard_dir, shard_dir / f"{sanitized}{_RAW_SUFFIX}", shard_dir / f"{sanitized}{_META_SUFFIX}"


def _sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
# ReversibleStore
# ═══════════════════════════════════════════════════════════════════════

class ReversibleStore:
    """Local, file-based archive of original (uncompressed) memory content.

    The store maps a ``memory_id`` to its raw text payload and provides
    symmetric ``store_original`` / ``retrieve_original`` operations.  It
    is the durability layer that makes compression *reversible*: a
    :class:`~neuromem.compression.models.MemorySnapshot` can be expanded
    back to full fidelity at any time by resolving its
    ``raw_reference`` through this store.

    The store is **storage-agnostic with respect to the cognitive
    backends** — it has no dependency on ChromaDB or Kuzu.  Those
    backends reference a ``memory_id`` in their node properties /
    vector metadata, and the compressed snapshot carries the same id in
    its ``raw_reference`` field.  Decoupling in this direction means the
    store can be developed, tested, and evolved independently.

    Lifecycle
    ---------
    1. Construct with a filesystem path.
    2. Call :meth:`initialize` to create the directory tree (idempotent).
    3. ``store_original`` / ``retrieve_original`` / ``exists`` / ``delete``.
    4. Call :meth:`close` when finished (also usable as a context manager).

    Parameters
    ----------
    storage_path:
        Filesystem directory used as the archive root.  Created on
        :meth:`initialize` if it does not exist.  Accepts ``str`` or
        :class:`~pathlib.Path`.
    encoding:
        Text encoding used for all payloads.  Defaults to ``"utf-8"``;
        almost never needs changing.

    Example
    -------
    ::

        with ReversibleStore("./neuromem_data/raw_archive") as store:
            store.initialize()
            store.store_original("snap_abc123", long_raw_conversation)
            ...
            original = store.retrieve_original("snap_abc123")
            assert original == long_raw_conversation
    """

    __slots__ = ("_path", "_encoding", "_state", "_objects_root")

    def __init__(
        self,
        storage_path: str | Path,
        *,
        encoding: str = "utf-8",
    ) -> None:
        self._path: Path = Path(storage_path)
        self._encoding: str = encoding
        self._state: str = "uninitialized"
        self._objects_root: Path = self._path / _OBJECTS_DIR

    # ── Lifecycle ────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create the on-disk directory tree.

        Idempotent — calling on an already-initialised store is a no-op.

        Raises
        ------
        ReversibleStoreError
            If the storage path exists but is not a directory, or if
            directory creation fails for any other reason.
        """
        if self._state in {"ready", "closed"}:
            self._state = "ready"
            return
        try:
            if self._path.exists() and not self._path.is_dir():
                raise ReversibleStoreError(
                    f"Storage path {self._path!s} exists but is not a directory",
                    context={"storage_path": str(self._path)},
                )
            self._path.mkdir(parents=True, exist_ok=True)
            self._objects_root.mkdir(parents=True, exist_ok=True)
        except ReversibleStoreError:
            raise
        except OSError as exc:
            raise ReversibleStoreError(
                f"Failed to initialise reversible store at {self._path!s}: {exc}",
                context={"storage_path": str(self._path), "os_error": str(exc)},
            ) from exc
        self._state = "ready"
        logger.debug(
            "ReversibleStore initialised at {} (encoding={})",
            self._path,
            self._encoding,
        )

    def close(self) -> None:
        """Release resources.

        The file-based store opens no long-lived handles, so this is
        effectively a no-op that simply marks the store as closed.
        Idempotent.
        """
        self._state = "closed"

    @property
    def storage_path(self) -> Path:
        """Absolute path to the archive root."""
        return self._path

    @property
    def state(self) -> str:
        """Current lifecycle state: ``uninitialized`` / ``ready`` / ``closed``."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """``True`` when the store has been initialised and not closed."""
        return self._state == "ready"

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> ReversibleStore:
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # ── Internals ─────────────────────────────────────────────────────────

    def _require_ready(self) -> None:
        """Guard: raise if the store has not been initialised."""
        if self._state != "ready":
            raise ReversibleStoreError(
                f"ReversibleStore is not ready (state={self._state!r}); "
                "call initialize() first",
                context={"state": self._state, "storage_path": str(self._path)},
            )

    def _resolve(self, memory_id: str) -> tuple[Path, Path, Path]:
        """Validate *memory_id* and return its on-disk paths.

        Raises
        ------
        InvalidMemoryIdError
            If the id is rejected by validation.
        """
        sanitized = _validate_and_sanitize(memory_id)
        return _entry_paths(self._path, sanitized)

    # ── Public API ─────────────────────────────────────────────────────────

    def store_original(self, memory_id: str, raw_content: str) -> None:
        """Persist *raw_content* under *memory_id*.

        Uses upsert semantics — storing under an existing id overwrites
        the previous payload and refreshes its metadata.  The write is
        atomic: each file is written to a temporary name and then moved
        into place with :func:`os.replace`, so a crash mid-write never
        leaves a partial or corrupt entry visible to readers.

        Parameters
        ----------
        memory_id:
            Unique identifier for this content.  Must be a non-empty
            string; filesystem-unsafe characters are replaced with ``_``.
        raw_content:
            The original, uncompressed text to archive.

        Raises
        ------
        InvalidMemoryIdError
            If *memory_id* fails validation.
        ReversibleStoreError
            If the store is not ready or an I/O error occurs.
        """
        self._require_ready()
        if not isinstance(raw_content, str):
            raise ReversibleStoreError(
                "raw_content must be a string",
                context={"memory_id": memory_id, "raw_type": type(raw_content).__name__},
            )

        shard_dir, raw_path, meta_path = self._resolve(memory_id)
        shard_dir.mkdir(parents=True, exist_ok=True)

        payload = raw_content.encode(self._encoding)
        digest = _sha256_hex(payload)

        try:
            self._atomic_write(raw_path, payload)
            meta = {
                "memory_id": memory_id,
                "sanitized_id": raw_path.stem,
                "size": len(payload),
                "sha256": digest,
                "encoding": self._encoding,
                "created_at": _utcnow().isoformat(),
            }
            self._atomic_write(meta_path, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        except OSError as exc:
            raise ReversibleStoreError(
                f"Failed to store original for memory_id={memory_id!r}: {exc}",
                context={"memory_id": memory_id, "os_error": str(exc)},
            ) from exc

        logger.debug(
            "Stored original for memory_id={!r} ({} bytes, sha256={}…)",
            memory_id, len(payload), digest[:12],
        )

    def retrieve_original(self, memory_id: str) -> str:
        """Return the raw content previously stored under *memory_id*.

        After reading the payload the stored SHA-256 digest (from the
        ``.meta`` sidecar) is re-computed and compared; a mismatch raises
        :class:`ReversibleStoreError`, signalling on-disk corruption or
        tampering.

        Parameters
        ----------
        memory_id:
            Identifier whose original content to recover.

        Returns
        -------
        str
            The exact original text passed to :meth:`store_original`.

        Raises
        ------
        InvalidMemoryIdError
            If *memory_id* fails validation.
        MemoryNotFoundError
            If no content is stored under *memory_id*.
        ReversibleStoreError
            If the store is not ready, an I/O error occurs, or the
            integrity check fails.
        """
        self._require_ready()
        _, raw_path, meta_path = self._resolve(memory_id)

        if not raw_path.exists():
            raise MemoryNotFoundError(memory_id)

        try:
            payload = raw_path.read_bytes()
        except OSError as exc:
            raise ReversibleStoreError(
                f"Failed to read original for memory_id={memory_id!r}: {exc}",
                context={"memory_id": memory_id, "os_error": str(exc)},
            ) from exc

        # Integrity verification — best-effort, degraded if sidecar absent.
        expected_digest: str | None = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_bytes().decode("utf-8"))
                expected_digest = meta.get("sha256")
            except (OSError, ValueError, json.JSONDecodeError):
                logger.warning(
                    "Unreadable metadata sidecar for memory_id={!r}; "
                    "skipping integrity check",
                    memory_id,
                )

        if expected_digest is not None:
            actual_digest = _sha256_hex(payload)
            if actual_digest != expected_digest:
                raise ReversibleStoreError(
                    f"Integrity check failed for memory_id={memory_id!r}: "
                    f"sha256 mismatch (expected {expected_digest[:12]}…, "
                    f"got {actual_digest[:12]}…)",
                    context={
                        "memory_id": memory_id,
                        "expected_sha256": expected_digest,
                        "actual_sha256": actual_digest,
                    },
                )

        try:
            return payload.decode(self._encoding)
        except UnicodeDecodeError as exc:
            raise ReversibleStoreError(
                f"Failed to decode original for memory_id={memory_id!r} "
                f"as {self._encoding}: {exc}",
                context={"memory_id": memory_id, "encoding": self._encoding},
            ) from exc

    def exists(self, memory_id: str) -> bool:
        """Return ``True`` if original content is stored under *memory_id*.

        This is a cheap existence check (``stat`` only); it does not read
        or verify the payload.  An :class:`InvalidMemoryIdError` is still
        raised for malformed ids, but a missing key returns ``False``
        rather than raising.
        """
        self._require_ready()
        try:
            _, raw_path, _ = self._resolve(memory_id)
        except InvalidMemoryIdError:
            raise
        return raw_path.exists()

    def delete(self, memory_id: str) -> bool:
        """Delete the original content stored under *memory_id*.

        Both the ``.raw`` payload and the ``.meta`` sidecar are removed.

        Parameters
        ----------
        memory_id:
            Identifier whose content to remove.

        Returns
        -------
        bool
            ``True`` if an entry existed and was removed, ``False`` if no
            entry existed under *memory_id* (idempotent delete).
        """
        self._require_ready()
        _, raw_path, meta_path = self._resolve(memory_id)

        removed = False
        try:
            if raw_path.exists():
                raw_path.unlink()
                removed = True
            if meta_path.exists():
                meta_path.unlink()
        except OSError as exc:
            raise ReversibleStoreError(
                f"Failed to delete original for memory_id={memory_id!r}: {exc}",
                context={"memory_id": memory_id, "os_error": str(exc)},
            ) from exc

        if removed:
            logger.debug("Deleted original for memory_id={!r}", memory_id)
        return removed

    def count(self) -> int:
        """Return the total number of stored entries.

        Walks the shard directories and counts ``.raw`` files.  This is
        an O(n) scan intended for diagnostics and housekeeping, not hot
        paths.
        """
        self._require_ready()
        if not self._objects_root.exists():
            return 0
        total = 0
        for shard_dir in self._objects_root.iterdir():
            if shard_dir.is_dir():
                total += sum(1 for f in shard_dir.iterdir() if f.suffix == _RAW_SUFFIX)
        return total

    def iter_memory_ids(self) -> Iterator[str]:
        """Yield the original (unsanitised) ``memory_id`` of every entry.

        Reads each ``.meta`` sidecar to recover the original id.  Entries
        with a missing or unreadable sidecar are skipped with a warning,
        so iteration never fails part-way through.

        Yields
        ------
        str
            The original ``memory_id`` of each stored entry, in
            unspecified order.
        """
        self._require_ready()
        if not self._objects_root.exists():
            return
        for shard_dir in self._objects_root.iterdir():
            if not shard_dir.is_dir():
                continue
            for meta_path in shard_dir.glob(f"*{_META_SUFFIX}"):
                try:
                    meta = json.loads(meta_path.read_bytes().decode("utf-8"))
                    memory_id = meta.get("memory_id")
                    if isinstance(memory_id, str):
                        yield memory_id
                except (OSError, ValueError, json.JSONDecodeError):
                    logger.warning("Skipping unreadable sidecar: {}", meta_path)

    # ── Low-level I/O ─────────────────────────────────────────────────────

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        """Write *data* to *target* atomically.

        Writes to a temporary file in the same directory as *target* and
        then moves it into place.  Same-directory moves via
        :func:`os.replace` are atomic on both POSIX and Windows, so
        concurrent readers observe either the old file or the new file —
        never a partial write.
        """
        target_dir = target.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        # NamedTemporaryFile is closed before the move to satisfy Windows,
        # which holds an exclusive lock on open file handles.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.stem}_",
            suffix=".tmp",
            dir=str(target_dir),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp_path, target)
        except BaseException:
            # Clean up the temp file on any failure (including
            # KeyboardInterrupt) to avoid leaving litter behind.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise


# ═══════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "ReversibleStore",
    "ReversibleStoreError",
    "MemoryNotFoundError",
    "InvalidMemoryIdError",
]

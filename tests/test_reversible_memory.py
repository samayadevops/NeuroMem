"""Tests for :class:`neuromem.compression.reversible_store.ReversibleStore`.

The core guarantee under test is **lossless reversibility**: any text
written via ``store_original`` must be recoverable **byte-for-byte**
through ``retrieve_original`` using the same structural id, regardless of
content (Unicode, whitespace, newlines, large payloads).  The id is the
*structural key* the test refers to — it determines the on-disk shard
layout and is the sole handle a compressed snapshot carries in its
``raw_reference`` field.

Beyond the round-trip, these tests cover the integrity sidecar (SHA-256
tamper detection), sanitisation of filesystem-unsafe ids, upsert
semantics, idempotent delete, lifecycle guards, and the ``zz`` shard
fallback used for ids shorter than the shard width.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuromem.compression.reversible_store import (
    InvalidMemoryIdError,
    MemoryNotFoundError,
    ReversibleStore,
    ReversibleStoreError,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def store(tmp_path: Path) -> ReversibleStore:
    """Return an initialised store rooted in a per-test tmp directory."""
    s = ReversibleStore(tmp_path / "archive")
    s.initialize()
    return s


@pytest.fixture()
def sample_texts() -> list[tuple[str, str]]:
    """Return ``(memory_id, raw_content)`` pairs covering tricky content."""
    return [
        ("plain_ascii", "The quick brown fox jumps over the lazy dog."),
        ("with_newlines", "line one\nline two\n\nline four after blank"),
        ("unicode", "héllo 世界 🧠 — em‐dash and naïve résumé"),
        ("tabs_crlf", "col1\tcol2\r\nrow2\tcol2"),
        ("json_payload", '{"a": [1, 2, {"b": "x"}], "c": true}'),
        ("code_like", "def f(x, y=2):\n    return x + y\n"),
        ("empty_single_line", "\n"),  # one newline, no other content
        ("long_text", "word " * 5000),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Core round-trip: write then read by structural id
# ═══════════════════════════════════════════════════════════════════════

class TestExactRoundTrip:
    """``store_original`` / ``retrieve_original`` must be exact inverses."""

    @pytest.mark.parametrize("memory_id,payload", [
        ("plain_ascii", "The quick brown fox jumps over the lazy dog."),
        ("with_newlines", "line one\nline two\n\nline four after blank"),
        ("unicode", "héllo 世界 🧠 — em‐dash and naïve résumé"),
        ("tabs_crlf", "col1\tcol2\r\nrow2\tcol2"),
        ("json_payload", '{"a": [1, 2, {"b": "x"}], "c": true}'),
        ("code_like", "def f(x, y=2):\n    return x + y\n"),
        ("long_text", "word " * 5000),
    ])
    def test_roundtrip_exact(
        self, store: ReversibleStore, memory_id: str, payload: str
    ) -> None:
        store.store_original(memory_id, payload)
        assert store.retrieve_original(memory_id) == payload

    def test_roundtrip_batch_all_match(
        self, store: ReversibleStore, sample_texts: list[tuple[str, str]]
    ) -> None:
        """Every sample, written together, reads back identically."""
        for memory_id, payload in sample_texts:
            store.store_original(memory_id, payload)
        for memory_id, payload in sample_texts:
            assert store.retrieve_original(memory_id) == payload

    def test_roundtrip_preserves_byte_length(self, store: ReversibleStore) -> None:
        payload = "héllo"  # multi-byte UTF-8
        store.store_original("u", payload)
        assert len(store.retrieve_original("u").encode("utf-8")) == len(payload.encode("utf-8"))

    def test_raw_reference_resolves(
        self, store: ReversibleStore, tmp_path: Path
    ) -> None:
        """The id a caller stores under is the *same* id used to resolve it.

        This mirrors how ``MemorySnapshot.raw_reference`` is consumed: the
        snapshot stores an id; the caller hands that id back to the store
        to recover the original.
        """
        original = "recoverable original text\nwith newlines"
        structural_id = "snap_abc123"
        store.store_original(structural_id, original)
        # The caller only has `structural_id`; it must be sufficient.
        assert store.retrieve_original(structural_id) == original


# ═══════════════════════════════════════════════════════════════════════
# Structural id → shard layout
# ═══════════════════════════════════════════════════════════════════════

class TestShardLayout:
    """Ids map deterministically to a two-character shard directory."""

    def test_id_uses_first_two_chars_as_shard(self, store: ReversibleStore) -> None:
        store.store_original("snapshot_001", "x")
        shard_dir = store.storage_path / "objects" / "sn"
        assert shard_dir.is_dir()
        assert (shard_dir / "snapshot_001.raw").exists()
        assert (shard_dir / "snapshot_001.meta").exists()

    def test_short_id_uses_zz_fallback_shard(self, store: ReversibleStore) -> None:
        """Ids shorter than the shard width (2) land in ``zz``."""
        store.store_original("x", "short")
        assert (store.storage_path / "objects" / "zz" / "x.raw").exists()
        assert store.retrieve_original("x") == "short"

    def test_unsafe_chars_sanitised_not_rejected(
        self, store: ReversibleStore
    ) -> None:
        """Path separators etc. are replaced with ``_``; the id still works."""
        raw_id = "ns/snap:0001?x=1"
        store.store_original(raw_id, "payload")
        # The *original* (unsanitised) id is the structural handle.
        assert store.retrieve_original(raw_id) == "payload"
        assert store.exists(raw_id)

    def test_meta_sidecar_records_original_id(
        self, store: ReversibleStore
    ) -> None:
        """The ``.meta`` sidecar preserves the unsanitised id for iteration."""
        store.store_original("ns/snap:1", "payload")
        # Find the meta file (sanitised filename) and read the original id back.
        metas = list(store.storage_path.rglob("*.meta"))
        assert len(metas) == 1
        meta = json.loads(metas[0].read_text(encoding="utf-8"))
        assert meta["memory_id"] == "ns/snap:1"
        assert meta["sha256"]
        assert meta["size"] == len("payload".encode("utf-8"))


# ═══════════════════════════════════════════════════════════════════════
# Integrity / tamper detection
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrityVerification:
    """SHA-256 sidecar catches on-disk corruption."""

    def test_tampered_payload_raises(self, store: ReversibleStore) -> None:
        store.store_original("victim", "original payload")
        raw_file = next(store.storage_path.rglob("victim.raw"))
        raw_file.write_bytes(b"TAMPERED")
        with pytest.raises(ReversibleStoreError, match="Integrity check failed"):
            store.retrieve_original("victim")

    def test_intact_payload_passes(self, store: ReversibleStore) -> None:
        store.store_original("clean", "untouched")
        # No tampering -> reads back without raising.
        assert store.retrieve_original("clean") == "untouched"

    def test_missing_meta_skips_check_gracefully(
        self, store: ReversibleStore
    ) -> None:
        """If the sidecar is deleted, retrieval still returns the payload."""
        store.store_original("orphan", "data")
        meta_file = next(store.storage_path.rglob("orphan.meta"))
        meta_file.unlink()
        # No sidecar -> integrity check is skipped, payload still returned.
        assert store.retrieve_original("orphan") == "data"


# ═══════════════════════════════════════════════════════════════════════
# Upsert, existence, delete, iteration
# ═══════════════════════════════════════════════════════════════════════

class TestCrudSemantics:
    def test_upsert_overwrites(self, store: ReversibleStore) -> None:
        store.store_original("k", "first")
        store.store_original("k", "second")
        assert store.retrieve_original("k") == "second"

    def test_upsert_refreshes_digest(self, store: ReversibleStore) -> None:
        """After overwrite, the stored SHA-256 matches the new payload."""
        store.store_original("k", "first")
        store.store_original("k", "second")
        # A clean retrieval proves the sidecar digest was refreshed.
        assert store.retrieve_original("k") == "second"

    def test_exists_true_after_write(self, store: ReversibleStore) -> None:
        store.store_original("k", "v")
        assert store.exists("k") is True

    def test_exists_false_for_missing(self, store: ReversibleStore) -> None:
        assert store.exists("absent") is False

    def test_delete_returns_true_then_false(self, store: ReversibleStore) -> None:
        store.store_original("k", "v")
        assert store.delete("k") is True
        assert store.exists("k") is False
        # Idempotent: deleting again returns False, no raise.
        assert store.delete("k") is False

    def test_retrieve_after_delete_raises_not_found(
        self, store: ReversibleStore
    ) -> None:
        store.store_original("k", "v")
        store.delete("k")
        with pytest.raises(MemoryNotFoundError):
            store.retrieve_original("k")

    def test_count_reflects_writes_and_deletes(
        self, store: ReversibleStore
    ) -> None:
        assert store.count() == 0
        store.store_original("a", "1")
        store.store_original("b", "2")
        store.store_original("c", "3")
        assert store.count() == 3
        store.delete("b")
        assert store.count() == 2

    def test_iter_memory_ids_yields_original_ids(
        self, store: ReversibleStore
    ) -> None:
        store.store_original("snap_1", "x")
        store.store_original("ns/snap:2", "y")
        ids = set(store.iter_memory_ids())
        assert ids == {"snap_1", "ns/snap:2"}


# ═══════════════════════════════════════════════════════════════════════
# Id validation
# ═══════════════════════════════════════════════════════════════════════

class TestIdValidation:
    @pytest.mark.parametrize("bad_id", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_ids_rejected(
        self, store: ReversibleStore, bad_id: str
    ) -> None:
        with pytest.raises(InvalidMemoryIdError):
            store.store_original(bad_id, "x")

    @pytest.mark.parametrize("bad_id", [None, 123, 4.5, ["a"], object()])
    def test_non_string_ids_rejected(
        self, store: ReversibleStore, bad_id: object
    ) -> None:
        with pytest.raises(InvalidMemoryIdError):
            store.store_original(bad_id, "x")  # type: ignore[arg-type]

    def test_invalid_id_on_retrieve(self, store: ReversibleStore) -> None:
        with pytest.raises(InvalidMemoryIdError):
            store.retrieve_original("")

    def test_invalid_id_on_exists(self, store: ReversibleStore) -> None:
        with pytest.raises(InvalidMemoryIdError):
            store.exists("   ")

    def test_invalid_id_on_delete(self, store: ReversibleStore) -> None:
        with pytest.raises(InvalidMemoryIdError):
            store.delete("")

    def test_non_string_content_rejected(self, store: ReversibleStore) -> None:
        with pytest.raises(ReversibleStoreError, match="raw_content must be a string"):
            store.store_original("k", 123)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# Lifecycle + context manager
# ═══════════════════════════════════════════════════════════════════════

class TestLifecycle:
    def test_uninitialized_state(self, tmp_path: Path) -> None:
        s = ReversibleStore(tmp_path / "x")
        assert s.state == "uninitialized"
        assert s.is_ready is False

    def test_initialize_makes_ready(self, tmp_path: Path) -> None:
        s = ReversibleStore(tmp_path / "x")
        s.initialize()
        assert s.state == "ready"
        assert s.is_ready is True

    def test_initialize_is_idempotent(self, tmp_path: Path) -> None:
        s = ReversibleStore(tmp_path / "x")
        s.initialize()
        s.initialize()  # second call is a no-op
        assert s.is_ready is True

    def test_close_marks_closed(self, tmp_path: Path) -> None:
        s = ReversibleStore(tmp_path / "x")
        s.initialize()
        s.close()
        assert s.state == "closed"
        assert s.is_ready is False

    def test_operations_require_ready(self, tmp_path: Path) -> None:
        s = ReversibleStore(tmp_path / "x")
        with pytest.raises(ReversibleStoreError, match="not ready"):
            s.store_original("k", "v")
        with pytest.raises(ReversibleStoreError, match="not ready"):
            s.retrieve_original("k")
        with pytest.raises(ReversibleStoreError, match="not ready"):
            s.exists("k")
        with pytest.raises(ReversibleStoreError, match="not ready"):
            s.delete("k")
        with pytest.raises(ReversibleStoreError, match="not ready"):
            s.count()

    def test_context_manager_initialises_and_closes(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ctx"
        with ReversibleStore(path) as s:
            assert s.is_ready is True
            s.store_original("k", "v")
            assert s.retrieve_original("k") == "v"
        assert s.state == "closed"

    def test_initialize_rejects_existing_file_at_path(self, tmp_path: Path) -> None:
        blocking_file = tmp_path / "blocker"
        blocking_file.write_text("not a directory")
        s = ReversibleStore(blocking_file)
        with pytest.raises(ReversibleStoreError, match="not a directory"):
            s.initialize()

    def test_storage_path_property(self, tmp_path: Path) -> None:
        s = ReversibleStore(tmp_path / "archive")
        assert s.storage_path == tmp_path / "archive"


# ═══════════════════════════════════════════════════════════════════════
# Empty / edge stores
# ═══════════════════════════════════════════════════════════════════════

class TestEmptyStore:
    def test_count_zero_before_any_write(self, store: ReversibleStore) -> None:
        assert store.count() == 0

    def test_iter_yields_nothing_when_empty(self, store: ReversibleStore) -> None:
        assert list(store.iter_memory_ids()) == []

    def test_retrieve_missing_raises(self, store: ReversibleStore) -> None:
        with pytest.raises(MemoryNotFoundError) as exc_info:
            store.retrieve_original("never_stored")
        assert exc_info.value.memory_id == "never_stored"


# ═══════════════════════════════════════════════════════════════════════
# Branch-coverage: rare paths (validation limits, iter/scan edges, I/O faults)
# ═══════════════════════════════════════════════════════════════════════

from neuromem.compression.reversible_store import (
    _SAFE_ID_MAX_LEN,
    _validate_and_sanitize,
)


class TestValidationAndSanitisation:
    def test_id_exceeding_max_length_rejected(self, store: ReversibleStore) -> None:
        too_long = "x" * (_SAFE_ID_MAX_LEN + 1)
        with pytest.raises(InvalidMemoryIdError, match="maximum length"):
            store.store_original(too_long, "v")

    def test_id_exceeding_max_length_on_retrieve(self, store: ReversibleStore) -> None:
        too_long = "x" * (_SAFE_ID_MAX_LEN + 1)
        with pytest.raises(InvalidMemoryIdError):
            store.retrieve_original(too_long)

    def test_id_exceeding_max_length_on_exists(self, store: ReversibleStore) -> None:
        too_long = "y" * (_SAFE_ID_MAX_LEN + 1)
        with pytest.raises(InvalidMemoryIdError):
            store.exists(too_long)

    def test_validate_and_sanitize_normal(self) -> None:
        assert _validate_and_sanitize("snap_001") == "snap_001"

    def test_validate_and_sanitize_replaces_unsafe_chars(self) -> None:
        assert _validate_and_sanitize("a/b:c") == "a_b_c"

    def test_validate_and_sanitize_strips_whitespace(self) -> None:
        assert _validate_and_sanitize("  snap  ") == "snap"

    def test_validate_and_sanitize_rejects_non_string(self) -> None:
        with pytest.raises(InvalidMemoryIdError, match="must be a string"):
            _validate_and_sanitize(None)  # type: ignore[arg-type]


class TestIterationAndCountEdges:
    def test_count_when_objects_root_missing(self, tmp_path: Path) -> None:
        """``count`` returns 0 gracefully when the objects dir doesn't exist."""
        s = ReversibleStore(tmp_path / "fresh")
        s.initialize()
        # Remove the objects dir to simulate a pristine/emptied store.
        import shutil

        shutil.rmtree(s.storage_path / "objects")
        assert s.count() == 0

    def test_iter_yields_nothing_when_objects_root_missing(
        self, tmp_path: Path
    ) -> None:
        s = ReversibleStore(tmp_path / "fresh")
        s.initialize()
        import shutil

        shutil.rmtree(s.storage_path / "objects")
        assert list(s.iter_memory_ids()) == []

    def test_iter_skips_unreadable_sidecar(
        self, store: ReversibleStore
    ) -> None:
        """A corrupt ``.meta`` file is skipped, not fatal."""
        store.store_original("good", "v1")
        store.store_original("broken", "v2")
        # Corrupt the broken sidecar.
        meta = next(store.storage_path.rglob("broken.meta"))
        meta.write_bytes(b"{ not valid json")
        ids = set(store.iter_memory_ids())
        assert "good" in ids
        assert "broken" not in ids

    def test_iter_skips_non_dict_meta(self, store: ReversibleStore) -> None:
        """A sidecar whose JSON is a non-string ``memory_id`` is skipped."""
        store.store_original("valid", "v")
        meta = next(store.storage_path.rglob("valid.meta"))
        meta.write_bytes(b'{"memory_id": 123}')  # non-string id
        assert "valid" not in set(store.iter_memory_ids())

    def test_count_ignores_non_raw_files(self, store: ReversibleStore) -> None:
        store.store_original("k", "v")
        # Drop a stray non-.raw file in the shard dir; it must not be counted.
        shard = next(p.parent for p in store.storage_path.rglob("k.raw"))
        (shard / "stray.txt").write_text("noise")
        assert store.count() == 1


class TestInitializeIdempotency:
    def test_initialize_after_close_returns_to_ready(
        self, tmp_path: Path
    ) -> None:
        """``initialize`` on a closed store brings it back to ``ready``."""
        s = ReversibleStore(tmp_path / "x")
        s.initialize()
        s.close()
        assert s.state == "closed"
        s.initialize()  # re-init from closed state
        assert s.state == "ready"
        assert s.is_ready is True

    def test_double_initialize_no_op(self, tmp_path: Path) -> None:
        s = ReversibleStore(tmp_path / "x")
        s.initialize()
        first_path = s.storage_path / "objects"
        s.initialize()  # idempotent
        assert first_path.exists()
        assert s.is_ready


class TestIOFaultPaths:
    """Cover the OSError → ReversibleStoreError translation branches."""

    def test_store_original_oserror_wrapped(
        self, store: ReversibleStore, monkeypatch
    ) -> None:
        """A write failure during ``store_original`` is wrapped in ReversibleStoreError."""
        import neuromem.compression.reversible_store as mod

        def boom(target, data):
            raise OSError("disk full")

        # _atomic_write is a staticmethod — patch it on the class.
        monkeypatch.setattr(ReversibleStore, "_atomic_write", staticmethod(boom))
        with pytest.raises(ReversibleStoreError, match="Failed to store"):
            store.store_original("k", "v")

    def test_retrieve_oserror_wrapped(
        self, store: ReversibleStore, monkeypatch
    ) -> None:
        store.store_original("k", "v")
        import neuromem.compression.reversible_store as mod

        real_read = Path.read_bytes

        def boom(self):
            if self.suffix == ".raw":
                raise OSError("read fault")
            return real_read(self)

        monkeypatch.setattr(Path, "read_bytes", boom)
        with pytest.raises(ReversibleStoreError, match="Failed to read"):
            store.retrieve_original("k")

    def test_delete_oserror_wrapped(
        self, store: ReversibleStore, monkeypatch
    ) -> None:
        store.store_original("k", "v")
        import neuromem.compression.reversible_store as mod

        real_unlink = Path.unlink

        def boom(self, *args, **kwargs):
            if self.suffix == ".raw":
                raise OSError("unlink denied")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", boom)
        with pytest.raises(ReversibleStoreError, match="Failed to delete"):
            store.delete("k")

    def test_decode_error_wrapped(
        self, store: ReversibleStore
    ) -> None:
        """A payload that cannot decode under the configured encoding raises."""
        # Write raw bytes that are invalid UTF-8 directly, bypassing the encoder.
        store.store_original("k", "v")
        raw = next(store.storage_path.rglob("k.raw"))
        raw.write_bytes(b"\xff\xfe\x00")  # invalid UTF-8
        # Refresh the sidecar digest to match the new bytes so the integrity
        # check passes and we reach the decode step.
        import hashlib
        import json

        data = raw.read_bytes()
        meta = next(store.storage_path.rglob("k.meta"))
        meta_obj = json.loads(meta.read_text())
        meta_obj["sha256"] = hashlib.sha256(data).hexdigest()
        meta.write_text(json.dumps(meta_obj))
        with pytest.raises(ReversibleStoreError, match="Failed to decode"):
            store.retrieve_original("k")


class TestAtomicWriteInternals:
    def test_atomic_write_cleans_up_temp_on_failure(
        self, store: ReversibleStore, monkeypatch
    ) -> None:
        """On ``os.replace`` failure the temp file is removed, not left behind."""
        import neuromem.compression.reversible_store as mod

        real_replace = mod.os.replace
        call_count = {"n": 0}

        def boom(src, dst):
            call_count["n"] += 1
            # Count how many temp files exist in the shard dir before failing.
            raise OSError("replace denied")

        monkeypatch.setattr(mod.os, "replace", boom)
        with pytest.raises(ReversibleStoreError):
            store.store_original("k", "v")
        # No leftover .tmp files in any shard dir.
        leftovers = list(store.storage_path.rglob("*.tmp"))
        assert leftovers == []
        assert call_count["n"] >= 1

    def test_atomic_write_unlink_cleanup_oserror_suppressed(
        self, store: ReversibleStore, monkeypatch
    ) -> None:
        """If cleanup after a failed ``os.replace`` also fails, the original error wins.

        Covers the ``except OSError: pass`` branch (line 643-644).
        """
        import neuromem.compression.reversible_store as mod

        def boom_replace(src, dst):
            raise OSError("replace failed")

        real_unlink = Path.unlink

        def boom_unlink(self, missing_ok=False):
            if str(self).endswith(".tmp"):
                raise OSError("unlink also failed")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(mod.os, "replace", boom_replace)
        monkeypatch.setattr(Path, "unlink", boom_unlink)
        # The replace OSError propagates; the unlink OSError is caught and suppressed.
        with pytest.raises(ReversibleStoreError):
            store.store_original("k", "v")


class TestInitializeOSError:
    """Cover the OSError → ReversibleStoreError translation in ``initialize``."""

    def test_oserror_on_mkdir_wrapped(self, tmp_path: Path, monkeypatch) -> None:
        """If ``mkdir`` raises OSError during ``initialize``, it is wrapped."""
        real_mkdir = Path.mkdir

        def boom_mkdir(self, *args, **kwargs):
            if "objects" in str(self):
                raise OSError("permission denied")
            return real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", boom_mkdir)
        s = ReversibleStore(tmp_path / "faulty")
        with pytest.raises(ReversibleStoreError, match="Failed to initialise"):
            s.initialize()


class TestRetrieveCorruptMeta:
    """Cover the unreadable meta sidecar branch during retrieve."""

    def test_corrupt_meta_skips_integrity_check(
        self, store: ReversibleStore
    ) -> None:
        """A non-JSON meta sidecar causes the integrity check to be skipped gracefully."""
        store.store_original("k", "valid payload")
        meta = next(store.storage_path.rglob("k.meta"))
        meta.write_bytes(b"corrupt not json at all")
        # Retrieval succeeds (integrity skipped) and returns the payload.
        assert store.retrieve_original("k") == "valid payload"


class TestCountAndIterNonDirShards:
    """Cover branches where shard entries are not directories."""

    def test_count_skips_non_dir_shard(self, store: ReversibleStore) -> None:
        """A non-directory file inside ``objects/`` is ignored by ``count``."""
        # Drop a stray file that is NOT a directory into objects/.
        stray = store.storage_path / "objects" / "stray_file.txt"
        stray.write_text("noise")
        store.store_original("k", "v")
        # The stray file is not a directory and must not affect the count.
        assert store.count() == 1

    def test_iter_skips_non_dir_shard(self, store: ReversibleStore) -> None:
        """Non-directory entries inside ``objects/`` are skipped by ``iter``."""
        stray = store.storage_path / "objects" / "stray_file.txt"
        stray.write_text("noise")
        store.store_original("k", "v")
        ids = set(store.iter_memory_ids())
        assert "k" in ids

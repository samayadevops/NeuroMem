"""Quick import/logic check for all 5 new features."""
import sys, traceback  # noqa: E401
sys.path.insert(0, ".")
errors = []

def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as exc:
        errors.append(f"{name}: {exc}")
        print(f"[FAIL] {name}:")
        traceback.print_exc()

# Feature 1 — Belief Reinforcement: EngineConfig
def _f1a():
    from neuromem.core.engine import EngineConfig
    cfg = EngineConfig(reinforce_on_duplicate=False)
    assert not cfg.reinforce_on_duplicate
    cfg2 = EngineConfig()
    assert cfg2.reinforce_on_duplicate  # default True
check("Feature 1 — EngineConfig.reinforce_on_duplicate", _f1a)

# Feature 2 — Guardrails: NegativeMemoryPatternType in models
def _f2a():
    from neuromem.core.models import NegativeMemoryPatternType, NegativeMemory
    nm = NegativeMemory(id="n1", pattern="foo", pattern_type="regex")
    assert nm.pattern_type == NegativeMemoryPatternType.REGEX
    nm2 = NegativeMemory(id="n2", pattern="bar", fuzzy_threshold=0.75)
    assert nm2.fuzzy_threshold == 0.75
    assert nm2.pattern_type == NegativeMemoryPatternType.EXACT
check("Feature 2 — NegativeMemory pattern_type/fuzzy_threshold fields", _f2a)

# Feature 2 — Pattern matching helper
def _f2b():
    from neuromem.core.engine import _matches_negative_pattern
    from neuromem.core.models import NegativeMemory

    # exact
    neg = NegativeMemory(id="e1", pattern="foo bar", pattern_type="exact")
    assert _matches_negative_pattern("foo bar", neg)
    assert not _matches_negative_pattern("foo", neg)

    # regex
    neg_re = NegativeMemory(id="e2", pattern=r"port \d+", pattern_type="regex")
    assert _matches_negative_pattern("connect on port 3000", neg_re)
    assert not _matches_negative_pattern("no match here", neg_re)

    # invalid regex — must not raise
    neg_bad = NegativeMemory(id="e3", pattern="[invalid", pattern_type="regex")
    assert not _matches_negative_pattern("anything", neg_bad)

    # fuzzy
    neg_fz = NegativeMemory(id="e4", pattern="database connection failed", pattern_type="fuzzy", fuzzy_threshold=0.5)
    assert _matches_negative_pattern("database connection error", neg_fz)
    assert not _matches_negative_pattern("completely unrelated", neg_fz)
check("Feature 2 — _matches_negative_pattern (exact/regex/fuzzy)", _f2b)

# Feature 3 — Providers imports
def _f3():
    from neuromem.providers import (
        BaseEmbedProvider,
        OpenAIEmbedProvider,
        OllamaEmbedProvider,
        SentenceTransformerEmbedProvider,
    )
    assert callable(OpenAIEmbedProvider)
    assert callable(OllamaEmbedProvider)
    assert callable(SentenceTransformerEmbedProvider)
    assert issubclass(OpenAIEmbedProvider, BaseEmbedProvider)
check("Feature 3 — providers package imports", _f3)

# Feature 4 — Async client imports
def _f4():
    from neuromem.async_client import AsyncNeuroMemClient
    assert hasattr(AsyncNeuroMemClient, "create")
    assert hasattr(AsyncNeuroMemClient, "learn")
    assert hasattr(AsyncNeuroMemClient, "recall")
    assert hasattr(AsyncNeuroMemClient, "guard")
    assert hasattr(AsyncNeuroMemClient, "is_blocked")
check("Feature 4 — AsyncNeuroMemClient imports and API", _f4)

# Feature 5 — MCP server imports (lazy — mcp package not required yet)
def _f5a():
    from neuromem.mcp_server import _json, _require_mcp
    from datetime import datetime, timezone
    result = _json({"tick": 1})
    assert result == '{"tick": 1}'
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result2 = _json({"at": dt})
    assert "2026" in result2
check("Feature 5 — mcp_server._json helper", _f5a)

# Feature 5 — CLI serve sub-command
def _f5b():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("cli", os.path.join("neuromem", "cli.py"))
    assert spec is not None, "Failed to load module spec"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, "Module spec has no loader"
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_cmd_serve"), "_cmd_serve handler missing"
    assert "serve" in mod._DISPATCH, "serve not in _DISPATCH"
check("Feature 5 — CLI serve sub-command registered", _f5b)

# Top-level __init__ exports
def _finit():
    import neuromem
    assert hasattr(neuromem, "AsyncNeuroMemClient"), "AsyncNeuroMemClient not exported"
    assert hasattr(neuromem, "NegativeMemoryPatternType"), "NegativeMemoryPatternType not exported"
    assert hasattr(neuromem, "providers"), "providers sub-package not exported"
check("neuromem/__init__.py — new exports", _finit)

print()
if errors:
    print(f"FAILED: {len(errors)} check(s)")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print(f"All {7} checks passed ✓")

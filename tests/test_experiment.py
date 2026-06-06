from __future__ import annotations
import asyncio
import pytest
from peekr.experiment import experiment
from peekr.exporters import _exporters


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class CollectingExporter:
    def __init__(self):
        self.spans = []

    def export(self, span):
        self.spans.append(span)


@pytest.fixture(autouse=True)
def isolated_exporters():
    _exporters.clear()
    collector = CollectingExporter()
    _exporters.append(collector)
    yield collector
    _exporters.clear()


# ---------------------------------------------------------------------------
# List-variant tests
# ---------------------------------------------------------------------------


def test_list_variant_injected(isolated_exporters):
    """The 'variant' kwarg is injected automatically."""
    received = []

    @experiment(variants=["control", "test_v2"])
    def run(query: str, variant: str):
        received.append(variant)
        return variant

    run("hello")
    assert received[0] in ("control", "test_v2")


def test_list_variant_tagged_on_span(isolated_exporters):
    """experiment_variant is recorded in span attributes."""

    @experiment(variants=["control", "test_v2"])
    def run(query: str, variant: str):
        return variant

    run("hello")
    span = isolated_exporters.spans[0]
    assert span.attributes.get("experiment_variant") in ("control", "test_v2")


def test_list_equal_split_default(isolated_exporters):
    """Without explicit split, equal weights are used (both variants appear)."""
    seen = set()

    @experiment(variants=["a", "b"])
    def run(query: str, variant: str):
        seen.add(variant)
        return variant

    # Run enough times that probability of only seeing one variant is negligible
    for _ in range(50):
        run("q")

    assert "a" in seen
    assert "b" in seen


def test_list_explicit_split_always_control(isolated_exporters):
    """A split of [1.0, 0.0] always picks the first variant."""

    @experiment(variants=["control", "test_v2"], split=[1.0, 0.0])
    def run(query: str, variant: str):
        return variant

    for _ in range(10):
        result = run("q")
        assert result == "control"


def test_list_no_variant_config_injected(isolated_exporters):
    """For list variants, variant_config is NOT injected."""

    @experiment(variants=["a"])
    def run(variant: str):
        return variant

    # Should not raise even though the function doesn't accept variant_config
    run()


# ---------------------------------------------------------------------------
# Dict-variant tests
# ---------------------------------------------------------------------------


def test_dict_variant_and_config_injected(isolated_exporters):
    """Both variant and variant_config are injected for dict variants."""
    received = {}

    @experiment(variants={"control": {"model": "gpt-4o"}, "test": {"model": "claude"}})
    def run(variant: str, variant_config: dict):
        received["variant"] = variant
        received["config"] = variant_config
        return variant

    run()
    assert received["variant"] in ("control", "test")
    assert "model" in received["config"]


def test_dict_variant_config_matches_key(isolated_exporters):
    """The injected variant_config corresponds to the chosen variant key."""
    configs = {"control": {"model": "gpt-4o"}, "test": {"model": "claude"}}
    results = []

    @experiment(variants=configs, split=[1.0, 0.0])
    def run(variant: str, variant_config: dict):
        results.append((variant, variant_config))
        return variant

    run()
    key, cfg = results[0]
    assert cfg == configs[key]


def test_dict_tagged_on_span(isolated_exporters):
    """experiment_variant is set to the chosen key, not the config dict."""

    @experiment(variants={"ctrl": {"x": 1}, "exp": {"x": 2}})
    def run(variant: str, variant_config: dict):
        return variant

    run()
    span = isolated_exporters.spans[0]
    assert span.attributes.get("experiment_variant") in ("ctrl", "exp")
    assert isinstance(span.attributes["experiment_variant"], str)


# ---------------------------------------------------------------------------
# Span / export tests
# ---------------------------------------------------------------------------


def test_span_exported(isolated_exporters):
    """Exactly one experiment span is exported per call."""

    @experiment(variants=["a"])
    def run(variant: str):
        return variant

    run()
    assert len(isolated_exporters.spans) == 1


def test_span_name(isolated_exporters):
    """The span name follows the pattern 'experiment.<function_name>'."""

    @experiment(variants=["a"])
    def my_agent(variant: str):
        return variant

    my_agent()
    assert isolated_exporters.spans[0].name == "experiment.my_agent"


def test_span_status_ok_on_success(isolated_exporters):

    @experiment(variants=["a"])
    def run(variant: str):
        return "ok"

    run()
    assert isolated_exporters.spans[0].status == "ok"


def test_span_status_error_on_exception(isolated_exporters):

    @experiment(variants=["a"])
    def run(variant: str):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run()

    span = isolated_exporters.spans[0]
    assert span.status == "error"
    assert "boom" in span.attributes.get("error", "")


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


def test_async_variant_injected(isolated_exporters):

    received = []

    @experiment(variants=["a", "b"])
    async def run(variant: str):
        received.append(variant)
        return variant

    asyncio.run(run())
    assert received[0] in ("a", "b")


def test_async_span_exported(isolated_exporters):

    @experiment(variants=["a"])
    async def run(variant: str):
        await asyncio.sleep(0)
        return variant

    asyncio.run(run())
    assert len(isolated_exporters.spans) == 1
    assert isolated_exporters.spans[0].status == "ok"


def test_async_error_captured(isolated_exporters):

    @experiment(variants=["a"])
    async def run(variant: str):
        raise ValueError("async boom")

    with pytest.raises(ValueError):
        asyncio.run(run())

    span = isolated_exporters.spans[0]
    assert span.status == "error"
    assert "async boom" in span.attributes.get("error", "")


# ---------------------------------------------------------------------------
# Decorator form without parentheses is not supported (by design) — no test
# needed, the spec uses explicit kwargs only.
# ---------------------------------------------------------------------------

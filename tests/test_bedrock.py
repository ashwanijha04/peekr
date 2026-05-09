"""
Tests for Bedrock patch. Mocks botocore directly — no AWS credentials needed.
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from peekr.exporters import _exporters
from peekr.patches.bedrock_patch import _BedrockStreamWrapper, patch_bedrock
from peekr.context import start_span


class CollectingExporter:
    def __init__(self):
        self.spans = []

    def export(self, span):
        self.spans.append(span)


@pytest.fixture(autouse=True)
def isolated_exporters():
    _exporters.clear()
    col = CollectingExporter()
    _exporters.append(col)
    yield col
    _exporters.clear()


def _make_client(service="bedrock-runtime"):
    client = MagicMock()
    client.meta.service_model.service_name = service
    return client


def _span_and_token():
    return start_span("test")


botocore = pytest.importorskip("botocore", reason="botocore not installed")

# ── converse (non-streaming) ──────────────────────────────────────────────────

def test_converse_captures_tokens(isolated_exporters):
    import botocore.client
    patch_bedrock()

    fake_response = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "usage": {"inputTokens": 15, "outputTokens": 6, "totalTokens": 21},
        "stopReason": "end_turn",
    }

    client = _make_client()
    with patch.object(
        botocore.client.BaseClient, "_make_api_call", return_value=fake_response
    ):
        result = botocore.client.BaseClient._make_api_call(
            client,
            "Converse",
            {"modelId": "anthropic.claude-3-haiku-20240307-v1:0",
             "messages": [{"role": "user", "content": [{"text": "Hi"}]}]},
        )

    span = isolated_exporters.spans[0]
    assert span.attributes["tokens_input"]  == 15
    assert span.attributes["tokens_output"] == 6
    assert span.attributes["tokens_total"]  == 21
    assert span.attributes["model"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert span.attributes["output"] == "Hello!"
    assert span.status == "ok"


def test_converse_captures_input(isolated_exporters):
    import botocore.client
    patch_bedrock()

    client = _make_client()
    with patch.object(
        botocore.client.BaseClient, "_make_api_call",
        return_value={"output": {"message": {"content": [{"text": "ok"}]}}, "usage": {}}
    ):
        botocore.client.BaseClient._make_api_call(
            client, "Converse",
            {"modelId": "amazon.titan-text-lite-v1",
             "messages": [{"role": "user", "content": [{"text": "hello"}]}]},
        )

    assert "input" in isolated_exporters.spans[0].attributes


def test_converse_error(isolated_exporters):
    import botocore.client
    patch_bedrock()

    client = _make_client()
    with patch.object(
        botocore.client.BaseClient, "_make_api_call",
        side_effect=RuntimeError("throttled")
    ):
        with pytest.raises(RuntimeError):
            botocore.client.BaseClient._make_api_call(
                client, "Converse", {"modelId": "x", "messages": []}
            )

    span = isolated_exporters.spans[0]
    assert span.status == "error"
    assert "throttled" in span.attributes["error"]


def test_non_bedrock_passthrough(isolated_exporters):
    """Calls to other AWS services must not be intercepted."""
    import botocore.client
    patch_bedrock()

    s3_response = {"Buckets": []}
    client = _make_client(service="s3")
    with patch.object(
        botocore.client.BaseClient, "_make_api_call", return_value=s3_response
    ):
        result = botocore.client.BaseClient._make_api_call(
            client, "ListBuckets", {}
        )

    assert result == s3_response
    assert len(isolated_exporters.spans) == 0


# ── converse_stream ───────────────────────────────────────────────────────────

def _make_stream_events(input_tokens=10, output_tokens=5):
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " there"}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": input_tokens, "outputTokens": output_tokens}}},
    ]


def test_stream_wrapper_captures_tokens(isolated_exporters):
    span, token = _span_and_token()
    events = _make_stream_events(input_tokens=20, output_tokens=8)
    wrapper = _BedrockStreamWrapper(iter(events), span, token)

    list(wrapper)

    span = isolated_exporters.spans[0]
    assert span.attributes["tokens_input"]  == 20
    assert span.attributes["tokens_output"] == 8
    assert span.attributes["tokens_total"]  == 28


def test_stream_wrapper_yields_all_events(isolated_exporters):
    span, token = _span_and_token()
    events = _make_stream_events()
    wrapper = _BedrockStreamWrapper(iter(events), span, token)

    received = list(wrapper)
    assert len(received) == len(events)


def test_stream_wrapper_status_ok(isolated_exporters):
    span, token = _span_and_token()
    wrapper = _BedrockStreamWrapper(iter(_make_stream_events()), span, token)
    list(wrapper)
    assert isolated_exporters.spans[0].status == "ok"


def test_stream_wrapper_error(isolated_exporters):
    def bad():
        yield {"messageStart": {}}
        raise ConnectionError("stream cut")

    span, token = _span_and_token()
    wrapper = _BedrockStreamWrapper(bad(), span, token)

    with pytest.raises(ConnectionError):
        list(wrapper)

    assert isolated_exporters.spans[0].status == "error"


def test_stream_wrapper_no_double_export(isolated_exporters):
    span, token = _span_and_token()
    wrapper = _BedrockStreamWrapper(iter(_make_stream_events()), span, token)
    list(wrapper)
    list(wrapper)
    assert len(isolated_exporters.spans) == 1

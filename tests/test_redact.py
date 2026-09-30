from app.obs.tracer import Tracer
from app.security import redact, redact_value


def test_redact_masks_keys():
    assert "sk-***REDACTED***" in redact("key=sk-abcdef123456")
    assert "REDACTED" in redact("Authorization: Bearer abc123")


def test_redact_value_nested():
    out = redact_value({"a": "sk-abcdef123456", "b": ["token=xyz789"]})
    assert "REDACTED" in out["a"]
    assert "REDACTED" in out["b"][0]


def test_tracer_redacts_metadata_and_attributes():
    tracer = Tracer()
    trace = tracer.new_trace("t", api_key="sk-abcdef123456")
    assert "REDACTED" in trace.metadata["api_key"]
    with tracer.span(trace, "s", note="token=xyz789"):
        pass
    assert "REDACTED" in trace.spans[0].attributes["note"]

"""Unit tests for streaming engine with Redis and thread-safe in-memory fallback."""

import pytest
from resq.stream import ResQStreamManager


def test_in_memory_stream_fallback():
    """Verify in-memory fallback stream functions reliably when Redis is absent."""
    manager = ResQStreamManager(enable_redis=False)
    status = manager.get_status()
    assert status["backend"] == "in_memory_fallback"

    event_id = manager.publish_event("test_stream", {"incident_id": "INC-001", "level": "HIGH"})
    assert event_id is not None
    assert "mem-" in event_id

    events = manager.consume_events("test_stream", count=10)
    assert len(events) == 1
    assert events[0]["incident_id"] == "INC-001"
    assert events[0]["level"] == "HIGH"

"""Streaming engine supporting Redis Streams with automatic thread-safe in-memory fallback."""

from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try importing redis
try:
    import redis
    HAS_REDIS_LIB = True
except ImportError:
    HAS_REDIS_LIB = False


class ResQStreamManager:
    """Unified streaming manager supporting Redis Streams with seamless in-memory fallback."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        enable_redis: bool = True,
    ):
        self.redis_client: Optional[Any] = None
        self.use_redis: bool = False
        self._lock = threading.Lock()
        self._in_memory_streams: Dict[str, deque] = {}

        if enable_redis and HAS_REDIS_LIB:
            try:
                client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    socket_connect_timeout=0.5,
                )
                if client.ping():
                    self.redis_client = client
                    self.use_redis = True
                    logger.info("Connected to Redis Streams at %s:%d", redis_host, redis_port)
            except Exception as exc:
                logger.debug("Redis connection unavailable (%s); using in-memory stream fallback.", exc)
                self.redis_client = None
                self.use_redis = False

    def publish_event(self, stream_name: str, payload: Dict[str, Any]) -> str:
        """Publish an event to the stream."""
        if "published_at" not in payload:
            payload["published_at"] = datetime.now(timezone.utc).isoformat()

        if self.use_redis and self.redis_client is not None:
            try:
                # Convert dict to string mapping for Redis
                redis_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in payload.items()}
                event_id = self.redis_client.xadd(stream_name, redis_data)
                return str(event_id)
            except Exception as exc:
                logger.warning("Redis xadd failed (%s); failing over to in-memory queue.", exc)
                self.use_redis = False

        # In-memory stream fallback
        with self._lock:
            if stream_name not in self._in_memory_streams:
                self._in_memory_streams[stream_name] = deque(maxlen=2000)
            event_id = f"mem-{len(self._in_memory_streams[stream_name]) + 1}-{int(datetime.now(timezone.utc).timestamp()*1000)}"
            record = {"_id": event_id, **payload}
            self._in_memory_streams[stream_name].append(record)
            return event_id

    def consume_events(self, stream_name: str, count: int = 20) -> List[Dict[str, Any]]:
        """Consume the latest events from the stream."""
        if self.use_redis and self.redis_client is not None:
            try:
                entries = self.redis_client.xrevrange(stream_name, max="+", min="-", count=count)
                events = []
                for msg_id, fields in entries:
                    item = {"_id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)}
                    for k, v in fields.items():
                        key_str = k.decode() if isinstance(k, bytes) else str(k)
                        val_str = v.decode() if isinstance(v, bytes) else str(v)
                        try:
                            item[key_str] = json.loads(val_str)
                        except Exception:
                            item[key_str] = val_str
                    events.append(item)
                return events
            except Exception as exc:
                logger.warning("Redis consume failed (%s); reading in-memory stream.", exc)
                self.use_redis = False

        # In-memory stream fallback
        with self._lock:
            q = self._in_memory_streams.get(stream_name, deque())
            items = list(q)[-count:]
            items.reverse()
            return items

    def get_status(self) -> Dict[str, Any]:
        """Return streaming backend operational status."""
        return {
            "backend": "redis_streams" if self.use_redis else "in_memory_fallback",
            "redis_library_installed": HAS_REDIS_LIB,
            "connected_to_redis": self.use_redis,
            "active_streams": list(self._in_memory_streams.keys()) if not self.use_redis else "redis_managed",
        }


# Global stream manager singleton instance
stream_manager = ResQStreamManager()

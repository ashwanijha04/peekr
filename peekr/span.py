from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    # First-class so the hosted backend can route + index without JSON extraction.
    # tenant_id = customer org (B2B), distinct from end-user (attributes.user_id).
    # retention_class = "default" | "short" | "long" | "pii" — interpreted by storage tier.
    tenant_id: str | None = None
    retention_class: str | None = None

    def finish(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_ms"] = self.duration_ms
        return d

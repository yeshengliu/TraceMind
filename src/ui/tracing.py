"""Local Phoenix/OpenInference tracing setup for TraceMind."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

import httpx
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict


DEFAULT_PHOENIX_URL = "http://localhost:6006"
DEFAULT_PROJECT_NAME = "TraceMind"


class TracingStatus(BaseModel):
    """Serializable tracing state displayed by the dashboard."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    enabled: bool
    collector_online: bool
    ui_url: str
    project_name: str
    detail: str
    tracer_provider: Any | None = None


def _collector_online(ui_url: str) -> bool:
    try:
        response = httpx.get(ui_url, timeout=0.75)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


@lru_cache(maxsize=1)
def setup_tracing() -> TracingStatus:
    """Register Phoenix once per process and degrade cleanly when unavailable."""
    ui_url = os.getenv("TRACEMIND_PHOENIX_URL", DEFAULT_PHOENIX_URL).rstrip("/")
    project_name = os.getenv("TRACEMIND_PHOENIX_PROJECT", DEFAULT_PROJECT_NAME)
    enabled = os.getenv("TRACEMIND_TRACING_ENABLED", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    online = _collector_online(ui_url)
    if not enabled:
        return TracingStatus(
            enabled=False,
            collector_online=online,
            ui_url=ui_url,
            project_name=project_name,
            detail="Tracing disabled by TRACEMIND_TRACING_ENABLED.",
        )

    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", ui_url)
    try:
        from phoenix.otel import register

        provider = register(
            project_name=project_name,
            auto_instrument=True,
            batch=False,
            endpoint=f"{ui_url}/v1/traces",
            protocol="http/protobuf",
        )
    except Exception as exc:
        return TracingStatus(
            enabled=False,
            collector_online=online,
            ui_url=ui_url,
            project_name=project_name,
            detail=f"Tracing registration failed: {type(exc).__name__}: {exc}",
        )

    detail = (
        "Phoenix collector connected."
        if online
        else "Instrumentation active; start the local Phoenix collector to view traces."
    )
    return TracingStatus(
        enabled=True,
        collector_online=online,
        ui_url=ui_url,
        project_name=project_name,
        detail=detail,
        tracer_provider=provider,
    )


@contextmanager
def agent_run_span(run_id: str, prompt: str) -> Iterator[Any]:
    """Create a parent OpenTelemetry span for one dashboard-triggered run."""
    tracer = trace.get_tracer("tracemind.dashboard")
    with tracer.start_as_current_span("tracemind.agent.run") as span:
        span.set_attribute("tracemind.run_id", run_id)
        span.set_attribute("tracemind.prompt_length", len(prompt))
        span.set_attribute("openinference.span.kind", "AGENT")
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def record_node_event(
    span: Any,
    *,
    node: str,
    status: str,
    retry_count: int,
    context_chars: int,
) -> None:
    """Attach bounded node metrics to the active run span."""
    if span is None:
        return
    span.add_event(
        f"node.{node}",
        attributes={
            "tracemind.node": node,
            "tracemind.status": status,
            "tracemind.retry_count": retry_count,
            "tracemind.context_chars": context_chars,
        },
    )

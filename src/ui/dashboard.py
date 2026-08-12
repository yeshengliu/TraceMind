"""Streamlit dashboard and asynchronous execution controller for TraceMind."""

from __future__ import annotations

import base64
import binascii
import html
import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict, Field
from pyvis.network import Network

from src.agent.graph import AgentState, create_graph, initial_state
from src.memory.pruner import ContextPruner
from src.ui.tracing import agent_run_span, record_node_event, setup_tracing


NODE_LABELS = {
    "planner": "Planner",
    "coder_agent": "Coder",
    "sandbox_executor": "Sandbox",
    "error_detector": "Error Check",
    "reflect_and_heal": "Self-Healing",
    "complete": "Success",
    "dashboard": "Dashboard",
}

NODE_ORDER = (
    "planner",
    "coder_agent",
    "sandbox_executor",
    "error_detector",
    "reflect_and_heal",
)

RUNNING_STATES = {"queued", "running"}


class RunEvent(BaseModel):
    """One observable LangGraph state transition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequence: int
    node: str
    label: str
    status: Literal["pending", "active", "success", "error", "healing"]
    summary: str
    timestamp: float
    elapsed_seconds: float = 0.0
    context_chars: int = 0
    estimated_tokens: int = 0
    retry_count: int = 0
    artifact_count: int = 0
    state: dict[str, Any] = Field(default_factory=dict)


class RunSnapshot(BaseModel):
    """Thread-safe, immutable view of a dashboard run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    active_node: str | None = None
    events: tuple[RunEvent, ...] = ()
    final_state: dict[str, Any] | None = None
    error: str | None = None


class DashboardArtifact(BaseModel):
    """Artifact extracted from a sandbox terminal stream."""

    kind: Literal["svg", "png", "markdown", "json"]
    title: str
    content: str | bytes | dict[str, Any]


@dataclass
class _RunRecord:
    run_id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    active_node: str | None = None
    events: list[RunEvent] = field(default_factory=list)
    final_state: dict[str, Any] | None = None
    error: str | None = None
    future: Future[None] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


GraphFactory = Callable[[], Any]
DashboardGraphFactory = Callable[[int], Any]


class AgentRunController:
    """Run a compiled TraceMind graph without blocking Streamlit rendering."""

    def __init__(self, max_workers: int = 2, max_runs: int = 20) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tracemind-ui",
        )
        self._runs: dict[str, _RunRecord] = {}
        self._runs_lock = threading.RLock()
        self._max_runs = max_runs
        self._pruner = ContextPruner()

    def start(
        self,
        prompt: str,
        graph_factory: GraphFactory = create_graph,
    ) -> str:
        """Queue an agent run and return its dashboard identifier."""
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("A non-empty prompt is required.")

        run_id = uuid.uuid4().hex
        record = _RunRecord(run_id=run_id)
        with self._runs_lock:
            self._runs[run_id] = record
            self._evict_finished_runs()
        record.future = self._executor.submit(
            self._execute,
            record,
            clean_prompt,
            graph_factory,
        )
        return run_id

    def _evict_finished_runs(self) -> None:
        """Drop the oldest finished runs so long-lived sessions stay bounded."""
        finished_ids = [
            run_id
            for run_id, candidate in self._runs.items()
            if candidate.status in {"completed", "failed"}
        ]
        overflow = len(self._runs) - self._max_runs
        if overflow > 0:
            for run_id in finished_ids[:overflow]:
                del self._runs[run_id]

    def snapshot(self, run_id: str) -> RunSnapshot:
        """Return the latest immutable snapshot for a run."""
        record = self._get_record(run_id)
        with record.lock:
            return RunSnapshot(
                run_id=record.run_id,
                status=record.status,
                active_node=record.active_node,
                events=tuple(event.model_copy(deep=True) for event in record.events),
                final_state=_copy_state(record.final_state),
                error=record.error,
            )

    def poll(self, run_id: str, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        """Return events emitted after ``after_sequence``."""
        snapshot = self.snapshot(run_id)
        return tuple(
            event for event in snapshot.events if event.sequence > after_sequence
        )

    def wait(self, run_id: str, timeout: float = 120.0) -> RunSnapshot:
        """Wait for a run to finish; useful for tests and non-UI callers."""
        record = self._get_record(run_id)
        if record.future is not None:
            record.future.result(timeout=timeout)
        return self.snapshot(run_id)

    def shutdown(self, wait: bool = True) -> None:
        """Release worker threads."""
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _get_record(self, run_id: str) -> _RunRecord:
        with self._runs_lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise KeyError(f"Unknown TraceMind run: {run_id}") from exc

    def _execute(
        self,
        record: _RunRecord,
        prompt: str,
        graph_factory: GraphFactory,
    ) -> None:
        started_at = time.monotonic()
        with record.lock:
            record.status = "running"
            record.active_node = "planner"

        tracing = setup_tracing()
        try:
            graph = graph_factory()
            latest_state: dict[str, Any] = dict(initial_state(prompt))
            with agent_run_span(record.run_id, prompt) as span:
                for node, state in _stream_updates(graph, latest_state):
                    latest_state = dict(state)
                    event = self._make_event(
                        record,
                        node,
                        latest_state,
                        started_at,
                    )
                    with record.lock:
                        record.events.append(event)
                        record.active_node = _next_node(node, latest_state)
                    record_node_event(
                        span,
                        node=node,
                        status=str(latest_state.get("status", "unknown")),
                        retry_count=int(latest_state.get("retry_count", 0) or 0),
                        context_chars=event.context_chars,
                    )

            final_status = str(latest_state.get("status", "completed"))
            with record.lock:
                record.final_state = _copy_state(latest_state)
                record.status = (
                    "completed" if final_status == "completed" else "failed"
                )
                record.active_node = None
                if record.status == "failed":
                    final_output = latest_state.get("final_output")
                    errors = latest_state.get("error_stack")
                    latest_error = (
                        errors[-1]
                        if isinstance(errors, list) and errors
                        else errors
                    )
                    record.error = str(
                        final_output or latest_error or "Agent run failed."
                    )
        except Exception as exc:
            failure_state = {"status": "failed", "error_stack": str(exc)}
            event = self._make_event(
                record,
                "dashboard",
                failure_state,
                started_at,
            )
            with record.lock:
                record.events.append(event)
                record.status = "failed"
                record.active_node = None
                record.final_state = failure_state
                record.error = str(exc)
        finally:
            # Retain the collector state in the final record without making a
            # local dashboard dependent on Phoenix availability.
            with record.lock:
                if record.final_state is not None:
                    record.final_state.setdefault(
                        "observability",
                        {
                            "enabled": tracing.enabled,
                            "online": tracing.collector_online,
                            "project": tracing.project_name,
                        },
                    )

    def _make_event(
        self,
        record: _RunRecord,
        node: str,
        state: dict[str, Any],
        started_at: float,
    ) -> RunEvent:
        with record.lock:
            sequence = len(record.events) + 1
        context_chars = _measure_context(self._pruner, state)
        return RunEvent(
            sequence=sequence,
            node=node,
            label=NODE_LABELS.get(node, node.replace("_", " ").title()),
            status=_event_status(node, state),
            summary=_event_summary(node, state),
            timestamp=time.time(),
            elapsed_seconds=round(time.monotonic() - started_at, 3),
            context_chars=context_chars,
            estimated_tokens=max(1, context_chars // 4) if context_chars else 0,
            retry_count=int(state.get("retry_count", 0) or 0),
            artifact_count=len(state.get("execution_artifacts", []) or []),
            state=_copy_state(state) or {},
        )


def _stream_updates(
    graph: Any,
    state: dict[str, Any],
):
    """Normalize LangGraph v2 and legacy update streams."""
    stream = graph.stream(state, stream_mode="updates", version="v2")
    for chunk in stream:
        if isinstance(chunk, dict) and chunk.get("type") == "updates":
            updates = chunk.get("data", {})
        else:
            updates = chunk
        if not isinstance(updates, dict):
            continue
        for node, update in updates.items():
            if isinstance(update, dict):
                yield str(node), update


def _measure_context(pruner: ContextPruner, state: dict[str, Any]) -> int:
    try:
        measurement = pruner.measure_state(state)
        if isinstance(measurement, int):
            return measurement
        if isinstance(measurement, dict):
            return int(
                measurement.get("characters")
                or measurement.get("char_count")
                or 0
            )
        return int(getattr(measurement, "characters", 0))
    except (AttributeError, TypeError, ValueError):
        return len(json.dumps(state, default=str))


def _copy_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return json.loads(json.dumps(state, default=str))


def _event_status(
    node: str,
    state: dict[str, Any],
) -> Literal["pending", "active", "success", "error", "healing"]:
    if node == "reflect_and_heal":
        return "healing"
    if node == "dashboard" or str(state.get("status")) == "failed":
        return "error"
    if node == "error_detector" and state.get("status") == "healing":
        return "error"
    return "success"


def _event_summary(node: str, state: dict[str, Any]) -> str:
    if node == "planner":
        plan = state.get("current_plan") or []
        if isinstance(plan, str):
            return plan
        return f"Prepared {len(plan)} executable plan step(s)."
    if node == "coder_agent":
        if state.get("status") == "failed":
            failure = str(state.get("final_output") or "Code healing failed.")
            return failure.strip().splitlines()[0]
        code = str(state.get("generated_code") or "")
        lines = len(code.splitlines())
        return f"Generated a focused Python program ({lines} line(s))."
    if node == "sandbox_executor":
        artifacts = state.get("execution_artifacts") or []
        latest = artifacts[-1] if artifacts else {}
        result = latest.get("result", {}) if isinstance(latest, dict) else {}
        exit_code = result.get("exit_code", "?")
        return f"Sandbox execution finished with exit code {exit_code}."
    if node == "error_detector":
        status = state.get("status")
        if status in {"healing", "failed"} and state.get("error_stack"):
            errors = state["error_stack"]
            latest_error = errors[-1] if isinstance(errors, list) else errors
            first_error = str(latest_error).strip().splitlines()[-1]
            if status == "failed":
                return f"Retry limit reached; execution failed: {first_error}"
            return f"Runtime failure detected: {first_error}"
        if status == "completed":
            return "Execution passed error and traceback checks."
        return f"Execution check ended in unexpected state: {status or 'unknown'}."
    if node == "reflect_and_heal":
        patches = state.get("patch_history") or []
        latest = patches[-1] if patches else {}
        if isinstance(latest, dict):
            cause = latest.get("root_cause") or latest.get("explanation")
            if cause:
                return f"Targeted repair prepared: {cause}"
        return "Traceback analyzed and a targeted code repair prepared."
    return str(state.get("error_stack") or "Dashboard execution failed.")


def _next_node(node: str, state: dict[str, Any]) -> str | None:
    if node == "planner":
        return "coder_agent"
    if node == "coder_agent":
        return "sandbox_executor"
    if node == "sandbox_executor":
        return "error_detector"
    if node == "error_detector":
        return "reflect_and_heal" if state.get("status") == "healing" else None
    if node == "reflect_and_heal":
        return "coder_agent" if state.get("status") == "coding" else "sandbox_executor"
    return None


def extract_artifacts(output: str) -> list[DashboardArtifact]:
    """Extract explicitly marked, renderable artifacts from sandbox stdout."""
    artifacts: list[DashboardArtifact] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("TREND_SVG="):
            svg = line.removeprefix("TREND_SVG=").strip()
            if "<svg" in svg.lower():
                artifacts.append(
                    DashboardArtifact(
                        kind="svg",
                        title="Generated trend chart",
                        content=svg,
                    )
                )
        elif line.startswith("ARTIFACT_PNG_BASE64="):
            encoded = line.removeprefix("ARTIFACT_PNG_BASE64=").strip()
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                continue
            if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                artifacts.append(
                    DashboardArtifact(
                        kind="png",
                        title="Generated image",
                        content=decoded,
                    )
                )
        elif line.startswith("ARTIFACT_MARKDOWN="):
            artifacts.append(
                DashboardArtifact(
                    kind="markdown",
                    title="Generated report",
                    content=line.removeprefix("ARTIFACT_MARKDOWN=").strip(),
                )
            )
        elif line.startswith("ARTIFACT_JSON="):
            value = line.removeprefix("ARTIFACT_JSON=").strip()
            try:
                payload = json.loads(value)
            except json.JSONDecodeError:
                continue
            artifacts.append(
                DashboardArtifact(
                    kind="json",
                    title="Generated data",
                    content=payload,
                )
            )
    return artifacts


def build_graph_html(
    events: tuple[RunEvent, ...] | list[RunEvent],
    active_node: str | None = None,
) -> str:
    """Build a self-contained PyVis execution graph."""
    observed = {event.node: event.status for event in events}
    network = Network(
        height="390px",
        width="100%",
        directed=True,
        bgcolor="#08111f",
        font_color="#e2e8f0",
        cdn_resources="in_line",
    )
    colors = {
        "pending": "#334155",
        "active": "#38bdf8",
        "success": "#10b981",
        "error": "#fb7185",
        "healing": "#a78bfa",
    }
    for node in NODE_ORDER:
        status = "active" if node == active_node else observed.get(node, "pending")
        network.add_node(
            node,
            label=NODE_LABELS[node],
            color=colors[status],
            shape="dot",
            size=28 if status == "active" else 22,
            borderWidth=3 if status == "active" else 1,
            title=f"{NODE_LABELS[node]} · {status}",
            level={
                "planner": 0,
                "coder_agent": 1,
                "sandbox_executor": 2,
                "error_detector": 3,
                "reflect_and_heal": 4,
            }[node],
        )
    network.add_node(
        "complete",
        label="Success",
        color=colors["success"] if events and active_node is None else colors["pending"],
        shape="star",
        size=24,
        level=4,
    )
    for source, target, label in (
        ("planner", "coder_agent", "plan"),
        ("coder_agent", "sandbox_executor", "execute"),
        ("sandbox_executor", "error_detector", "inspect"),
        ("error_detector", "complete", "clean"),
        ("error_detector", "reflect_and_heal", "error"),
        ("reflect_and_heal", "sandbox_executor", "patch"),
    ):
        network.add_edge(
            source,
            target,
            label=label,
            color="#64748b",
            arrows="to",
            smooth={"type": "curvedCW", "roundness": 0.15},
        )
    network.set_options(
        """
        {
          "layout": {"hierarchical": {
            "enabled": true,
            "direction": "LR",
            "sortMethod": "directed",
            "nodeSpacing": 115,
            "levelSeparation": 130
          }},
          "physics": {"enabled": false},
          "interaction": {"dragNodes": false, "zoomView": true},
          "edges": {
            "font": {"color": "#94a3b8", "size": 11, "strokeWidth": 0},
            "width": 1.5
          },
          "nodes": {
            "font": {"color": "#f8fafc", "face": "Inter", "size": 13}
          }
        }
        """
    )
    return network.generate_html(notebook=False)


def build_metrics_figure(events: tuple[RunEvent, ...] | list[RunEvent]) -> go.Figure:
    """Create context, token, retry, and artifact telemetry curves."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    x_values = [event.sequence for event in events]
    labels = [event.label for event in events]
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=[event.context_chars for event in events],
            mode="lines+markers",
            name="Context chars",
            line={"color": "#38bdf8", "width": 3},
            customdata=labels,
            hovertemplate="%{customdata}<br>%{y:,} chars<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=[event.estimated_tokens for event in events],
            mode="lines+markers",
            name="Estimated tokens",
            line={"color": "#a78bfa", "width": 2, "dash": "dot"},
            customdata=labels,
            hovertemplate="%{customdata}<br>~%{y:,} tokens<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=x_values,
            y=[event.retry_count for event in events],
            name="Retries",
            marker_color="#fb7185",
            opacity=0.5,
            customdata=labels,
            hovertemplate="%{customdata}<br>%{y} retries<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        height=310,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,17,31,0.55)",
        font={"color": "#cbd5e1"},
        legend={"orientation": "h", "y": 1.14, "x": 0},
        hovermode="x unified",
        xaxis={"title": "State transition", "gridcolor": "#1e293b"},
    )
    figure.update_yaxes(
        title_text="Context size",
        gridcolor="#1e293b",
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Retries",
        gridcolor="rgba(0,0,0,0)",
        secondary_y=True,
        rangemode="tozero",
    )
    return figure


@st.cache_resource
def _controller_resource() -> AgentRunController:
    return AgentRunController()


def render_dashboard(
    *,
    configure_page: bool = True,
    default_prompt: str | None = None,
    graph_factory: DashboardGraphFactory | None = None,
    auto_launch: bool = False,
    scenario_note: str | None = None,
    artifact_first: bool = False,
) -> None:
    """Render Agent Studio standalone, embedded, or as a recording scenario."""
    if configure_page:
        st.set_page_config(
            page_title="TraceMind · Agent Observatory",
            page_icon="◈",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    st.markdown(_DASHBOARD_CSS, unsafe_allow_html=True)

    tracing = setup_tracing()
    controller = _controller_resource()

    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="eyebrow">LOCAL AGENT OBSERVATORY</div>
            <h1>TraceMind</h1>
            <p>Watch plans become code, code enter the sandbox, and failures heal.</p>
          </div>
          <div class="live-badge"><span></span> LIVE STATE GRAPH</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_surface = (
        st.sidebar
        if configure_page
        else st.expander("Agent Studio controls & telemetry", expanded=False)
    )
    launch = False
    with control_surface:
        st.markdown("### Mission control")
        prompt = st.text_area(
            "Agent task",
            value=default_prompt
            or (
                "Calculate the 50th Fibonacci number and produce an SVG trend "
                "chart for the first 50 values."
            ),
            height=150,
        )
        max_retries = st.slider("Maximum healing attempts", 1, 5, 3)
        if scenario_note:
            st.info(scenario_note)
        if configure_page:
            launch = st.button(
                "Launch TraceMind",
                type="primary",
                width="stretch",
            )
        st.divider()
        st.markdown("#### Phoenix telemetry")
        if tracing.collector_online:
            st.success(f"Connected · {tracing.project_name}")
            st.link_button(
                "Open Phoenix",
                tracing.ui_url,
                width="stretch",
            )
        elif tracing.enabled:
            st.warning("Collector offline; the dashboard remains fully local.")
            st.code("docker compose --profile observability up -d phoenix")
        else:
            st.info("Tracing disabled with TRACEMIND_TRACING_ENABLED=0.")
        st.caption("Observable rationale is shown; hidden chain-of-thought is not.")

    if not configure_page:
        launch = st.button(
            "Launch TraceMind",
            type="primary",
            width="stretch",
        )

    should_auto_launch = (
        auto_launch
        and "tracemind_run_id" not in st.session_state
        and not st.session_state.get("tracemind_auto_launch_started", False)
    )
    if launch or should_auto_launch:
        try:
            selected_graph_factory = graph_factory or (
                lambda retries: create_graph(max_retries=retries)
            )
            run_id = controller.start(
                prompt,
                graph_factory=lambda: selected_graph_factory(max_retries),
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["tracemind_run_id"] = run_id
            if should_auto_launch:
                st.session_state["tracemind_auto_launch_started"] = True

    run_id = st.session_state.get("tracemind_run_id")
    if not run_id:
        _render_empty_dashboard()
        return

    snapshot = controller.snapshot(run_id)
    left_column, right_column = st.columns([0.88, 1.12], gap="large")
    left_placeholder = left_column.empty()
    right_placeholder = right_column.empty()
    render_iteration = 0

    while snapshot.status in RUNNING_STATES:
        _render_left(left_placeholder, snapshot)
        _render_right(
            right_placeholder,
            snapshot,
            render_iteration=render_iteration,
            artifact_first=artifact_first,
            show_metrics=not artifact_first,
        )
        render_iteration += 1
        time.sleep(0.15)
        snapshot = controller.snapshot(run_id)

    _render_left(left_placeholder, snapshot)
    _render_right(
        right_placeholder,
        snapshot,
        render_iteration=render_iteration,
        artifact_first=artifact_first,
        show_metrics=not artifact_first,
    )


def _render_empty_dashboard() -> None:
    left_column, right_column = st.columns([0.88, 1.12], gap="large")
    with left_column:
        st.markdown("### Execution trace")
        st.info("Launch a task to stream state transitions here.")
        _render_node_grid(None, ())
    with right_column:
        st.markdown("### Live state topology")
        st.iframe(build_graph_html([]), height=410)
        st.markdown("### Context telemetry")
        st.plotly_chart(
            build_metrics_figure([]),
            width="stretch",
            config={"displayModeBar": False},
            key="dashboard-empty-context-telemetry",
        )


def _render_left(placeholder: Any, snapshot: RunSnapshot) -> None:
    with placeholder.container():
        heading_class = "spinner" if snapshot.status in RUNNING_STATES else "done-dot"
        st.markdown(
            f"""
            <div class="section-heading">
              <div><span class="{heading_class}"></span>
                Execution trace
              </div>
              <code>{html.escape(snapshot.run_id[:8])}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_node_grid(snapshot.active_node, snapshot.events)
        if not snapshot.events:
            st.markdown(
                '<div class="trace-card active">Preparing local runtime…</div>',
                unsafe_allow_html=True,
            )
        for event in snapshot.events:
            css_status = html.escape(event.status)
            latest_class = (
                " latest" if event.sequence == len(snapshot.events) else ""
            )
            st.markdown(
                f"""
                <div class="trace-card {css_status}{latest_class}">
                  <div class="trace-meta">
                    <span>{event.sequence:02d} · {html.escape(event.label)}</span>
                    <span>{event.elapsed_seconds:.2f}s</span>
                  </div>
                  <div class="trace-summary">{html.escape(event.summary)}</div>
                  <div class="trace-stats">
                    ~{event.estimated_tokens:,} tokens ·
                    retry {event.retry_count} ·
                    {event.artifact_count} execution artifact(s)
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if snapshot.status == "completed":
            st.success("Run completed successfully.")
        elif snapshot.status == "failed":
            st.error(snapshot.error or "Run failed.")


def _render_node_grid(
    active_node: str | None,
    events: tuple[RunEvent, ...] | list[RunEvent],
) -> None:
    statuses = {event.node: event.status for event in events}
    cards = []
    for node in NODE_ORDER:
        status = "active" if node == active_node else statuses.get(node, "pending")
        cards.append(
            f'<div class="node-pill {status}"><span></span>'
            f"{html.escape(NODE_LABELS[node])}</div>"
        )
    st.markdown(
        f'<div class="node-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_right(
    placeholder: Any,
    snapshot: RunSnapshot,
    *,
    render_iteration: int,
    artifact_first: bool = False,
    show_metrics: bool = True,
) -> None:
    with placeholder.container():
        state = snapshot.final_state or _latest_state(snapshot)
        if artifact_first and extract_artifacts(_latest_stdout(state)):
            _render_execution_output(state)
        st.markdown("### Live state topology")
        st.iframe(
            build_graph_html(snapshot.events, snapshot.active_node),
            height=410,
        )
        if show_metrics:
            st.markdown("### Context telemetry")
            st.plotly_chart(
                build_metrics_figure(snapshot.events),
                width="stretch",
                config={"displayModeBar": False},
                key=(
                    f"dashboard-context-telemetry-{snapshot.run_id}-"
                    f"{render_iteration}"
                ),
            )
        if not artifact_first or not extract_artifacts(_latest_stdout(state)):
            _render_execution_output(state)


def _latest_state(snapshot: RunSnapshot) -> dict[str, Any]:
    return snapshot.events[-1].state if snapshot.events else {}


def _latest_stdout(state: dict[str, Any]) -> str:
    artifacts = state.get("execution_artifacts") or []
    if not artifacts or not isinstance(artifacts[-1], dict):
        return ""
    result = artifacts[-1].get("result", {})
    return str(result.get("logs") or "") if isinstance(result, dict) else ""


def _render_execution_output(state: dict[str, Any]) -> None:
    artifacts = state.get("execution_artifacts") or []
    if not artifacts:
        return
    latest = artifacts[-1]
    if not isinstance(latest, dict):
        return
    result = latest.get("result", {})
    stdout = str(result.get("logs") or "")
    stderr = str(
        result.get("traceback")
        or result.get("error_message")
        or ""
    )
    succeeded = result.get("status") == "success"
    rendered = extract_artifacts(stdout) if succeeded else []

    st.markdown("### Sandbox artifacts" if succeeded else "### Failed sandbox attempt")
    if not succeeded:
        st.error(
            "No successful artifact was produced. The terminal below contains "
            "the final sandbox failure."
        )
    for artifact in rendered:
        st.caption(artifact.title)
        if artifact.kind == "svg":
            # Render in Streamlit's isolated component iframe so an SVG can be
            # displayed without granting it access to the parent dashboard.
            st.iframe(
                str(artifact.content),
                height=330,
            )
        elif artifact.kind == "png":
            st.image(artifact.content, width="stretch")
        elif artifact.kind == "markdown":
            st.markdown(str(artifact.content))
        else:
            st.json(artifact.content)

    with st.expander("Live terminal", expanded=not succeeded or not rendered):
        st.code(stdout or "(no stdout)", language="text")
        if stderr:
            st.code(stderr, language="text")


_DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
:root {
  --ink: #e8eef8;
  --muted: #8492a8;
  --cyan: #38bdf8;
  --violet: #a78bfa;
  --green: #10b981;
  --rose: #fb7185;
}
.stApp {
  background:
    radial-gradient(circle at 84% 8%, rgba(56,189,248,.13), transparent 29rem),
    radial-gradient(circle at 11% 40%, rgba(167,139,250,.10), transparent 28rem),
    #050a12;
  color: var(--ink);
  font-family: Manrope, sans-serif;
}
[data-testid="stSidebar"] {
  background: rgba(7, 14, 25, .96);
  border-right: 1px solid rgba(148,163,184,.13);
}
[data-testid="stHeader"] { background: rgba(5,10,18,.82); }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 { color: var(--ink); }
h1, h2, h3, h4 { font-family: Manrope, sans-serif; letter-spacing: -.025em; }
code, pre { font-family: "DM Mono", monospace !important; }
.hero {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1.4rem 1.65rem; margin: .1rem 0 1.4rem;
  border: 1px solid rgba(148,163,184,.14); border-radius: 22px;
  background: linear-gradient(135deg, rgba(15,23,42,.86), rgba(8,17,31,.54));
  box-shadow: 0 24px 80px rgba(0,0,0,.25);
}
.hero h1 { margin: .15rem 0 .2rem; font-size: 2.7rem; }
.hero p { margin: 0; color: var(--muted); }
.eyebrow { color: var(--cyan); font: 500 .71rem "DM Mono"; letter-spacing: .18em; }
.live-badge {
  font: 500 .72rem "DM Mono"; letter-spacing: .08em; color: #b8c7db;
  padding: .65rem .85rem; border: 1px solid rgba(56,189,248,.24);
  border-radius: 999px; background: rgba(56,189,248,.06);
}
.live-badge span, .node-pill span {
  display: inline-block; width: 7px; height: 7px; margin-right: .45rem;
  border-radius: 50%; background: var(--green);
  box-shadow: 0 0 14px var(--green); animation: pulse 1.4s infinite;
}
.section-heading {
  display: flex; justify-content: space-between; align-items: center;
  font-weight: 700; font-size: 1.18rem; margin: .65rem 0 .85rem;
}
.section-heading code { color: var(--muted); font-size: .72rem; font-weight: 400; }
.spinner {
  display: inline-block; width: 12px; height: 12px; margin-right: .48rem;
  border: 2px solid rgba(56,189,248,.2); border-top-color: var(--cyan);
  border-radius: 50%; animation: spin .8s linear infinite;
}
.done-dot {
  display: inline-block; width: 9px; height: 9px; margin-right: .52rem;
  background: var(--green); border-radius: 50%; box-shadow: 0 0 14px var(--green);
}
.node-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: .38rem; margin-bottom: 1rem; }
.node-pill {
  padding: .58rem .35rem; text-align: center; color: var(--muted);
  font: 500 .66rem "DM Mono"; border: 1px solid rgba(148,163,184,.13);
  border-radius: 10px; background: rgba(15,23,42,.52);
}
.node-pill span { width: 6px; height: 6px; background: #475569; box-shadow: none; animation: none; }
.node-pill.active { border-color: rgba(56,189,248,.6); color: #dff5ff; box-shadow: 0 0 23px rgba(56,189,248,.15); }
.node-pill.active span { background: var(--cyan); box-shadow: 0 0 12px var(--cyan); animation: pulse 1s infinite; }
.node-pill.success span { background: var(--green); }
.node-pill.error span { background: var(--rose); }
.node-pill.healing span { background: var(--violet); box-shadow: 0 0 10px var(--violet); }
.trace-card {
  position: relative; margin: .62rem 0; padding: .9rem 1rem;
  border: 1px solid rgba(148,163,184,.13); border-left: 3px solid #334155;
  border-radius: 13px; background: rgba(10,19,34,.74);
  animation: reveal .32s ease both;
}
.trace-card.active { border-left-color: var(--cyan); }
.trace-card.success { border-left-color: var(--green); }
.trace-card.error { border-left-color: var(--rose); }
.trace-card.healing { border-left-color: var(--violet); box-shadow: 0 0 25px rgba(167,139,250,.08); }
.trace-meta { display: flex; justify-content: space-between; color: var(--muted); font: 500 .68rem "DM Mono"; }
.trace-summary { color: #e2e8f0; margin: .38rem 0 .3rem; font-size: .89rem; line-height: 1.45; }
.trace-card.latest .trace-summary {
  animation: typewrite .7s steps(28, end) both;
}
.trace-stats { color: #65758c; font: 400 .65rem "DM Mono"; }
[data-testid="stPlotlyChart"], iframe {
  border: 1px solid rgba(148,163,184,.12); border-radius: 15px;
  background: rgba(8,17,31,.45);
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100% { opacity: .55; transform: scale(.85); } 50% { opacity: 1; transform: scale(1.18); } }
@keyframes reveal { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes typewrite { from { clip-path: inset(0 100% 0 0); } to { clip-path: inset(0 0 0 0); } }
@media (max-width: 900px) {
  .hero { align-items: flex-start; gap: 1rem; flex-direction: column; }
  .node-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""


__all__ = [
    "AgentRunController",
    "DashboardArtifact",
    "RunEvent",
    "RunSnapshot",
    "build_graph_html",
    "build_metrics_figure",
    "extract_artifacts",
    "render_dashboard",
]

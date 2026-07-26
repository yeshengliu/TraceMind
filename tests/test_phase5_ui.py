"""Phase 5 dashboard, async execution, and artifact rendering tests."""

from __future__ import annotations

import base64
import os
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

os.environ["TRACEMIND_TRACING_ENABLED"] = "0"

from src.agent.graph import create_graph  # noqa: E402
from src.agent.tools import (  # noqa: E402
    CodeGenerationOutput,
    PlanOutput,
    PlanStep,
)
from src.sandbox.test_sandbox import SandboxResult  # noqa: E402
from src.ui.dashboard import (  # noqa: E402
    AgentRunController,
    RunEvent,
    build_graph_html,
    build_metrics_figure,
    extract_artifacts,
)
from src.ui.tracing import setup_tracing  # noqa: E402


PROMPT = "Calculate the 50th Fibonacci number and render an SVG trend."


class DashboardPlanner:
    def invoke(self, input: object, **kwargs: Any) -> PlanOutput:
        return PlanOutput(
            objective=PROMPT,
            steps=[
                PlanStep(
                    index=1,
                    instruction="Calculate F(0) through F(50).",
                    expected_result="F(50) is printed.",
                ),
                PlanStep(
                    index=2,
                    instruction="Render the values as SVG.",
                    expected_result="A marked SVG artifact is printed.",
                ),
            ],
        )


class DashboardCoder:
    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        return CodeGenerationOutput(
            code=(
                "values = [0, 1]\n"
                "for _ in range(2, 51):\n"
                "    values.append(values[-1] + values[-2])\n"
                'print(f"FIBONACCI_50={values[50]}")\n'
            ),
            timeout_seconds=3,
            summary="Calculate and display the Fibonacci series.",
        )


def dashboard_sandbox(code: str, *, timeout_seconds: float) -> SandboxResult:
    assert "values.append" in code
    assert timeout_seconds == 3
    return SandboxResult(
        status="success",
        exit_code=0,
        logs=(
            "FIBONACCI_50=12586269025\n"
            'TREND_SVG=<svg xmlns="http://www.w3.org/2000/svg">'
            '<polyline points="0,10 10,1"/></svg>\n'
        ),
        duration_seconds=0.01,
    )


def test_async_controller_streams_complete_graph_run() -> None:
    setup_tracing.cache_clear()
    controller = AgentRunController(max_workers=1)
    try:
        run_id = controller.start(
            PROMPT,
            graph_factory=lambda: create_graph(
                planner=DashboardPlanner(),
                coder=DashboardCoder(),
                sandbox_runner=dashboard_sandbox,
            ),
        )
        snapshot = controller.wait(run_id, timeout=10)
    finally:
        controller.shutdown()

    assert snapshot.status == "completed"
    assert [event.node for event in snapshot.events] == [
        "planner",
        "coder_agent",
        "sandbox_executor",
        "error_detector",
    ]
    assert snapshot.final_state is not None
    assert "FIBONACCI_50=12586269025" in snapshot.final_state["final_output"]
    assert snapshot.final_state["observability"]["enabled"] is False
    assert all(event.context_chars > 0 for event in snapshot.events)
    assert controller.poll(run_id, after_sequence=2)[0].sequence == 3


def test_artifact_extractors_reject_invalid_images_and_keep_safe_types() -> None:
    one_pixel_png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + b"test payload"
    ).decode()
    output = "\n".join(
        [
            "TREND_SVG=<svg><path d='M0 0'/></svg>",
            f"ARTIFACT_PNG_BASE64={one_pixel_png}",
            "ARTIFACT_PNG_BASE64=not-base64",
            "ARTIFACT_MARKDOWN=## Result",
            'ARTIFACT_JSON={"value": 12586269025}',
        ]
    )

    artifacts = extract_artifacts(output)

    assert [artifact.kind for artifact in artifacts] == [
        "svg",
        "png",
        "markdown",
        "json",
    ]
    assert artifacts[-1].content == {"value": 12586269025}


def test_graph_and_metrics_render_from_events() -> None:
    events = [
        RunEvent(
            sequence=1,
            node="planner",
            label="Planner",
            status="success",
            summary="Prepared two steps.",
            timestamp=1,
            context_chars=800,
            estimated_tokens=200,
        ),
        RunEvent(
            sequence=2,
            node="reflect_and_heal",
            label="Self-Healing",
            status="healing",
            summary="Applied a targeted patch.",
            timestamp=2,
            context_chars=1_000,
            estimated_tokens=250,
            retry_count=1,
        ),
    ]

    graph_html = build_graph_html(events, active_node="sandbox_executor")
    figure = build_metrics_figure(events)

    assert "Planner" in graph_html
    assert "Self-Healing" in graph_html
    assert len(figure.data) == 3
    assert list(figure.data[0].y) == [800, 1_000]
    assert list(figure.data[2].y) == [0, 1]


def test_streamlit_app_boots_without_starting_an_agent() -> None:
    setup_tracing.cache_clear()
    app = AppTest.from_file("app.py").run(timeout=15)

    assert not app.exception
    assert "TraceMind" in " ".join(item.value for item in app.markdown)
    assert app.text_area[0].label == "Agent task"
    assert app.button[0].label == "Launch TraceMind"


@pytest.mark.integration
def test_async_dashboard_can_use_phase1_docker_binding() -> None:
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
        client.images.get("python:3.12-slim")
    except docker.errors.DockerException:
        pytest.skip("Docker daemon or python:3.12-slim image is unavailable")

    controller = AgentRunController(max_workers=1)
    try:
        run_id = controller.start(
            PROMPT,
            graph_factory=lambda: create_graph(
                planner=DashboardPlanner(),
                coder=DashboardCoder(),
            ),
        )
        snapshot = controller.wait(run_id, timeout=30)
    finally:
        controller.shutdown()

    assert snapshot.status == "completed"
    assert snapshot.final_state is not None
    assert "FIBONACCI_50=12586269025" in snapshot.final_state["final_output"]

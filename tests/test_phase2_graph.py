"""Phase 2 graph integration and retry guardrail tests."""

from __future__ import annotations

import json
from typing import Any

import docker
import pytest
from docker.errors import DockerException, ImageNotFound
from pydantic import ValidationError

from src.agent.graph import AgentState, create_graph, initial_state
from src.agent.reflection import (
    CodePatch,
    PatchEdit,
    ReflectionResult,
    apply_code_patch,
    parse_traceback,
)
from src.agent.tools import (
    CodeGenerationOutput,
    PlanOutput,
    PlanStep,
    PythonExecutionInput,
    validate_generated_code,
)
from src.sandbox.test_sandbox import SandboxResult, _CONTAINER_RUNNER


USER_PROMPT = "Calculate the 50th Fibonacci number and plot a trend chart."


class StaticPlanner:
    def invoke(self, input: object, **kwargs: Any) -> PlanOutput:
        return PlanOutput(
            objective=USER_PROMPT,
            steps=[
                PlanStep(
                    index=1,
                    instruction="Calculate Fibonacci values from F(0) through F(50).",
                    expected_result="F(50) is printed.",
                ),
                PlanStep(
                    index=2,
                    instruction="Render the trend with a standard-library SVG polyline.",
                    expected_result="An SVG trend chart is printed.",
                ),
            ],
        )


class StaticCoder:
    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        code = """\
values = [0, 1]
for _ in range(2, 51):
    values.append(values[-1] + values[-2])

sample = values[::5]
width, height = 500, 160
maximum = max(sample)
points = " ".join(
    f"{index * width / (len(sample) - 1):.1f},{height - value * height / maximum:.1f}"
    for index, value in enumerate(sample)
)
svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
    f'<polyline fill="none" stroke="#38bdf8" stroke-width="3" points="{points}"/>'
    "</svg>"
)
print(f"FIBONACCI_50={values[50]}")
print(f"TREND_SVG={svg}")
"""
        return CodeGenerationOutput(
            code=code,
            timeout_seconds=5,
            summary="Calculate F(50) and print a dependency-free SVG trend chart.",
        )


def _docker_is_ready() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        client.images.get("python:3.12-slim")
    except (DockerException, ImageNotFound):
        return False
    return True


@pytest.mark.integration
def test_fibonacci_prompt_runs_through_graph_and_real_sandbox() -> None:
    assert _docker_is_ready(), "Docker daemon and python:3.12-slim are required"

    app = create_graph(planner=StaticPlanner(), coder=StaticCoder())
    final_state: AgentState = app.invoke(initial_state(USER_PROMPT))

    assert final_state["status"] == "completed"
    assert final_state["retry_count"] == 0
    assert len(final_state["execution_artifacts"]) == 1
    assert final_state["execution_artifacts"][0]["result"]["status"] == "success"
    assert "FIBONACCI_50=12586269025" in final_state["final_output"]
    assert "TREND_SVG=<svg" in final_state["final_output"]


class AlwaysBrokenCoder:
    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        return CodeGenerationOutput(
            code="raise RuntimeError('still broken')",
            timeout_seconds=1,
            summary="Exercise retry guardrails.",
        )


class ProgressiveReflector:
    def __init__(self) -> None:
        self.calls = 0

    def reflect(self, failed_code: str, traceback_text: str) -> ReflectionResult:
        self.calls += 1
        patch = CodePatch(
            root_cause="The deliberately broken expression still raises.",
            edits=[
                PatchEdit(
                    old_text="still broken",
                    new_text=f"still broken {self.calls}",
                    reason="Change the failed code while exercising retry routing.",
                )
            ],
            validation_notes="The fake sandbox remains configured to fail.",
        )
        patched_code, unified_diff = apply_code_patch(failed_code, patch)
        return ReflectionResult(
            parsed_traceback=parse_traceback(traceback_text),
            patch=patch,
            patched_code=patched_code,
            unified_diff=unified_diff,
        )


def _always_fails(code: str, *, timeout_seconds: float) -> SandboxResult:
    return SandboxResult(
        status="error",
        exit_code=1,
        error_type="RuntimeError",
        error_message="still broken",
        traceback="RuntimeError: still broken",
        duration_seconds=0.01,
    )


def test_retry_guardrail_stops_after_configured_corrections() -> None:
    app = create_graph(
        planner=StaticPlanner(),
        coder=AlwaysBrokenCoder(),
        reflector=ProgressiveReflector(),
        sandbox_runner=_always_fails,
        max_retries=3,
    )
    final_state: AgentState = app.invoke(initial_state(USER_PROMPT))

    assert final_state["status"] == "failed"
    assert final_state["retry_count"] == 3
    assert len(final_state["execution_artifacts"]) <= 2
    assert len(final_state["error_stack"]) == 1
    assert len(final_state["patch_history"]) == 1
    assert "RuntimeError: still broken" in final_state["final_output"]


def test_unchanged_fallback_code_retries_without_repeating_the_sandbox() -> None:
    sandbox_calls = 0

    class RejectingReflector:
        def reflect(self, failed_code: str, traceback_text: str) -> ReflectionResult:
            raise ValueError("patch did not pass syntax validation")

    def counted_failure(code: str, *, timeout_seconds: float) -> SandboxResult:
        nonlocal sandbox_calls
        sandbox_calls += 1
        return _always_fails(code, timeout_seconds=timeout_seconds)

    app = create_graph(
        planner=StaticPlanner(),
        coder=AlwaysBrokenCoder(),
        reflector=RejectingReflector(),
        sandbox_runner=counted_failure,
        max_retries=3,
    )
    final_state: AgentState = app.invoke(initial_state(USER_PROMPT))

    assert final_state["status"] == "failed"
    assert final_state["retry_count"] == 3
    assert sandbox_calls == 1
    assert len(final_state["execution_artifacts"]) == 1
    assert "retry limit" in final_state["final_output"]
    assert "RuntimeError: still broken" in final_state["final_output"]


def test_tool_schema_rejects_unknown_fields_and_unsafe_timeout() -> None:
    with pytest.raises(ValidationError):
        PythonExecutionInput.model_validate(
            {"code": "print('hello')", "timeout_seconds": 1, "network": True}
        )
    with pytest.raises(ValidationError):
        PythonExecutionInput(code="print('hello')", timeout_seconds=31)


def test_validate_generated_code_rejects_non_stdlib_imports() -> None:
    assert validate_generated_code("import numpy as np\nprint(np.zeros(1))") is not None
    assert (
        validate_generated_code("from matplotlib import pyplot as plt\nplt.plot([1])")
        is not None
    )
    assert (
        validate_generated_code(
            "def chart():\n    import plotly.express as px\n    return px"
        )
        is not None
    )


def test_validate_generated_code_accepts_stdlib_only_code() -> None:
    code = (
        "import json\n"
        "from datetime import datetime\n"
        "import statistics as stats\n"
        'print(json.dumps({"day": datetime.now().isoformat()}))\n'
        "print(stats.median([1, 2, 3]))\n"
    )
    assert validate_generated_code(code) is None
    assert validate_generated_code("this is not python :::") is not None


class InvalidThenValidCoder:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        self.calls += 1
        if self.calls == 1:
            return CodeGenerationOutput.model_validate(
                {
                    "code": "print('invalid timeout')",
                    "timeout_seconds": 60,
                    "summary": "Deliberately violate the timeout schema.",
                }
            )
        return StaticCoder().invoke(input, **kwargs)


def _always_succeeds(code: str, *, timeout_seconds: float) -> SandboxResult:
    return SandboxResult(
        status="success",
        exit_code=0,
        logs="FIBONACCI_50=12586269025\nTREND_SVG=<svg></svg>\n",
        duration_seconds=0.01,
    )


def test_coder_schema_violation_enters_bounded_healing_loop() -> None:
    coder = InvalidThenValidCoder()
    app = create_graph(
        planner=StaticPlanner(),
        coder=coder,
        sandbox_runner=_always_succeeds,
        max_retries=3,
    )
    final_state: AgentState = app.invoke(initial_state(USER_PROMPT))

    assert final_state["status"] == "completed"
    assert final_state["retry_count"] == 1
    assert len(final_state["execution_artifacts"]) == 1
    assert "Coder schema validation failed" in final_state["error_stack"][0]
    assert coder.calls == 2


class FailingPlanner:
    def invoke(self, input: object, **kwargs: Any) -> PlanOutput:
        raise RuntimeError("ollama offline")


class ExplodingCoder:
    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        raise AssertionError("coder must not be called after a planner failure")


def test_planner_failure_ends_run_without_calling_coder() -> None:
    app = create_graph(
        planner=FailingPlanner(),
        coder=ExplodingCoder(),
        sandbox_runner=_always_succeeds,
    )
    final_state: AgentState = app.invoke(initial_state(USER_PROMPT))

    assert final_state["status"] == "failed"
    assert "Planner invocation failed: RuntimeError: ollama offline" in final_state[
        "error_stack"
    ][-1]
    assert "ollama offline" in final_state["final_output"]


class AlwaysRaisingCoder:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        self.calls += 1
        raise RuntimeError("model timeout")


def test_coder_invocation_exception_enters_bounded_healing_loop() -> None:
    coder = AlwaysRaisingCoder()
    app = create_graph(
        planner=StaticPlanner(),
        coder=coder,
        sandbox_runner=_always_succeeds,
        max_retries=3,
    )
    final_state: AgentState = app.invoke(initial_state(USER_PROMPT))

    assert final_state["status"] == "failed"
    assert final_state["retry_count"] == 3
    assert coder.calls == 4
    assert "Coder invocation failed: RuntimeError: model timeout" in final_state[
        "error_stack"
    ][-1]
    assert "model timeout" in final_state["final_output"]


def test_sandbox_runner_treats_clean_system_exit_as_success(
    capsys: Any,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TRACEMIND_SNIPPET", "import sys\nsys.exit(0)\n")
    with pytest.raises(SystemExit) as excinfo:
        exec(compile(_CONTAINER_RUNNER, "<runner>", "exec"), {"__name__": "__runner__"})

    assert excinfo.value.code == 0
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert payload["status"] == "success"


def test_sandbox_runner_reports_nonzero_system_exit_as_error(
    capsys: Any,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TRACEMIND_SNIPPET", "import sys\nsys.exit(3)\n")
    with pytest.raises(SystemExit) as excinfo:
        exec(compile(_CONTAINER_RUNNER, "<runner>", "exec"), {"__name__": "__runner__"})

    assert excinfo.value.code == 3
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert payload["status"] == "error"
    assert payload["error_type"] == "SystemExit"


@pytest.mark.ollama
def test_live_ollama_graph() -> None:
    assert _docker_is_ready(), "Docker daemon and python:3.12-slim are required"

    final_state: AgentState = create_graph().invoke(initial_state(USER_PROMPT))
    assert final_state["status"] == "completed", final_state["final_output"]

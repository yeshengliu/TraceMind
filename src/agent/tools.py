"""Strict tool schemas and bindings for TraceMind."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from src.sandbox.test_sandbox import SandboxResult, run_in_sandbox


class StrictModel(BaseModel):
    """Base schema that rejects undeclared model or tool fields."""

    model_config = ConfigDict(extra="forbid")


class PlanStep(StrictModel):
    """One observable, implementation-oriented step."""

    index: int = Field(ge=1)
    instruction: str = Field(min_length=1, max_length=500)
    expected_result: str = Field(min_length=1, max_length=500)


class PlanOutput(StrictModel):
    """Schema enforced on planner model output."""

    objective: str = Field(min_length=1, max_length=1_000)
    steps: list[PlanStep] = Field(min_length=1, max_length=12)


class PythonExecutionInput(StrictModel):
    """Validated arguments accepted by the Python sandbox tool."""

    code: str = Field(
        min_length=1,
        max_length=50_000,
        description="Complete Python source code to execute.",
    )
    timeout_seconds: float = Field(
        ge=0.1,
        le=30.0,
        description="Hard wall-clock execution limit.",
    )


class CodeGenerationOutput(PythonExecutionInput):
    """Schema enforced on coder and healing model output."""

    summary: str = Field(
        min_length=1,
        max_length=1_000,
        description="Concise implementation summary; never hidden chain-of-thought.",
    )


class PythonExecutionOutput(StrictModel):
    """Stable output contract exposed at the tool boundary."""

    status: Literal["success", "error", "timeout", "infrastructure_error"]
    exit_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    logs: str = ""
    duration_seconds: float = Field(ge=0)

    @classmethod
    def from_sandbox_result(cls, result: SandboxResult) -> "PythonExecutionOutput":
        return cls.model_validate(result.model_dump())


SandboxRunner = Callable[..., SandboxResult]


def execute_python(
    request: PythonExecutionInput,
    *,
    runner: SandboxRunner = run_in_sandbox,
) -> PythonExecutionOutput:
    """Validate a call, execute it in Phase 1, and validate the response."""
    result = runner(request.code, timeout_seconds=request.timeout_seconds)
    return PythonExecutionOutput.from_sandbox_result(result)


def _python_sandbox_tool(code: str, timeout_seconds: float) -> dict[str, Any]:
    request = PythonExecutionInput(code=code, timeout_seconds=timeout_seconds)
    return execute_python(request).model_dump(mode="json")


python_sandbox_tool = StructuredTool.from_function(
    func=_python_sandbox_tool,
    name="python_sandbox",
    description=(
        "Execute Python in an offline, unprivileged Docker container with strict "
        "CPU, memory, PID, filesystem, and wall-clock limits."
    ),
    args_schema=PythonExecutionInput,
)

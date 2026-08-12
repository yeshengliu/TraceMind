"""Strict tool schemas and bindings for TraceMind."""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.sandbox.test_sandbox import SandboxResult, run_in_sandbox


_STDLIB_MODULES = frozenset(sys.stdlib_module_names)


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


def validate_generated_code(code: str) -> str | None:
    """Reject imports the offline stdlib-only sandbox cannot satisfy."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return (
            "Sandbox contract violation: generated code is not valid Python "
            f"(line {exc.lineno}: {exc.msg}). Return one complete, parseable "
            "program; never place a literal newline inside a quoted string."
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _STDLIB_MODULES:
                    return (
                        "Sandbox contract violation: the program imports "
                        f"non-stdlib module {alias.name!r}, which is unavailable "
                        "in the offline python:3.12-slim container. Replace it "
                        "with a standard-library or self-contained implementation."
                    )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
        ):
            root = node.module.split(".", 1)[0]
            if root not in _STDLIB_MODULES:
                return (
                    "Sandbox contract violation: the program imports "
                    f"non-stdlib module {node.module!r}, which is unavailable "
                    "in the offline python:3.12-slim container. Replace it "
                    "with a standard-library or self-contained implementation."
                )
    return None


SandboxRunner = Callable[..., SandboxResult]


def execute_python(
    request: PythonExecutionInput,
    *,
    runner: SandboxRunner = run_in_sandbox,
) -> PythonExecutionOutput:
    """Validate a call, execute it in Phase 1, and validate the response."""
    result = runner(request.code, timeout_seconds=request.timeout_seconds)
    return PythonExecutionOutput.from_sandbox_result(result)

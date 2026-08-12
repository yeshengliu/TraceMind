"""Transparent LangGraph workflow for planning and sandboxed code execution."""

from __future__ import annotations

import json
from typing import Any, Literal, NotRequired, Protocol, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from src.agent.llm import ModelRole, get_structured_llm
from src.agent.reflection import (
    ReflectionResult,
    TracebackReflector,
)
from src.agent.tools import (
    CodeGenerationOutput,
    PlanOutput,
    PythonExecutionInput,
    PythonExecutionOutput,
    SandboxRunner,
    execute_python,
    validate_generated_code,
)
from src.memory.pruner import ContextPruner, EpisodicMemory
from src.sandbox.test_sandbox import run_in_sandbox


DEFAULT_MAX_RETRIES = 3

PLANNER_SYSTEM_PROMPT = """\
You are TraceMind's planning node. Convert the user request into a small,
ordered implementation plan for a Python program. Every step must be observable
and testable. The program runs in an offline container with only the Python
standard library. Never plan third-party imports, GUI display, network access,
or exponential brute-force algorithms. Prefer efficient iterative algorithms
and dependency-free SVG or ASCII output for charts. Return only data matching
the supplied JSON schema.
"""

CODER_SYSTEM_PROMPT = """\
You are TraceMind's code generation node. Produce one complete Python program
that fulfills the supplied plan. The program runs in an offline python:3.12-slim
container with only the Python standard library installed. Never import
matplotlib, numpy, pandas, seaborn, plotly, or any other third-party package.
For charts, generate and print a literal SVG string or an ASCII chart; do not
open a GUI or depend on a file surviving the container. Print every requested
result or artifact to stdout. Prefer SVG for charts. Use triple-quoted Python
strings for multiline SVG templates; never split a single- or double-quoted
string literal across physical lines. If using ASCII, build a list of lines and
print each line. The complete program must pass ast.parse before you return it.
Do not access the network, spawn processes, or read host files. Use a
timeout_seconds value from 0.1 through 30 inclusive. Return only data matching
the supplied JSON schema.
"""

REGENERATION_FALLBACK_PROMPT = f"""\
{CODER_SYSTEM_PROMPT}
A targeted edit could not produce valid Python. Use the supplied failed_code
and latest_error to return one complete corrected replacement. Eliminate the
reported failure instead of preserving the malformed block.
"""

class StructuredModel(Protocol):
    """Minimal interface required from structured LangChain model runnables."""

    def invoke(self, input: object, **kwargs: Any) -> BaseModel | dict[str, Any]:
        ...


class Reflector(Protocol):
    def reflect(self, failed_code: str, traceback_text: str) -> ReflectionResult:
        ...


class AgentState(TypedDict):
    """Shared, inspectable state persisted across every graph node."""

    messages: list[AnyMessage]
    current_plan: list[str]
    error_stack: list[str]
    retry_count: int
    execution_artifacts: list[dict[str, Any]]
    patch_history: list[dict[str, Any]]
    history_summary: list[str]
    generated_code: NotRequired[str]
    code_summary: NotRequired[str]
    requested_timeout_seconds: NotRequired[float]
    status: NotRequired[Literal["planning", "coding", "executing", "healing", "completed", "failed"]]
    final_output: NotRequired[str]
    episode_id: NotRequired[str]


class GraphDependencies(TypedDict):
    planner: StructuredModel
    coder: StructuredModel
    reflector: Reflector
    sandbox_runner: SandboxRunner
    max_retries: int
    pruner: ContextPruner
    episodic_memory: EpisodicMemory


def initial_state(user_prompt: str) -> AgentState:
    """Build a complete graph input with stable collection defaults."""
    if not user_prompt.strip():
        raise ValueError("user_prompt must not be empty")
    return AgentState(
        messages=[HumanMessage(content=user_prompt)],
        current_plan=[],
        error_stack=[],
        retry_count=0,
        execution_artifacts=[],
        patch_history=[],
        history_summary=[],
        status="planning",
    )


def _validate_response(schema: type[BaseModel], response: object) -> BaseModel:
    if isinstance(response, schema):
        return response
    return schema.model_validate(response)


def _latest_user_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("AgentState.messages must contain a HumanMessage")


def _artifact_error(artifact: dict[str, Any]) -> str:
    result = artifact["result"]
    details = result.get("traceback") or result.get("error_message") or result["status"]
    return f"Attempt {artifact['attempt']}: {details}"


def _latest_runtime_error(state: AgentState) -> str | None:
    artifacts = state.get("execution_artifacts") or []
    if not artifacts:
        return None
    result = artifacts[-1].get("result", {})
    return str(
        result.get("traceback")
        or result.get("error_message")
        or result.get("status")
        or ""
    ) or None


def create_graph(
    *,
    planner: StructuredModel | None = None,
    coder: StructuredModel | None = None,
    reflector: Reflector | None = None,
    sandbox_runner: SandboxRunner = run_in_sandbox,
    max_retries: int = DEFAULT_MAX_RETRIES,
    pruner: ContextPruner | None = None,
    episodic_memory: EpisodicMemory | None = None,
):
    """Build and compile the graph with injectable dependencies for testing."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    context_pruner = pruner or ContextPruner()
    episode_store = episodic_memory or EpisodicMemory()
    dependencies: GraphDependencies = {
        "planner": planner or get_structured_llm(ModelRole.PLANNER, PlanOutput),
        "coder": coder or get_structured_llm(ModelRole.CODER, CodeGenerationOutput),
        "reflector": reflector or TracebackReflector(),
        "sandbox_runner": sandbox_runner,
        "max_retries": max_retries,
        "pruner": context_pruner,
        "episodic_memory": episode_store,
    }

    def transition(
        state: AgentState,
        updates: dict[str, Any],
        *,
        messages: list[AnyMessage] | None = None,
        prune: bool = True,
    ) -> AgentState:
        merged: dict[str, Any] = dict(state)
        merged.update(updates)
        merged["messages"] = [*state["messages"], *(messages or [])]
        if prune:
            merged = dependencies["pruner"].prune_state(merged)
        return AgentState(**merged)

    def planner_node(state: AgentState) -> AgentState:
        request = _latest_user_text(state["messages"])
        try:
            response = dependencies["planner"].invoke(
                [
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=request),
                ]
            )
            plan = _validate_response(PlanOutput, response)
        except Exception as exc:
            planner_error = f"Planner invocation failed: {type(exc).__name__}: {exc}"
            return transition(
                state,
                {
                    "error_stack": [*state["error_stack"], planner_error],
                    "status": "failed",
                    "final_output": planner_error,
                },
                messages=[
                    AIMessage(content="Planning failed; the run cannot continue.")
                ],
            )
        assert isinstance(plan, PlanOutput)
        visible_plan = [
            f"{step.index}. {step.instruction} -> {step.expected_result}"
            for step in plan.steps
        ]
        return transition(
            state,
            {"current_plan": visible_plan, "status": "coding"},
            messages=[AIMessage(content=f"Plan created with {len(plan.steps)} steps.")],
        )

    def coder_node(state: AgentState) -> AgentState:
        request = _latest_user_text(state["messages"])
        runtime_error = _latest_runtime_error(state)
        context = {
            "user_request": request,
            "plan": state["current_plan"],
            "retry_count": state["retry_count"],
            "latest_error": (
                runtime_error
                or (state["error_stack"][-1] if state["error_stack"] else None)
            ),
            "failed_code": state.get("generated_code") if state["error_stack"] else None,
            "regeneration_requirement": (
                "Return syntactically valid Python that materially changes the "
                "failed block and passes ast.parse."
                if state["error_stack"]
                else None
            ),
        }
        system_prompt = (
            REGENERATION_FALLBACK_PROMPT
            if state["error_stack"] and state.get("generated_code")
            else CODER_SYSTEM_PROMPT
        )
        generated: CodeGenerationOutput | None = None
        coder_error: str | None = None
        try:
            response = dependencies["coder"].invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=json.dumps(context, indent=2)),
                ]
            )
            generated = _validate_response(CodeGenerationOutput, response)
        except Exception as exc:
            if isinstance(exc, ValidationError):
                coder_error = f"Coder schema validation failed: {exc}"
            else:
                coder_error = f"Coder invocation failed: {type(exc).__name__}: {exc}"
        if coder_error is None and generated is not None:
            coder_error = validate_generated_code(generated.code)
        if coder_error is not None:
            exhausted = state["retry_count"] >= dependencies["max_retries"]
            return transition(
                state,
                {
                    "error_stack": [*state["error_stack"], coder_error],
                    "status": "failed" if exhausted else "healing",
                    "final_output": coder_error,
                },
                messages=[
                    AIMessage(
                        content=(
                            "Coder failed; "
                            f"{'retry limit reached' if exhausted else 'requesting correction'}."
                        )
                    )
                ],
            )
        assert isinstance(generated, CodeGenerationOutput)
        previous_code = state.get("generated_code")
        if (
            state["error_stack"]
            and previous_code
            and generated.code.strip() == previous_code.strip()
        ):
            exhausted = state["retry_count"] >= dependencies["max_retries"]
            stalled_error = (
                "Healing failed after the retry limit because fallback regeneration "
                "kept returning unchanged code."
                if exhausted
                else "Fallback regeneration returned unchanged code; retrying repair "
                "without re-running the same sandbox failure."
            )
            unresolved_error = runtime_error or state["error_stack"][-1]
            return transition(
                state,
                {
                    "error_stack": [*state["error_stack"], stalled_error],
                    "status": "failed" if exhausted else "healing",
                    "final_output": f"{stalled_error}\n\n{unresolved_error}",
                },
                messages=[
                    AIMessage(
                        content=(
                            "Fallback regeneration made no code change; "
                            + (
                                "retry limit reached."
                                if exhausted
                                else "requesting another bounded repair without "
                                "re-running unchanged code."
                            )
                        )
                    )
                ],
            )
        return transition(
            state,
            {
                "generated_code": generated.code,
                "code_summary": generated.summary,
                "requested_timeout_seconds": generated.timeout_seconds,
                "status": "executing",
            },
            messages=[AIMessage(content=f"Code prepared: {generated.summary}")],
        )

    def route_after_coding(
        state: AgentState,
    ) -> Literal["sandbox_executor", "reflect_and_heal", "__end__"]:
        if state["status"] == "executing":
            return "sandbox_executor"
        if state["status"] == "healing":
            return "reflect_and_heal"
        return END

    def route_after_planning(
        state: AgentState,
    ) -> Literal["coder_agent", "__end__"]:
        return "coder_agent" if state["status"] != "failed" else END

    def sandbox_executor_node(state: AgentState) -> AgentState:
        request = PythonExecutionInput(
            code=state["generated_code"],
            timeout_seconds=state["requested_timeout_seconds"],
        )
        result = execute_python(request, runner=dependencies["sandbox_runner"])
        artifact = {
            "attempt": state["retry_count"] + 1,
            "tool": "python_sandbox",
            "request": request.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
        }
        return transition(
            state,
            {
                "execution_artifacts": [*state["execution_artifacts"], artifact],
                "status": "executing",
            },
            prune=False,
        )

    def error_detector_node(state: AgentState) -> AgentState:
        artifact = state["execution_artifacts"][-1]
        result = PythonExecutionOutput.model_validate(artifact["result"])
        if result.status == "success":
            completed = transition(
                state,
                {"status": "completed", "final_output": result.logs},
                messages=[AIMessage(content="Sandbox execution completed successfully.")],
            )
            episode = dependencies["episodic_memory"].record_success(completed)
            return transition(completed, {"episode_id": episode.episode_id})

        errors = [*state["error_stack"], _artifact_error(artifact)]
        exhausted = state["retry_count"] >= dependencies["max_retries"]
        return transition(
            state,
            {
                "error_stack": errors,
                "status": "failed" if exhausted else "healing",
                "final_output": result.traceback or result.error_message or result.logs,
            },
            messages=[
                AIMessage(
                    content=(
                        f"Execution failed with {result.error_type or result.status}; "
                        f"{'retry limit reached' if exhausted else 'starting correction'}."
                    )
                )
            ],
        )

    def route_after_detection(state: AgentState) -> Literal["reflect_and_heal", "__end__"]:
        return "reflect_and_heal" if state["status"] == "healing" else END

    def reflect_and_heal_node(state: AgentState) -> AgentState:
        retry_count = state["retry_count"] + 1
        artifacts = list(state["execution_artifacts"])

        if not artifacts:
            return transition(
                state,
                {"retry_count": retry_count, "status": "coding"},
                messages=[
                    AIMessage(
                        content=f"Schema correction {retry_count} of {max_retries}."
                    )
                ],
            )

        latest_artifact = artifacts[-1]
        result = PythonExecutionOutput.model_validate(latest_artifact["result"])
        traceback_text = (
            result.traceback
            or (
                f"{result.error_type or 'SandboxError'}: "
                f"{result.error_message or result.status}"
            )
        )
        failed_code = str(latest_artifact["request"]["code"])
        try:
            reflection = dependencies["reflector"].reflect(failed_code, traceback_text)
        except Exception as exc:
            reflection_error = f"Reflector failed: {type(exc).__name__}: {exc}"
            return transition(
                state,
                {
                    "retry_count": retry_count,
                    "error_stack": [*state["error_stack"], reflection_error],
                    "status": "coding",
                    "final_output": reflection_error,
                },
                messages=[
                    AIMessage(
                        content=(
                            f"Targeted patch failed validation on correction {retry_count}; "
                            "falling back to schema-constrained regeneration."
                        )
                    )
                ],
            )

        latest_artifact = {
            **latest_artifact,
            "reflection": reflection.model_dump(mode="json"),
        }
        artifacts[-1] = latest_artifact
        patch_record = {
            "attempt": retry_count,
            "root_cause": reflection.patch.root_cause,
            "edits": [
                edit.model_dump(mode="json") for edit in reflection.patch.edits
            ],
            "unified_diff": reflection.unified_diff,
        }
        return transition(
            state,
            {
                "retry_count": retry_count,
                "execution_artifacts": artifacts,
                "patch_history": [*state["patch_history"], patch_record],
                "generated_code": reflection.patched_code,
                "code_summary": f"Targeted patch: {reflection.patch.root_cause}",
                "status": "executing",
            },
            messages=[
                AIMessage(
                    content=(
                        f"Applied targeted correction {retry_count} of {max_retries}: "
                        f"{reflection.patch.root_cause}"
                    )
                )
            ],
        )

    def route_after_reflection(
        state: AgentState,
    ) -> Literal["sandbox_executor", "coder_agent", "__end__"]:
        if state["status"] == "executing":
            return "sandbox_executor"
        if state["status"] == "coding":
            return "coder_agent"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("coder_agent", coder_node)
    builder.add_node("sandbox_executor", sandbox_executor_node)
    builder.add_node("error_detector", error_detector_node)
    builder.add_node("reflect_and_heal", reflect_and_heal_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planning,
        {"coder_agent": "coder_agent", END: END},
    )
    builder.add_conditional_edges(
        "coder_agent",
        route_after_coding,
        {
            "sandbox_executor": "sandbox_executor",
            "reflect_and_heal": "reflect_and_heal",
            END: END,
        },
    )
    builder.add_edge("sandbox_executor", "error_detector")
    builder.add_conditional_edges(
        "error_detector",
        route_after_detection,
        {"reflect_and_heal": "reflect_and_heal", END: END},
    )
    builder.add_conditional_edges(
        "reflect_and_heal",
        route_after_reflection,
        {
            "sandbox_executor": "sandbox_executor",
            "coder_agent": "coder_agent",
            END: END,
        },
    )
    return builder.compile()


graph = create_graph()

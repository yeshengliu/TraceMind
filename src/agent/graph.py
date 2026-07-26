"""Transparent LangGraph workflow for planning and sandboxed code execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, Literal, NotRequired, Protocol, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ValidationError

from src.agent.llm import ModelRole, get_structured_llm
from src.agent.tools import (
    CodeGenerationOutput,
    PlanOutput,
    PythonExecutionInput,
    PythonExecutionOutput,
    SandboxRunner,
    execute_python,
)
from src.sandbox.test_sandbox import run_in_sandbox


DEFAULT_MAX_RETRIES = 3

PLANNER_SYSTEM_PROMPT = """\
You are TraceMind's planning node. Convert the user request into a small,
ordered implementation plan for a Python program. Every step must be observable
and testable. Return only data matching the supplied JSON schema.
"""

CODER_SYSTEM_PROMPT = """\
You are TraceMind's code generation node. Produce one complete Python program
that fulfills the supplied plan. The program runs in an offline python:3.12-slim
container with only the Python standard library installed. Never import
matplotlib, numpy, pandas, seaborn, plotly, or any other third-party package.
For charts, generate and print a literal SVG string or an ASCII chart; do not
open a GUI or depend on a file surviving the container. Print every requested
result or artifact to stdout. Do not access the network, spawn processes, or
read host files. Use a timeout_seconds value from 0.1 through 30 inclusive.
Return only data matching the supplied JSON schema.
"""

HEALING_SYSTEM_PROMPT = """\
The previous sandbox execution failed. Produce a corrected, complete replacement
program. Use the error type, message, and traceback as evidence. Preserve the
original objective and do not repeat the failing implementation. The sandbox
contains no third-party packages: never import matplotlib, numpy, pandas,
seaborn, plotly, or similar libraries. Replace unavailable plotting packages
with a printed literal SVG string or ASCII chart. Use a timeout_seconds value
from 0.1 through 30 inclusive. Return only data matching the supplied JSON
schema.
"""


class StructuredModel(Protocol):
    """Minimal interface required from structured LangChain model runnables."""

    def invoke(self, input: object, **kwargs: Any) -> BaseModel | dict[str, Any]:
        ...


class AgentState(TypedDict):
    """Shared, inspectable state persisted across every graph node."""

    messages: Annotated[list[AnyMessage], add_messages]
    current_plan: list[str]
    error_stack: list[str]
    retry_count: int
    execution_artifacts: list[dict[str, Any]]
    generated_code: NotRequired[str]
    code_summary: NotRequired[str]
    requested_timeout_seconds: NotRequired[float]
    status: NotRequired[Literal["planning", "coding", "executing", "healing", "completed", "failed"]]
    final_output: NotRequired[str]


class GraphDependencies(TypedDict):
    planner: StructuredModel
    coder: StructuredModel
    sandbox_runner: SandboxRunner
    max_retries: int


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


def create_graph(
    *,
    planner: StructuredModel | None = None,
    coder: StructuredModel | None = None,
    sandbox_runner: SandboxRunner = run_in_sandbox,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    """Build and compile the graph with injectable dependencies for testing."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    dependencies: GraphDependencies = {
        "planner": planner or get_structured_llm(ModelRole.PLANNER, PlanOutput),
        "coder": coder or get_structured_llm(ModelRole.CODER, CodeGenerationOutput),
        "sandbox_runner": sandbox_runner,
        "max_retries": max_retries,
    }

    def planner_node(state: AgentState) -> dict[str, Any]:
        request = _latest_user_text(state["messages"])
        response = dependencies["planner"].invoke(
            [SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=request)]
        )
        plan = _validate_response(PlanOutput, response)
        assert isinstance(plan, PlanOutput)
        visible_plan = [
            f"{step.index}. {step.instruction} -> {step.expected_result}"
            for step in plan.steps
        ]
        return {
            "current_plan": visible_plan,
            "status": "coding",
            "messages": [AIMessage(content=f"Plan created with {len(plan.steps)} steps.")],
        }

    def coder_node(state: AgentState) -> dict[str, Any]:
        request = _latest_user_text(state["messages"])
        context = {
            "user_request": request,
            "plan": state["current_plan"],
            "retry_count": state["retry_count"],
            "latest_error": state["error_stack"][-1] if state["error_stack"] else None,
        }
        system_prompt = HEALING_SYSTEM_PROMPT if state["error_stack"] else CODER_SYSTEM_PROMPT
        try:
            response = dependencies["coder"].invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=json.dumps(context, indent=2)),
                ]
            )
            generated = _validate_response(CodeGenerationOutput, response)
        except ValidationError as exc:
            schema_error = f"Coder schema validation failed: {exc}"
            exhausted = state["retry_count"] >= dependencies["max_retries"]
            return {
                "error_stack": [*state["error_stack"], schema_error],
                "status": "failed" if exhausted else "healing",
                "final_output": schema_error,
                "messages": [
                    AIMessage(
                        content=(
                            "Coder output violated the tool schema; "
                            f"{'retry limit reached' if exhausted else 'requesting correction'}."
                        )
                    )
                ],
            }
        assert isinstance(generated, CodeGenerationOutput)
        return {
            "generated_code": generated.code,
            "code_summary": generated.summary,
            "requested_timeout_seconds": generated.timeout_seconds,
            "status": "executing",
            "messages": [AIMessage(content=f"Code prepared: {generated.summary}")],
        }

    def route_after_coding(
        state: AgentState,
    ) -> Literal["sandbox_executor", "reflect_and_heal", "__end__"]:
        if state["status"] == "executing":
            return "sandbox_executor"
        if state["status"] == "healing":
            return "reflect_and_heal"
        return END

    def sandbox_executor_node(state: AgentState) -> dict[str, Any]:
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
        return {
            "execution_artifacts": [*state["execution_artifacts"], artifact],
            "status": "executing",
        }

    def error_detector_node(state: AgentState) -> dict[str, Any]:
        artifact = state["execution_artifacts"][-1]
        result = PythonExecutionOutput.model_validate(artifact["result"])
        if result.status == "success":
            return {
                "status": "completed",
                "final_output": result.logs,
                "messages": [AIMessage(content="Sandbox execution completed successfully.")],
            }

        errors = [*state["error_stack"], _artifact_error(artifact)]
        exhausted = state["retry_count"] >= dependencies["max_retries"]
        return {
            "error_stack": errors,
            "status": "failed" if exhausted else "healing",
            "final_output": result.traceback or result.error_message or result.logs,
            "messages": [
                AIMessage(
                    content=(
                        f"Execution failed with {result.error_type or result.status}; "
                        f"{'retry limit reached' if exhausted else 'starting correction'}."
                    )
                )
            ],
        }

    def route_after_detection(state: AgentState) -> Literal["reflect_and_heal", "__end__"]:
        return "reflect_and_heal" if state["status"] == "healing" else END

    def reflect_and_heal_node(state: AgentState) -> dict[str, Any]:
        return {
            "retry_count": state["retry_count"] + 1,
            "status": "coding",
            "messages": [
                AIMessage(
                    content=f"Correction attempt {state['retry_count'] + 1} of {max_retries}."
                )
            ],
        }

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("coder_agent", coder_node)
    builder.add_node("sandbox_executor", sandbox_executor_node)
    builder.add_node("error_detector", error_detector_node)
    builder.add_node("reflect_and_heal", reflect_and_heal_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "coder_agent")
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
    builder.add_edge("reflect_and_heal", "coder_agent")
    return builder.compile()


graph = create_graph()

"""Phase 3/4 traceback reflection and bounded-memory integration tests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import docker
import pytest
from docker.errors import DockerException, ImageNotFound
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.graph import AgentState, create_graph, initial_state
from src.agent.reflection import (
    CodePatch,
    PatchApplicationError,
    PatchEdit,
    ReflectionResult,
    TracebackReflector,
    apply_code_patch,
    parse_traceback,
)
from src.agent.tools import CodeGenerationOutput, PlanOutput, PlanStep
from src.memory.pruner import ContextPruner, EpisodicMemory
from src.sandbox.test_sandbox import SandboxResult


RUNTIME_ERROR_PROMPT = "Sum the amount column in the records and print the result."


class RuntimePlanner:
    def invoke(self, input: object, **kwargs: Any) -> PlanOutput:
        return PlanOutput(
            objective=RUNTIME_ERROR_PROMPT,
            steps=[
                PlanStep(
                    index=1,
                    instruction="Sum the amount field in all records.",
                    expected_result="The total is printed.",
                )
            ],
        )


class BrokenColumnCoder:
    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        return CodeGenerationOutput(
            code=(
                'records = [{"amount": 13}, {"amount": 21}]\n'
                'total = sum(row["missing_amount"] for row in records)\n'
                'print(f"TOTAL={total}")\n'
            ),
            timeout_seconds=3,
            summary="Sum a record field and print the total.",
        )


class ColumnPatchModel:
    def invoke(self, input: object, **kwargs: Any) -> CodePatch:
        return CodePatch(
            root_cause=(
                "The code reads missing_amount, but each record contains the amount key."
            ),
            edits=[
                PatchEdit(
                    old_text='row["missing_amount"]',
                    new_text='row["amount"]',
                    reason="Use the key that exists in every record.",
                )
            ],
            validation_notes="The replacement preserves the summation and output format.",
        )


def _docker_is_ready() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        client.images.get("python:3.12-slim")
    except (DockerException, ImageNotFound):
        return False
    return True


def test_parse_traceback_extracts_frames_and_root_exception() -> None:
    traceback_text = """\
Traceback (most recent call last):
  File "<sandbox>", line 8, in <module>
    transform(records)
  File "<sandbox>", line 5, in transform
    return rows[0]["missing"]
KeyError: 'missing'
"""
    parsed = parse_traceback(traceback_text)

    assert parsed.exception_type == "KeyError"
    assert parsed.exception_message == "'missing'"
    assert len(parsed.frames) == 2
    assert parsed.frames[-1].function == "transform"
    assert parsed.frames[-1].source_line == 'return rows[0]["missing"]'
    assert parsed.raw == traceback_text


def test_patch_gate_rejects_a_syntax_error_before_sandbox_execution() -> None:
    patch = CodePatch(
        root_cause="The proposed replacement removes a closing quote.",
        edits=[
            PatchEdit(
                old_text='print("ok")',
                new_text='print("broken)',
                reason="Exercise the syntax validation boundary.",
            )
        ],
        validation_notes="This deliberately invalid patch must be rejected.",
    )
    with pytest.raises(PatchApplicationError, match="not valid Python"):
        apply_code_patch('print("ok")\n', patch)


def test_working_memory_preserves_policy_intent_and_latest_evidence() -> None:
    pruner = ContextPruner(max_chars=4_000, max_messages=4)
    state = {
        "messages": [
            SystemMessage(content="Never access the network."),
            HumanMessage(content="Repair this calculation."),
            *[AIMessage(content=f"Intermediate attempt {index}") for index in range(8)],
        ],
        "current_plan": ["Calculate", "Verify"],
        "error_stack": ["KeyError: old", "KeyError: latest"],
        "generated_code": 'print(record["latest"])\n',
        "patch_history": [
            {"root_cause": "old"},
            {"root_cause": "latest", "unified_diff": "-old\n+latest\n"},
        ],
        "execution_artifacts": [],
        "history_summary": [],
        "retry_count": 2,
    }
    working = pruner.working_memory(state)

    assert working.system_prompts == ["Never access the network."]
    assert working.user_intent == "Repair this calculation."
    assert working.latest_error == "KeyError: latest"
    assert working.latest_code == 'print(record["latest"])\n'
    assert working.latest_patch["root_cause"] == "latest"
    assert len(working.recent_messages) <= 2


def test_pruner_truncation_makes_progress_for_large_single_artifact() -> None:
    pruner = ContextPruner(
        max_chars=1_200,
        max_messages=2,
        max_artifacts=1,
        max_output_chars=400,
    )
    state = {
        "messages": [HumanMessage(content="Keep the run bounded.")],
        "current_plan": ["Execute"],
        "error_stack": ["RuntimeError: synthetic"],
        "retry_count": 1,
        "execution_artifacts": [
            {
                "attempt": 1,
                "request": {"code": "raise RuntimeError('synthetic')"},
                "result": {
                    "status": "error",
                    "logs": "diagnostic\n" * 1_000,
                    "traceback": "RuntimeError: synthetic",
                },
            }
        ],
        "patch_history": [],
        "history_summary": [],
        "final_output": "diagnostic\n" * 1_000,
    }

    pruned = pruner.prune_state(state)

    assert len(pruned["execution_artifacts"][0]["result"]["logs"]) <= 256
    assert len(pruned["final_output"]) <= pruner.max_output_chars
    assert pruner.measure_state(pruned) <= pruner.max_chars


@pytest.mark.integration
def test_missing_column_self_corrects_with_targeted_patch_in_docker() -> None:
    assert _docker_is_ready(), "Docker daemon and python:3.12-slim are required"

    episodic_memory = EpisodicMemory()
    app = create_graph(
        planner=RuntimePlanner(),
        coder=BrokenColumnCoder(),
        reflector=TracebackReflector(model=ColumnPatchModel()),
        episodic_memory=episodic_memory,
    )
    final_state: AgentState = app.invoke(initial_state(RUNTIME_ERROR_PROMPT))

    assert final_state["status"] == "completed"
    assert final_state["retry_count"] == 1
    assert "TOTAL=34" in final_state["final_output"]
    assert len(final_state["execution_artifacts"]) == 2
    reflection = final_state["execution_artifacts"][0]["reflection"]
    assert reflection["parsed_traceback"]["exception_type"] == "KeyError"
    assert "missing_amount" in reflection["patch"]["root_cause"]
    assert '-total = sum(row["missing_amount"]' in reflection["unified_diff"]
    assert '+total = sum(row["amount"]' in reflection["unified_diff"]
    assert len(episodic_memory.episodes) == 1
    assert episodic_memory.episodes[0].episode_id == final_state["episode_id"]


class LoopCoder:
    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        return CodeGenerationOutput(
            code='counter = 0\nprint(f"COUNTER={counter}")\n',
            timeout_seconds=1,
            summary="Print a counter after repeated targeted corrections.",
        )


class IncrementingReflector:
    def reflect(self, failed_code: str, traceback_text: str) -> ReflectionResult:
        match = re.search(r"counter = (\d+)", failed_code)
        assert match is not None
        old_value = int(match.group(1))
        patch = CodePatch(
            root_cause=f"Synthetic failure requires counter value {old_value + 1}.",
            edits=[
                PatchEdit(
                    old_text=f"counter = {old_value}",
                    new_text=f"counter = {old_value + 1}",
                    reason="Advance the deterministic healing-loop fixture.",
                )
            ],
            validation_notes="Only the counter literal changes.",
        )
        patched_code, unified_diff = apply_code_patch(failed_code, patch)
        return ReflectionResult(
            parsed_traceback=parse_traceback(traceback_text),
            patch=patch,
            patched_code=patched_code,
            unified_diff=unified_diff,
        )


class FourFailuresThenSuccess:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, code: str, *, timeout_seconds: float) -> SandboxResult:
        self.calls += 1
        if self.calls <= 4:
            return SandboxResult(
                status="error",
                exit_code=1,
                error_type="KeyError",
                error_message=f"'missing_{self.calls}'",
                traceback=(
                    "Traceback (most recent call last):\n"
                    f'  File "<sandbox>", line {self.calls}, in <module>\n'
                    f"KeyError: 'missing_{self.calls}'\n"
                ),
                logs="diagnostic\n" * 500,
                duration_seconds=0.01,
            )
        return SandboxResult(
            status="success",
            exit_code=0,
            logs="COUNTER=4\n",
            duration_seconds=0.01,
        )


def test_state_history_stays_bounded_after_four_healing_loops(tmp_path: Path) -> None:
    pruner = ContextPruner(
        max_chars=6_000,
        max_messages=5,
        max_artifacts=2,
        max_output_chars=400,
    )
    episodic_memory = EpisodicMemory()
    runner = FourFailuresThenSuccess()
    app = create_graph(
        planner=RuntimePlanner(),
        coder=LoopCoder(),
        reflector=IncrementingReflector(),
        sandbox_runner=runner,
        max_retries=5,
        pruner=pruner,
        episodic_memory=episodic_memory,
    )
    final_state: AgentState = app.invoke(initial_state("Heal four failures, then succeed."))

    assert final_state["status"] == "completed"
    assert final_state["retry_count"] == 4
    assert runner.calls == 5
    assert pruner.measure_state(final_state) <= pruner.max_chars
    assert len(final_state["messages"]) <= pruner.max_messages
    assert len(final_state["execution_artifacts"]) <= pruner.max_artifacts
    assert len(final_state["error_stack"]) == 1
    assert len(final_state["patch_history"]) == 1
    assert "missing_4" in final_state["error_stack"][0]
    assert "chars pruned" not in final_state["error_stack"][0]
    assert "counter = 4" in final_state["generated_code"]
    assert episodic_memory.episodes[0].retry_count == 4
    export_path = episodic_memory.export_jsonl(tmp_path / "episodes.jsonl")
    exported = json.loads(export_path.read_text(encoding="utf-8").strip())
    assert exported["episode_id"] == final_state["episode_id"]
    assert exported["retry_count"] == 4


@pytest.mark.ollama
def test_live_qwen_reflector_generates_applicable_keyerror_patch() -> None:
    failed_code = (
        'records = [{"amount": 13}, {"amount": 21}]\n'
        'total = sum(row["missing_amount"] for row in records)\n'
        'print(f"TOTAL={total}")\n'
    )
    traceback_text = """\
Traceback (most recent call last):
  File "<sandbox>", line 2, in <module>
  File "<sandbox>", line 2, in <genexpr>
KeyError: 'missing_amount'
"""
    reflection = TracebackReflector().reflect(failed_code, traceback_text)

    assert reflection.patch.root_cause
    assert 'row["amount"]' in reflection.patched_code
    assert 'row["missing_amount"]' not in reflection.patched_code
    assert "--- failed.py" in reflection.unified_diff

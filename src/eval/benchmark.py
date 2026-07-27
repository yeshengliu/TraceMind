"""Reproducible single-pass versus self-healing benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from src.agent.graph import AgentState, create_graph, initial_state
from src.agent.llm import ModelRole, get_structured_llm
from src.agent.reflection import (
    CodePatch,
    PatchEdit,
    ReflectionResult,
    TracebackReflector,
    apply_code_patch,
    parse_traceback,
)
from src.agent.tools import (
    CodeGenerationOutput,
    PlanOutput,
    PlanStep,
    PythonExecutionInput,
    StrictModel,
    execute_python,
)
from src.memory.pruner import ContextPruner
from src.sandbox.test_sandbox import SandboxResult, run_in_sandbox


DEFAULT_DATASET = Path("data/benchmark.jsonl")
DEFAULT_JSON_REPORT = Path("docs/benchmark_report.json")
DEFAULT_MARKDOWN_REPORT = Path("docs/benchmark_report.md")


class BenchmarkMode(StrEnum):
    SINGLE_PASS = "single_pass"
    SELF_HEALING = "self_healing"


class ReflectorMode(StrEnum):
    DATASET = "dataset"
    OLLAMA = "ollama"


class BenchmarkRepair(StrictModel):
    root_cause: str = Field(min_length=1, max_length=2_000)
    edits: list[PatchEdit] = Field(min_length=1, max_length=8)
    validation_notes: str = Field(min_length=1, max_length=1_000)

    def as_code_patch(self) -> CodePatch:
        return CodePatch.model_validate(self.model_dump())


class BenchmarkTask(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    category: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=1, max_length=1_000)
    program: str = Field(min_length=1, max_length=50_000)
    expected_stdout_regex: str = Field(min_length=1, max_length=2_000)
    timeout_seconds: float = Field(default=3.0, ge=0.1, le=30)
    repair: BenchmarkRepair | None = None


class JudgeVerdict(StrictModel):
    passed: bool
    score: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=1_000)


class TaskResult(StrictModel):
    task_id: str
    category: str
    mode: BenchmarkMode
    success: bool
    judge_score: float
    judge_rationale: str
    execution_status: str
    healing_attempts: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    raw_context_chars: int = Field(ge=0)
    pruned_context_chars: int = Field(ge=0)
    output: str


class ModeSummary(StrictModel):
    mode: BenchmarkMode
    tasks: int = Field(ge=0)
    successful_tasks: int = Field(ge=0)
    success_rate_pct: float = Field(ge=0, le=100)
    average_healing_attempts: float = Field(ge=0)
    average_raw_context_tokens: float = Field(ge=0)
    average_pruned_context_tokens: float = Field(ge=0)
    token_reduction_pct: float = Field(ge=0, le=100)
    average_duration_seconds: float = Field(ge=0)


class BenchmarkReport(StrictModel):
    schema_version: str = "1.0"
    generated_at: datetime
    dataset: str
    judge: str
    reflector: ReflectorMode
    task_count: int = Field(ge=0)
    summaries: list[ModeSummary]
    results: list[TaskResult]
    methodology_notes: list[str]


class Judge(Protocol):
    name: str

    def evaluate(
        self,
        task: BenchmarkTask,
        *,
        execution_status: str,
        output: str,
    ) -> JudgeVerdict:
        ...


class ExpectedOutputJudge:
    """Deterministic correctness oracle used for committed reports and CI."""

    name = "expected-output-regex"

    def evaluate(
        self,
        task: BenchmarkTask,
        *,
        execution_status: str,
        output: str,
    ) -> JudgeVerdict:
        matched = (
            execution_status == "success"
            and re.search(task.expected_stdout_regex, output, re.MULTILINE) is not None
        )
        return JudgeVerdict(
            passed=matched,
            score=1.0 if matched else 0.0,
            rationale=(
                "Sandbox succeeded and output matched the task oracle."
                if matched
                else "Execution failed or output did not match the task oracle."
            ),
        )


class OllamaJudge:
    """Optional local LLM-as-a-judge for semantic output evaluation."""

    name = "ollama-qwen-structured-judge"

    _SYSTEM_PROMPT = """\
You are TraceMind's local benchmark judge. Decide whether the sandbox output
semantically satisfies the requested coding task. Execution failures never
pass. Use the expected output regex as a strong correctness signal, but inspect
the request and output for semantic correctness. Return only the supplied JSON
schema. Do not reward commentary, style, or hidden reasoning.
"""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model or get_structured_llm(ModelRole.PLANNER, JudgeVerdict)

    def evaluate(
        self,
        task: BenchmarkTask,
        *,
        execution_status: str,
        output: str,
    ) -> JudgeVerdict:
        response = self.model.invoke(
            [
                SystemMessage(content=self._SYSTEM_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "task": task.prompt,
                            "expected_stdout_regex": task.expected_stdout_regex,
                            "execution_status": execution_status,
                            "sandbox_output": output[-4_000:],
                        },
                        indent=2,
                    )
                ),
            ]
        )
        if isinstance(response, JudgeVerdict):
            return response
        return JudgeVerdict.model_validate(response)


class RecordingPruner(ContextPruner):
    """Measure raw and retained state at every graph transition."""

    def __init__(self) -> None:
        super().__init__(
            max_chars=4_000,
            max_messages=4,
            max_artifacts=1,
            max_output_chars=400,
        )
        self.max_raw_chars = 0
        self.max_pruned_chars = 0

    def prune_state(self, state: dict[str, Any]) -> dict[str, Any]:
        self.max_raw_chars = max(self.max_raw_chars, self.measure_state(state))
        pruned = super().prune_state(state)
        self.max_pruned_chars = max(
            self.max_pruned_chars,
            self.measure_state(pruned),
        )
        return pruned


class _TaskPlanner:
    def __init__(self, task: BenchmarkTask) -> None:
        self.task = task

    def invoke(self, input: object, **kwargs: Any) -> PlanOutput:
        return PlanOutput(
            objective=self.task.prompt,
            steps=[
                PlanStep(
                    index=1,
                    instruction="Execute the supplied benchmark program.",
                    expected_result="Output satisfies the benchmark oracle.",
                )
            ],
        )


class _TaskCoder:
    def __init__(self, task: BenchmarkTask) -> None:
        self.task = task

    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        return CodeGenerationOutput(
            code=self.task.program,
            timeout_seconds=self.task.timeout_seconds,
            summary="Run the deterministic benchmark program.",
        )


class _DatasetReflector:
    def __init__(self, task: BenchmarkTask) -> None:
        self.task = task

    def reflect(self, failed_code: str, traceback_text: str) -> ReflectionResult:
        if self.task.repair is None:
            raise ValueError(f"Task {self.task.id} has no declared repair")
        patch = self.task.repair.as_code_patch()
        patched_code, unified_diff = apply_code_patch(failed_code, patch)
        return ReflectionResult(
            parsed_traceback=parse_traceback(traceback_text),
            patch=patch,
            patched_code=patched_code,
            unified_diff=unified_diff,
        )


SandboxRunner = Callable[..., SandboxResult]


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[BenchmarkTask]:
    """Load and validate a JSONL benchmark without allowing duplicate IDs."""
    source = Path(path)
    tasks: list[BenchmarkTask] = []
    identifiers: set[str] = set()
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            task = BenchmarkTask.model_validate_json(raw_line)
        except Exception as exc:
            raise ValueError(f"Invalid benchmark row {line_number}: {exc}") from exc
        if task.id in identifiers:
            raise ValueError(f"Duplicate benchmark task id: {task.id}")
        identifiers.add(task.id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"Benchmark dataset is empty: {source}")
    return tasks


def _serialized_size(payload: object) -> int:
    return len(json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True))


def run_single_pass(
    task: BenchmarkTask,
    *,
    judge: Judge,
    sandbox_runner: SandboxRunner = run_in_sandbox,
) -> TaskResult:
    """Execute one generated program with no reflection or retry."""
    started = time.monotonic()
    result = execute_python(
        PythonExecutionInput(
            code=task.program,
            timeout_seconds=task.timeout_seconds,
        ),
        runner=sandbox_runner,
    )
    verdict = judge.evaluate(
        task,
        execution_status=result.status,
        output=result.logs,
    )
    context_chars = _serialized_size(
        {
            "prompt": task.prompt,
            "program": task.program,
            "result": result.model_dump(mode="json"),
        }
    )
    return TaskResult(
        task_id=task.id,
        category=task.category,
        mode=BenchmarkMode.SINGLE_PASS,
        success=verdict.passed,
        judge_score=verdict.score,
        judge_rationale=verdict.rationale,
        execution_status=result.status,
        healing_attempts=0,
        duration_seconds=time.monotonic() - started,
        raw_context_chars=context_chars,
        pruned_context_chars=context_chars,
        output=result.logs[-4_000:],
    )


def run_self_healing(
    task: BenchmarkTask,
    *,
    judge: Judge,
    sandbox_runner: SandboxRunner = run_in_sandbox,
    reflector_mode: ReflectorMode = ReflectorMode.DATASET,
) -> TaskResult:
    """Execute a task through the complete planning and healing graph."""
    started = time.monotonic()
    pruner = RecordingPruner()
    reflector = (
        TracebackReflector()
        if reflector_mode == ReflectorMode.OLLAMA
        else _DatasetReflector(task)
    )
    graph = create_graph(
        planner=_TaskPlanner(task),
        coder=_TaskCoder(task),
        reflector=reflector,
        sandbox_runner=sandbox_runner,
        max_retries=3,
        pruner=pruner,
    )
    final_state: AgentState = graph.invoke(initial_state(task.prompt))
    output = str(final_state.get("final_output", ""))
    execution_status = (
        "success" if final_state.get("status") == "completed" else "error"
    )
    verdict = judge.evaluate(
        task,
        execution_status=execution_status,
        output=output,
    )
    fallback_size = pruner.measure_state(final_state)
    return TaskResult(
        task_id=task.id,
        category=task.category,
        mode=BenchmarkMode.SELF_HEALING,
        success=verdict.passed,
        judge_score=verdict.score,
        judge_rationale=verdict.rationale,
        execution_status=execution_status,
        healing_attempts=int(final_state.get("retry_count", 0)),
        duration_seconds=time.monotonic() - started,
        raw_context_chars=max(pruner.max_raw_chars, fallback_size),
        pruned_context_chars=max(pruner.max_pruned_chars, fallback_size),
        output=output[-4_000:],
    )


def _summarize(mode: BenchmarkMode, results: list[TaskResult]) -> ModeSummary:
    selected = [result for result in results if result.mode == mode]
    count = len(selected)
    if not count:
        return ModeSummary(
            mode=mode,
            tasks=0,
            successful_tasks=0,
            success_rate_pct=0,
            average_healing_attempts=0,
            average_raw_context_tokens=0,
            average_pruned_context_tokens=0,
            token_reduction_pct=0,
            average_duration_seconds=0,
        )
    successes = sum(result.success for result in selected)
    average_raw_chars = sum(result.raw_context_chars for result in selected) / count
    average_pruned_chars = (
        sum(result.pruned_context_chars for result in selected) / count
    )
    token_reduction = (
        (average_raw_chars - average_pruned_chars) / average_raw_chars * 100
        if average_raw_chars
        else 0
    )
    return ModeSummary(
        mode=mode,
        tasks=count,
        successful_tasks=successes,
        success_rate_pct=round(successes / count * 100, 2),
        average_healing_attempts=round(
            sum(result.healing_attempts for result in selected) / count,
            2,
        ),
        average_raw_context_tokens=round(average_raw_chars / 4, 2),
        average_pruned_context_tokens=round(average_pruned_chars / 4, 2),
        token_reduction_pct=round(max(0, token_reduction), 2),
        average_duration_seconds=round(
            sum(result.duration_seconds for result in selected) / count,
            3,
        ),
    )


def run_benchmark(
    tasks: Iterable[BenchmarkTask],
    *,
    judge: Judge | None = None,
    sandbox_runner: SandboxRunner = run_in_sandbox,
    dataset_name: str = str(DEFAULT_DATASET),
    reflector_mode: ReflectorMode = ReflectorMode.DATASET,
) -> BenchmarkReport:
    """Run both benchmark modes and return a validated quantitative report."""
    selected = list(tasks)
    evaluator = judge or ExpectedOutputJudge()
    results: list[TaskResult] = []
    for task in selected:
        results.append(
            run_single_pass(
                task,
                judge=evaluator,
                sandbox_runner=sandbox_runner,
            )
        )
    for task in selected:
        results.append(
            run_self_healing(
                task,
                judge=evaluator,
                sandbox_runner=sandbox_runner,
                reflector_mode=reflector_mode,
            )
        )
    return BenchmarkReport(
        generated_at=datetime.now(UTC),
        dataset=dataset_name,
        judge=evaluator.name,
        reflector=reflector_mode,
        task_count=len(selected),
        summaries=[
            _summarize(BenchmarkMode.SINGLE_PASS, results),
            _summarize(BenchmarkMode.SELF_HEALING, results),
        ],
        results=results,
        methodology_notes=[
            "Both modes execute inside the same offline Docker sandbox.",
            "Single-pass executes the generated program once with no correction.",
            (
                "Self-healing uses the full LangGraph retry path with "
                + (
                    "deterministic dataset-backed exact-text patches for reproducibility."
                    if reflector_mode == ReflectorMode.DATASET
                    else "live qwen2.5-coder reflection through Ollama."
                )
            ),
            (
                "Context tokens are estimated at four characters per token; raw and "
                "pruned maxima are measured immediately around each pruning step."
            ),
            (
                "The committed report uses exact expected-output oracles. "
                "Use --judge ollama for optional local semantic LLM judging."
            ),
            "This fixture benchmark is evidence for these tasks, not a universal model claim.",
        ],
    )


def render_markdown_report(report: BenchmarkReport) -> str:
    summaries = {summary.mode: summary for summary in report.summaries}
    single = summaries[BenchmarkMode.SINGLE_PASS]
    healed = summaries[BenchmarkMode.SELF_HEALING]
    improvement = healed.success_rate_pct - single.success_rate_pct
    lines = [
        "# TraceMind Benchmark Report",
        "",
        f"Generated: `{report.generated_at.isoformat()}`<br>",
        f"Dataset: `{report.dataset}` ({report.task_count} tasks)<br>",
        f"Judge: `{report.judge}`<br>",
        f"Reflector: `{report.reflector}`",
        "",
        "## Aggregate results",
        "",
        "| Mode | Success | Success rate | Avg. healing attempts | "
        "Avg. raw tokens | Avg. retained tokens | Token reduction | Avg. duration |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.summaries:
        label = (
            "Single-pass"
            if summary.mode == BenchmarkMode.SINGLE_PASS
            else "TraceMind self-healing"
        )
        lines.append(
            f"| {label} | {summary.successful_tasks}/{summary.tasks} | "
            f"{summary.success_rate_pct:.2f}% | "
            f"{summary.average_healing_attempts:.2f} | "
            f"{summary.average_raw_context_tokens:.2f} | "
            f"{summary.average_pruned_context_tokens:.2f} | "
            f"{summary.token_reduction_pct:.2f}% | "
            f"{summary.average_duration_seconds:.3f}s |"
        )
    lines.extend(
        [
            "",
            f"**Measured completion-rate lift: +{improvement:.2f} percentage points.**",
            "",
            "## Task-level results",
            "",
            "| Task | Category | Single-pass | Self-healing | Repairs |",
            "|---|---|---:|---:|---:|",
        ]
    )
    indexed = {
        (result.task_id, result.mode): result for result in report.results
    }
    task_ids = [
        result.task_id
        for result in report.results
        if result.mode == BenchmarkMode.SINGLE_PASS
    ]
    for task_id in task_ids:
        direct = indexed[(task_id, BenchmarkMode.SINGLE_PASS)]
        full = indexed[(task_id, BenchmarkMode.SELF_HEALING)]
        lines.append(
            f"| `{task_id}` | {direct.category} | "
            f"{'✅' if direct.success else '❌'} | "
            f"{'✅' if full.success else '❌'} | "
            f"{full.healing_attempts} |"
        )
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            *[f"- {note}" for note in report.methodology_notes],
            "",
            "Reproduce:",
            "",
            "```bash",
            "python -m src.eval.benchmark",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    report: BenchmarkReport,
    *,
    json_path: str | Path = DEFAULT_JSON_REPORT,
    markdown_path: str | Path = DEFAULT_MARKDOWN_REPORT,
) -> tuple[Path, Path]:
    json_destination = Path(json_path)
    markdown_destination = Path(markdown_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    markdown_destination.write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )
    return json_destination, markdown_destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=DEFAULT_MARKDOWN_REPORT,
    )
    parser.add_argument(
        "--judge",
        choices=("exact", "ollama"),
        default="exact",
        help="Use reproducible output oracles or the optional local Qwen judge.",
    )
    parser.add_argument(
        "--reflector",
        choices=("dataset", "ollama"),
        default="dataset",
        help="Use reproducible declared patches or live local Qwen reflection.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = load_dataset(args.dataset)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        tasks = tasks[: args.limit]
    judge: Judge = OllamaJudge() if args.judge == "ollama" else ExpectedOutputJudge()
    report = run_benchmark(
        tasks,
        judge=judge,
        dataset_name=str(args.dataset),
        reflector_mode=ReflectorMode(args.reflector),
    )
    json_path, markdown_path = write_report(
        report,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    for summary in report.summaries:
        print(
            f"{summary.mode}: {summary.successful_tasks}/{summary.tasks} "
            f"({summary.success_rate_pct:.2f}%)"
        )
    print(f"Wrote {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

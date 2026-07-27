"""Phase 6 benchmark dataset, evaluation, and reporting tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.benchmark import (
    BenchmarkMode,
    ExpectedOutputJudge,
    JudgeVerdict,
    OllamaJudge,
    load_dataset,
    run_benchmark,
    write_report,
)
from src.sandbox.test_sandbox import SandboxResult


def test_committed_dataset_has_eighteen_diverse_valid_tasks() -> None:
    tasks = load_dataset()

    assert len(tasks) == 18
    assert len({task.id for task in tasks}) == 18
    assert sum(task.repair is not None for task in tasks) == 12
    assert sum(task.repair is None for task in tasks) == 6
    assert len({task.category for task in tasks}) >= 10


def _fixture_sandbox(code: str, *, timeout_seconds: float) -> SandboxResult:
    if 'row["missing_amount"]' in code:
        return SandboxResult(
            status="error",
            exit_code=1,
            error_type="KeyError",
            error_message="'missing_amount'",
            traceback=(
                "Traceback (most recent call last):\n"
                '  File "<sandbox>", line 2, in <module>\n'
                "KeyError: 'missing_amount'\n"
            ),
            logs="fixture diagnostic\n" * 100,
            duration_seconds=0.01,
        )
    if 'row["amount"]' in code:
        output = "TOTAL=34\n"
    elif 'print(f"FIB={a}")' in code:
        output = "FIB=55\n"
    else:
        raise AssertionError(f"Unexpected benchmark program: {code}")
    return SandboxResult(
        status="success",
        exit_code=0,
        logs=output,
        duration_seconds=0.01,
    )


def test_benchmark_compares_single_pass_with_full_healing_graph(
    tmp_path: Path,
) -> None:
    tasks_by_id = {task.id: task for task in load_dataset()}
    selected = [
        tasks_by_id["missing-dict-column"],
        tasks_by_id["fibonacci-baseline"],
    ]

    report = run_benchmark(
        selected,
        judge=ExpectedOutputJudge(),
        sandbox_runner=_fixture_sandbox,
        dataset_name="fixture",
    )

    summaries = {summary.mode: summary for summary in report.summaries}
    assert summaries[BenchmarkMode.SINGLE_PASS].success_rate_pct == 50
    assert summaries[BenchmarkMode.SELF_HEALING].success_rate_pct == 100
    healed = [
        result
        for result in report.results
        if result.task_id == "missing-dict-column"
        and result.mode == BenchmarkMode.SELF_HEALING
    ][0]
    assert healed.healing_attempts == 1
    assert healed.pruned_context_chars < healed.raw_context_chars

    json_path, markdown_path = write_report(
        report,
        json_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["task_count"] == 2
    assert payload["reflector"] == "dataset"
    assert "Measured completion-rate lift: +50.00" in markdown
    assert "`missing-dict-column`" in markdown


class _PassingJudgeModel:
    def invoke(self, input: object, **kwargs: Any) -> JudgeVerdict:
        return JudgeVerdict(
            passed=True,
            score=0.95,
            rationale="The result semantically satisfies the task.",
        )


def test_optional_ollama_judge_uses_structured_verdict_contract() -> None:
    task = load_dataset()[0]
    verdict = OllamaJudge(model=_PassingJudgeModel()).evaluate(
        task,
        execution_status="success",
        output="TOTAL=34",
    )

    assert verdict.passed is True
    assert verdict.score == 0.95

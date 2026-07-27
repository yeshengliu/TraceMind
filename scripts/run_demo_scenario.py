#!/usr/bin/env python3
"""Launch the deterministic TraceMind sales self-healing recording scenario.

Run from the repository root:

    python scripts/run_demo_scenario.py

The default path routes the real Docker traceback through the configured
qwen2.5-coder:7b reflector. If Ollama is unavailable or returns a patch that
does not satisfy the demo invariant, a validated deterministic repair keeps
the recording outcome reproducible. Use ``--fixture-reflector`` to skip Ollama
entirely when rehearsing framing or testing the UI offline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from src.agent.graph import create_graph  # noqa: E402
from src.agent.llm import ModelRole, get_llm  # noqa: E402
from src.agent.reflection import (  # noqa: E402
    REFLECTOR_SYSTEM_PROMPT,
    CodePatch,
    PatchEdit,
    ReflectionResult,
    apply_code_patch,
    parse_traceback,
)
from src.agent.tools import CodeGenerationOutput, PlanOutput, PlanStep  # noqa: E402


DEMO_PROMPT = "Analyze sales data and plot profit trends."

FAILED_PROGRAM = '''\
import html

sales = [
    {"month": "Jan", "revenue": 120000, "cost": 90000},
    {"month": "Feb", "revenue": 138000, "cost": 96000},
    {"month": "Mar", "revenue": 151000, "cost": 101000},
    {"month": "Apr", "revenue": 169000, "cost": 108000},
    {"month": "May", "revenue": 187000, "cost": 116000},
    {"month": "Jun", "revenue": 210000, "cost": 124000},
]

profit_trend = [
    (row["month"], row["profit_margin"])
    for row in sales
]

width, height, pad = 720, 320, 48
values = [value for _, value in profit_trend]
low, high = min(values), max(values)
spread = high - low or 1
points = " ".join(
    f"{pad + index * (width - 2 * pad) / (len(values) - 1):.1f},"
    f"{height - pad - (value - low) * (height - 2 * pad) / spread:.1f}"
    for index, value in enumerate(values)
)
labels = "".join(
    f'<text x="{pad + index * (width - 2 * pad) / (len(values) - 1):.1f}" '
    f'y="{height - 17}" text-anchor="middle" fill="#94a3b8" '
    f'font-size="13">{html.escape(month)}</text>'
    for index, (month, _) in enumerate(profit_trend)
)
svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
    f'viewBox="0 0 {width} {height}">'
    '<rect width="100%" height="100%" rx="18" fill="#08111f"/>'
    '<text x="48" y="34" fill="#e2e8f0" font-size="20" '
    'font-family="sans-serif" font-weight="700">Monthly profit margin</text>'
    f'<polyline points="{points}" fill="none" stroke="#10b981" '
    'stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
    + "".join(
        f'<circle cx="{point.split(",")[0]}" cy="{point.split(",")[1]}" r="6" '
        f'fill="#38bdf8" stroke="#e2e8f0" stroke-width="2"/>'
        for point in points.split()
    )
    + labels
    + "</svg>"
)
print("PROFIT_TREND=" + ", ".join(
    f"{month}:{margin:.1f}%" for month, margin in profit_trend
))
print(f"TREND_SVG={svg}")
'''

BROKEN_EXPRESSION = 'row["profit_margin"]'
REPAIRED_EXPRESSION = (
    'round((row["revenue"] - row["cost"]) / row["revenue"] * 100, 2)'
)


class DemoPlanner:
    """Stable visible plan; no model latency before the failure is shown."""

    def invoke(self, input: object, **kwargs: Any) -> PlanOutput:
        return PlanOutput(
            objective=DEMO_PROMPT,
            steps=[
                PlanStep(
                    index=1,
                    instruction="Load the monthly sales records and calculate profit margin.",
                    expected_result="A month-by-month profit series is available.",
                ),
                PlanStep(
                    index=2,
                    instruction="Render the profit-margin trend as a chart artifact.",
                    expected_result="The right panel displays a green trend chart.",
                ),
            ],
        )


class DemoCoder:
    """Return the intentional edge case exactly once."""

    def invoke(self, input: object, **kwargs: Any) -> CodeGenerationOutput:
        return CodeGenerationOutput(
            code=FAILED_PROGRAM,
            timeout_seconds=5,
            summary=(
                "Analyze six months of sales and render a profit-margin trend."
            ),
        )


def _fixture_reflection(
    failed_code: str,
    traceback_text: str,
    *,
    reason: str | None = None,
) -> ReflectionResult:
    root_cause = (
        "The sales rows contain revenue and cost, but no profit_margin key; "
        "derive the margin from the available fields."
    )
    if reason:
        root_cause = f"{root_cause} Deterministic fallback: {reason}"
    patch = CodePatch(
        root_cause=root_cause,
        edits=[
            PatchEdit(
                old_text=BROKEN_EXPRESSION,
                new_text=REPAIRED_EXPRESSION,
                reason="Calculate profit margin from revenue minus cost.",
            )
        ],
        validation_notes=(
            "The exact missing-key access is replaced and the patched program "
            "passes Python syntax validation."
        ),
    )
    patched_code, unified_diff = apply_code_patch(failed_code, patch)
    return ReflectionResult(
        parsed_traceback=parse_traceback(traceback_text),
        patch=patch,
        patched_code=patched_code,
        unified_diff=unified_diff,
    )


class DemoReflector:
    """Route to Qwen, then enforce the recording scenario's safe invariant."""

    def __init__(self, *, use_qwen: bool) -> None:
        self.use_qwen = use_qwen
        self.model = get_llm(ModelRole.CODER) if use_qwen else None

    def reflect(self, failed_code: str, traceback_text: str) -> ReflectionResult:
        if self.model is None:
            return _fixture_reflection(failed_code, traceback_text)
        try:
            reflection = self._reflect_with_qwen(failed_code, traceback_text)
        except Exception as exc:
            return _fixture_reflection(
                failed_code,
                traceback_text,
                reason=f"{type(exc).__name__}",
            )
        if (
            BROKEN_EXPRESSION in reflection.patched_code
            or "TREND_SVG=" not in reflection.patched_code
        ):
            return _fixture_reflection(
                failed_code,
                traceback_text,
                reason="model patch did not preserve the demo invariant",
            )
        return reflection

    def _reflect_with_qwen(
        self,
        failed_code: str,
        traceback_text: str,
    ) -> ReflectionResult:
        """Use plain JSON mode because some Ollama builds reject schema grammar."""
        parsed = parse_traceback(traceback_text)
        payload = {
            "failed_code": failed_code,
            "traceback": parsed.model_dump(mode="json"),
            "required_output_schema": CodePatch.model_json_schema(),
        }
        response = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        f"{REFLECTOR_SYSTEM_PROMPT}\n"
                        "Return one raw JSON object only. Do not use Markdown fences."
                    )
                ),
                HumanMessage(content=json.dumps(payload, indent=2)),
            ]
        )
        content = response.content
        if not isinstance(content, str):
            content = json.dumps(content)
        start = content.find("{")
        if start < 0:
            raise ValueError("Qwen response did not contain a JSON object")
        patch_payload, _ = json.JSONDecoder().raw_decode(content[start:])
        patch = CodePatch.model_validate(patch_payload)
        patched_code, unified_diff = apply_code_patch(failed_code, patch)
        return ReflectionResult(
            parsed_traceback=parsed,
            patch=patch,
            patched_code=patched_code,
            unified_diff=unified_diff,
        )


class PacedGraph:
    """Add short, consistent pauses so state changes remain legible on video."""

    def __init__(self, graph: Any, delay_seconds: float) -> None:
        self.graph = graph
        self.delay_seconds = delay_seconds

    def stream(self, *args: Any, **kwargs: Any):
        for update in self.graph.stream(*args, **kwargs):
            time.sleep(self.delay_seconds)
            yield update


def create_demo_graph(
    max_retries: int = 2,
    *,
    use_qwen: bool = True,
    pace_seconds: float = 0.4,
) -> PacedGraph:
    """Create the real TraceMind graph with deterministic demo dependencies."""
    graph = create_graph(
        planner=DemoPlanner(),
        coder=DemoCoder(),
        reflector=DemoReflector(use_qwen=use_qwen),
        max_retries=max_retries,
    )
    return PacedGraph(graph, delay_seconds=pace_seconds)


def _render_app() -> None:
    import streamlit as st

    from src.ui.dashboard import render_dashboard

    use_qwen = os.getenv("TRACEMIND_DEMO_FIXTURE_REFLECTOR") != "1"
    mode = (
        "Live qwen2.5-coder:7b reflection · deterministic validated fallback"
        if use_qwen
        else "Fixture reflection · offline rehearsal mode"
    )
    st.session_state.setdefault("tracemind_demo_mode", mode)
    render_dashboard(
        default_prompt=DEMO_PROMPT,
        graph_factory=lambda retries: create_demo_graph(
            retries,
            use_qwen=use_qwen,
        ),
        auto_launch=True,
        scenario_note=mode,
        artifact_first=True,
    )


def _launch_streamlit(args: argparse.Namespace) -> int:
    environment = os.environ.copy()
    environment["TRACEMIND_TRACING_ENABLED"] = "0"
    environment.setdefault("TRACEMIND_LLM_TIMEOUT_SECONDS", "12")
    if args.fixture_reflector:
        environment["TRACEMIND_DEMO_FIXTURE_REFLECTOR"] = "1"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.port",
        str(args.port),
        "--server.headless",
        "false",
        "--",
        "--streamlit-app",
    ]
    return subprocess.call(command, cwd=REPOSITORY_ROOT, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit port (default: 8501).",
    )
    parser.add_argument(
        "--fixture-reflector",
        action="store_true",
        help="Skip Ollama and use the deterministic repair fixture.",
    )
    parser.add_argument("--streamlit-app", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.streamlit_app:
        _render_app()
        return 0
    return _launch_streamlit(args)


if __name__ == "__main__":
    raise SystemExit(main())

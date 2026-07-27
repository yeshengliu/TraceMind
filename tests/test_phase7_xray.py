"""Phase 7 LLM X-Ray capture, probability, vector, and KV-cache tests."""

from __future__ import annotations

import json
import os
from typing import Any

os.environ["TRACEMIND_TRACING_ENABLED"] = "0"

import httpx
import pytest
from pydantic import BaseModel
from streamlit.testing.v1 import AppTest

from src.agent.llm_inspector import (
    GenerationSettings,
    OllamaInspector,
    capture_prompt,
    parse_logprob_step,
)
from src.ui.token_vis import (
    attention_focus_proxy,
    build_probability_animation,
    build_top_tokens_figure,
    prompt_diff_html,
    render_confidence_heatmap,
)
from src.ui.vector_3d import (
    KVCacheConfig,
    MemoryVector,
    build_embedding_figure,
    build_kv_cache_figure,
    deterministic_demo_embeddings,
    estimate_kv_cache,
    project_vector_space,
    pruning_animation_html,
)


class DemoSchema(BaseModel):
    code: str
    confidence: float


class FakeStreamResponse:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads

    def __enter__(self) -> "FakeStreamResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        return iter(json.dumps(payload) for payload in self.payloads)


class FakeClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.request: dict[str, Any] | None = None

    def stream(self, method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        self.request = {"method": method, "url": url, **kwargs}
        return FakeStreamResponse(self.payloads)


def _logprob(
    token: str,
    selected_logprob: float,
    alternatives: list[tuple[str, float]],
) -> dict[str, Any]:
    return {
        "token": token,
        "logprob": selected_logprob,
        "top_logprobs": [
            {"token": candidate, "logprob": logprob}
            for candidate, logprob in alternatives
        ],
    }


def test_prompt_capture_includes_system_schema_memory_and_raw_input() -> None:
    capture = capture_prompt(
        "Return safe Python.",
        system_prompt="Stay offline.",
        schema=DemoSchema,
        memory_context={"latest_error": "TimeoutError"},
    )

    assert capture.raw_prompt == "Return safe Python."
    assert [section.kind for section in capture.sections] == [
        "system",
        "schema",
        "memory",
        "user",
    ]
    assert "latest_error" in capture.processed_prompt
    assert "confidence" in capture.processed_prompt
    assert capture.messages[-1] == {
        "role": "user",
        "content": "Return safe Python.",
    }
    assert "Assembled API input" in prompt_diff_html(capture)


def test_inspector_requests_and_parses_streamed_top_five_logprobs() -> None:
    payloads = [
        {
            "message": {"content": "print"},
            "logprobs": [
                _logprob(
                    "print",
                    -0.10,
                    [("print", -0.10), ("return", -1.7), ("def", -2.1)],
                )
            ],
            "done": False,
        },
        {
            "message": {"content": "(42)"},
            "logprobs": [
                _logprob(
                    "(42)",
                    -0.25,
                    [("(42)", -0.25), ("()", -1.5), ("(0)", -2.0)],
                )
            ],
            "done": True,
            "prompt_eval_count": 91,
            "eval_count": 2,
            "total_duration": 123_000,
        },
    ]
    client = FakeClient(payloads)
    inspector = OllamaInspector(base_url="http://localhost:11434/v1", client=client)
    observed = []
    settings = GenerationSettings(temperature=0.7, top_p=0.8, top_k=25)
    trace = inspector.generate(
        capture_prompt("Show the answer."),
        settings=settings,
        model="fixture-model",
        on_token=lambda step, partial: observed.append((step, partial)),
    )

    assert trace.content == "print(42)"
    assert trace.logprobs_available is True
    assert trace.prompt_tokens == 91
    assert trace.generated_tokens == 2
    assert len(trace.token_steps) == 2
    assert trace.token_steps[0].probability == pytest.approx(0.904837, rel=1e-5)
    assert observed[-1][1] == "print(42)"
    assert client.request is not None
    assert client.request["url"] == "http://localhost:11434/api/chat"
    assert client.request["json"]["logprobs"] is True
    assert client.request["json"]["top_logprobs"] == 5
    assert client.request["json"]["options"]["top_k"] == 25


def test_token_heatmap_histogram_animation_and_attention_proxy() -> None:
    step = parse_logprob_step(
        _logprob(
            "<unsafe>",
            -0.4,
            [
                ("<unsafe>", -0.4),
                (" offline", -0.8),
                (" local", -1.2),
                (" network", -1.6),
                (" sandbox", -2.0),
            ],
        ),
        0,
    )
    capture = capture_prompt(
        "Create code.",
        system_prompt="Keep generated code offline and avoid network access.",
    )

    heatmap = render_confidence_heatmap([step])
    histogram = build_top_tokens_figure(step)
    animation = build_probability_animation([step, step.model_copy(update={"index": 1})])
    focus = attention_focus_proxy(
        "The generated code stays offline without network access.",
        capture.sections,
    )

    assert "&lt;unsafe&gt;" in heatmap
    assert "<unsafe>" not in heatmap
    assert "Moderate Uncertainty" in heatmap
    assert len(histogram.data[0].x) == 5
    assert "raw softmax probability distribution" in histogram.data[0].hovertemplate
    assert len(animation.frames) == 2
    assert "raw softmax probability distribution" in animation.data[0].hovertemplate
    assert (
        "raw softmax probability distribution"
        in animation.frames[0].data[0].hovertemplate
    )
    assert focus[0].section == "System metaprompt"
    assert {"offline", "network"} <= set(focus[0].matched_terms)


def test_vector_projection_retrieval_and_3d_connections() -> None:
    texts = [
        "offline sandbox python",
        "weather forecast network",
        "safe python execution",
        "active offline python query",
    ]
    embeddings = deterministic_demo_embeddings(texts)
    memory = [
        MemoryVector(id="sandbox", label="Sandbox", text=texts[0], embedding=embeddings[0]),
        MemoryVector(id="weather", label="Weather", text=texts[1], embedding=embeddings[1]),
        MemoryVector(id="safety", label="Safety", text=texts[2], embedding=embeddings[2]),
    ]
    projection = project_vector_space(
        memory,
        embeddings[3],
        query_text=texts[3],
        top_k=2,
    )
    figure = build_embedding_figure(projection)
    query_trace = next(trace for trace in figure.data if trace.name == "Query pulse")

    assert len(projection.memory_xyz) == 3
    assert len(projection.memory_similarities) == 3
    assert len(projection.query_xyz) == 3
    assert len(projection.neighbors) == 2
    assert projection.neighbors[0].id == "sandbox"
    assert len(figure.data) >= 1 + 3 + 1 + 2
    assert figure.data[0].customdata[0][0] == "Memory Item"
    assert "Cosine similarity" in figure.data[0].hovertemplate
    assert query_trace.customdata[0][0] == "Query"
    assert "active offline python query" in query_trace.customdata[0][2]


def test_kv_cache_estimate_and_pruning_animation_are_transparent() -> None:
    config = KVCacheConfig(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        bytes_per_element=2,
        context_window=128,
        vram_budget_gb=1,
    )
    estimate = estimate_kv_cache(168, config)
    figure = build_kv_cache_figure(estimate)

    assert estimate.bytes_per_token == 64
    assert estimate.retained_tokens == 128
    assert estimate.pruned_tokens == 40
    assert estimate.estimated_bytes == 8_192
    assert "40 historical tokens evicted" in pruning_animation_html(estimate)
    assert figure.data[0].value == estimate.estimated_gb
    assert "Estimated GPU VRAM consumed" in figure.data[1].hovertemplate


def test_inspector_surfaces_missing_logprob_metadata_without_fabricating_it() -> None:
    client = FakeClient(
        [
            {
                "message": {"content": "plain response"},
                "done": True,
                "eval_count": 2,
            }
        ]
    )
    trace = OllamaInspector(client=client).generate(capture_prompt("Test metadata."))

    assert trace.content == "plain response"
    assert trace.logprobs_available is False
    assert trace.token_steps == []
    assert trace.notice is not None
    assert "returned no token logprobs" in trace.notice


def test_xray_streamlit_controls_expose_educational_help() -> None:
    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    sliders = {slider.label: slider for slider in app.slider}
    assert "Controls randomness" in sliders["Temperature · A"].help
    assert "Cumulative probability cutoff" in sliders["Top-P · A"].help
    assert "K most likely tokens" in sliders["Top-K · A"].help
    assert any(
        expander.label == "ℹ️ How to read this X-Ray Lab"
        for expander in app.expander
    )
    assert any(
        caption.help and "observable model telemetry" in caption.help
        for caption in app.caption
    )

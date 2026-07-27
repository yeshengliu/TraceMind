"""Streamlit assembly for the interactive LLM X-Ray Lab."""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

import httpx
import streamlit as st

from src.agent.graph import CODER_SYSTEM_PROMPT
from src.agent.llm import ModelRole, model_name_for
from src.agent.llm_inspector import (
    GenerationSettings,
    GenerationTrace,
    OllamaInspector,
    PromptCapture,
    TokenStep,
    capture_prompt,
)
from src.agent.tools import CodeGenerationOutput
from src.ui.token_vis import (
    HIGH_CONFIDENCE_EXPLANATION,
    HIGH_ENTROPY_EXPLANATION,
    MODERATE_UNCERTAINTY_EXPLANATION,
    TOP_FIVE_EXPLANATION,
    attention_focus_proxy,
    build_probability_animation,
    build_top_tokens_figure,
    prompt_diff_html,
    render_confidence_heatmap,
)
from src.ui.vector_3d import (
    KV_CACHE_EXPLANATION,
    VECTOR_PLOT_EXPLANATION,
    KVCacheConfig,
    MemoryVector,
    build_embedding_figure,
    build_kv_cache_figure,
    deterministic_demo_embeddings,
    estimate_kv_cache,
    project_vector_space,
    pruning_animation_html,
)

_TEMPERATURE_HELP = (
    "Controls randomness in token selection. Lower values (e.g., 0.1) make output "
    "deterministic; higher values (e.g., 1.0) encourage creative/diverse sampling."
)
_TOP_P_HELP = (
    "Cumulative probability cutoff. The model only considers the smallest set of "
    "top tokens whose combined probability exceeds P (e.g., 0.90)."
)
_TOP_K_HELP = (
    "Limits candidate token pool to the K most likely tokens before applying softmax."
)
_METAPROMPT_HELP = (
    "Explains how raw developer prompts are wrapped with System Instructions, "
    "Pydantic JSON schemas, and Memory Context before hitting the GPU."
)
_ATTENTION_PROXY_HELP = (
    "Ranks prompt sections by lexical overlap with generated output. This is an "
    "explainable attribution proxy, not measured Transformer attention weight, "
    "because Ollama does not expose attention tensors."
)


_DEFAULT_XRAY_PROMPT = (
    "Write a small offline Python function that validates a list of temperatures "
    "and returns their mean. Explain the observable design choices briefly."
)

_MEMORY_DOCUMENTS = [
    (
        "sandbox-policy",
        "Sandbox policy",
        "Generated Python runs offline, without network access or third-party packages.",
    ),
    (
        "artifact-contract",
        "Artifact protocol",
        "Programs print requested results and may emit marked SVG, JSON, or Markdown artifacts.",
    ),
    (
        "healing-memory",
        "Healing memory",
        "Keep the newest traceback and latest validated patch; compact superseded failures.",
    ),
    (
        "schema-contract",
        "Coder schema",
        "Return complete Python code, a bounded timeout_seconds value, and a concise summary.",
    ),
    (
        "safety-rule",
        "Execution safety",
        "Never spawn processes, access host files, open a GUI, or depend on the network.",
    ),
]

_XRAY_CSS = """
<style>
.xray-hero {
  border: 1px solid rgba(56,189,248,.22); border-radius: 18px;
  padding: 1.25rem 1.4rem; margin: .25rem 0 1rem;
  background: radial-gradient(circle at 90% 0%, rgba(56,189,248,.14), transparent 38%),
              rgba(8,17,31,.48);
}
.xray-hero h2 { margin: 0 0 .3rem; }
.xray-hero p { color: #94a3b8; margin: 0; }
.xray-heatmap {
  white-space: pre-wrap; overflow-wrap: anywhere; line-height: 2.05;
  padding: 1rem; border: 1px solid rgba(148,163,184,.15);
  border-radius: 13px; background: rgba(8,17,31,.55);
  font: .82rem/2.05 "DM Mono", monospace;
}
.xray-token { border-radius: 4px; padding: .13rem .08rem; }
.xray-legend { color: #94a3b8; font-size: .76rem; margin: .35rem 0 1rem; }
.xray-legend i { display:inline-block; width:.75rem; height:.75rem; border-radius:3px; margin:0 .25rem 0 .7rem; }
.xray-tip {
  position: relative; display: inline-flex; align-items: center; cursor: help;
  border-bottom: 1px dotted rgba(148,163,184,.65); outline: none;
}
.xray-tip::after {
  content: attr(data-tooltip); position: absolute; z-index: 999; left: 50%;
  bottom: calc(100% + 9px); width: min(310px, 70vw); padding: .65rem .75rem;
  border: 1px solid rgba(56,189,248,.3); border-radius: 9px;
  background: #0f172a; color: #e2e8f0; box-shadow: 0 12px 35px rgba(0,0,0,.35);
  font: .74rem/1.45 sans-serif; opacity: 0; visibility: hidden;
  transform: translate(-50%, 5px); transition: opacity .15s ease, transform .15s ease;
  pointer-events: none;
}
.xray-tip:hover::after, .xray-tip:focus-visible::after {
  opacity: 1; visibility: visible; transform: translate(-50%, 0);
}
.xray-concept-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: .65rem; margin: .25rem 0 .8rem;
}
.xray-concept-card {
  padding: .8rem .9rem; border: 1px solid rgba(148,163,184,.16);
  border-radius: 12px; background: rgba(15,23,42,.48); color: #cbd5e1;
  font-size: .82rem; line-height: 1.5;
}
.xray-concept-card strong { color: #e2e8f0; display: block; margin-bottom: .2rem; }
.diff { width: 100%; border-collapse: collapse; font: .72rem "DM Mono", monospace; }
.diff th { color:#e2e8f0; padding:.5rem; background:rgba(51,65,85,.6); }
.diff td { padding:.13rem .35rem; vertical-align:top; }
.diff_header { color:#64748b; }
.diff_add { background:rgba(16,185,129,.28); }
.diff_chg { background:rgba(245,158,11,.3); }
.diff_sub { background:rgba(244,63,94,.25); }
.kv-stable, .kv-prune { color:#94a3b8; padding:.7rem 0; }
.kv-shred { display:flex; gap:4px; margin-top:.55rem; height:22px; overflow:hidden; }
.kv-shred span { width:4.5%; background:#fb7185; border-radius:2px;
  animation:kv-shred .9s ease-in forwards; }
@keyframes kv-shred {
  0% { opacity:.8; transform:translateY(0) rotate(0); }
  100% { opacity:0; transform:translateY(28px) rotate(30deg); }
}
</style>
"""


def _tooltip_label(label: str, explanation: str) -> str:
    """Return an accessible CSS tooltip for compact inline annotations."""
    return (
        "<span class='xray-tip' tabindex='0' data-tooltip='"
        f"{html.escape(explanation, quote=True)}'>{html.escape(label)}</span>"
    )


def _render_reading_guide() -> None:
    with st.expander("ℹ️ How to read this X-Ray Lab", expanded=False):
        st.markdown(
            """
            <div class="xray-concept-grid">
              <div class="xray-concept-card"><strong>🎛️ Sampling controls</strong>
              Compare identical prompts while changing randomness and the candidate-token pool.</div>
              <div class="xray-concept-card"><strong>🎯 Token competition</strong>
              Each colored token shows selected-token confidence; the histogram reveals its top alternatives.</div>
              <div class="xray-concept-card"><strong>🧠 Prompt metamorphosis</strong>
              Inspect the system policy, output schema, and memory injected around the raw request.</div>
              <div class="xray-concept-card"><strong>🌌 Retrieval & cache</strong>
              Follow nearest-memory links and the estimated GPU cost of retaining prior tokens.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Hover chart marks, colored tokens, technical labels, and Streamlit `?` "
            "icons for concise explanations.",
            help=(
                "The lab visualizes observable model telemetry and clearly labeled "
                "estimates or proxies; it does not expose private chain-of-thought."
            ),
        )


def _capture_for_prompt(prompt: str) -> PromptCapture:
    memory = {
        "retrieval_policy": "Top-K local semantic memories are injected when available.",
        "candidate_memories": [
            {"id": item_id, "label": label, "text": text}
            for item_id, label, text in _MEMORY_DOCUMENTS
        ],
    }
    return capture_prompt(
        prompt,
        system_prompt=CODER_SYSTEM_PROMPT,
        schema=CodeGenerationOutput,
        memory_context=memory,
        extra_system_sections={
            "X-Ray observability rule": (
                "Return concise observable rationale only. Do not expose or invent "
                "private hidden chain-of-thought."
            )
        },
    )


def _settings_controls(prefix: str, defaults: GenerationSettings) -> GenerationSettings:
    st.markdown(f"#### Configuration {prefix}")
    temperature = st.slider(
        f"Temperature · {prefix}",
        0.0,
        2.0,
        defaults.temperature,
        0.05,
        key=f"xray_temperature_{prefix}",
        help=_TEMPERATURE_HELP,
    )
    top_p = st.slider(
        f"Top-P · {prefix}",
        0.05,
        1.0,
        defaults.top_p,
        0.05,
        key=f"xray_top_p_{prefix}",
        help=_TOP_P_HELP,
    )
    top_k = st.slider(
        f"Top-K · {prefix}",
        1,
        100,
        defaults.top_k,
        1,
        key=f"xray_top_k_{prefix}",
        help=_TOP_K_HELP,
    )
    return GenerationSettings(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=defaults.seed,
        max_tokens=defaults.max_tokens,
    )


def _run_live(
    inspector: OllamaInspector,
    capture: PromptCapture,
    settings: GenerationSettings,
    *,
    model: str,
    output_placeholder: Any,
    chart_placeholder: Any,
    cache_placeholder: Any,
    context_window: int,
) -> GenerationTrace:
    """Run synchronously while updating Streamlit from the main script thread."""

    def on_token(step: TokenStep, partial_text: str) -> None:
        output_placeholder.code(partial_text or "…", language="python")
        chart_placeholder.plotly_chart(
            build_top_tokens_figure(step),
            width="stretch",
            config={"displayModeBar": False},
            key=f"live-{id(output_placeholder)}-{step.index}",
        )
        live_tokens = len(capture_words(capture.processed_prompt)) + step.index + 1
        cache = estimate_kv_cache(
            live_tokens,
            KVCacheConfig(context_window=context_window),
        )
        cache_placeholder.progress(
            min(1.0, cache.utilization),
            text=(
                f"Live KV estimate · {cache.retained_tokens:,} tokens · "
                f"{cache.estimated_gb:.3f} GiB"
            )
        )

    trace = inspector.generate(
        capture,
        settings=settings,
        model=model,
        on_token=on_token,
    )
    output_placeholder.code(trace.content or "(empty model response)", language="python")
    if trace.token_steps:
        chart_placeholder.plotly_chart(
            build_top_tokens_figure(trace.token_steps[-1]),
            width="stretch",
            config={"displayModeBar": False},
            key=f"live-final-{id(output_placeholder)}",
        )
    return trace


def _render_trace(trace: GenerationTrace, label: str) -> None:
    st.markdown(f"### Run {label}")
    mean_confidence = (
        sum(step.probability for step in trace.token_steps) / len(trace.token_steps)
        if trace.token_steps
        else 0.0
    )
    metrics = st.columns(3)
    metrics[0].metric(
        "Output tokens",
        trace.generated_tokens,
        help="Number of tokens generated by the local model for this run.",
    )
    metrics[1].metric(
        "Mean confidence",
        f"{mean_confidence:.1%}",
        help="Average selected-token softmax probability across the observed output.",
    )
    metrics[2].metric(
        "Temperature",
        f"{trace.settings.temperature:.2f}",
        help=_TEMPERATURE_HELP,
    )
    if trace.notice:
        st.warning(trace.notice)
    if trace.token_steps:
        st.markdown(
            render_confidence_heatmap(trace.token_steps),
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='xray-legend'>"
            "<i style='background:#10b981'></i>"
            + _tooltip_label("green · high >80%", HIGH_CONFIDENCE_EXPLANATION)
            + "<i style='background:#f59e0b'></i>"
            + _tooltip_label(
                "yellow · moderate 40–80%", MODERATE_UNCERTAINTY_EXPLANATION
            )
            + "<i style='background:#f43f5e'></i>"
            + _tooltip_label("red · high entropy ≤40%", HIGH_ENTROPY_EXPLANATION)
            + "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Top-5 candidate distribution", help=TOP_FIVE_EXPLANATION)
        st.plotly_chart(
            build_probability_animation(trace.token_steps),
            width="stretch",
            config={"displayModeBar": False},
            key=f"xray-animation-{label}",
        )
    else:
        st.code(trace.content or "(empty model response)", language="python")


def _render_prompt_and_focus(
    capture: PromptCapture,
    traces: Sequence[GenerationTrace],
) -> None:
    st.markdown("### Prompt metamorphosis")
    st.caption(
        "The right side is the exact role-tagged API input assembled by this lab, "
        "including system policy, schema, and memory context. Ollama applies the "
        "selected model's chat template after receiving it.",
        help=_METAPROMPT_HELP,
    )
    st.markdown(
        f"<div style='overflow:auto'>{prompt_diff_html(capture)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Attention focus inspector")
    st.caption("Prompt-section attribution proxy", help=_ATTENTION_PROXY_HELP)
    st.info(
        "Ollama does not expose transformer attention matrices. These matches are "
        "an explainable lexical input/output attribution proxy, not hidden attention."
    )
    if not traces:
        st.caption("Run the comparison to inspect which prompt sections align with output.")
        return
    for index, trace in enumerate(traces):
        matches = attention_focus_proxy(trace.content, capture.sections)
        with st.expander(f"Run {'AB'[index]} prompt-section alignment", expanded=True):
            if not matches:
                st.caption("No meaningful lexical alignment was detected.")
            for match in matches:
                st.progress(
                    match.score,
                    text=(
                        f"{match.section} · {match.score:.1%} · "
                        f"{', '.join(match.matched_terms)}"
                    ),
                )
                st.caption(match.excerpt, help=_ATTENTION_PROXY_HELP)


def _memory_projection(
    inspector: OllamaInspector,
    prompt: str,
    *,
    use_ollama: bool,
) -> tuple[list[MemoryVector], list[float], bool]:
    texts = [document[2] for document in _MEMORY_DOCUMENTS]
    if not use_ollama:
        vectors = deterministic_demo_embeddings([*texts, prompt])
        synthetic = True
    else:
        try:
            vectors = inspector.embed([*texts, prompt])
            synthetic = False
        except (httpx.HTTPError, OSError, ValueError):
            vectors = deterministic_demo_embeddings([*texts, prompt])
            synthetic = True
    memory = [
        MemoryVector(id=item_id, label=label, text=text, embedding=vectors[index])
        for index, (item_id, label, text) in enumerate(_MEMORY_DOCUMENTS)
    ]
    return memory, vectors[-1], synthetic


def _render_memory_and_kv(
    inspector: OllamaInspector,
    prompt: str,
    traces: Sequence[GenerationTrace],
    *,
    context_window: int,
) -> None:
    st.markdown("### 3D local memory retrieval")
    memory, query_embedding, synthetic = _memory_projection(
        inspector,
        prompt,
        use_ollama=bool(traces),
    )
    if synthetic:
        st.warning(
            "This scene is explicitly using deterministic synthetic demo vectors. "
            "After a successful A/B run the lab requests Ollama embeddings; pull "
            "`nomic-embed-text` to enable real retrieval."
        )
    projection = project_vector_space(
        memory,
        query_embedding,
        query_label="Active query",
        query_text=prompt,
        top_k=3,
    )
    st.caption("Hover any node to inspect its type, similarity, and text.", help=VECTOR_PLOT_EXPLANATION)
    st.plotly_chart(
        build_embedding_figure(projection),
        width="stretch",
        config={"displayModeBar": False},
        key="xray-vector-space",
    )
    st.caption(
        "Top-K: "
        + " · ".join(
            f"{neighbor.label} ({neighbor.similarity:.3f})"
            for neighbor in projection.neighbors
        ),
        help=VECTOR_PLOT_EXPLANATION,
    )

    st.markdown("### KV cache memory gauge")
    st.caption("Estimated retained-attention memory pressure", help=KV_CACHE_EXPLANATION)
    measured_prompt_tokens = max(
        [trace.prompt_tokens for trace in traces] or [len(capture_words(prompt))]
    )
    generated = max([trace.generated_tokens for trace in traces] or [0])
    estimate = estimate_kv_cache(
        measured_prompt_tokens + generated,
        KVCacheConfig(context_window=context_window),
    )
    st.plotly_chart(
        build_kv_cache_figure(estimate),
        width="stretch",
        config={"displayModeBar": False},
        key="xray-kv-cache",
    )
    st.caption(
        "Estimate: 2 × layers × KV heads × head dimension × tokens × dtype bytes. "
        "Actual allocation depends on model architecture and runtime quantization.",
        help=KV_CACHE_EXPLANATION,
    )
    st.markdown(pruning_animation_html(estimate), unsafe_allow_html=True)


def capture_words(text: str) -> list[str]:
    """Small tokenizer estimate used only when Ollama usage metadata is absent."""
    return text.split()


def render_xray_tab(*, inspector: OllamaInspector | None = None) -> None:
    """Render the complete Phase 7 lab inside an existing Streamlit page."""
    st.markdown(_XRAY_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="xray-hero">
          <h2>🔬 LLM X-Ray Lab</h2>
          <p>Inspect prompt injection, token sampling, retrieval geometry, and
          estimated KV-cache pressure from your local Ollama model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Observable API telemetry only. The lab does not reveal private hidden "
        "chain-of-thought or claim access to unexposed attention tensors.",
        help=(
            "Token probabilities come from the local endpoint when available. "
            "Retrieval and KV-cache views state when values are estimates or proxies."
        ),
    )
    _render_reading_guide()

    prompt = st.text_input(
        "Shared comparison prompt",
        value=_DEFAULT_XRAY_PROMPT,
        key="xray_prompt",
        help="This exact prompt is sent to both A and B so sampling settings are the independent variable.",
    )
    model = st.text_input(
        "Local model",
        value=model_name_for(ModelRole.CODER),
        key="xray_model",
        help="Local Ollama model used for both sides of the comparison.",
    )
    controls_a, controls_b = st.columns(2)
    with controls_a:
        settings_a = _settings_controls(
            "A", GenerationSettings(temperature=0.1, top_p=0.9, top_k=20)
        )
    with controls_b:
        settings_b = _settings_controls(
            "B", GenerationSettings(temperature=1.0, top_p=0.95, top_k=60)
        )
    context_window = st.slider(
        "KV context window (tokens)",
        128,
        32_768,
        8_192,
        128,
        key="xray_context_window",
        help=(
            "Maximum retained token count used by the KV-cache estimate. Tokens "
            "beyond this boundary trigger the pruning animation."
        ),
    )

    try:
        capture = _capture_for_prompt(prompt)
    except ValueError as exc:
        st.error(str(exc))
        return

    active_inspector = inspector or OllamaInspector()
    trace_context_key = f"{model}\0{capture.processed_prompt}"
    if st.button(
        "Run A/B X-Ray comparison",
        type="primary",
        width="stretch",
        key="xray_run_comparison",
    ):
        st.session_state.pop("xray_traces", None)
        st.session_state.pop("xray_trace_context", None)
        traces = []
        live_columns = st.columns(2)
        try:
            for label, settings, column in zip(
                ("A", "B"),
                (settings_a, settings_b),
                live_columns,
                strict=True,
            ):
                with column:
                    st.markdown(f"#### Live stream {label}")
                    traces.append(
                        _run_live(
                            active_inspector,
                            capture,
                            settings,
                            model=model,
                            output_placeholder=st.empty(),
                            chart_placeholder=st.empty(),
                            cache_placeholder=st.empty(),
                            context_window=context_window,
                        )
                    )
        except (httpx.HTTPError, OSError, ValueError, RuntimeError) as exc:
            st.error(
                "Local generation failed. Confirm Ollama is running and the selected "
                f"model is installed. Details: {exc}"
            )
        else:
            st.session_state["xray_traces"] = [
                trace.model_dump(mode="json") for trace in traces
            ]
            st.session_state["xray_trace_context"] = trace_context_key

    stored_traces = (
        st.session_state.get("xray_traces", [])
        if st.session_state.get("xray_trace_context") == trace_context_key
        else []
    )
    traces = [GenerationTrace.model_validate(item) for item in stored_traces]
    probability_tab, prompt_tab, memory_tab = st.tabs(
        ["🎯 Token probabilities", "🧠 Prompt & focus", "🌌 Vector & KV cache"]
    )
    with probability_tab:
        if traces:
            trace_columns = st.columns(2)
            for index, trace in enumerate(traces[:2]):
                with trace_columns[index]:
                    _render_trace(trace, "AB"[index])
        else:
            st.info("Run the A/B comparison to populate live Top-5 and confidence views.")
    with prompt_tab:
        _render_prompt_and_focus(capture, traces)
    with memory_tab:
        _render_memory_and_kv(
            active_inspector,
            prompt,
            traces,
            context_window=context_window,
        )


__all__ = ["render_xray_tab"]

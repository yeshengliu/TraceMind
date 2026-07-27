"""Token probability, confidence, prompt-diff, and attribution visuals."""

from __future__ import annotations

import difflib
import html
import re
from collections.abc import Sequence

import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, Field

from src.agent.llm_inspector import PromptCapture, PromptSection, TokenStep


HIGH_CONFIDENCE_EXPLANATION = (
    "High Confidence (P > 80%). The model is very certain about this token."
)
MODERATE_UNCERTAINTY_EXPLANATION = (
    "Moderate Uncertainty (40% < P <= 80%). Alternatives were strongly considered."
)
HIGH_ENTROPY_EXPLANATION = (
    "High Entropy / Potential Hallucination (P <= 40%). "
    "High risk of random guessing or error."
)
TOP_FIVE_EXPLANATION = (
    "Displays the raw softmax probability distribution for the top 5 competing "
    "tokens considered at this exact generation step."
)


class AttentionMatch(BaseModel):
    """An explainable lexical alignment between output and a prompt section."""

    model_config = ConfigDict(extra="forbid")

    section: str
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    excerpt: str = ""


def confidence_color(probability: float) -> str:
    """Map confidence to an accessible red/amber/green background."""
    bounded = min(1.0, max(0.0, probability))
    if bounded > 0.80:
        return f"rgba(16,185,129,{0.18 + bounded * 0.42:.3f})"
    if bounded >= 0.40:
        return f"rgba(245,158,11,{0.20 + bounded * 0.35:.3f})"
    return f"rgba(244,63,94,{0.22 + (1.0 - bounded) * 0.34:.3f})"


def confidence_explanation(probability: float) -> str:
    """Return the educational interpretation for one selected-token probability."""
    if probability > 0.80:
        return HIGH_CONFIDENCE_EXPLANATION
    if probability > 0.40:
        return MODERATE_UNCERTAINTY_EXPLANATION
    return HIGH_ENTROPY_EXPLANATION


def render_confidence_heatmap(steps: Sequence[TokenStep]) -> str:
    """Return safe HTML whose token backgrounds encode selected probability."""
    spans = []
    for step in steps:
        token = html.escape(step.token)
        title = html.escape(
            f"step {step.index} · p={step.probability:.3f} · "
            f"logp={step.logprob:.3f} · {confidence_explanation(step.probability)}"
        )
        spans.append(
            "<span class='xray-token' "
            f"style='background:{confidence_color(step.probability)}' "
            f"title='{title}'>{token}</span>"
        )
    return (
        "<div class='xray-heatmap' aria-label='Token confidence heatmap'>"
        + "".join(spans)
        + "</div>"
    )


def build_top_tokens_figure(step: TokenStep) -> go.Figure:
    """Build the live horizontal Top-5 candidate histogram for one step."""
    candidates = list(reversed(step.candidates[:5]))
    colors = [
        "#38bdf8" if candidate.token == step.token else "#64748b"
        for candidate in candidates
    ]
    figure = go.Figure(
        go.Bar(
            x=[candidate.probability for candidate in candidates],
            y=[candidate.token.replace(" ", "·") or "∅" for candidate in candidates],
            orientation="h",
            marker={"color": colors},
            text=[f"{candidate.probability:.1%}" for candidate in candidates],
            textposition="auto",
            customdata=[
                [
                    candidate.token or "∅",
                    confidence_explanation(candidate.probability),
                    "Sampled token" if candidate.token == step.token else "Alternative",
                ]
                for candidate in candidates
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[2]}"
                "<br>Softmax probability: %{x:.4f} (%{x:.1%})"
                "<br>%{customdata[1]}"
                f"<br><br>{TOP_FIVE_EXPLANATION}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"Step {step.index + 1} · sampled {step.token!r}",
        xaxis={"range": [0, 1], "tickformat": ".0%", "title": "softmax probability"},
        yaxis={"title": ""},
        height=285,
        margin={"l": 25, "r": 15, "t": 48, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,17,31,.45)",
        font={"color": "#cbd5e1"},
        showlegend=False,
    )
    return figure


def build_probability_animation(steps: Sequence[TokenStep]) -> go.Figure:
    """Build a Plotly playback animation across all observed token steps."""
    if not steps:
        figure = go.Figure()
        figure.add_annotation(text="No token logprobs returned", showarrow=False)
        return figure

    def frame_bar(step: TokenStep) -> go.Bar:
        candidates = list(reversed(step.candidates[:5]))
        return go.Bar(
            x=[candidate.probability for candidate in candidates],
            y=[candidate.token.replace(" ", "·") or "∅" for candidate in candidates],
            orientation="h",
            marker={
                "color": [
                    "#38bdf8" if candidate.token == step.token else "#64748b"
                    for candidate in candidates
                ]
            },
            text=[f"{candidate.probability:.1%}" for candidate in candidates],
            textposition="auto",
            customdata=[
                [
                    candidate.token or "∅",
                    confidence_explanation(candidate.probability),
                    "Sampled token" if candidate.token == step.token else "Alternative",
                ]
                for candidate in candidates
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[2]}"
                "<br>Softmax probability: %{x:.4f} (%{x:.1%})"
                "<br>%{customdata[1]}"
                f"<br><br>{TOP_FIVE_EXPLANATION}<extra></extra>"
            ),
        )

    frames = [
        go.Frame(data=[frame_bar(step)], name=str(step.index)) for step in steps
    ]
    figure = go.Figure(data=[frame_bar(steps[0])], frames=frames)
    figure.update_layout(
        xaxis={"range": [0, 1], "tickformat": ".0%", "title": "softmax probability"},
        height=330,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,17,31,.45)",
        font={"color": "#cbd5e1"},
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "▶ Play token stream",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 180, "redraw": True},
                                "transition": {"duration": 80},
                                "fromcurrent": True,
                            },
                        ],
                    }
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "label": str(step.index + 1),
                        "method": "animate",
                        "args": [
                            [str(step.index)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for step in steps
                ]
            }
        ],
    )
    return figure


def prompt_diff_html(capture: PromptCapture) -> str:
    """Render raw versus fully injected prompt as a side-by-side HTML diff."""
    differ = difflib.HtmlDiff(wrapcolumn=72)
    return differ.make_table(
        capture.raw_prompt.splitlines(),
        capture.processed_prompt.splitlines(),
        fromdesc="Raw developer/user prompt",
        todesc="Assembled API input (before model chat template)",
        context=False,
        numlines=2,
    )


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{2,}")
_STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "only",
    "that",
    "the",
    "this",
    "with",
    "you",
    "your",
}


def _terms(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _WORD.finditer(text)
        if match.group(0).lower() not in _STOP_WORDS
    }


def attention_focus_proxy(
    generated_text: str,
    sections: Sequence[PromptSection],
    *,
    limit: int = 5,
) -> list[AttentionMatch]:
    """Rank prompt sections by lexical alignment with the generated response.

    This is deliberately named a proxy: it is explainable input/output
    attribution, not a model attention tensor.
    """
    output_terms = _terms(generated_text)
    if not output_terms:
        return []
    matches = []
    for section in sections:
        if section.kind == "user":
            continue
        section_terms = _terms(section.content)
        overlap = sorted(output_terms & section_terms)
        if not overlap:
            continue
        score = len(overlap) / max(1, min(len(output_terms), len(section_terms)))
        matches.append(
            AttentionMatch(
                section=section.name,
                score=min(1.0, score),
                matched_terms=overlap[:12],
                excerpt=section.content.strip().replace("\n", " ")[:220],
            )
        )
    return sorted(matches, key=lambda match: match.score, reverse=True)[:limit]


__all__ = [
    "AttentionMatch",
    "attention_focus_proxy",
    "build_probability_animation",
    "build_top_tokens_figure",
    "confidence_color",
    "confidence_explanation",
    "HIGH_CONFIDENCE_EXPLANATION",
    "HIGH_ENTROPY_EXPLANATION",
    "MODERATE_UNCERTAINTY_EXPLANATION",
    "prompt_diff_html",
    "render_confidence_heatmap",
    "TOP_FIVE_EXPLANATION",
]

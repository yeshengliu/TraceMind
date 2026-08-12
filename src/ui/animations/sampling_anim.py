"""Interactive Temperature & Top-P Sampler Animation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_DEFAULT_LOGITS = [
    ("def", 4.5),
    ("return", 3.2),
    ("print", 2.6),
    ("import", 1.9),
    ("class", 1.2),
    ("for", 0.6),
    ("if", 0.2),
    ("while", -0.5),
]


def softmax_with_temp(logits: list[float], temp: float) -> np.ndarray:
    """Compute softmax probabilities with temperature scaling: p_i = exp(z_i / T) / sum(exp(z_j / T))."""
    scaled = np.array(logits, dtype=float) / max(0.01, temp)
    # Numerical stability
    scaled -= np.max(scaled)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled)


def build_sampling_temperature_figure(
    tokens_with_logits: list[tuple[str, float]] | None = None,
    current_temp: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 5,
) -> go.Figure:
    """Build a Plotly frame-by-frame animation showing Softmax distribution dynamics across temperatures."""
    data = tokens_with_logits or _DEFAULT_LOGITS
    tokens = [item[0] for item in data]
    logits = [item[1] for item in data]

    # Temperatures for frame-by-frame slider transition
    temps = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
    # Ensure current_temp is in frames
    if not any(np.isclose(current_temp, t, atol=0.04) for t in temps):
        temps.append(current_temp)
        temps.sort()

    frames: list[go.Frame] = []
    for t in temps:
        probs = softmax_with_temp(logits, t)
        
        # Apply Top-K & Top-P filter coloring logic
        sorted_indices = np.argsort(probs)[::-1]
        cumulative = 0.0
        colors = []
        for idx in range(len(probs)):
            orig_idx = idx
            rank = int(np.where(sorted_indices == orig_idx)[0][0])
            prob_val = probs[orig_idx]
            
            # Cumulative sum check for Top-P
            top_p_indices = []
            cum_val = 0.0
            for s_idx in sorted_indices:
                top_p_indices.append(s_idx)
                cum_val += probs[s_idx]
                if cum_val >= top_p:
                    break

            if rank < top_k and orig_idx in top_p_indices:
                colors.append("#38bdf8")  # Highlighted blue: sampled candidate
            elif rank >= top_k:
                colors.append("#64748b")  # Dark gray: pruned by Top-K
            else:
                colors.append("#f43f5e")  # Reddish: pruned by Top-P

        frame_data = go.Bar(
            x=tokens,
            y=probs,
            marker={"color": colors, "line": {"color": "rgba(255,255,255,0.2)", "width": 1}},
            text=[f"{p:.1%}" for p in probs],
            textposition="auto",
            customdata=[[t, logits[i], f"Rank {int(np.where(sorted_indices == i)[0][0]) + 1}"] for i in range(len(tokens))],
            hovertemplate=(
                "Token: <b>%{x}</b><br>"
                "Probability: %{y:.2%}<br>"
                "Raw Logit: %{customdata[1]:.2f}<br>"
                "%{customdata[2]} at Temp=%{customdata[0]:.2f}"
                "<extra></extra>"
            ),
        )
        frames.append(go.Frame(data=[frame_data], name=f"temp_{t:.2f}"))

    # Initial frame for current_temp
    init_probs = softmax_with_temp(logits, current_temp)
    init_sorted = np.argsort(init_probs)[::-1]
    cum_val = 0.0
    top_p_set = set()
    for s_idx in init_sorted:
        top_p_set.add(s_idx)
        cum_val += init_probs[s_idx]
        if cum_val >= top_p:
            break

    init_colors = []
    for idx in range(len(init_probs)):
        rank = int(np.where(init_sorted == idx)[0][0])
        if rank < top_k and idx in top_p_set:
            init_colors.append("#38bdf8")
        elif rank >= top_k:
            init_colors.append("#64748b")
        else:
            init_colors.append("#f43f5e")

    fig = go.Figure(
        data=[
            go.Bar(
                x=tokens,
                y=init_probs,
                marker={"color": init_colors, "line": {"color": "rgba(255,255,255,0.2)", "width": 1}},
                text=[f"{p:.1%}" for p in init_probs],
                textposition="auto",
                customdata=[[current_temp, logits[i], f"Rank {int(np.where(init_sorted == i)[0][0]) + 1}"] for i in range(len(tokens))],
                hovertemplate=(
                    "Token: <b>%{x}</b><br>"
                    "Probability: %{y:.2%}<br>"
                    "Raw Logit: %{customdata[1]:.2f}<br>"
                    "%{customdata[2]} at Temp=%{customdata[0]:.2f}"
                    "<extra></extra>"
                ),
            )
        ],
        frames=frames,
    )

    # Add slider control for animation frames
    slider_steps = []
    for t in temps:
        slider_steps.append(
            {
                "args": [
                    [f"temp_{t:.2f}"],
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 200},
                    },
                ],
                "label": f"{t:.2f}",
                "method": "animate",
            }
        )

    fig.update_layout(
        height=380,
        margin={"l": 20, "r": 20, "t": 40, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,17,31,0.5)",
        font={"color": "#cbd5e1"},
        xaxis={"title": "Candidate Tokens", "gridcolor": "rgba(148,163,184,0.1)"},
        yaxis={"title": "Softmax Probability", "range": [0, 1.05], "gridcolor": "rgba(148,163,184,0.1)"},
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 400, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 200},
                            },
                        ],
                        "label": "▶ Play Temperature Sweep",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "⏸ Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": next((i for i, t in enumerate(temps) if np.isclose(t, current_temp, atol=0.04)), 0),
                "currentvalue": {"prefix": "Temperature T = ", "visible": True, "xanchor": "right"},
                "pad": {"b": 10, "t": 20},
                "steps": slider_steps,
                "x": 0.0,
                "len": 1.0,
            }
        ],
    )
    return fig


def render_sampling_animation_ui(key_prefix: str = "") -> None:
    """Render interactive sampling animation UI with user controls."""
    prefix = f"{key_prefix}_" if key_prefix else ""
    st.markdown("#### 🎛️ Temperature & Softmax Dynamics")
    st.caption(
        "Adjust Temperature to see how candidate probabilities sharpen or flatten. "
        "Blue bars are sampled candidates; gray bars are cut off by Top-K; red bars by Top-P."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.slider("Temperature (T)", 0.1, 2.0, 0.7, 0.05, key=f"{prefix}anim_sampling_temp")
    with col2:
        top_p = st.slider("Top-P Cutoff", 0.1, 1.0, 0.9, 0.05, key=f"{prefix}anim_sampling_top_p")
    with col3:
        top_k = st.slider("Top-K Cutoff", 1, 8, 5, 1, key=f"{prefix}anim_sampling_top_k")

    fig = build_sampling_temperature_figure(current_temp=temp, top_p=top_p, top_k=top_k)
    st.plotly_chart(fig, width="stretch", key=f"{prefix}anim_sampling_chart")


__all__ = ["build_sampling_temperature_figure", "render_sampling_animation_ui", "softmax_with_temp"]

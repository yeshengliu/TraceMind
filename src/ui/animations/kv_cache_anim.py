"""Interactive KV Cache & Memory Pruning Dynamics Animation."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

_KV_CACHE_ANIM_CSS = """
<style>
.kv-vram-container {
  border: 1px solid rgba(56,189,248,.25); border-radius: 14px;
  padding: 1.2rem; background: rgba(8,17,31,.65);
  box-shadow: inset 0 0 25px rgba(56,189,248,.05); margin: .5rem 0 1rem;
}
.kv-vram-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: .8rem; font-size: .88rem; color: #94a3b8;
}
.kv-vram-header strong { color: #38bdf8; }
.kv-stack-grid {
  display: flex; flex-wrap: wrap; gap: 6px; padding: .6rem;
  background: rgba(15,23,42,.6); border-radius: 10px; min-height: 90px;
  align-items: flex-end; overflow: hidden; position: relative;
}
.kv-block {
  width: 32px; height: 48px; border-radius: 6px;
  background: linear-gradient(135deg, #0284c7, #38bdf8);
  border: 1px solid rgba(255,255,255,.2);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  font-size: .65rem; color: #f8fafc; font-family: monospace;
  box-shadow: 0 4px 12px rgba(56,189,248,.25);
  animation: kv-slide-in 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
}
.kv-block.pruned {
  background: linear-gradient(135deg, #be123c, #fb7185);
  animation: kv-shred-evict 0.8s ease-out forwards;
}
.kv-block .kv-label { font-weight: bold; }
.kv-block .kv-sub { font-size: .55rem; opacity: .8; }
@keyframes kv-slide-in {
  0% { transform: translateY(-30px) scale(0.6); opacity: 0; }
  100% { transform: translateY(0) scale(1); opacity: 1; }
}
@keyframes kv-shred-evict {
  0% { transform: scale(1) translateY(0) rotate(0deg); opacity: 1; filter: blur(0px); }
  50% { transform: scale(1.1) translateY(12px) rotate(8deg); opacity: 0.7; filter: blur(2px); }
  100% { transform: scale(0.2) translateY(50px) rotate(45deg); opacity: 0; filter: blur(6px); }
}
.kv-metrics-row {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin-top: .8rem;
  font-size: .8rem; text-align: center;
}
.kv-metric-box {
  background: rgba(15,23,42,.4); padding: .5rem; border-radius: 8px; border: 1px solid rgba(148,163,184,.1);
}
</style>
"""


def build_kv_stack_figure(
    retained_tokens: int,
    pruned_tokens: int,
    context_window: int = 100,
) -> go.Figure:
    """Build a Plotly stacked horizontal VRAM capacity figure."""
    retained = min(retained_tokens, context_window)
    pruned = max(0, pruned_tokens)
    free = max(0, context_window - retained)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Retained KV Tensors",
            y=["VRAM Cache"],
            x=[retained],
            orientation="h",
            marker={"color": "#38bdf8"},
            hovertemplate="Retained: <b>%{x} tokens</b><extra></extra>",
        )
    )
    if pruned > 0:
        fig.add_trace(
            go.Bar(
                name="Pruned / Shredded",
                y=["VRAM Cache"],
                x=[pruned],
                orientation="h",
                marker={"color": "#fb7185"},
                hovertemplate="Pruned: <b>%{x} tokens</b><extra></extra>",
            )
        )
    if free > 0:
        fig.add_trace(
            go.Bar(
                name="Available Window",
                y=["VRAM Cache"],
                x=[free],
                orientation="h",
                marker={"color": "rgba(51,65,85,0.4)"},
                hovertemplate="Free Space: <b>%{x} tokens</b><extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        height=140,
        margin={"l": 10, "r": 10, "t": 25, "b": 25},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,17,31,0.5)",
        font={"color": "#cbd5e1"},
        xaxis={"title": "Token Capacity", "gridcolor": "rgba(148,163,184,0.1)"},
        yaxis={"visible": False},
        legend={"orientation": "h", "y": 1.2, "x": 0},
    )
    return fig


def render_kv_cache_animation_ui(key_prefix: str = "") -> None:
    """Render the interactive KV cache memory stack & pruning transition UI."""
    prefix = f"{key_prefix}_" if key_prefix else ""
    st.markdown(_KV_CACHE_ANIM_CSS, unsafe_allow_html=True)
    st.markdown("#### 🧠 KV Cache & Memory Pruning Dynamics")
    st.caption(
        "Observe how new KV tensors slide into VRAM during generation. "
        "When context limit is exceeded, historical tokens are shredded/evicted."
    )

    col1, col2 = st.columns(2)
    with col1:
        max_context = st.slider("Context Window (Max Tokens)", 10, 50, 20, 5, key=f"{prefix}anim_kv_max_context")
    with col2:
        token_count = st.slider("Generated Stream Count", 5, 40, 24, 1, key=f"{prefix}anim_kv_token_count")

    retained_count = min(token_count, max_context)
    pruned_count = max(0, token_count - max_context)

    # Render CSS Animated VRAM Block Stack
    blocks_html = []
    # Show active retained tokens sliding in
    for i in range(1, retained_count + 1):
        delay = (i % 5) * 0.05
        blocks_html.append(
            f"<div class='kv-block' style='animation-delay:{delay:.2f}s'>"
            f"<span class='kv-label'>K{i}</span>"
            f"<span class='kv-sub'>V{i}</span>"
            "</div>"
        )
    # Show pruned tokens shredding out
    for i in range(1, pruned_count + 1):
        delay = (i % 5) * 0.08
        blocks_html.append(
            f"<div class='kv-block pruned' style='animation-delay:{delay:.2f}s'>"
            f"<span class='kv-label'>P{i}</span>"
            f"<span class='kv-sub'>EVICT</span>"
            "</div>"
        )

    grid_content = "".join(blocks_html)
    vram_status = (
        f"<span>Active Tokens: <strong>{retained_count}/{max_context}</strong></span>"
        f"<span>Pruned: <strong style='color:#fb7185;'>{pruned_count}</strong></span>"
    )

    st.markdown(
        f"""
        <div class="kv-vram-container">
          <div class="kv-vram-header">
            <span><strong>GPU VRAM KV Stack Visualizer</strong></span>
            {vram_status}
          </div>
          <div class="kv-stack-grid">
            {grid_content}
          </div>
          <div class="kv-metrics-row">
            <div class="kv-metric-box">
              <span style="color:#94a3b8; display:block;">Total Streamed</span>
              <strong>{token_count} Tokens</strong>
            </div>
            <div class="kv-metric-box">
              <span style="color:#94a3b8; display:block;">VRAM Retained</span>
              <strong style="color:#38bdf8;">{retained_count} Tokens</strong>
            </div>
            <div class="kv-metric-box">
              <span style="color:#94a3b8; display:block;">Pruned Evictions</span>
              <strong style="color:#fb7185;">{pruned_count} Tokens</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Plotly Stack gauge chart below
    fig = build_kv_stack_figure(retained_count, pruned_count, max_context)
    st.plotly_chart(fig, width="stretch", key=f"{prefix}anim_kv_plotly_chart")


__all__ = ["build_kv_stack_figure", "render_kv_cache_animation_ui"]

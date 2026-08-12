"""Interactive 3D Vector Search Pulse Animation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.ui.vector_3d import (
    MemoryVector,
    VectorProjection,
    deterministic_demo_embeddings,
    project_vector_space,
)

_SAMPLE_DOCS = [
    ("sandbox", "Sandbox Rules", "Generated code stays strictly offline without external network."),
    ("artifact", "Artifact Format", "Prints formatted output and emits structured markdown/JSON."),
    ("healing", "Self-Healing Memory", "Preserves error tracebacks and compacts past failures."),
    ("coder", "Coder Schema", "Returns Python code, timeout bounds, and executive summary."),
    ("safety", "Safety Guardrails", "Prevents process spawning, file overwrites, or socket connections."),
]


def _build_default_projection() -> VectorProjection:
    texts = [doc[2] for doc in _SAMPLE_DOCS]
    query_text = "offline python code execution without network"
    embeddings = deterministic_demo_embeddings([*texts, query_text])
    memory = [
        MemoryVector(id=doc[0], label=doc[1], text=doc[2], embedding=embeddings[idx])
        for idx, doc in enumerate(_SAMPLE_DOCS)
    ]
    return project_vector_space(memory, embeddings[-1], query_label="Query Node", query_text=query_text, top_k=3)


def build_vector_pulse_figure(
    projection: VectorProjection | None = None,
    frame_count: int = 15,
) -> go.Figure:
    """Build a 3D Plotly animation with a radiating pulse wave expanding from Query to Memory nodes."""
    proj = projection or _build_default_projection()
    q_x, q_y, q_z = proj.query_xyz
    mem_xyz = np.array(proj.memory_xyz)

    # Distances from query
    distances = np.linalg.norm(mem_xyz - np.array([q_x, q_y, q_z]), axis=1)
    max_dist = float(np.max(distances)) if len(distances) > 0 else 1.0
    neighbor_ids = {n.id for n in proj.neighbors}

    frames: list[go.Frame] = []
    r_steps = np.linspace(0.05, max_dist * 1.15, frame_count)

    # Pre-generate sphere wireframe template centered at query
    phi = np.linspace(0, np.pi, 12)
    theta = np.linspace(0, 2 * np.pi, 24)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    for step_idx, radius in enumerate(r_steps):
        # Pulse sphere coordinates
        s_x = q_x + radius * np.sin(phi_grid) * np.cos(theta_grid)
        s_y = q_y + radius * np.sin(phi_grid) * np.sin(theta_grid)
        s_z = q_z + radius * np.cos(phi_grid)

        # Highlight memory nodes reached by pulse
        colors = []
        sizes = []
        for i, item in enumerate(proj.memory):
            dist = distances[i]
            is_neighbor = item.id in neighbor_ids
            if radius >= dist:
                if is_neighbor:
                    colors.append("#a78bfa")  # Bright purple for Top-K hit
                    sizes.append(15)
                else:
                    colors.append("#38bdf8")  # Reached blue
                    sizes.append(11)
            else:
                colors.append("#334155")  # Dark gray unreached
                sizes.append(7)

        # Traces for frame:
        # 0: Memory nodes
        # 1: Pulse Wave Sphere Wireframe
        # 2: Query Node
        # 3+: Active connection lines reached so far
        frame_traces = []
        frame_traces.append(
            go.Scatter3d(
                x=mem_xyz[:, 0],
                y=mem_xyz[:, 1],
                z=mem_xyz[:, 2],
                mode="markers+text",
                text=[item.label for item in proj.memory],
                textposition="top center",
                marker={"size": sizes, "color": colors, "opacity": 0.9},
                customdata=[
                    [item.label, proj.memory_similarities[i], item.text[:100]]
                    for i, item in enumerate(proj.memory)
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Cosine Sim: %{customdata[1]:.3f}<br>"
                    "Text: %{customdata[2]}..."
                    "<extra></extra>"
                ),
                name="Memory Nodes",
            )
        )
        # Pulse shell
        frame_traces.append(
            go.Scatter3d(
                x=s_x.flatten(),
                y=s_y.flatten(),
                z=s_z.flatten(),
                mode="markers",
                marker={"size": 3, "color": "#38bdf8", "opacity": max(0.05, 0.4 - 0.25 * (radius / max_dist))},
                hoverinfo="skip",
                name="Pulse Front",
            )
        )
        # Query Node
        frame_traces.append(
            go.Scatter3d(
                x=[q_x],
                y=[q_y],
                z=[q_z],
                mode="markers+text",
                text=[proj.query_label],
                textposition="top center",
                marker={"size": 14, "color": "#f8fafc", "line": {"color": "#38bdf8", "width": 6}},
                name="Query Pulse Node",
            )
        )
        # Active hit connection lines
        for neighbor in proj.neighbors:
            target_idx = next(i for i, item in enumerate(proj.memory) if item.id == neighbor.id)
            target_xyz = mem_xyz[target_idx]
            if radius >= distances[target_idx]:
                frame_traces.append(
                    go.Scatter3d(
                        x=[q_x, target_xyz[0]],
                        y=[q_y, target_xyz[1]],
                        z=[q_z, target_xyz[2]],
                        mode="lines",
                        line={"color": "#a78bfa", "width": 6},
                        name=f"Top-K Link: {neighbor.label}",
                        hoverinfo="skip",
                    )
                )

        frames.append(go.Frame(data=frame_traces, name=f"pulse_{step_idx}"))

    # Base Figure initialization with Frame 0
    fig = go.Figure(data=frames[0].data, frames=frames)

    slider_steps = [
        {
            "args": [
                [f"pulse_{idx}"],
                {
                    "frame": {"duration": 200, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 100},
                },
            ],
            "label": f"{idx + 1}",
            "method": "animate",
        }
        for idx in range(frame_count)
    ]

    fig.update_layout(
        height=500,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        scene={
            "bgcolor": "rgba(8,17,31,0.5)",
            "xaxis": {"showgrid": False, "showticklabels": False, "title": "X"},
            "yaxis": {"showgrid": False, "showticklabels": False, "title": "Y"},
            "zaxis": {"showgrid": False, "showticklabels": False, "title": "Z"},
        },
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 250, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 150},
                            },
                        ],
                        "label": "▶ Fire Radiating Pulse Wave",
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
                "y": 1.12,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Pulse Wave Step ", "visible": True},
                "pad": {"b": 10, "t": 20},
                "steps": slider_steps,
                "x": 0.0,
                "len": 1.0,
            }
        ],
    )
    return fig


def render_vector_pulse_ui(key_prefix: str = "") -> None:
    """Render the interactive 3D Vector Search Pulse animation UI."""
    prefix = f"{key_prefix}_" if key_prefix else ""
    st.markdown("#### 🌌 Radiating 3D Vector Search Pulse Wave")
    st.caption(
        "Click ▶ to trigger a radiating search pulse wave expanding outward from the Query node. "
        "When the pulse wave collides with nodes, Top-K memories illuminate and lock connections."
    )
    fig = build_vector_pulse_figure()
    st.plotly_chart(fig, width="stretch", key=f"{prefix}anim_vector_pulse_chart")


__all__ = ["build_vector_pulse_figure", "render_vector_pulse_ui"]

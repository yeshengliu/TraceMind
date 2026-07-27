"""3D embedding retrieval and KV-cache pressure visualizations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, Field

VECTOR_PLOT_EXPLANATION = (
    "Maps the active query and stored memory chunks into a shared 3D projection. "
    "Connections identify the top-K memories by cosine similarity."
)
KV_CACHE_EXPLANATION = (
    "Estimated GPU VRAM consumed by Key-Value tensors stored across Transformer "
    "attention layers. Prevents recomputing past prompt tokens during multi-turn generation."
)


class MemoryVector(BaseModel):
    """A labeled memory document and its full-dimensional embedding."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    text: str
    embedding: list[float] = Field(min_length=1)


class NeighborMatch(BaseModel):
    """A cosine-similar memory retrieved for the active query."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    similarity: float


class VectorProjection(BaseModel):
    """PCA coordinates and retrieval edges used by the Plotly scene."""

    model_config = ConfigDict(extra="forbid")

    memory: list[MemoryVector]
    memory_xyz: list[tuple[float, float, float]]
    memory_similarities: list[float]
    query_label: str
    query_text: str = ""
    query_xyz: tuple[float, float, float]
    neighbors: list[NeighborMatch]


class KVCacheConfig(BaseModel):
    """Transparent approximation inputs for a transformer KV cache."""

    model_config = ConfigDict(extra="forbid")

    num_layers: int = Field(default=28, ge=1)
    num_kv_heads: int = Field(default=4, ge=1)
    head_dim: int = Field(default=128, ge=1)
    bytes_per_element: int = Field(default=2, ge=1, le=8)
    context_window: int = Field(default=32_768, ge=128)
    vram_budget_gb: float = Field(default=8.0, gt=0)


class KVCacheEstimate(BaseModel):
    """Estimated cache allocation and optional context-pruning event."""

    model_config = ConfigDict(extra="forbid")

    tokens_before_pruning: int
    retained_tokens: int
    pruned_tokens: int
    bytes_per_token: int
    estimated_bytes: int
    estimated_gb: float
    budget_gb: float
    utilization: float = Field(ge=0.0)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def project_vector_space(
    memory: Sequence[MemoryVector],
    query_embedding: Sequence[float],
    *,
    query_label: str = "Active query",
    query_text: str = "",
    top_k: int = 3,
) -> VectorProjection:
    """Project embeddings to 3D with PCA and select cosine nearest neighbors."""
    if not memory:
        raise ValueError("memory must contain at least one vector")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    dimensions = len(memory[0].embedding)
    if len(query_embedding) != dimensions:
        raise ValueError("query and memory embeddings must have equal dimensions")
    if any(len(item.embedding) != dimensions for item in memory):
        raise ValueError("all memory embeddings must have equal dimensions")

    memory_matrix = np.asarray([item.embedding for item in memory], dtype=float)
    query = np.asarray(query_embedding, dtype=float)
    matrix = np.vstack([memory_matrix, query])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[: min(3, axes.shape[0])].T
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])))

    similarities = [
        _cosine(query, memory_matrix[index]) for index, _item in enumerate(memory)
    ]
    ranked = sorted(
        (
            NeighborMatch(
                id=item.id,
                label=item.label,
                similarity=similarities[index],
            )
            for index, item in enumerate(memory)
        ),
        key=lambda match: match.similarity,
        reverse=True,
    )[: min(top_k, len(memory))]
    return VectorProjection(
        memory=list(memory),
        memory_xyz=[tuple(map(float, row)) for row in projected[:-1]],
        memory_similarities=similarities,
        query_label=query_label,
        query_text=query_text,
        query_xyz=tuple(map(float, projected[-1])),
        neighbors=ranked,
    )


def build_embedding_figure(projection: VectorProjection) -> go.Figure:
    """Render memory nodes, a luminous query, pulse shells, and retrieval edges."""
    xyz = np.asarray(projection.memory_xyz)
    neighbor_ids = {neighbor.id for neighbor in projection.neighbors}
    colors = [
        "#a78bfa" if item.id in neighbor_ids else "#334155"
        for item in projection.memory
    ]
    sizes = [12 if item.id in neighbor_ids else 7 for item in projection.memory]
    memory_customdata = [
        ["Memory Item", similarity, item.text[:180]]
        for item, similarity in zip(
            projection.memory,
            projection.memory_similarities,
            strict=True,
        )
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers+text",
            text=[item.label for item in projection.memory],
            textposition="top center",
            marker={"size": sizes, "color": colors, "opacity": 0.85},
            customdata=memory_customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Node type: %{customdata[0]}<br>"
                "Cosine similarity: %{customdata[1]:.3f}<br>"
                "Chunk preview: %{customdata[2]}"
                "<extra></extra>"
            ),
            name="Local memory",
        )
    )
    query_x, query_y, query_z = projection.query_xyz
    for size, opacity in ((42, 0.08), (29, 0.14), (17, 0.25)):
        figure.add_trace(
            go.Scatter3d(
                x=[query_x],
                y=[query_y],
                z=[query_z],
                mode="markers",
                marker={"size": size, "color": "#38bdf8", "opacity": opacity},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter3d(
            x=[query_x],
            y=[query_y],
            z=[query_z],
            mode="markers+text",
            text=[projection.query_label],
            textposition="top center",
            marker={"size": 13, "color": "#f8fafc", "line": {"color": "#38bdf8", "width": 5}},
            name="Query pulse",
            customdata=[
                [
                    "Query",
                    1.0,
                    projection.query_text[:180] or projection.query_label,
                ]
            ],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Node type: %{customdata[0]}<br>"
                "Cosine similarity: %{customdata[1]:.3f}<br>"
                "Chunk preview: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )
    coordinates = {
        item.id: projection.memory_xyz[index]
        for index, item in enumerate(projection.memory)
    }
    for neighbor in projection.neighbors:
        target = coordinates[neighbor.id]
        figure.add_trace(
            go.Scatter3d(
                x=[query_x, target[0]],
                y=[query_y, target[1]],
                z=[query_z, target[2]],
                mode="lines",
                line={
                    "color": "#38bdf8",
                    "width": max(2, 8 * max(0.0, neighbor.similarity)),
                },
                name=f"{neighbor.label} · {neighbor.similarity:.3f}",
                customdata=[
                    [neighbor.label, neighbor.similarity],
                    [neighbor.label, neighbor.similarity],
                ],
                hovertemplate=(
                    "Connected memory: %{customdata[0]}<br>"
                    "Cosine similarity: %{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        height=540,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        scene={
            "bgcolor": "rgba(8,17,31,.5)",
            "xaxis": {"showgrid": False, "showticklabels": False, "title": "PC1"},
            "yaxis": {"showgrid": False, "showticklabels": False, "title": "PC2"},
            "zaxis": {"showgrid": False, "showticklabels": False, "title": "PC3"},
        },
        legend={"orientation": "h"},
    )
    return figure


def estimate_kv_cache(
    token_count: int,
    config: KVCacheConfig | None = None,
) -> KVCacheEstimate:
    """Estimate decoder KV memory using ``2*L*Hkv*D*tokens*dtype``."""
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    settings = config or KVCacheConfig()
    retained = min(token_count, settings.context_window)
    bytes_per_token = (
        2
        * settings.num_layers
        * settings.num_kv_heads
        * settings.head_dim
        * settings.bytes_per_element
    )
    estimated_bytes = retained * bytes_per_token
    budget_bytes = settings.vram_budget_gb * (1024**3)
    return KVCacheEstimate(
        tokens_before_pruning=token_count,
        retained_tokens=retained,
        pruned_tokens=max(0, token_count - retained),
        bytes_per_token=bytes_per_token,
        estimated_bytes=estimated_bytes,
        estimated_gb=estimated_bytes / (1024**3),
        budget_gb=settings.vram_budget_gb,
        utilization=estimated_bytes / budget_bytes,
    )


def build_kv_cache_figure(estimate: KVCacheEstimate) -> go.Figure:
    """Build a gauge for retained KV-cache VRAM pressure."""
    maximum = max(estimate.budget_gb, estimate.estimated_gb, 0.001)
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=estimate.estimated_gb,
            number={"suffix": " GiB", "valueformat": ".3f"},
            delta={"reference": estimate.budget_gb, "relative": False},
            title={
                "text": (
                    f"Estimated KV cache · {estimate.retained_tokens:,} retained tokens"
                )
            },
            gauge={
                "axis": {"range": [0, maximum], "tickformat": ".2f"},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [0, maximum * 0.6], "color": "rgba(16,185,129,.18)"},
                    {
                        "range": [maximum * 0.6, maximum * 0.85],
                        "color": "rgba(245,158,11,.22)",
                    },
                    {
                        "range": [maximum * 0.85, maximum],
                        "color": "rgba(244,63,94,.24)",
                    },
                ],
                "threshold": {
                    "line": {"color": "#fb7185", "width": 4},
                    "value": min(estimate.budget_gb, maximum),
                },
            },
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0.5],
            y=[0.5],
            mode="markers",
            marker={"size": 130, "color": "rgba(255,255,255,0.01)"},
            customdata=[[estimate.estimated_gb, estimate.budget_gb]],
            hovertemplate=(
                f"{KV_CACHE_EXPLANATION}<br><br>"
                "Estimated use: %{customdata[0]:.3f} GiB<br>"
                "Configured budget: %{customdata[1]:.2f} GiB"
                "<extra></extra>"
            ),
            showlegend=False,
            name="KV cache explanation",
        )
    )
    figure.update_layout(
        height=290,
        margin={"l": 30, "r": 30, "t": 65, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1]},
        hovermode="closest",
    )
    return figure


def pruning_animation_html(estimate: KVCacheEstimate) -> str:
    """Return a CSS fading/shredding animation for evicted historical tokens."""
    if estimate.pruned_tokens <= 0:
        return (
            "<div class='kv-stable'>Context retained · no historical KV blocks "
            "were pruned.</div>"
        )
    blocks = "".join(
        f"<span style='animation-delay:{index * 55}ms'></span>" for index in range(18)
    )
    return (
        "<div class='kv-prune' aria-label='Context pruning animation'>"
        f"<strong>{estimate.pruned_tokens:,} historical tokens evicted</strong>"
        f"<div class='kv-shred'>{blocks}</div></div>"
    )


_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def deterministic_demo_embeddings(
    texts: Sequence[str],
    *,
    dimensions: int = 48,
) -> list[list[float]]:
    """Create explicitly synthetic local vectors for offline UI demonstration."""
    if dimensions < 3:
        raise ValueError("dimensions must be at least 3")
    output: list[list[float]] = []
    for text in texts:
        vector = np.zeros(dimensions, dtype=float)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        output.append((vector / norm if norm else vector).tolist())
    return output


__all__ = [
    "KV_CACHE_EXPLANATION",
    "KVCacheConfig",
    "KVCacheEstimate",
    "MemoryVector",
    "NeighborMatch",
    "VectorProjection",
    "VECTOR_PLOT_EXPLANATION",
    "build_embedding_figure",
    "build_kv_cache_figure",
    "deterministic_demo_embeddings",
    "estimate_kv_cache",
    "project_vector_space",
    "pruning_animation_html",
]

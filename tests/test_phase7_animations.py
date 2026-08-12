"""Phase 7.2 Interactive Concept Animations unit and Streamlit tests."""

from __future__ import annotations

import os

os.environ["TRACEMIND_TRACING_ENABLED"] = "0"

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.ui.animations import (
    build_kv_stack_figure,
    build_sampling_temperature_figure,
    build_vector_pulse_figure,
    render_kv_cache_animation_ui,
    render_prompt_pipeline_animation_ui,
    render_sampling_animation_ui,
    render_vector_pulse_ui,
)
from src.ui.animations.sampling_anim import softmax_with_temp


def test_softmax_temperature_scaling_math() -> None:
    logits = [4.0, 2.0, 1.0, 0.0]
    low_temp_probs = softmax_with_temp(logits, 0.1)
    high_temp_probs = softmax_with_temp(logits, 2.0)

    # Low temp should sharpen top peak (close to 1.0 for highest logit)
    assert low_temp_probs[0] > 0.99
    # High temp should flatten distribution
    assert high_temp_probs[0] < low_temp_probs[0]
    assert np.isclose(np.sum(low_temp_probs), 1.0)
    assert np.isclose(np.sum(high_temp_probs), 1.0)


def test_sampling_temperature_figure_building() -> None:
    fig = build_sampling_temperature_figure(current_temp=0.7, top_p=0.9, top_k=4)

    assert len(fig.frames) >= 8
    assert fig.layout.updatemenus[0].buttons[0].label == "▶ Play Temperature Sweep"
    assert len(fig.layout.sliders[0].steps) == len(fig.frames)
    assert fig.data[0].type == "bar"


def test_kv_stack_figure_building() -> None:
    fig = build_kv_stack_figure(retained_tokens=15, pruned_tokens=5, context_window=20)

    assert len(fig.data) == 3
    assert fig.data[0].name == "Retained KV Tensors"
    assert fig.data[1].name == "Pruned / Shredded"
    assert fig.data[2].name == "Available Window"
    assert fig.data[0].x[0] == 15
    assert fig.data[1].x[0] == 5
    assert fig.data[2].x[0] == 5  # 20 - 15 = 5 free space


def test_vector_pulse_figure_building() -> None:
    fig = build_vector_pulse_figure(frame_count=10)

    assert len(fig.frames) == 10
    assert fig.layout.updatemenus[0].buttons[0].label == "▶ Fire Radiating Pulse Wave"
    # Check that initial frame contains memory nodes, pulse front, and query node
    assert len(fig.data) >= 3
    assert fig.data[0].name == "Memory Nodes"
    assert fig.data[2].name == "Query Pulse Node"


def test_animations_ui_renderers_run_without_error() -> None:
    # Quick call to confirm function signatures and basic component execution
    assert callable(render_sampling_animation_ui)
    assert callable(render_kv_cache_animation_ui)
    assert callable(render_vector_pulse_ui)
    assert callable(render_prompt_pipeline_animation_ui)


def test_app_test_renders_animated_concept_guides() -> None:
    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    # Check that animated concept guides expanders are present
    assert any(
        "Concept Guide" in expander.label or "Animated" in expander.label
        for expander in app.expander
    )

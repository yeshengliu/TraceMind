"""Interactive Prompt Assembly Pipeline Animation."""

from __future__ import annotations

import streamlit as st

_PROMPT_PIPELINE_CSS = """
<style>
.pipeline-container {
  border: 1px solid rgba(56,189,248,.25); border-radius: 16px;
  padding: 1.4rem; background: rgba(8,17,31,.7);
  margin: .5rem 0 1rem; position: relative; overflow: hidden;
}
.pipeline-title {
  font-size: .95rem; font-weight: bold; color: #e2e8f0; margin-bottom: 1rem;
  display: flex; align-items: center; gap: .5rem;
}
.pipeline-assembly-line {
  display: grid; grid-template-columns: repeat(4, 1fr) 40px 1fr; gap: .8rem;
  align-items: center; position: relative;
}
@media (max-width: 800px) {
  .pipeline-assembly-line { grid-template-columns: 1fr; }
}
.pipeline-stage {
  background: rgba(15,23,42,.7); border: 1px solid rgba(148,163,184,.2);
  border-radius: 12px; padding: .85rem; text-align: center;
  position: relative; transition: all .3s ease;
}
.pipeline-stage:hover {
  border-color: #38bdf8; transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(56,189,248,.2);
}
.pipeline-stage .stage-icon { font-size: 1.5rem; margin-bottom: .4rem; }
.pipeline-stage .stage-name { font-size: .8rem; font-weight: bold; color: #cbd5e1; }
.pipeline-stage .stage-desc { font-size: .68rem; color: #94a3b8; margin-top: .2rem; }

.pipeline-arrow {
  font-size: 1.4rem; color: #38bdf8; text-align: center;
  animation: pulse-arrow 1.5s infinite ease-in-out;
}
@keyframes pulse-arrow {
  0%, 100% { opacity: 0.4; transform: translateX(0); }
  50% { opacity: 1; transform: translateX(5px); }
}

.pipeline-gpu-target {
  background: linear-gradient(135deg, rgba(56,189,248,.18), rgba(167,139,250,.22));
  border: 1.5px solid #a78bfa; border-radius: 14px; padding: 1rem; text-align: center;
  box-shadow: 0 0 25px rgba(167,139,250,.25);
  animation: glow-gpu 2.5s infinite alternate ease-in-out;
}
@keyframes glow-gpu {
  0% { box-shadow: 0 0 15px rgba(56,189,248,.2); }
  100% { box-shadow: 0 0 30px rgba(167,139,250,.4); }
}
.pipeline-particle-track {
  width: 100%; height: 4px; background: rgba(51,65,85,.5); margin-top: 1rem;
  border-radius: 2px; position: relative; overflow: hidden;
}
.pipeline-particle {
  width: 25%; height: 100%;
  background: linear-gradient(90deg, transparent, #38bdf8, #a78bfa, transparent);
  position: absolute; animation: particle-flow 2s linear infinite;
}
@keyframes particle-flow {
  0% { left: -25%; }
  100% { left: 100%; }
}
</style>
"""


def render_prompt_pipeline_animation_ui() -> None:
    """Render interactive Prompt Assembly Pipeline visual assembly line."""
    st.markdown(_PROMPT_PIPELINE_CSS, unsafe_allow_html=True)
    st.markdown("#### ⚙️ Prompt Assembly Pipeline Animation")
    st.caption(
        "Trace how raw user inputs, system constraints, JSON schemas, and vector memory chunks "
        "are combined into the final GPU Metaprompt."
    )

    # Try optional streamlit_lottie if installed, else fallback to custom SVG/CSS assembly line
    lottie_loaded = False
    try:
        from streamlit_lottie import st_lottie
        # Lottie placeholder or sample animation if user provides standard json
        # Since offline/builtin environment might not have remote Lottie URL, we fall back to CSS assembly line smoothly
    except ImportError:
        pass

    st.markdown(
        """
        <div class="pipeline-container">
          <div class="pipeline-title">
            <span>🏭 Metaprompt Manufacturing Assembly Line</span>
          </div>
          <div class="pipeline-assembly-line">
            <div class="pipeline-stage">
              <div class="stage-icon">💬</div>
              <div class="stage-name">1. User Query</div>
              <div class="stage-desc">Raw user prompt request</div>
            </div>
            <div class="pipeline-stage">
              <div class="stage-icon">🛡️</div>
              <div class="stage-name">2. System Guardrails</div>
              <div class="stage-desc">Sandbox & system rules</div>
            </div>
            <div class="pipeline-stage">
              <div class="stage-icon">📋</div>
              <div class="stage-name">3. Pydantic Schema</div>
              <div class="stage-desc">Structured output rules</div>
            </div>
            <div class="pipeline-stage">
              <div class="stage-icon">🌌</div>
              <div class="stage-name">4. Memory Context</div>
              <div class="stage-desc">Retrieved vector chunks</div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-gpu-target">
              <div style="font-size: 1.6rem; margin-bottom: .2rem;">⚡</div>
              <div style="font-weight: bold; color: #f8fafc; font-size: .85rem;">Final GPU Metaprompt</div>
              <div style="font-size: .68rem; color: #a78bfa; margin-top: .2rem;">Ready for LLM Inference</div>
            </div>
          </div>
          <div class="pipeline-particle-track">
            <div class="pipeline-particle"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Inspect Metaprompt Composition Breakdown", expanded=False):
        st.markdown(
            """
            - **User Query**: The initial intent or question supplied by the user.
            - **System Guardrails**: System instructions enforcing security policies (e.g. offline execution, no subprocesses).
            - **Pydantic Schemas**: Structural requirements ensuring output adheres to JSON contracts.
            - **Memory Context**: Top-K retrieved documents or previous conversation turns injected for context.
            - **Final GPU Metaprompt**: The role-tagged string compiled and sent to Ollama's `chat` endpoint.
            """
        )


__all__ = ["render_prompt_pipeline_animation_ui"]

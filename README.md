<div align="center">
  <img src="docs/assets/tracemind-hero.svg" alt="TraceMind — transparent, local, self-healing agents" width="100%">

  <p align="center">
  <img src="docs/hero_demo.gif" alt="TraceMind Self-Healing Agent Demo" width="100%">
  </p>

  <p><sub><strong>Query → Plan → Fail → Heal:</strong> TraceMind plans the sales analysis, surfaces a red Docker <code>KeyError</code>, applies a traceback-guided repair, then turns the node green and renders the profit trend.</sub></p>

  <p><strong>A local-first Python agent that plans visibly, executes in Docker, reads its own tracebacks, and repairs failed code.</strong></p>

  <p>
    <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white"></a>
    <a href="https://www.docker.com/"><img alt="Docker isolated" src="https://img.shields.io/badge/Docker-Isolated-2496ED?logo=docker&logoColor=white"></a>
    <a href="https://github.com/langchain-ai/langgraph"><img alt="LangGraph" src="https://img.shields.io/badge/Orchestration-LangGraph-18A558"></a>
    <a href="https://ollama.com/"><img alt="Ollama local models" src="https://img.shields.io/badge/Models-Ollama_Local-111111?logo=ollama&logoColor=white"></a>
    <a href="https://streamlit.io/"><img alt="Streamlit dashboard" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white"></a>
    <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-8B5CF6"></a>
  </p>
</div>

---

TraceMind turns agent failures into inspectable state transitions. The planner,
coder, sandbox, error detector, and reflector are explicit LangGraph nodes.
Every generated program runs inside an offline, read-only Docker container.
When execution fails, TraceMind parses the real traceback, applies a minimal
validated patch, prunes stale context, and tries again—with a visible retry
limit. The **LLM X-Ray Lab** adds prompt, token-probability, vector-retrieval,
and estimated KV-cache inspection for local Ollama generation.

## Measured benchmark

The committed 18-task fixture benchmark covers missing columns, schema drift,
zero division, index bounds, missing dependencies, malformed syntax, null
handling, parsing errors, type mismatches, and clean baselines.

| Mode | Success | Success rate | Avg. repairs | Avg. raw tokens¹ | Avg. retained tokens¹ | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| Single-pass execution | 6/18 | **33.33%** | 0.00 | 196.32 | 196.32 | 0.00% |
| TraceMind self-healing | 18/18 | **100.00%** | 0.67 | 962.33 | 830.47 | **13.70%** |

> **+66.67 percentage points in measured task completion.** Both modes used the
> same local Docker sandbox. These results describe the committed deterministic
> fixture benchmark; they are not a universal claim about every model or coding
> task.

¹ Token counts are transparent estimates at four characters per token.
[Read the task-level report](docs/benchmark_report.md) or inspect the
[machine-readable results](docs/benchmark_report.json).

## Architecture

```mermaid
flowchart LR
    User["User prompt"] --> UI["Streamlit observatory"]
    UI --> Planner

    subgraph Graph["LangGraph agent state machine"]
        Planner["Planner"] --> Coder["Coder"]
        Coder --> Sandbox["Sandbox executor"]
        Sandbox --> Detector{"Error detector"}
        Detector -->|clean| Success["Success"]
        Detector -->|traceback| Reflector["Reflect + patch"]
        Reflector -->|targeted repair| Sandbox
        Reflector -->|schema recovery| Coder
    end

    Sandbox <--> Docker["Offline Docker sandbox<br/>read-only · non-root · capped"]
    Graph <--> Memory["Working memory<br/>context pruning"]
    Success --> Episodes["Episodic memory<br/>JSONL export"]
    Graph -. OpenInference spans .-> Phoenix["Local Phoenix tracing"]
    Graph -. live state updates .-> UI
    UI --> XRay["LLM X-Ray Lab"]
    XRay --> Logprobs["Token logprobs<br/>Top-5 + confidence"]
    XRay --> Vectors["3D memory retrieval<br/>KV-cache estimate"]
```

## Why TraceMind

- **Transparent by construction** — plans, tool inputs, node transitions,
  sandbox output, tracebacks, patches, retries, and context pressure are
  inspectable. The dashboard presents observable rationale, not private hidden
  chain-of-thought.
- **Self-healing execution** — structured traceback parsing identifies the
  terminal exception and stack frames; exact-text edits are syntax-validated
  before re-execution.
- **Hardened Docker isolation** — generated code has no network, runs as an
  unprivileged user, uses a read-only root filesystem, and receives CPU,
  memory, PID, and wall-clock limits.
- **Local models, no per-token API bill** — Qwen 2.5 planning and coding models
  run through Ollama on your machine. Hardware and electricity costs still
  apply.
- **Bounded memory** — system policy, user intent, the newest traceback, and the
  latest patch survive while stale attempts and verbose logs are compacted.
- **Live observability** — Streamlit animates node activity, state topology,
  estimated token pressure, retries, artifacts, and terminal output. Phoenix
  adds local OpenInference traces.
- **Rigid tool contracts** — Pydantic schemas reject malformed model output and
  unsafe execution parameters before they cross the tool boundary.
- **LLM X-Ray mode** — compare Temperature, Top-P, and Top-K settings against
  the same prompt; replay Top-5 token probabilities; inspect a confidence
  heatmap; diff raw versus injected prompts; explore local embeddings in 3D;
  and watch estimated KV-cache pressure and context pruning.

## LLM X-Ray Lab

Open the **🔬 LLM X-Ray Lab** tab after starting Streamlit. The lab provides:

### Phase 7 in motion

**Concept guide, sampling controls, and inline help**

<p align="center">
  <img src="docs/assets/phase7-xray-guide.gif"
       alt="LLM X-Ray Lab concept guide and sampling parameter help"
       width="88%">
</p>

**Prompt metamorphosis and focus attribution**

<p align="center">
  <img src="docs/assets/phase7-prompt-focus.gif"
       alt="Raw prompt versus injected system prompt, schema, memory, and focus attribution"
       width="88%">
</p>

**Interactive 3D retrieval and KV-cache pressure**

<p align="center">
  <img src="docs/assets/phase7-vector-kv.gif"
       alt="Rotating 3D memory retrieval graph and KV-cache gauge"
       width="88%">
</p>

The vector demo above uses the lab's explicitly labeled deterministic offline
vectors. After a successful A/B run, TraceMind requests local Ollama embeddings
for the same interactive scene.

- a streamed Top-5 candidate histogram from Ollama `logprobs` and
  `top_logprobs`, plus a replay control for every observed token;
- green/amber/red generated-token confidence highlighting based on the
  model-reported selected-token probability;
- a raw-versus-processed prompt diff containing system policy, the Pydantic
  response schema, memory context, and the user prompt;
- an explainable prompt-section alignment view. This is a lexical attribution
  proxy—Ollama does not expose internal transformer attention tensors;
- a PCA-projected Plotly 3D memory scene connected to the Top-K cosine nearest
  neighbors returned by local embeddings;
- a transparent KV-cache estimate using
  `2 × layers × KV heads × head dimension × tokens × dtype bytes`, including
  a pruning animation when the configured context window is exceeded; and
- side-by-side A/B sampling controls for Temperature, Top-P, and Top-K.

The lab never fabricates missing probability metadata. Older Ollama/model
combinations that omit logprobs show a capability notice. If the configured
embedding model is unavailable, the vector panel is explicitly labeled as a
deterministic synthetic demo; install the embedding model with:

```bash
ollama pull nomic-embed-text
```

TraceMind displays observable model output and bounded rationale, not private
hidden chain-of-thought.

## Quickstart

Prerequisites: Python 3.11+, Docker Desktop, and
[Ollama](https://ollama.com/).

### 1. Install and start the isolated services

```bash
git clone https://github.com/yeshengliu/TraceMind.git
cd TraceMind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose pull sandbox
docker compose --profile observability up -d phoenix
```

### 2. Pull the local models

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
```

### 3. Launch the observatory

```bash
streamlit run app.py
```

Open:

- TraceMind Agent Studio and LLM X-Ray Lab: `http://localhost:8501`
- Phoenix traces: `http://localhost:6006`

Phoenix is optional. The dashboard continues to work if the collector is
offline.

## The healing loop

1. **Plan** — `qwen2.5:7b` returns a schema-constrained execution plan.
2. **Code** — `qwen2.5-coder:7b` returns complete Python plus a bounded timeout.
3. **Execute** — the Phase 1 tool runs the program in the locked-down container.
4. **Detect** — non-zero exits and Python tracebacks become structured state.
5. **Reflect** — the coder model diagnoses the root cause and proposes minimal
   exact-text edits.
6. **Validate** — TraceMind verifies edit targets and runs `ast.parse` before
   accepting the patch.
7. **Retry** — the graph re-enters the sandbox, capped at three corrections.
8. **Remember** — successful trajectories enter episodic memory; noisy working
   context is pruned.

## Benchmark pipeline

Run the reproducible exact-oracle benchmark:

```bash
python -m src.eval.benchmark
```

This performs 36 top-level evaluations—18 single-pass and 18 full-engine
runs—plus patched retries for failed full-engine attempts. It rewrites:

- `docs/benchmark_report.json`
- `docs/benchmark_report.md`

Use live local model reflection and the structured LLM judge:

```bash
python -m src.eval.benchmark --reflector ollama --judge ollama
```

The dataset is plain JSONL, so new cases can be reviewed and added without
changing evaluator code:

```json
{
  "id": "missing-dict-column",
  "category": "missing column",
  "prompt": "Sum the amount field.",
  "program": "total = sum(row[\"missing_amount\"] for row in records)",
  "expected_stdout_regex": "^TOTAL=34$",
  "repair": {
    "root_cause": "The records contain amount, not missing_amount.",
    "edits": [{
      "old_text": "row[\"missing_amount\"]",
      "new_text": "row[\"amount\"]",
      "occurrence": 1,
      "reason": "Use the available column."
    }],
    "validation_notes": "The aggregation and output format are preserved."
  }
}
```

Every repair contains at least one reviewable, exact-text edit.

## Dashboard artifact protocol

Sandbox programs can publish rich output by printing marked lines:

```text
TREND_SVG=<svg ...>...</svg>
ARTIFACT_PNG_BASE64=<base64 PNG>
ARTIFACT_MARKDOWN=## Report
ARTIFACT_JSON={"value": 42}
```

SVG and PNG images, Markdown reports, JSON data, and raw terminal logs appear in
the right-hand dashboard pane.

## Record the hero demo

Start Docker Desktop, make sure the sandbox image and local coder model are
available, then launch the auto-running recording scenario:

```bash
docker compose pull sandbox
ollama pull qwen2.5-coder:7b
python scripts/run_demo_scenario.py
```

The fixed plan, data, initial program, and model seed make the capture
repeatable. The first execution raises `KeyError: 'profit_margin'` in the real
Docker sandbox; the traceback is routed to `qwen2.5-coder:7b`, the validated
patch is re-executed, and the right pane renders a dependency-free SVG trend
chart. For an offline framing rehearsal, use
`python scripts/run_demo_scenario.py --fixture-reflector`.

In Kap or CleanShot X, select only the browser window, record at **1920×1080 and
30 fps**, hide the cursor, and stop just after the green chart appears
(approximately 10–15 seconds). Export the source recording to
`/tmp/tracemind-hero.mov`, then create the README asset:

```bash
brew install ffmpeg gifsicle
mkdir -p docs
ffmpeg -y -i /tmp/tracemind-hero.mov -t 15 \
  -vf "fps=15,scale=1280:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  -loop 0 docs/hero_demo.raw.gif
gifsicle -O3 --lossy=80 --colors 128 \
  -o docs/hero_demo.gif docs/hero_demo.raw.gif
du -h docs/hero_demo.gif
```

If the GIF is still over 5 MB, use 10 fps and 960 px instead:

```bash
ffmpeg -y -i /tmp/tracemind-hero.mov -t 15 \
  -vf "fps=10,scale=960:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4" \
  -loop 0 docs/hero_demo.gif
```

For a sharper, usually much smaller MP4 alternative:

```bash
ffmpeg -y -i /tmp/tracemind-hero.mov -t 15 \
  -vf "fps=30,scale=1920:-2:flags=lanczos" \
  -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p \
  -movflags +faststart -an docs/hero_demo.mp4
```

## Project map

```text
TraceMind/
├── app.py                         # Streamlit entry point
├── scripts/run_demo_scenario.py   # Reproducible self-healing hero demo
├── data/benchmark.jsonl           # 18 reproducible edge cases
├── docs/
│   ├── assets/tracemind-hero.svg  # Animated GitHub hero
│   ├── hero_demo.gif              # Recorded README demo (generated locally)
│   ├── benchmark_report.json      # Machine-readable measurements
│   └── benchmark_report.md        # Human-readable results
├── src/
│   ├── agent/
│   │   ├── graph.py               # LangGraph state machine
│   │   ├── llm.py                 # Ollama model router
│   │   ├── llm_inspector.py       # Prompt + native logprob/embed hooks
│   │   ├── reflection.py          # Traceback parser + patch gate
│   │   └── tools.py               # Pydantic tool contracts
│   ├── eval/benchmark.py          # Dual-mode benchmark runner
│   ├── memory/pruner.py           # Working + episodic memory
│   ├── sandbox/test_sandbox.py    # Docker isolation boundary
│   └── ui/
│       ├── dashboard.py           # Animated dashboard
│       ├── token_vis.py           # Probability, confidence, and prompt views
│       ├── tracing.py             # Phoenix/OpenInference setup
│       ├── vector_3d.py           # 3D retrieval + KV-cache visualization
│       └── xray_tab.py            # Interactive Phase 7 lab
└── tests/                         # Unit, integration, Docker, and UI tests
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TRACEMIND_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama-compatible API |
| `TRACEMIND_PLANNER_MODEL` | `qwen2.5:7b` | Planning and optional judging |
| `TRACEMIND_CODER_MODEL` | `qwen2.5-coder:7b` | Code generation and reflection |
| `TRACEMIND_EMBEDDING_MODEL` | `nomic-embed-text` | X-Ray local memory embeddings |
| `TRACEMIND_OLLAMA_NATIVE_URL` | derived from base URL | Native chat/embed API override |
| `TRACEMIND_LLM_TIMEOUT_SECONDS` | `120` | Local model request timeout |
| `TRACEMIND_PHOENIX_URL` | `http://localhost:6006` | Phoenix UI/collector |
| `TRACEMIND_PHOENIX_PROJECT` | `TraceMind` | Trace project name |
| `TRACEMIND_TRACING_ENABLED` | `1` | Set to `0` to disable telemetry |

## Verification

```bash
pytest -q
```

The suite runs deterministic model doubles, the real Docker sandbox, and live
local Ollama tests. Docker Desktop, `python:3.12-slim`, and both configured Qwen
models must be available; missing integration infrastructure fails verification
instead of silently skipping coverage.

## Security boundary

TraceMind is a research and local-development project, not a guarantee that
arbitrary generated code is harmless. Keep Docker Desktop updated, review
resource limits before deployment, do not mount sensitive host directories into
the sandbox, and do not expose Ollama, Phoenix, or Streamlit directly to an
untrusted network.

## License

[MIT](LICENSE)

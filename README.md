# TraceMind

TraceMind is an offline-first autonomous reasoning agent engine focused on
transparent execution and self-healing.

## Phase 1: Docker sandbox probe

Prerequisites: Docker Desktop and Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose pull sandbox

python src/sandbox/test_sandbox.py --case error
python src/sandbox/test_sandbox.py --case timeout --timeout 1
```

The sandbox runs without network access, as an unprivileged user, with a
read-only root filesystem and explicit CPU, memory, PID, and wall-clock limits.
Each run returns structured JSON, including a formatted Python traceback for
runtime errors.

## Phase 2: Local agent graph

Phase 2 adds Pydantic-enforced planning and code-generation schemas, a
LangGraph state machine, and an explicit correction loop capped at three
retries:

```text
planner -> coder_agent -> sandbox_executor -> error_detector
                  ^                                  |
                  +-------- reflect_and_heal <-------+
```

Install the requested local models and run the live integration:

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
TRACEMIND_RUN_OLLAMA_TESTS=1 pytest -m ollama -q
```

The regular suite uses deterministic structured-model doubles while still
executing the Fibonacci example in the real Docker sandbox:

```bash
pytest -q
```

Configuration can be overridden with `TRACEMIND_OLLAMA_BASE_URL`,
`TRACEMIND_PLANNER_MODEL`, `TRACEMIND_CODER_MODEL`, and
`TRACEMIND_LLM_TIMEOUT_SECONDS`.

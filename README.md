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

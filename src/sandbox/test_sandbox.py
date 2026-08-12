"""Minimal Docker sandbox integration probe.

This module is intentionally executable despite its ``test_`` prefix:

    python src/sandbox/test_sandbox.py --case error
    python src/sandbox/test_sandbox.py --case timeout --timeout 1
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Literal

import docker
from docker.errors import DockerException, ImageNotFound
from pydantic import BaseModel
from requests.exceptions import ReadTimeout, RequestException
from urllib3.exceptions import ReadTimeoutError


DEFAULT_IMAGE = "python:3.12-slim"

_CONTAINER_RUNNER = r"""
import json
import os
import traceback

source = os.environ["TRACEMIND_SNIPPET"]
try:
    exec(compile(source, "<sandbox>", "exec"), {"__name__": "__sandbox__"})
except SystemExit as exc:
    code = 0 if exc.code in (None, 0) else exc.code
    if code == 0:
        print(json.dumps({
            "status": "success",
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }))
    else:
        print(json.dumps({
            "status": "error",
            "error_type": "SystemExit",
            "error_message": str(code),
            "traceback": traceback.format_exc(),
        }))
    raise SystemExit(code)
except BaseException as exc:
    print(json.dumps({
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }))
    raise SystemExit(1)
else:
    print(json.dumps({
        "status": "success",
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }))
"""


class SandboxResult(BaseModel):
    """Stable, serializable result returned by the sandbox boundary."""

    status: Literal["success", "error", "timeout", "infrastructure_error"]
    exit_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    logs: str = ""
    duration_seconds: float


def _is_read_timeout(exc: BaseException) -> bool:
    """Recognize timeouts even when requests wraps urllib3's exception."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()

    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, (TimeoutError, ReadTimeout, ReadTimeoutError)):
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        pending.extend(arg for arg in current.args if isinstance(arg, BaseException))

    return False


def _result_from_logs(logs: str, exit_code: int, duration: float) -> SandboxResult:
    """Extract the runner's final JSON record while retaining all raw logs."""
    for line in reversed(logs.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("status") in {"success", "error"}:
            return SandboxResult(
                **payload,
                exit_code=exit_code,
                logs=logs,
                duration_seconds=duration,
            )

    return SandboxResult(
        status="infrastructure_error",
        exit_code=exit_code,
        error_type="InvalidSandboxOutput",
        error_message="Container exited without a valid result record.",
        logs=logs,
        duration_seconds=duration,
    )


def run_in_sandbox(
    snippet: str,
    *,
    timeout_seconds: float = 2.0,
    image: str = DEFAULT_IMAGE,
) -> SandboxResult:
    """Execute Python in a locked-down container and return a structured result."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    started = time.monotonic()
    container = None

    try:
        client = docker.from_env()
        client.ping()
        try:
            client.images.get(image)
        except ImageNotFound as exc:
            return SandboxResult(
                status="infrastructure_error",
                error_type="ImageNotFound",
                error_message=f"Pull the sandbox image first: docker compose pull sandbox ({exc})",
                duration_seconds=time.monotonic() - started,
            )

        container = client.containers.run(
            image=image,
            command=["python", "-c", _CONTAINER_RUNNER],
            detach=True,
            network_disabled=True,
            read_only=True,
            user="65534:65534",
            working_dir="/tmp",
            environment={"TRACEMIND_SNIPPET": snippet},
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            pids_limit=64,
            mem_limit="256m",
            nano_cpus=500_000_000,
            tmpfs={"/tmp": "rw,noexec,nosuid,nodev,size=64m"},
        )

        try:
            wait_result = container.wait(timeout=timeout_seconds)
        except RequestException as exc:
            if not _is_read_timeout(exc):
                raise
            container.kill()
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            return SandboxResult(
                status="timeout",
                error_type="SandboxTimeout",
                error_message=f"Execution exceeded {timeout_seconds:.2f} seconds.",
                logs=logs,
                duration_seconds=time.monotonic() - started,
            )

        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
        return _result_from_logs(
            logs=logs,
            exit_code=int(wait_result["StatusCode"]),
            duration=time.monotonic() - started,
        )
    except (DockerException, RequestException) as exc:
        return SandboxResult(
            status="infrastructure_error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            duration_seconds=time.monotonic() - started,
        )
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except DockerException:
                pass


def _example(case: str) -> str:
    examples = {
        "success": "print('sandbox execution succeeded')",
        "error": "numerator = 1\ndenominator = 0\nprint(numerator / denominator)",
        "timeout": "import time\ntime.sleep(10)",
    }
    return examples[case]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("success", "error", "timeout"), default="error")
    parser.add_argument("--timeout", type=float, default=2.0, help="Wall-clock limit in seconds")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()

    result = run_in_sandbox(
        _example(args.case),
        timeout_seconds=args.timeout,
        image=args.image,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.status in {"success", "error", "timeout"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded working memory and successful-trajectory episodic memory."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field


def _message_record(message: AnyMessage) -> dict[str, str]:
    return {"role": message.type, "content": str(message.content)}


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "type") and hasattr(value, "content"):
        return _message_record(value)  # type: ignore[arg-type]
    return str(value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... <{omitted} chars pruned>"


class WorkingMemory(BaseModel):
    """Prompt-ready view of the current execution trajectory."""

    model_config = ConfigDict(extra="forbid")

    system_prompts: list[str] = Field(default_factory=list)
    user_intent: str
    current_plan: list[str] = Field(default_factory=list)
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    latest_code: str | None = None
    latest_error: str | None = None
    latest_patch: dict[str, Any] | None = None
    history_summary: list[str] = Field(default_factory=list)


class EpisodicEpisode(BaseModel):
    """One successful, exportable agent trajectory."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    created_at: datetime
    user_intent: str
    plan: list[str]
    retry_count: int
    patches: list[dict[str, Any]]
    artifact_summaries: list[dict[str, Any]]
    final_output: str


class EpisodicMemory:
    """Thread-safe in-memory store for successful trajectories."""

    def __init__(self) -> None:
        self._episodes: list[EpisodicEpisode] = []
        self._lock = Lock()

    @property
    def episodes(self) -> tuple[EpisodicEpisode, ...]:
        with self._lock:
            return tuple(self._episodes)

    def record_success(self, state: dict[str, Any]) -> EpisodicEpisode:
        if state.get("status") != "completed":
            raise ValueError("Only completed trajectories can enter episodic memory")

        messages = state.get("messages", [])
        user_intent = next(
            (str(message.content) for message in messages if isinstance(message, HumanMessage)),
            "",
        )
        artifact_summaries = [
            {
                "attempt": artifact.get("attempt"),
                "status": artifact.get("result", {}).get("status"),
                "error_type": artifact.get("result", {}).get("error_type"),
                "duration_seconds": artifact.get("result", {}).get("duration_seconds"),
            }
            for artifact in state.get("execution_artifacts", [])
        ]
        episode = EpisodicEpisode(
            episode_id=str(uuid4()),
            created_at=datetime.now(UTC),
            user_intent=user_intent,
            plan=list(state.get("current_plan", [])),
            retry_count=int(state.get("retry_count", 0)),
            patches=copy.deepcopy(state.get("patch_history", [])),
            artifact_summaries=artifact_summaries,
            final_output=str(state.get("final_output", "")),
        )
        with self._lock:
            self._episodes.append(episode)
        return episode

    def export_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        with self._lock:
            payload = "\n".join(
                episode.model_dump_json() for episode in self._episodes
            )
        destination.write_text(f"{payload}\n" if payload else "", encoding="utf-8")
        return destination


class ContextPruner:
    """Trim verbose history while retaining the active debugging evidence."""

    def __init__(
        self,
        *,
        max_chars: int = 16_000,
        max_messages: int = 8,
        max_artifacts: int = 2,
        max_summaries: int = 10,
        max_output_chars: int = 1_000,
    ) -> None:
        if max_chars < 1_000:
            raise ValueError("max_chars must be at least 1000")
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        if max_artifacts < 1:
            raise ValueError("max_artifacts must be at least 1")
        self.max_chars = max_chars
        self.max_messages = max_messages
        self.max_artifacts = max_artifacts
        self.max_summaries = max_summaries
        self.max_output_chars = max_output_chars

    def measure_state(self, state: dict[str, Any]) -> int:
        return len(
            json.dumps(
                state,
                default=_json_default,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def prune_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a bounded copy without mutating the caller's state."""
        pruned = copy.deepcopy(state)
        summaries = list(pruned.get("history_summary", []))

        messages: list[AnyMessage] = list(pruned.get("messages", []))
        retained_indexes = {
            index for index, message in enumerate(messages) if isinstance(message, SystemMessage)
        }
        first_user = next(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        if first_user is not None:
            retained_indexes.add(first_user)
        remaining_slots = max(0, self.max_messages - len(retained_indexes))
        candidate_indexes = [
            index for index in range(len(messages)) if index not in retained_indexes
        ]
        if remaining_slots:
            retained_indexes.update(candidate_indexes[-remaining_slots:])
        dropped_messages = [
            message for index, message in enumerate(messages) if index not in retained_indexes
        ]
        if dropped_messages:
            summaries.append(f"Pruned {len(dropped_messages)} intermediate messages.")
        pruned["messages"] = [
            message for index, message in enumerate(messages) if index in retained_indexes
        ]

        errors = list(pruned.get("error_stack", []))
        if len(errors) > 1:
            summaries.append(f"Pruned {len(errors) - 1} superseded error tracebacks.")
            pruned["error_stack"] = errors[-1:]

        artifacts = list(pruned.get("execution_artifacts", []))
        if len(artifacts) > self.max_artifacts:
            for artifact in artifacts[: -self.max_artifacts]:
                result = artifact.get("result", {})
                summaries.append(
                    "Attempt "
                    f"{artifact.get('attempt')}: {result.get('status')}"
                    f"/{result.get('error_type') or 'none'}."
                )
            artifacts = artifacts[-self.max_artifacts :]
        for index, artifact in enumerate(artifacts):
            if index < len(artifacts) - 1:
                request = artifact.get("request", {})
                if "code" in request:
                    request["code"] = "<older failed code pruned>"
            result = artifact.get("result", {})
            result["logs"] = _truncate(str(result.get("logs", "")), self.max_output_chars)
            if result.get("traceback"):
                result["traceback"] = _truncate(
                    str(result["traceback"]), self.max_output_chars
                )
        pruned["execution_artifacts"] = artifacts

        patches = list(pruned.get("patch_history", []))
        if len(patches) > 1:
            summaries.append(f"Pruned {len(patches) - 1} superseded code patches.")
            pruned["patch_history"] = patches[-1:]

        pruned["history_summary"] = summaries[-self.max_summaries :]
        if "final_output" in pruned:
            pruned["final_output"] = _truncate(
                str(pruned["final_output"]), self.max_output_chars
            )

        self._shrink_optional_history(pruned)
        return pruned

    def _shrink_optional_history(self, state: dict[str, Any]) -> None:
        while self.measure_state(state) > self.max_chars:
            summaries = state.get("history_summary", [])
            if summaries:
                summaries.pop(0)
                continue

            messages = state.get("messages", [])
            removable = next(
                (
                    index
                    for index, message in enumerate(messages)
                    if not isinstance(message, (SystemMessage, HumanMessage))
                ),
                None,
            )
            if removable is not None:
                messages.pop(removable)
                continue

            artifacts = state.get("execution_artifacts", [])
            if len(artifacts) > 1:
                artifacts.pop(0)
                continue

            if artifacts:
                result = artifacts[-1].get("result", {})
                logs = str(result.get("logs", ""))
                if len(logs) > 256:
                    result["logs"] = _truncate(logs, 256)
                    continue

            final_output = str(state.get("final_output", ""))
            if len(final_output) > 256:
                state["final_output"] = _truncate(final_output, 256)
                continue
            break

    def working_memory(self, state: dict[str, Any]) -> WorkingMemory:
        bounded = self.prune_state(state)
        messages: list[AnyMessage] = bounded.get("messages", [])
        user_intent = next(
            (str(message.content) for message in messages if isinstance(message, HumanMessage)),
            "",
        )
        system_prompts = [
            str(message.content) for message in messages if isinstance(message, SystemMessage)
        ]
        recent_messages = [
            _message_record(message)
            for message in messages
            if not isinstance(message, (SystemMessage, HumanMessage))
        ]
        patches = bounded.get("patch_history", [])
        return WorkingMemory(
            system_prompts=system_prompts,
            user_intent=user_intent,
            current_plan=list(bounded.get("current_plan", [])),
            recent_messages=recent_messages,
            latest_code=bounded.get("generated_code"),
            latest_error=(
                bounded.get("error_stack", [])[-1]
                if bounded.get("error_stack")
                else None
            ),
            latest_patch=patches[-1] if patches else None,
            history_summary=list(bounded.get("history_summary", [])),
        )

"""Local Ollama model configuration and routing."""

from __future__ import annotations

import os
from enum import StrEnum
from typing import TypeVar

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_PLANNER_MODEL = "qwen2.5:7b"
DEFAULT_CODER_MODEL = "qwen2.5-coder:7b"

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModelRole(StrEnum):
    """Logical model roles used by the TraceMind graph."""

    PLANNER = "planner"
    CODER = "coder"


def model_name_for(role: ModelRole) -> str:
    """Resolve a role to an overridable local Ollama model name."""
    environment_key = {
        ModelRole.PLANNER: "TRACEMIND_PLANNER_MODEL",
        ModelRole.CODER: "TRACEMIND_CODER_MODEL",
    }[role]
    default = {
        ModelRole.PLANNER: DEFAULT_PLANNER_MODEL,
        ModelRole.CODER: DEFAULT_CODER_MODEL,
    }[role]
    return os.getenv(environment_key, default)


def get_llm(role: ModelRole) -> ChatOpenAI:
    """Create a deterministic LangChain client for Ollama's local v1 endpoint."""
    return ChatOpenAI(
        model=model_name_for(role),
        base_url=os.getenv("TRACEMIND_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        api_key=SecretStr(os.getenv("TRACEMIND_OLLAMA_API_KEY", "ollama")),
        temperature=0,
        seed=42,
        timeout=float(os.getenv("TRACEMIND_LLM_TIMEOUT_SECONDS", "120")),
        max_retries=1,
    )


def get_structured_llm(
    role: ModelRole,
    schema: type[SchemaT],
) -> Runnable[object, SchemaT]:
    """Bind a Pydantic response schema to a role-specific Ollama model."""
    return get_llm(role).with_structured_output(schema, method="json_schema")

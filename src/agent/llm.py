"""Local Ollama model configuration and routing."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
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
    """Bind a Pydantic parser using Ollama-compatible JSON generation mode.

    Ollama's OpenAI compatibility layer can reject otherwise valid Pydantic
    JSON Schemas while compiling its token grammar. JSON mode keeps generation
    constrained to an object and still validates the result against ``schema``
    in LangChain.
    """
    schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    instruction = (
        "\n\nOUTPUT CONTRACT (mandatory): Return exactly one JSON object matching "
        "this JSON Schema. Use every required field, use no undeclared fields, "
        "and do not wrap the object in Markdown. If the schema contains a code "
        "field, emit every other required field before code and ensure the code "
        f"string is complete:\n{schema_text}"
    )

    def inject_schema(input_value: object) -> object:
        if not isinstance(input_value, list):
            return input_value
        messages = list(input_value)
        if messages and isinstance(messages[0], SystemMessage):
            first = messages[0]
            messages[0] = SystemMessage(content=f"{first.content}{instruction}")
        else:
            messages.insert(0, SystemMessage(content=instruction.lstrip()))
        return messages

    constrained = get_llm(role).with_structured_output(schema, method="json_mode")

    def invoke_with_repair(input_value: object) -> SchemaT:
        messages = inject_schema(input_value)
        for attempt in range(3):
            try:
                return constrained.invoke(messages)
            except OutputParserException as exc:
                if attempt == 2 or not isinstance(messages, list):
                    raise
                messages = [
                    *messages,
                    HumanMessage(
                        content=(
                            "Your previous JSON failed the mandatory output contract. "
                            "Regenerate the entire object, correcting every validation "
                            "error below. Do not explain the correction.\n\n"
                            f"{str(exc)[:4_000]}"
                        )
                    ),
                ]
        raise RuntimeError("structured output retry loop exited unexpectedly")

    return RunnableLambda(invoke_with_repair)

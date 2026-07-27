"""Observable Ollama generation hooks for TraceMind's LLM X-Ray Lab.

The inspector intentionally records inputs and output-token statistics only.
It does not claim access to hidden chain-of-thought or internal attention
matrices, neither of which Ollama exposes through its public API.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.agent.llm import DEFAULT_OLLAMA_BASE_URL, ModelRole, model_name_for


DEFAULT_NATIVE_OLLAMA_URL = "http://localhost:11434"


class PromptSection(BaseModel):
    """One named part of the fully assembled model input."""

    model_config = ConfigDict(extra="forbid")

    name: str
    content: str
    kind: Literal["system", "schema", "memory", "user"] = "system"


class PromptCapture(BaseModel):
    """Raw user input and the exact, inspectable prompt assembled around it."""

    model_config = ConfigDict(extra="forbid")

    raw_prompt: str
    sections: list[PromptSection]
    messages: list[dict[str, str]]
    processed_prompt: str


class GenerationSettings(BaseModel):
    """Sampling controls accepted by Ollama's native chat endpoint."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=200)
    seed: int = 42
    max_tokens: int = Field(default=512, ge=1, le=8_192)


class TokenCandidate(BaseModel):
    """A candidate token and its model-reported softmax probability."""

    model_config = ConfigDict(extra="forbid")

    token: str
    logprob: float
    probability: float = Field(ge=0.0, le=1.0)


class TokenStep(BaseModel):
    """The sampled token and up to five alternatives at one generation step."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    token: str
    logprob: float
    probability: float = Field(ge=0.0, le=1.0)
    candidates: list[TokenCandidate]


class GenerationTrace(BaseModel):
    """Complete inspectable result returned by one local generation."""

    model_config = ConfigDict(extra="forbid")

    model: str
    content: str
    prompt: PromptCapture
    settings: GenerationSettings
    token_steps: list[TokenStep] = Field(default_factory=list)
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_duration_ns: int = 0
    logprobs_available: bool = False
    notice: str | None = None


TokenCallback = Callable[[TokenStep, str], None]


def _native_base_url(openai_base_url: str) -> str:
    """Convert TraceMind's optional ``.../v1`` URL to Ollama's server root."""
    explicit = os.getenv("TRACEMIND_OLLAMA_NATIVE_URL")
    if explicit:
        return explicit.rstrip("/")
    base = openai_base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _schema_payload(schema: object | None) -> object | None:
    if schema is None:
        return None
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    if isinstance(schema, BaseModel):
        return type(schema).model_json_schema()
    if isinstance(schema, str):
        try:
            return json.loads(schema)
        except json.JSONDecodeError:
            return schema
    return schema


def capture_prompt(
    raw_prompt: str,
    *,
    system_prompt: str = "",
    schema: object | None = None,
    memory_context: object | None = None,
    extra_system_sections: Mapping[str, str] | None = None,
) -> PromptCapture:
    """Assemble and retain the exact messages sent to the local model."""
    if not raw_prompt.strip():
        raise ValueError("raw_prompt must not be empty")

    sections: list[PromptSection] = []
    if system_prompt:
        sections.append(
            PromptSection(name="System metaprompt", content=system_prompt, kind="system")
        )
    for name, content in (extra_system_sections or {}).items():
        if content:
            sections.append(PromptSection(name=name, content=content, kind="system"))

    schema_payload = _schema_payload(schema)
    if schema_payload is not None:
        sections.append(
            PromptSection(
                name="Pydantic JSON schema",
                content=_json_text(schema_payload),
                kind="schema",
            )
        )
    if memory_context is not None:
        sections.append(
            PromptSection(
                name="Memory context",
                content=_json_text(memory_context),
                kind="memory",
            )
        )
    sections.append(PromptSection(name="Raw user prompt", content=raw_prompt, kind="user"))

    system_blocks = [
        f"## {section.name}\n{section.content}"
        for section in sections
        if section.kind != "user"
    ]
    system_content = "\n\n".join(system_blocks)
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": raw_prompt})
    processed = "\n\n".join(
        f"[{message['role'].upper()}]\n{message['content']}" for message in messages
    )
    return PromptCapture(
        raw_prompt=raw_prompt,
        sections=sections,
        messages=messages,
        processed_prompt=processed,
    )


def _probability(logprob: float) -> float:
    if not math.isfinite(logprob):
        return 0.0
    return min(1.0, max(0.0, math.exp(logprob)))


def parse_logprob_step(payload: Mapping[str, Any], index: int) -> TokenStep:
    """Normalize one Ollama native logprob record."""
    token = str(payload.get("token", ""))
    logprob = float(payload.get("logprob", float("-inf")))
    raw_candidates = list(payload.get("top_logprobs") or [])
    if not any(str(item.get("token", "")) == token for item in raw_candidates):
        raw_candidates.append({"token": token, "logprob": logprob})

    candidates = [
        TokenCandidate(
            token=str(item.get("token", "")),
            logprob=float(item.get("logprob", float("-inf"))),
            probability=_probability(float(item.get("logprob", float("-inf")))),
        )
        for item in raw_candidates
    ]
    candidates.sort(key=lambda candidate: candidate.logprob, reverse=True)
    return TokenStep(
        index=index,
        token=token,
        logprob=logprob,
        probability=_probability(logprob),
        candidates=candidates[:5],
    )


class OllamaInspector:
    """Call Ollama with input capture and streamed token probability hooks."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        configured = base_url or os.getenv(
            "TRACEMIND_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL
        )
        self.base_url = _native_base_url(configured)
        self.timeout = timeout or float(
            os.getenv("TRACEMIND_LLM_TIMEOUT_SECONDS", "120")
        )
        self._client = client

    def _client_instance(self) -> httpx.Client:
        return self._client or httpx.Client(timeout=self.timeout)

    def generate(
        self,
        prompt: PromptCapture,
        *,
        settings: GenerationSettings | None = None,
        model: str | None = None,
        on_token: TokenCallback | None = None,
    ) -> GenerationTrace:
        """Stream one generation and call ``on_token`` as probabilities arrive."""
        sampling = settings or GenerationSettings()
        selected_model = model or model_name_for(ModelRole.CODER)
        body = {
            "model": selected_model,
            "messages": prompt.messages,
            "stream": True,
            "logprobs": True,
            "top_logprobs": 5,
            "options": {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "seed": sampling.seed,
                "num_predict": sampling.max_tokens,
            },
        }

        chunks: list[str] = []
        steps: list[TokenStep] = []
        final_payload: dict[str, Any] = {}
        client = self._client_instance()
        should_close = self._client is None
        try:
            with client.stream(
                "POST", f"{self.base_url}/api/chat", json=body, timeout=self.timeout
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    final_payload = payload
                    message = payload.get("message") or {}
                    chunks.append(str(message.get("content") or ""))
                    for item in payload.get("logprobs") or []:
                        step = parse_logprob_step(item, len(steps))
                        steps.append(step)
                        if on_token is not None:
                            on_token(step, "".join(chunks))
        finally:
            if should_close:
                client.close()

        content = "".join(chunks)
        notice = None
        if not steps:
            notice = (
                "This Ollama server/model returned no token logprobs. "
                "Update Ollama or choose a model/backend that exposes them."
            )
        return GenerationTrace(
            model=selected_model,
            content=content,
            prompt=prompt,
            settings=sampling,
            token_steps=steps,
            prompt_tokens=int(final_payload.get("prompt_eval_count") or 0),
            generated_tokens=int(final_payload.get("eval_count") or len(steps)),
            total_duration_ns=int(final_payload.get("total_duration") or 0),
            logprobs_available=bool(steps),
            notice=notice,
        )

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Return local embeddings from Ollama's native batch endpoint."""
        if not texts:
            return []
        selected_model = model or os.getenv(
            "TRACEMIND_EMBEDDING_MODEL", "nomic-embed-text"
        )
        client = self._client_instance()
        should_close = self._client is None
        try:
            response = client.post(
                f"{self.base_url}/api/embed",
                json={"model": selected_model, "input": list(texts)},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        finally:
            if should_close:
                client.close()
        embeddings = payload.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise ValueError("Ollama returned an unexpected embedding count")
        return [[float(value) for value in vector] for vector in embeddings]


__all__ = [
    "GenerationSettings",
    "GenerationTrace",
    "OllamaInspector",
    "PromptCapture",
    "PromptSection",
    "TokenCandidate",
    "TokenStep",
    "capture_prompt",
    "parse_logprob_step",
]

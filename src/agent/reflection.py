"""Deep traceback parsing and targeted code patch generation."""

from __future__ import annotations

import ast
import difflib
import json
import re
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field, model_validator

from src.agent.llm import ModelRole, get_structured_llm
from src.agent.tools import StrictModel


REFLECTOR_SYSTEM_PROMPT = """\
You are TraceMind's traceback reflector and patch generator. Diagnose the exact
root cause from the failed code and parsed Python traceback. Return the smallest
safe exact-text edits that fix that cause. Do not rewrite the whole program.
Each old_text must appear verbatim in the failed code at the requested
occurrence. Preserve unrelated behavior. The sandbox is offline and contains
only the Python standard library, so replace missing third-party packages with
standard-library implementations. For SyntaxError, replace the complete
malformed logical line or block and ensure the resulting program passes
ast.parse. Never reverse an edit: old_text is the broken text copied from
failed_code, and new_text is its corrected replacement. Return only the
supplied JSON schema.
"""

_FRAME_PATTERN = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)'
    r"(?:, in (?P<function>.+))?$"
)
_EXCEPTION_PATTERN = re.compile(
    r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit))"
    r"(?:: (?P<message>.*))?$"
)


class TracebackFrame(StrictModel):
    file: str
    line: int = Field(ge=1)
    function: str
    source_line: str | None = None


class ParsedTraceback(StrictModel):
    raw: str
    exception_type: str
    exception_message: str
    frames: list[TracebackFrame]


class PatchEdit(StrictModel):
    old_text: str = Field(min_length=1, max_length=10_000)
    new_text: str = Field(max_length=10_000)
    occurrence: int = Field(default=1, ge=1)
    reason: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def edit_must_change_code(self) -> "PatchEdit":
        if self.old_text == self.new_text:
            raise ValueError("old_text and new_text must differ")
        return self


class CodePatch(StrictModel):
    root_cause: str = Field(min_length=1, max_length=2_000)
    edits: list[PatchEdit] = Field(min_length=1, max_length=8)
    validation_notes: str = Field(min_length=1, max_length=1_000)


class ReflectionResult(StrictModel):
    parsed_traceback: ParsedTraceback
    patch: CodePatch
    patched_code: str
    unified_diff: str


class PatchModel(Protocol):
    def invoke(self, input: object, **kwargs: Any) -> CodePatch | dict[str, Any]:
        ...


class PatchApplicationError(ValueError):
    """Raised when a model-generated exact-text patch is unsafe to apply."""


def parse_traceback(traceback_text: str) -> ParsedTraceback:
    """Parse the terminal Python exception and every visible stack frame."""
    if not traceback_text.strip():
        raise ValueError("traceback_text must not be empty")

    lines = traceback_text.rstrip().splitlines()
    frames: list[TracebackFrame] = []
    for index, line in enumerate(lines):
        match = _FRAME_PATTERN.match(line)
        if not match:
            continue
        source_line = None
        if index + 1 < len(lines):
            candidate = lines[index + 1].strip()
            if candidate and not _FRAME_PATTERN.match(lines[index + 1]):
                source_line = candidate
        frames.append(
            TracebackFrame(
                file=match.group("file"),
                line=int(match.group("line")),
                function=(match.group("function") or "<module>").strip(),
                source_line=source_line,
            )
        )

    exception_type = "UnknownError"
    exception_message = ""
    for line in reversed(lines):
        match = _EXCEPTION_PATTERN.match(line.strip())
        if match:
            exception_type = match.group("type")
            exception_message = match.group("message") or ""
            break

    return ParsedTraceback(
        raw=traceback_text,
        exception_type=exception_type,
        exception_message=exception_message,
        frames=frames,
    )


def _replace_occurrence(source: str, edit: PatchEdit) -> str:
    starts: list[int] = []
    cursor = 0
    while True:
        index = source.find(edit.old_text, cursor)
        if index < 0:
            break
        starts.append(index)
        cursor = index + max(1, len(edit.old_text))
    if len(starts) < edit.occurrence:
        raise PatchApplicationError(
            f"Patch target occurrence {edit.occurrence} was not found: {edit.old_text!r}"
        )
    start = starts[edit.occurrence - 1]
    end = start + len(edit.old_text)
    return f"{source[:start]}{edit.new_text}{source[end:]}"


def apply_code_patch(source: str, patch: CodePatch) -> tuple[str, str]:
    """Apply validated exact-text edits and return code plus a unified diff."""
    patched = source
    for edit in patch.edits:
        patched = _replace_occurrence(patched, edit)
    if patched == source:
        raise PatchApplicationError("Patch did not change the source")
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        raise PatchApplicationError(
            f"Patched code is not valid Python at line {exc.lineno}: {exc.msg}"
        ) from exc
    unified_diff = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="failed.py",
            tofile="patched.py",
        )
    )
    return patched, unified_diff


class TracebackReflector:
    """Use the local coder model to diagnose and patch one failed program."""

    def __init__(self, model: PatchModel | None = None) -> None:
        self.model = model or get_structured_llm(ModelRole.CODER, CodePatch)

    def reflect(self, failed_code: str, traceback_text: str) -> ReflectionResult:
        parsed = parse_traceback(traceback_text)
        payload = {
            "failed_code": failed_code,
            "traceback": parsed.model_dump(mode="json"),
        }
        response = self.model.invoke(
            [
                SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, indent=2)),
            ]
        )
        patch = response if isinstance(response, CodePatch) else CodePatch.model_validate(response)
        patched_code, unified_diff = apply_code_patch(failed_code, patch)
        return ReflectionResult(
            parsed_traceback=parsed,
            patch=patch,
            patched_code=patched_code,
            unified_diff=unified_diff,
        )

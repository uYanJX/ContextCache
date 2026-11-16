"""Dialogue representation, envelope serialization, and prompt formatting utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

CONTEXT_ENVELOPE_PREFIX = "__VCACHE_DIALOGUE_V1__:"

DEFAULT_CONTEXT_INSTRUCTION = (
    "Given a conversation, represent the user's current query intent "
    "as a standalone semantic search vector."
)


@dataclass(frozen=True)
class DialogueInput:
    """Structured multi-turn input consisting of historical turns and the current query."""

    query: str
    context: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_turns(
        cls,
        query: str,
        context: Optional[Sequence[str]] = None,
        max_turns: int = 6,
    ) -> "DialogueInput":
        cleaned_ctx: List[str] = []
        if context:
            cleaned_ctx = [str(t).strip() for t in context if t and str(t).strip()]
            if max_turns > 0:
                cleaned_ctx = cleaned_ctx[-max_turns:]
        return cls(query=str(query).strip(), context=tuple(cleaned_ctx))

    def pack(self, max_turns: int = 6) -> str:
        return pack_dialogue(self.query, self.context, max_turns=max_turns)


def messages_to_turns(
    messages: Sequence[Dict[str, str]],
    include_assistant: bool = True,
    max_turns: int = 6,
    max_assistant_chars: int = 200,
) -> Tuple[List[str], str]:
    """Convert OpenAI-style chat messages into `(context_turns, current_query)`.

    The last user message becomes `current_query`, and preceding user (and optional
    truncated assistant) messages form `context_turns`.
    """
    if not messages:
        raise ValueError("messages must not be empty")

    filtered: List[Tuple[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user")).lower().strip()
        content = str(msg.get("content", "")).strip()
        if not content or role == "system":
            continue
        filtered.append((role, content))

    if not filtered:
        raise ValueError("messages must contain at least one non-system message")

    last_user_idx = -1
    for idx in range(len(filtered) - 1, -1, -1):
        if filtered[idx][0] == "user":
            last_user_idx = idx
            break
    if last_user_idx == -1:
        last_user_idx = len(filtered) - 1

    current_query = filtered[last_user_idx][1]
    history_items = filtered[:last_user_idx]

    context_turns: List[str] = []
    for role, content in history_items:
        if role == "assistant":
            if not include_assistant:
                continue
            clipped = content[:max_assistant_chars]
            context_turns.append(f"Assistant: {clipped}")
        else:
            context_turns.append(content)

    if max_turns > 0:
        context_turns = context_turns[-max_turns:]
    return context_turns, current_query


def pack_dialogue(
    query: Optional[str] = None,
    context: Optional[Sequence[str]] = None,
    messages: Optional[Sequence[Dict[str, str]]] = None,
    max_turns: int = 6,
    include_assistant: bool = True,
) -> str:
    """Serialize `(context, query)` or `messages` into a backward-compatible prompt string.

    When `context` is empty, returns `query` directly with zero envelope overhead so
    single-turn callers behave identically to upstream vCache.
    """
    if messages is not None:
        ctx_list, extracted_query = messages_to_turns(
            messages=messages,
            include_assistant=include_assistant,
            max_turns=max_turns,
        )
        if query is not None and str(query).strip():
            extracted_query = str(query).strip()
        query = extracted_query
        context = ctx_list

    if query is None:
        raise ValueError("Either `query` or `messages` must be provided")

    if isinstance(query, str) and query.startswith(CONTEXT_ENVELOPE_PREFIX) and not context:
        return query

    cleaned_query = str(query).strip()
    cleaned_ctx: List[str] = []
    if context:
        cleaned_ctx = [str(t).strip() for t in context if t and str(t).strip()]
        if max_turns > 0:
            cleaned_ctx = cleaned_ctx[-max_turns:]

    if not cleaned_ctx:
        return cleaned_query

    payload = json.dumps(
        {"context": cleaned_ctx, "query": cleaned_query},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{CONTEXT_ENVELOPE_PREFIX}{payload}"


def unpack_dialogue(
    prompt: Union[str, DialogueInput],
    max_turns: Optional[int] = None,
) -> Tuple[List[str], str]:
    """Unpack a string or `DialogueInput` into `(context_turns, current_query)`."""
    if isinstance(prompt, DialogueInput):
        ctx = list(prompt.context)
        if max_turns is not None and max_turns > 0:
            ctx = ctx[-max_turns:]
        return ctx, prompt.query

    text = str(prompt)
    if text.startswith(CONTEXT_ENVELOPE_PREFIX):
        raw_json = text[len(CONTEXT_ENVELOPE_PREFIX) :]
        data = json.loads(raw_json)
        ctx = [str(t).strip() for t in data.get("context", []) if str(t).strip()]
        if max_turns is not None and max_turns > 0:
            ctx = ctx[-max_turns:]
        return ctx, str(data.get("query", "")).strip()

    return [], text.strip()


def dialogue_to_chat_messages(
    prompt: Union[str, DialogueInput],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Reconstruct OpenAI-compatible chat `messages` from a packed or plain prompt."""
    ctx_turns, current_query = unpack_dialogue(prompt)
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in ctx_turns:
        if turn.startswith("Assistant:"):
            content = turn[len("Assistant:") :].strip()
            if content:
                messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": "user", "content": turn})
    messages.append({"role": "user", "content": current_query})
    return messages


def dialogue_to_text_prompt(
    prompt: Union[str, DialogueInput],
    system_prompt: Optional[str] = None,
) -> str:
    """Format a packed or plain multi-turn dialogue into a clean text completion prompt."""
    ctx_turns, current_query = unpack_dialogue(prompt)
    parts: List[str] = []
    if system_prompt:
        parts.append(f"System: {system_prompt}")
    for turn in ctx_turns:
        if turn.startswith("Assistant:"):
            parts.append(turn)
        else:
            parts.append(f"User: {turn}")
    if ctx_turns or system_prompt:
        parts.append(f"User: {current_query}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    return current_query


def format_qwen3_query_prompt(
    query: str,
    instruction: Optional[str] = DEFAULT_CONTEXT_INSTRUCTION,
) -> str:
    """Format a standalone query according to Qwen3-Embedding's instruction template."""
    q = query.strip()
    if instruction:
        return f"Instruct: {instruction}\nQuery: {q}"
    return q


def format_qwen3_context_prompt(
    context: Sequence[str],
    query: str,
    instruction: Optional[str] = DEFAULT_CONTEXT_INSTRUCTION,
    max_turns: int = 6,
    max_turn_chars: int = 320,
) -> str:
    """Format multi-turn context + current query with relative recency markers for Qwen3-Embedding.

    Relative markers (`[Turn -k] ... [Turn -1]`) ensure `[Turn -1]` consistently denotes
    the immediately preceding utterance regardless of conversation length.
    """
    cleaned_ctx = [str(t).strip()[:max_turn_chars] for t in context if t and str(t).strip()]
    if max_turns > 0:
        cleaned_ctx = cleaned_ctx[-max_turns:]

    if not cleaned_ctx:
        return format_qwen3_query_prompt(query=query, instruction=instruction)

    num_ctx = len(cleaned_ctx)
    lines: List[str] = []
    if instruction:
        lines.append(f"Instruct: {instruction}")
    lines.append("Conversation History:")
    for idx, turn_text in enumerate(cleaned_ctx):
        rel_idx = num_ctx - idx  # e.g., 2, 1 -> [Turn -2], [Turn -1]
        lines.append(f"[Turn -{rel_idx}]: {turn_text}")
    lines.append(f"Current Query: {query.strip()}")
    return "\n".join(lines)

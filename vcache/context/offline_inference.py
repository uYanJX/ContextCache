"""Offline / zero-API inference engines for retrieval-focused semantic cache evaluation."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from typing_extensions import override

from vcache.context.dialogue import unpack_dialogue
from vcache.inference_engine.inference_engine import InferenceEngine


class PassthroughInferenceEngine(InferenceEngine):
    """Zero-LLM-API inference engine for semantic cache retrieval benchmarking and testing.

    Eliminates the need for live LLM API calls during cache lookup/update cycles:
    1. If `set_next_response(...)` was called, returns that response (and simulated cost) once.
    2. Else if `response_lookup` contains the packed prompt or unpacked query, returns it.
    3. Else calls `fallback_fn(context, query)` or returns a deterministic placeholder string.
    """

    def __init__(
        self,
        response_lookup: Optional[Dict[str, str]] = None,
        fallback_fn: Optional[Callable[[list[str], str], str]] = None,
        default_cost: float = 0.5,
    ):
        super().__init__()
        self.response_lookup: Dict[str, str] = dict(response_lookup or {})
        self.fallback_fn = fallback_fn
        self.default_cost: float = default_cost
        self.next_response: Optional[str] = None
        self.next_cost: Optional[float] = default_cost
        self.call_count: int = 0

    def set_next_response(self, response: str, cost: Optional[float] = None) -> None:
        self.next_response = response
        self.next_cost = self.default_cost if cost is None else cost

    @override
    def create(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.call_count += 1
        if self.next_response is not None:
            resp = self.next_response
            self.next_response = None
            return resp

        if prompt in self.response_lookup:
            self.next_cost = self.default_cost
            return self.response_lookup[prompt]

        ctx, query = unpack_dialogue(prompt)
        if query in self.response_lookup:
            self.next_cost = self.default_cost
            return self.response_lookup[query]

        self.next_cost = self.default_cost
        if self.fallback_fn is not None:
            return self.fallback_fn(ctx, query)

        if ctx:
            return f"[OfflineResponse | ctx_turns={len(ctx)}] {query}"
        return f"[OfflineResponse] {query}"

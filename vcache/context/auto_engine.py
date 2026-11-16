"""Unified ContextAwareEmbeddingEngine interface for ContextCache and ContextCache+."""

from __future__ import annotations

import json
import os
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from vcache.context.dialogue import DialogueInput
from vcache.context.qwen3_embedding import Qwen3EmbeddingEngine
from vcache.context.turn_adapter import GatedTurnCrossAttentionEngine
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine

ContextBackendType = Literal[
    "auto",
    "contextcache",
    "contextcache+",
    "contextcache_plus",
    "turn_adapter",
    "qwen_lora",
    "query_only",
]

_BACKEND_ALIASES = {
    "contextcache": "turn_adapter",
    "turn_adapter": "turn_adapter",
    "contextcache+": "qwen_lora",
    "contextcache_plus": "qwen_lora",
    "qwen_lora": "qwen_lora",
    "query_only": "query_only",
}


class ContextAwareEmbeddingEngine(EmbeddingEngine):
    """Unified multi-turn context-aware embedding engine.

    Supported `backend` modes:
    - `"contextcache"` (or `"turn_adapter"`): ContextCache attention-based turn context modeling.
    - `"contextcache+"` (or `"qwen_lora"`): ContextCache+ LoRA-adapted embedding backbone with prefix KV reuse.
    - `"query_only"`: Single-turn query embedding baseline.
    - `"auto"`: Loads the default backend configured in `configs/default_model.json`.
    """

    def __init__(
        self,
        backend: ContextBackendType = "auto",
        base_model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        checkpoint_dir: Optional[str] = None,
        config_json_path: Optional[str] = None,
        device: Optional[str] = None,
        max_turns: int = 6,
    ):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if config_json_path is None:
            config_json_path = os.path.join(repo_root, "configs", "default_model.json")

        resolved_backend: str = backend
        resolved_ckpt = checkpoint_dir

        if resolved_backend == "auto" and os.path.isfile(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            resolved_backend = cfg.get("selected_backend", "turn_adapter")
            if resolved_ckpt is None:
                rel_or_abs = cfg.get("checkpoint_path")
                if rel_or_abs:
                    candidate = (
                        rel_or_abs
                        if os.path.isabs(rel_or_abs)
                        else os.path.join(repo_root, rel_or_abs)
                    )
                    if os.path.isdir(candidate):
                        resolved_ckpt = candidate

        if resolved_backend == "auto":
            turn_default = os.path.join(repo_root, "checkpoints", "turn_adapter")
            lora_default = os.path.join(repo_root, "checkpoints", "qwen3_context_lora")
            if os.path.isdir(turn_default):
                resolved_backend = "turn_adapter"
                resolved_ckpt = turn_default
            elif os.path.isdir(lora_default):
                resolved_backend = "qwen_lora"
                resolved_ckpt = lora_default
            else:
                resolved_backend = "turn_adapter"

        resolved_backend = _BACKEND_ALIASES.get(resolved_backend.lower(), resolved_backend)
        self.backend: str = resolved_backend
        self.checkpoint_dir: Optional[str] = resolved_ckpt

        if resolved_backend == "qwen_lora":
            from vcache.context.qwen_context_lora import Qwen3ContextLoRAEngine

            if resolved_ckpt is None:
                resolved_ckpt = os.path.join(repo_root, "checkpoints", "qwen3_context_lora")
            self.recommended_threshold: float = 0.78
            self.engine: EmbeddingEngine = Qwen3ContextLoRAEngine(
                lora_adapter_path=resolved_ckpt,
                base_model_name_or_path=base_model_name_or_path,
                device=device,
                max_turns=max_turns,
            )
        elif resolved_backend == "turn_adapter":
            if resolved_ckpt is None:
                candidate = os.path.join(repo_root, "checkpoints", "turn_adapter")
                resolved_ckpt = candidate if os.path.isdir(candidate) else None
            self.recommended_threshold = 0.85
            self.engine = GatedTurnCrossAttentionEngine(
                adapter_path=resolved_ckpt,
                model_name_or_path=base_model_name_or_path,
                device=device,
                max_turns=max_turns,
            )
        elif resolved_backend == "query_only":
            self.recommended_threshold = 0.85
            self.engine = Qwen3EmbeddingEngine(
                model_name_or_path=base_model_name_or_path,
                mode="query_only",
                device=device,
                max_turns=max_turns,
            )
        else:
            raise ValueError(f"Unknown ContextAwareEmbeddingEngine backend: {backend}")

    def encode_dialogues(
        self,
        dialogues: Sequence[Tuple[Sequence[str], str]],
        batch_size: int = 64,
    ) -> np.ndarray:
        if hasattr(self.engine, "encode_dialogues"):
            return self.engine.encode_dialogues(dialogues, batch_size=batch_size)
        vecs = [self.engine.get_embedding(DialogueInput.from_turns(q, ctx)) for ctx, q in dialogues]
        return np.asarray(vecs, dtype=np.float32)

    def get_embedding(self, text: Union[str, DialogueInput]) -> List[float]:
        return self.engine.get_embedding(text)

"""ContextCache multi-turn context-aware semantic caching modules.

Provides multi-turn dialogue parsing, turn embedding management, attention-based
context integration, and two-stage contextual cache retrieval.
"""

from vcache.context.auto_engine import ContextAwareEmbeddingEngine
from vcache.context.dialogue import (
    CONTEXT_ENVELOPE_PREFIX,
    DEFAULT_CONTEXT_INSTRUCTION,
    DialogueInput,
    dialogue_to_chat_messages,
    dialogue_to_text_prompt,
    format_qwen3_context_prompt,
    format_qwen3_query_prompt,
    messages_to_turns,
    pack_dialogue,
    unpack_dialogue,
)
from vcache.context.offline_inference import PassthroughInferenceEngine
from vcache.context.qwen3_embedding import Qwen3EmbeddingEngine, last_token_pool
from vcache.context.turn_adapter import (
    GatedTurnCrossAttentionEngine,
    GatedTurnCrossAttentionNet,
)
from vcache.context.two_stage_cache import TwoStageContextCache

try:
    from vcache.context.qwen_context_lora import Qwen3ContextLoRAEngine
except ImportError:
    Qwen3ContextLoRAEngine = None  # type: ignore[assignment,misc]

__all__ = [
    "CONTEXT_ENVELOPE_PREFIX",
    "DEFAULT_CONTEXT_INSTRUCTION",
    "DialogueInput",
    "dialogue_to_chat_messages",
    "dialogue_to_text_prompt",
    "messages_to_turns",
    "pack_dialogue",
    "unpack_dialogue",
    "format_qwen3_query_prompt",
    "format_qwen3_context_prompt",
    "PassthroughInferenceEngine",
    "Qwen3EmbeddingEngine",
    "last_token_pool",
    "GatedTurnCrossAttentionNet",
    "GatedTurnCrossAttentionEngine",
    "Qwen3ContextLoRAEngine",
    "TwoStageContextCache",
    "ContextAwareEmbeddingEngine",
]

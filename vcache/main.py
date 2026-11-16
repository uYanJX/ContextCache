import copy
from typing import Dict, List, Optional, Sequence, Tuple

from vcache.config import VCacheConfig
from vcache.context.dialogue import pack_dialogue
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_policy.strategies.verified import VerifiedDecisionPolicy
from vcache.vcache_policy.vcache_policy import VCachePolicy


class VCache:
    """Main ContextCache (vCache-compatible) entry point supporting both single-turn and multi-turn dialogues."""

    def __init__(
        self,
        config: Optional[VCacheConfig] = None,
        policy: Optional[VCachePolicy] = None,
    ):
        self.vcache_config: VCacheConfig = config if config is not None else VCacheConfig()
        self.vcache_policy: VCachePolicy = (
            policy if policy is not None else VerifiedDecisionPolicy(delta=0.02)
        )
        self.vcache_policy.setup(self.vcache_config)
        self._hits: int = 0
        self._misses: int = 0

    def infer(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        context: Optional[Sequence[str]] = None,
        messages: Optional[Sequence[Dict[str, str]]] = None,
    ) -> str:
        """Infer a response using context-aware semantic caching."""
        _, response, _, _ = self.infer_with_cache_info(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
            messages=messages,
        )
        return response

    def infer_with_cache_info(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        id_set: int = -1,
        context: Optional[Sequence[str]] = None,
        messages: Optional[Sequence[Dict[str, str]]] = None,
    ) -> Tuple[bool, str, EmbeddingMetadataObj, Optional[EmbeddingMetadataObj]]:
        """Infer a response and return `(is_cache_hit, response, response_meta, nn_meta)`.

        Accepts either:
        - `prompt` alone (single-turn or pre-packed envelope),
        - `prompt` + `context=[turn_1, ..., turn_{t-1}]` (multi-turn context list), or
        - `messages=[{"role": "user", "content": ...}, ...]` (OpenAI chat format).
        """
        packed_prompt = pack_dialogue(
            query=prompt,
            context=context,
            messages=messages,
            max_turns=self.vcache_config.max_context_turns,
        )

        if system_prompt is None:
            system_prompt = self.vcache_config.system_prompt

        if self.vcache_config.eviction_policy.is_evicting():
            response = self.__generate_response(packed_prompt, system_prompt)
            self._misses += 1
            return (
                False,
                response,
                EmbeddingMetadataObj(embedding_id=-1, response=response, id_set=id_set),
                EmbeddingMetadataObj(embedding_id=-1, response=response, id_set=id_set),
            )

        is_cache_hit, response, nn_metadata = self.vcache_policy.process_request(
            packed_prompt, system_prompt, id_set
        )

        if is_cache_hit:
            self._hits += 1
        else:
            self._misses += 1

        if nn_metadata is not None and nn_metadata.embedding_id != -1:
            self.vcache_config.eviction_policy.update_eviction_metadata(nn_metadata)

        nn_metadata_copy: Optional[EmbeddingMetadataObj] = (
            copy.deepcopy(nn_metadata) if nn_metadata is not None else None
        )

        if self.vcache_config.eviction_policy.ready_to_evict(self.vcache_policy.cache):
            self.vcache_config.eviction_policy.evict(self.vcache_policy.cache)

        if is_cache_hit:
            return is_cache_hit, response, nn_metadata_copy, nn_metadata_copy
        return (
            is_cache_hit,
            response,
            EmbeddingMetadataObj(embedding_id=-1, response=response, id_set=id_set),
            nn_metadata_copy,
        )

    def __generate_response(self, prompt: str, system_prompt: Optional[str]) -> str:
        return self.vcache_policy.inference_engine.create(prompt, system_prompt)

    def flush(self) -> bool:
        """Flush all entries and reset hit/miss statistics."""
        if getattr(self.vcache_policy, "cache", None) is not None:
            self.vcache_policy.cache.flush()
        self._hits = 0
        self._misses = 0
        return True

    def get_statistics(self) -> Dict[str, float | int]:
        """Return structured cache hit/miss and index size statistics."""
        total = self._hits + self._misses
        cache_obj = getattr(self.vcache_policy, "cache", None)
        size = cache_obj.vector_db_size() if cache_obj is not None else 0
        stats: Dict[str, float | int] = {
            "total_requests": total,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total > 0 else 0.0,
            "vector_db_size": size,
        }
        if cache_obj is not None and hasattr(cache_obj, "stats"):
            stats.update(cache_obj.stats)
        return stats

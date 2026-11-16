from typing import Optional

from vcache.context.offline_inference import PassthroughInferenceEngine
from vcache.inference_engine.inference_engine import InferenceEngine
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
    HNSWLibVectorDB,
)
from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import VectorDB
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import (
    NoEvictionPolicy,
)
from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)


class VCacheConfig:
    """Configuration class containing all settings for ContextCache / vCache."""

    def __init__(
        self,
        inference_engine: Optional[InferenceEngine] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_db: Optional[VectorDB] = None,
        embedding_metadata_storage: Optional[EmbeddingMetadataStorage] = None,
        eviction_policy: Optional[EvictionPolicy] = None,
        similarity_evaluator: Optional[SimilarityEvaluator] = None,
        system_prompt: Optional[str] = None,
        use_two_stage_cache: bool = False,
        coarse_k: int = 16,
        theta1: float = 0.40,
        theta2: Optional[float] = None,
        max_context_turns: int = 6,
        max_cache_entries: Optional[int] = None,
    ):
        """Initialize VCache configuration with fresh per-instance components by default."""
        self.inference_engine: InferenceEngine = (
            inference_engine
            if inference_engine is not None
            else PassthroughInferenceEngine()
        )
        if embedding_engine is None:
            # Lazily use ContextAwareEmbeddingEngine when requested, or Benchmark/OpenAI if specified
            from vcache.context.auto_engine import ContextAwareEmbeddingEngine

            self.embedding_engine: EmbeddingEngine = ContextAwareEmbeddingEngine()
        else:
            self.embedding_engine = embedding_engine

        self.vector_db: VectorDB = (
            vector_db if vector_db is not None else HNSWLibVectorDB()
        )
        self.eviction_policy: EvictionPolicy = (
            eviction_policy if eviction_policy is not None else NoEvictionPolicy()
        )
        self.embedding_metadata_storage: EmbeddingMetadataStorage = (
            embedding_metadata_storage
            if embedding_metadata_storage is not None
            else InMemoryEmbeddingMetadataStorage()
        )
        self.similarity_evaluator: SimilarityEvaluator = (
            similarity_evaluator
            if similarity_evaluator is not None
            else StringComparisonSimilarityEvaluator()
        )
        self.system_prompt = system_prompt
        self.use_two_stage_cache = use_two_stage_cache
        self.coarse_k = coarse_k
        self.theta1 = theta1
        self.theta2 = theta2
        self.max_context_turns = max_context_turns
        self.max_cache_entries = max_cache_entries

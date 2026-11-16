"""Two-Stage Context-Aware Cache integrating coarse HNSW recall + short-circuit + fine contextual reranking."""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class TwoStageContextCache(Cache):
    """Drop-in `Cache` subclass implementing Two-Stage Dynamic Contextual Retrieval:

    - **Stage 1 (Fast HNSW Candidate Recall & Empty-Set Short-Circuit)**:
      Queries the HNSW vector index for top-`coarse_k` candidates and filters by coarse
      similarity threshold `theta1`. If no candidate satisfies `sim >= theta1`, immediately
      short-circuits (`[]`) without fine reranking.
    - **Stage 2 (Exact Contextual Cosine Reranking)**:
      Recomputes exact float32 cosine similarity against cached L2-normalized contextual
      vectors (eliminating HNSW approximation jitter) and optionally filters by `theta2`,
      returning top-`k` `(similarity, embedding_id)` pairs to vCache's decision policy.
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        embedding_engine: EmbeddingEngine,
        eviction_policy: EvictionPolicy,
        coarse_k: int = 16,
        theta1: float = 0.40,
        theta2: Optional[float] = None,
        max_entries: Optional[int] = None,
    ):
        super().__init__(
            embedding_store=embedding_store,
            embedding_engine=embedding_engine,
            eviction_policy=eviction_policy,
        )
        self.coarse_k = max(1, coarse_k)
        self.theta1 = theta1
        self.theta2 = theta2
        self.max_entries = max_entries
        self._exact_vectors: Dict[int, np.ndarray] = {}
        self._lru_order: OrderedDict[int, int] = OrderedDict()
        self._id_set_counts: Counter[int] = Counter()
        self._last_prompt_vec: Optional[Tuple[str, np.ndarray]] = None
        self.stats = {
            "total_queries": 0,
            "stage1_short_circuits": 0,
            "stage2_reranks": 0,
            "lru_evictions": 0,
        }

    def has_id_set(self, id_set: int) -> bool:
        """Return True if at least one non-evicted cached entry belongs to `id_set`."""
        return self._id_set_counts.get(id_set, 0) > 0

    def add(self, prompt: str, response: str, id_set: int, cost: float = None) -> int:
        if self._last_prompt_vec is not None and self._last_prompt_vec[0] == prompt:
            normed = self._last_prompt_vec[1]
        else:
            embedding = self.embedding_engine.get_embedding(prompt)
            arr = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(arr) + 1e-12
            normed = (arr / norm).astype(np.float32)

        if self.max_entries is not None and self.max_entries > 0:
            while len(self._lru_order) >= self.max_entries:
                oldest_id, _ = next(iter(self._lru_order.items()))
                self.remove(oldest_id)
                self.stats["lru_evictions"] += 1

        emb_id = self.embedding_store.add_embedding(
            normed.tolist(), response, id_set, cost
        )
        self._exact_vectors[emb_id] = normed
        self._lru_order[emb_id] = id_set
        self._id_set_counts[id_set] += 1
        return emb_id

    def remove(self, embedding_id: int) -> int:
        self._exact_vectors.pop(embedding_id, None)
        old_id_set = self._lru_order.pop(embedding_id, None)
        if old_id_set is not None:
            new_c = self._id_set_counts[old_id_set] - 1
            if new_c <= 0:
                self._id_set_counts.pop(old_id_set, None)
            else:
                self._id_set_counts[old_id_set] = new_c
        return super().remove(embedding_id)

    def flush(self) -> None:
        self._exact_vectors.clear()
        self._lru_order.clear()
        self._id_set_counts.clear()
        self._last_prompt_vec = None
        self.stats = {
            "total_queries": 0,
            "stage1_short_circuits": 0,
            "stage2_reranks": 0,
            "lru_evictions": 0,
        }
        super().flush()

    def get_knn(self, prompt: str, k: int) -> List[Tuple[float, int]]:
        self.stats["total_queries"] += 1
        if self.is_empty():
            self.stats["stage1_short_circuits"] += 1
            return []

        raw_emb = self.embedding_engine.get_embedding(prompt)
        q_vec = np.asarray(raw_emb, dtype=np.float32)
        q_vec = (q_vec / (np.linalg.norm(q_vec) + 1e-12)).astype(np.float32)
        self._last_prompt_vec = (prompt, q_vec)

        # Stage 1: Coarse HNSW candidate recall + theta1 filter
        k_recall = max(k, self.coarse_k)
        coarse_candidates = self.embedding_store.get_knn(q_vec.tolist(), k_recall)
        filtered = [
            (sim, emb_id) for sim, emb_id in coarse_candidates if sim >= self.theta1
        ]
        if not filtered:
            self.stats["stage1_short_circuits"] += 1
            return []

        # Stage 2: Exact contextual vector reranking
        self.stats["stage2_reranks"] += 1
        reranked: List[Tuple[float, int]] = []
        for approx_sim, emb_id in filtered:
            exact_vec = self._exact_vectors.get(emb_id)
            if exact_vec is not None:
                exact_sim = float(np.clip(np.dot(q_vec, exact_vec), -1.0, 1.0))
            else:
                exact_sim = float(approx_sim)
            if self.theta2 is None or exact_sim >= self.theta2:
                reranked.append((exact_sim, emb_id))

        if not reranked:
            return []

        reranked.sort(key=lambda item: item[0], reverse=True)
        top_k = reranked[:k]
        if self.max_entries is not None and top_k:
            top_id = top_k[0][1]
            if top_id in self._lru_order:
                self._lru_order.move_to_end(top_id)
        return top_k

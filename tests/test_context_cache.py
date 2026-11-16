"""10 core unit tests for ContextCache (multi-turn context-aware semantic caching on Qwen3-Embedding-0.6B)."""

from __future__ import annotations

import os
import numpy as np
import pytest
import torch

from vcache import (
    BenchmarkComparisonSimilarityEvaluator,
    BenchmarkStaticDecisionPolicy,
    ContextAwareEmbeddingEngine,
    DialogueInput,
    GatedTurnCrossAttentionNet,
    HNSWLibVectorDB,
    InMemoryEmbeddingMetadataStorage,
    NoEvictionPolicy,
    PassthroughInferenceEngine,
    Qwen3EmbeddingEngine,
    TwoStageContextCache,
    VCache,
    VCacheConfig,
    VerifiedDecisionPolicy,
    pack_dialogue,
    unpack_dialogue,
)

BASE_MODEL = os.environ.get("VCACHE_BASE_MODEL", "Qwen/Qwen3-Embedding-0.6B")


@pytest.fixture(scope="module")
def qwen_query_only_engine() -> Qwen3EmbeddingEngine:
    return Qwen3EmbeddingEngine(
        model_name_or_path=BASE_MODEL,
        mode="query_only",
    )


@pytest.fixture(scope="module")
def context_engine() -> ContextAwareEmbeddingEngine:
    return ContextAwareEmbeddingEngine(
        backend="auto",
        base_model_name_or_path=BASE_MODEL,
        max_turns=6,
    )


def _make_static_vcache(engine, threshold: float = 0.80, two_stage: bool = True) -> VCache:
    cfg = VCacheConfig(
        inference_engine=PassthroughInferenceEngine(),
        embedding_engine=engine,
        vector_db=HNSWLibVectorDB(max_capacity=2048),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        eviction_policy=NoEvictionPolicy(),
        similarity_evaluator=BenchmarkComparisonSimilarityEvaluator(),
        use_two_stage_cache=two_stage,
        coarse_k=12,
        theta1=0.35,
    )
    return VCache(config=cfg, policy=BenchmarkStaticDecisionPolicy(threshold=threshold))


# 1. Dialogue Envelope & OpenAI Messages Conversion
def test_01_dialogue_envelope_and_openai_messages():
    single = pack_dialogue(query="What is vCache?")
    assert single == "What is vCache?"
    assert unpack_dialogue(single) == ([], "What is vCache?")

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Current Question?"},
    ]
    packed = pack_dialogue(messages=msgs, max_turns=3)
    ctx, q = unpack_dialogue(packed)
    assert q == "Current Question?"
    assert len(ctx) == 3
    assert ctx[-1] == "Assistant: Reply 2"

    d_obj = DialogueInput.from_turns("Final?", ["A", "B", "C"], max_turns=2)
    assert list(d_obj.context) == ["B", "C"]


# 2. Zero-LLM-API Offline Passthrough Inference Mode
def test_02_offline_inference_zero_llm_api():
    from vcache.vcache_core.cache.embedding_engine.strategies.benchmark import (
        BenchmarkEmbeddingEngine,
    )

    dummy_emb = BenchmarkEmbeddingEngine()
    p1 = "Explain PagedAttention"
    dummy_emb.set_next_embedding([1.0, 0.0, 0.0, 0.0])

    offline_llm = PassthroughInferenceEngine(
        response_lookup={p1: "PagedAttention manages KV blocks via virtual pages."}
    )
    cfg = VCacheConfig(
        inference_engine=offline_llm,
        embedding_engine=dummy_emb,
        similarity_evaluator=BenchmarkComparisonSimilarityEvaluator(),
    )
    vc = VCache(config=cfg, policy=BenchmarkStaticDecisionPolicy(threshold=0.9))

    hit1, ans1, _, _ = vc.infer_with_cache_info(prompt=p1, id_set=10)
    hit2, ans2, _, _ = vc.infer_with_cache_info(prompt=p1, id_set=10)
    assert hit1 is False
    assert hit2 is True
    assert ans1 == ans2 == "PagedAttention manages KV blocks via virtual pages."
    assert offline_llm.call_count == 1
    assert vc.get_statistics()["hits"] == 1


# 3. Qwen3-Embedding-0.6B Last-Token Pooling, Norm & Batch Consistency
def test_03_qwen3_embedding_last_token_pool_and_l2_norm(
    qwen_query_only_engine: Qwen3EmbeddingEngine,
):
    q1 = "What is PostgreSQL MVCC?"
    q2 = "How does B-tree indexing work in relational databases?"

    v_single = np.asarray(qwen_query_only_engine.get_embedding(q1), dtype=np.float32)
    assert v_single.shape == (1024,)
    assert np.isclose(np.linalg.norm(v_single), 1.0, atol=1e-4)

    batch_vecs = qwen_query_only_engine.encode_single_turns_cached([q1, q2])
    assert batch_vecs.shape == (2, 1024)
    cos_consistency = float(np.dot(v_single, batch_vecs[0]))
    assert cos_consistency > 0.999


# 4. GatedTurnCrossAttentionNet Empty-Context Identity & Dual Erase/Write Gating
def test_04_gated_turn_adapter_zero_context_identity():
    torch.manual_seed(7)
    net = GatedTurnCrossAttentionNet(embed_dim=1024, hidden_dim=1024, num_heads=8, num_layers=3).eval()
    q_raw = torch.randn(4, 1024)
    q_norm = torch.nn.functional.normalize(q_raw, p=2, dim=-1)

    # Empty context must return exact normalized q_vec
    out_empty, diag_empty = net(q_norm, None, None, return_diagnostics=True)
    assert torch.allclose(out_empty, q_norm, atol=1e-6)
    assert float(diag_empty["gate"].abs().max().item()) == 0.0
    assert float(diag_empty["erase_gate"].abs().max().item()) == 0.0
    assert float(diag_empty["write_gate"].abs().max().item()) == 0.0

    # Masked-out context must also return exact normalized q_vec
    ctx_dummy = torch.randn(4, 3, 1024)
    mask_false = torch.zeros(4, 3, dtype=torch.bool)
    out_masked = net(q_norm, ctx_dummy, mask_false)
    assert torch.allclose(out_masked, q_norm, atol=1e-6)

    # Active context produces unit-norm contextualized output
    mask_true = torch.ones(4, 3, dtype=torch.bool)
    out_ctx = net(q_norm, ctx_dummy, mask_true)
    norms = torch.linalg.norm(out_ctx, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# 5. English Multi-Turn Coreference, Paraphrase Hit & Prefix-KV Reuse
def test_05_multiturn_coreference_cache_hit_en(context_engine: ContextAwareEmbeddingEngine):
    vc = _make_static_vcache(context_engine, threshold=0.80)
    vc.vcache_config.inference_engine.set_next_response(
        "Apache Kafka provides high-throughput distributed event streaming and partitioned logs."
    )
    shared_ctx = ["What is Apache Kafka?"]
    hit1, _, _, _ = vc.infer_with_cache_info(
        context=shared_ctx,
        prompt="What are its key features?",
        id_set=101,
    )
    assert hit1 is False

    # Same-session follow-up exercises Prefix-KV reuse when Qwen3ContextLoRAEngine is active
    inner = getattr(context_engine, "engine", None)
    kv_hits_before = getattr(inner, "prefix_kv_hits", 0)
    hit_same_ctx, _, _, _ = vc.infer_with_cache_info(
        context=shared_ctx,
        prompt="Could you summarize its primary capabilities?",
        id_set=101,
    )
    assert hit_same_ctx is True
    if hasattr(inner, "_use_prefix_kv") and getattr(inner, "_use_prefix_kv", False):
        assert inner.prefix_kv_hits > kv_hits_before

    hit2, resp2, _, nn_meta = vc.infer_with_cache_info(
        context=[
            "Can you briefly introduce Apache Kafka?",
            "Assistant: Apache Kafka is a distributed event streaming platform.",
        ],
        prompt="Can you list its main technical capabilities?",
        id_set=101,
    )
    assert hit2 is True
    assert nn_meta is not None and nn_meta.id_set == 101
    assert "Apache Kafka" in resp2


# 6. Cross-Context Identical-Query Hard-Negative Rejection ("What are its key features?" Trap)
def test_06_cross_context_identical_query_rejection(
    context_engine: ContextAwareEmbeddingEngine,
    qwen_query_only_engine: Qwen3EmbeddingEngine,
):
    # Confirm vanilla query-only gives cosine=1.0 on identical surface queries across different contexts
    p_kafka = pack_dialogue(query="What are its key features?", context=["What is Apache Kafka?"])
    p_k8s = pack_dialogue(query="What are its key features?", context=["What is Kubernetes?"])

    v_qonly_1 = np.asarray(qwen_query_only_engine.get_embedding(p_kafka))
    v_qonly_2 = np.asarray(qwen_query_only_engine.get_embedding(p_k8s))
    assert float(np.dot(v_qonly_1, v_qonly_2)) > 0.999

    # ContextAwareEmbeddingEngine cleanly separates them and rejects false cache hit
    vc = _make_static_vcache(context_engine, threshold=0.80)
    vc.infer_with_cache_info(
        context=["What is Apache Kafka?"],
        prompt="What are its key features?",
        id_set=201,
    )
    is_hit_diff_ctx, _, _, _ = vc.infer_with_cache_info(
        context=["What is Kubernetes?"],
        prompt="What are its key features?",
        id_set=202,
    )
    assert is_hit_diff_ctx is False


# 7. Chinese Same-Context Different-Intent Rejection (Preventing History-Concat Pollution)
def test_07_same_context_different_intent_rejection_zh(
    context_engine: ContextAwareEmbeddingEngine,
):
    shared_ctx = [
        "请介绍一下 Redis 缓存系统。",
        "Assistant: Redis 是基于内存的高性能键值存储与缓存系统。",
    ]
    vc = _make_static_vcache(context_engine, threshold=0.80)
    vc.infer_with_cache_info(
        context=shared_ctx,
        prompt="它的核心持久化机制有哪些？",
        id_set=301,
    )
    is_hit_diff_aspect, _, _, _ = vc.infer_with_cache_info(
        context=shared_ctx,
        prompt="它在集群模式下如何做分片和水平扩容？",
        id_set=302,
    )
    assert is_hit_diff_aspect is False


# 8. Chinese Cross-Turn Multi-Turn <-> Standalone Semantic Equivalence Hit
def test_08_chinese_multiturn_to_standalone_equivalence(
    context_engine: ContextAwareEmbeddingEngine,
):
    vc = _make_static_vcache(context_engine, threshold=0.78)
    vc.vcache_config.inference_engine.set_next_response(
        "Elasticsearch 基于倒排索引、分片与主副分片协调节点架构。"
    )
    vc.infer_with_cache_info(
        context=["什么是 Elasticsearch？"],
        prompt="它的底层系统架构是怎么设计的？",
        id_set=401,
    )
    is_hit, resp, _, nn_meta = vc.infer_with_cache_info(
        context=[],
        prompt="Elasticsearch 的底层系统架构和核心组件是如何设计的？",
        id_set=401,
    )
    assert is_hit is True
    assert nn_meta is not None and nn_meta.id_set == 401
    assert "Elasticsearch" in resp


# 9. Mid-Conversation Topic-Switch Robustness
def test_09_topic_switch_mid_conversation_hit(
    context_engine: ContextAwareEmbeddingEngine,
):
    vc = _make_static_vcache(context_engine, threshold=0.78)
    vc.vcache_config.inference_engine.set_next_response("def binary_search(arr, x): ...")
    vc.infer_with_cache_info(
        context=[],
        prompt="How do I implement binary search on a sorted list in Python?",
        id_set=501,
    )
    # Ask inside an unrelated database conversation
    is_hit, resp, _, nn_meta = vc.infer_with_cache_info(
        context=[
            "What is Snowflake?",
            "Assistant: Snowflake is a cloud-native data warehousing platform.",
        ],
        prompt="By the way, how can I binary-search a sorted array in Python?",
        id_set=501,
    )
    assert is_hit is True
    assert nn_meta is not None and nn_meta.id_set == 501
    assert "binary_search" in resp


# 10. Two-Stage Retrieval Short-Circuit & vCache Verified Adaptive Policy Integration
def test_10_two_stage_cache_and_verified_adaptive_policy(
    context_engine: ContextAwareEmbeddingEngine,
):
    import random

    random.seed(42)
    np.random.seed(42)
    policy = VerifiedDecisionPolicy(delta=0.02, sync_updates=True)
    cfg = VCacheConfig(
        inference_engine=PassthroughInferenceEngine(),
        embedding_engine=context_engine,
        vector_db=HNSWLibVectorDB(max_capacity=2048),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        eviction_policy=NoEvictionPolicy(),
        similarity_evaluator=BenchmarkComparisonSimilarityEvaluator(),
        use_two_stage_cache=True,
        coarse_k=10,
        theta1=0.45,
    )
    vc = VCache(config=cfg, policy=policy)
    assert isinstance(vc.vcache_policy.cache, TwoStageContextCache)

    # Seed target entry (id_set=601)
    vc.infer_with_cache_info(
        context=["What is CockroachDB?"],
        prompt="How does it handle high availability and automatic failover?",
        id_set=601,
    )
    sc_before = vc.get_statistics()["stage1_short_circuits"]

    # Unrelated query below theta1 triggers non-empty Stage-1 short-circuit
    vc.infer_with_cache_info(
        context=[],
        prompt="撰写一首关于西湖春天的五言绝句。",
        id_set=999,
    )
    assert vc.get_statistics()["stage1_short_circuits"] > sc_before

    # Feed online observations (both positive paraphrases and cross-context hard negatives)
    pos_variants = [
        (["Can you introduce CockroachDB?"], "How does it handle high availability and automatic failover?"),
        (["Tell me about CockroachDB."], "How does it ensure high availability and automatic failover?"),
        ([], "How does CockroachDB handle high availability and automatic failover?"),
        (["We use CockroachDB."], "How does replication keep CockroachDB available with automatic failover?"),
    ]
    neg_variants = [
        (["What is Cassandra?"], "How does it handle high availability and automatic failover?"),
        (["What is CockroachDB?"], "How can I interact with it from Python?"),
    ]

    false_hits = 0
    exploited_hits = 0
    for round_idx in range(5):
        for idx_n, (ctx, q) in enumerate(neg_variants):
            is_h, _, _, nn_m = vc.infer_with_cache_info(
                context=ctx, prompt=q, id_set=700 + round_idx * 10 + idx_n
            )
            if is_h and nn_m is not None and nn_m.id_set == 601:
                false_hits += 1
        for ctx, q in pos_variants:
            is_h, _, _, nn_m = vc.infer_with_cache_info(context=ctx, prompt=q, id_set=601)
            if is_h and nn_m is not None and nn_m.id_set == 601:
                exploited_hits += 1

    root_meta = vc.vcache_policy.cache.get_metadata(0)
    assert len(root_meta.observations) >= 6
    assert root_meta.t_hat is not None
    assert exploited_hits >= 1
    assert false_hits == 0
    policy.shutdown()


# 11. Both ContextCache and ContextCache+ Backends Pass Quickstart Scenarios
def test_11_contextcache_and_contextcache_plus_quickstart():
    for backend_name in ("contextcache", "contextcache+"):
        engine = ContextAwareEmbeddingEngine(backend=backend_name)
        llm = PassthroughInferenceEngine()
        cfg = VCacheConfig(
            inference_engine=llm,
            embedding_engine=engine,
            use_two_stage_cache=True,
        )
        cache = VCache(
            config=cfg,
            policy=BenchmarkStaticDecisionPolicy(threshold=engine.recommended_threshold),
        )

        # Seed conversation (Apache Kafka)
        llm.set_next_response("Apache Kafka provides distributed event streaming and partitioned logs.")
        hit1, ans1, _, _ = cache.infer_with_cache_info(
            messages=[
                {"role": "user", "content": "What is Apache Kafka?"},
                {"role": "assistant", "content": "Apache Kafka is a distributed event streaming platform."},
                {"role": "user", "content": "What are its key features?"},
            ]
        )
        assert hit1 is False

        # Scenario 1: Paraphrased follow-up in equivalent context -> Cache Hit
        hit2, ans2, _, _ = cache.infer_with_cache_info(
            messages=[
                {"role": "user", "content": "Can you briefly introduce Apache Kafka?"},
                {"role": "assistant", "content": "It is a distributed event streaming platform."},
                {"role": "user", "content": "Could you list its main features and capabilities?"},
            ]
        )
        assert hit2 is True
        assert ans2 == ans1

        # Scenario 2: Identical surface query under different entity context -> Cache Miss
        llm.set_next_response("Kubernetes provides container orchestration and automated scaling.")
        hit3, ans3, _, _ = cache.infer_with_cache_info(
            messages=[
                {"role": "user", "content": "What is Kubernetes?"},
                {"role": "assistant", "content": "Kubernetes is a container orchestration system."},
                {"role": "user", "content": "What are its key features?"},
            ]
        )
        assert hit3 is False
        assert "Kubernetes" in ans3


# 12. Multi-Turn Chat Message Unpacking for LLM Inference Backends
def test_12_chat_messages_unpacking_for_llm_backends():
    from vcache import CONTEXT_ENVELOPE_PREFIX, dialogue_to_chat_messages, dialogue_to_text_prompt

    packed = pack_dialogue(
        messages=[
            {"role": "user", "content": "What is PostgreSQL?"},
            {"role": "assistant", "content": "PostgreSQL is an open-source relational database."},
            {"role": "user", "content": "How does MVCC work in it?"},
        ]
    )
    assert packed.startswith(CONTEXT_ENVELOPE_PREFIX)

    chat_msgs = dialogue_to_chat_messages(packed, system_prompt="Be concise.")
    assert chat_msgs == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is PostgreSQL?"},
        {"role": "assistant", "content": "PostgreSQL is an open-source relational database."},
        {"role": "user", "content": "How does MVCC work in it?"},
    ]

    txt = dialogue_to_text_prompt(packed, system_prompt="Be concise.")
    assert CONTEXT_ENVELOPE_PREFIX not in txt
    assert "User: What is PostgreSQL?" in txt
    assert "Assistant: PostgreSQL is an open-source relational database." in txt
    assert "User: How does MVCC work in it?" in txt

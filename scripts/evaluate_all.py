#!/usr/bin/env python3
"""Comprehensive multi-turn retrieval, latency & end-to-end vCache evaluation across 5 methods.

Compares:
1. `query_only`: Vanilla vCache (`Qwen/Qwen3-Embedding-0.6B` on current query only)
2. `concat`: Untrained history concatenation (`Qwen/Qwen3-Embedding-0.6B` on `ctx + query`)
3. `turn_mean`: Exponentially recency-weighted turn embedding pooling
4. `turn_adapter` (Model A): Trained `GatedTurnCrossAttentionEngine` (v2 Dual Erase-and-Write)
5. `qwen_lora` (Model B): Trained `Qwen3ContextLoRAEngine` (with Causal Prefix-KV Reuse)

Measures both amortized stream latency (`latency_ms`) and strict `batch_size=1` live GPU
single-request latency (`online_mean_ms`, `online_p50_ms`), plus `TwoStageContextCache` +
`VerifiedDecisionPolicy`, writing `configs/default_model.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vcache import (
    BenchmarkComparisonSimilarityEvaluator,
    BenchmarkStaticDecisionPolicy,
    GatedTurnCrossAttentionEngine,
    HNSWLibVectorDB,
    InMemoryEmbeddingMetadataStorage,
    NoEvictionPolicy,
    PassthroughInferenceEngine,
    Qwen3ContextLoRAEngine,
    Qwen3EmbeddingEngine,
    VCache,
    VCacheConfig,
    VerifiedDecisionPolicy,
    pack_dialogue,
)
from vcache.context.dialogue import DEFAULT_CONTEXT_INSTRUCTION, format_qwen3_query_prompt
from vcache.context.qwen3_embedding import select_torch_device
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class PrecomputedLookupEngine(EmbeddingEngine):
    """Fast lookup wrapper over pre-encoded embeddings for exact multi-threshold vCache sweeps."""

    def __init__(self, mapping: Dict[str, List[float]]):
        self.mapping = mapping

    def get_embedding(self, text: str) -> List[float]:
        return self.mapping[text]


def batch_encode_method(
    method_name: str,
    dialogues: Sequence[Tuple[Sequence[str], str]],
    base_engine: Qwen3EmbeddingEngine | None,
    turn_adapter_engine: GatedTurnCrossAttentionEngine | None,
    lora_engine: Qwen3ContextLoRAEngine | None,
    batch_size: int = 64,
) -> Tuple[np.ndarray, float]:
    """Encode `(context, query)` pairs from cold dynamic caches and measure per-query GPU latency (ms)."""
    if base_engine is not None:
        base_engine.clear_caches()
    if turn_adapter_engine is not None:
        turn_adapter_engine.clear_caches()
    if lora_engine is not None:
        lora_engine.clear_caches()

    _cuda_sync()
    t0 = time.perf_counter()
    if method_name == "query_only":
        assert base_engine is not None
        prompts = [
            format_qwen3_query_prompt(q, instruction=DEFAULT_CONTEXT_INSTRUCTION)
            for _, q in dialogues
        ]
        vecs = base_engine.encode_raw_texts(prompts, batch_size=batch_size, max_length=256)
    elif method_name == "concat":
        assert base_engine is not None
        prompts = [
            format_qwen3_query_prompt(
                " | ".join(list(ctx)[-6:] + [q]),
                instruction=DEFAULT_CONTEXT_INSTRUCTION,
            )
            for ctx, q in dialogues
        ]
        vecs = base_engine.encode_raw_texts(prompts, batch_size=batch_size, max_length=384)
    elif method_name == "turn_mean":
        assert base_engine is not None
        base_engine.mode = "turn_mean"
        packed = [pack_dialogue(query=q, context=ctx, max_turns=6) for ctx, q in dialogues]
        unique_utts = list({u for ctx, q in dialogues for u in list(ctx)[-6:] + [q]})
        base_engine.encode_single_turns_cached(unique_utts, use_instruction=True, batch_size=batch_size)
        vecs = np.asarray([base_engine.get_embedding(p) for p in packed], dtype=np.float32)
        base_engine.mode = "query_only"
    elif method_name == "turn_adapter":
        assert turn_adapter_engine is not None
        vecs = turn_adapter_engine.encode_dialogues(dialogues, batch_size=batch_size)
    elif method_name == "qwen_lora":
        assert lora_engine is not None
        vecs = lora_engine.encode_dialogues(dialogues, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    _cuda_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_query_ms = elapsed_ms / max(1, len(dialogues))
    return vecs, per_query_ms


def measure_online_single_request_latency(
    method_name: str,
    stream: List[Dict[str, Any]],
    threshold: float,
    base_engine: Qwen3EmbeddingEngine | None,
    turn_adapter_engine: GatedTurnCrossAttentionEngine | None,
    lora_engine: Qwen3ContextLoRAEngine | None,
) -> Dict[str, float]:
    """Measure true sequential `batch_size=1` end-to-end vCache latency on live GPU engines."""
    if method_name in {"query_only", "concat", "turn_mean"}:
        assert base_engine is not None
        base_engine.enable_torch_compile()
        base_engine.clear_caches()
        base_engine.mode = method_name  # type: ignore[assignment]
        live_engine: EmbeddingEngine = base_engine
    elif method_name == "turn_adapter":
        assert turn_adapter_engine is not None
        turn_adapter_engine.base_engine.enable_torch_compile()
        turn_adapter_engine.clear_caches()
        live_engine = turn_adapter_engine
    elif method_name == "qwen_lora":
        assert lora_engine is not None
        lora_engine.enable_torch_compile()
        lora_engine.clear_caches()
        live_engine = lora_engine
    else:
        raise ValueError(method_name)

    # Warmup single-turn & multi-turn shapes and clear dynamic caches so CUDA kernels are warm but cache is cold
    _ = live_engine.get_embedding(pack_dialogue(query="warmup standalone query", context=[], max_turns=6))
    for w_i in range(3):
        _ = live_engine.get_embedding(
            pack_dialogue(
                query=f"warmup query {w_i}",
                context=[f"warmup turn a {w_i}", f"warmup turn b {w_i}"],
                max_turns=6,
            )
        )
    if base_engine is not None:
        base_engine.clear_caches()
    if turn_adapter_engine is not None:
        turn_adapter_engine.clear_caches()
    if lora_engine is not None:
        lora_engine.clear_caches()

    inf_engine = PassthroughInferenceEngine()
    cfg = VCacheConfig(
        inference_engine=inf_engine,
        embedding_engine=live_engine,
        vector_db=HNSWLibVectorDB(max_capacity=20000),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        eviction_policy=NoEvictionPolicy(),
        similarity_evaluator=BenchmarkComparisonSimilarityEvaluator(),
        use_two_stage_cache=True,
        coarse_k=16,
        theta1=0.35,
    )
    vc = VCache(config=cfg, policy=BenchmarkStaticDecisionPolicy(threshold=threshold))

    latencies_ms: List[float] = []
    for req in stream:
        packed = pack_dialogue(query=req["query"], context=req["context"], max_turns=6)
        inf_engine.set_next_response(req["response"], cost=0.5)
        _cuda_sync()
        t_req = time.perf_counter()
        vc.infer_with_cache_info(prompt=packed, id_set=req["id_set"])
        _cuda_sync()
        latencies_ms.append((time.perf_counter() - t_req) * 1000.0)

    if base_engine is not None:
        base_engine.mode = "query_only"

    arr = np.asarray(latencies_ms, dtype=np.float64)
    out = {
        "online_mean_ms": round(float(np.mean(arr)), 3),
        "online_p50_ms": round(float(np.percentile(arr, 50)), 3),
        "online_p95_ms": round(float(np.percentile(arr, 95)), 3),
    }
    if method_name == "qwen_lora" and lora_engine is not None:
        tot_kv = lora_engine.prefix_kv_hits + lora_engine.prefix_kv_misses
        out["prefix_kv_hit_rate"] = round(
            lora_engine.prefix_kv_hits / max(1, tot_kv), 4
        )
    return out


def evaluate_canard_retrieval(
    method_name: str,
    queries: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    base_engine: Qwen3EmbeddingEngine | None,
    turn_adapter_engine: GatedTurnCrossAttentionEngine | None,
    lora_engine: Qwen3ContextLoRAEngine | None,
) -> Dict[str, float]:
    if not queries or not pool:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr_at_10": 0.0}

    q_dialogues = [(q["context"], q["query"]) for q in queries]
    p_dialogues = [(p["context"], p["query"]) for p in pool]

    q_vecs, q_ms = batch_encode_method(
        method_name, q_dialogues, base_engine, turn_adapter_engine, lora_engine
    )
    p_vecs, _ = batch_encode_method(
        method_name, p_dialogues, base_engine, turn_adapter_engine, lora_engine
    )

    sims = q_vecs @ p_vecs.T  # (Nq, Np)
    top10 = np.argsort(-sims, axis=1)[:, :10]

    r1 = 0
    r5 = 0
    mrr10 = 0.0
    for idx, q in enumerate(queries):
        gold = q["gold_pool_id"]
        ranked = top10[idx]
        if ranked[0] == gold:
            r1 += 1
        if gold in ranked[:5]:
            r5 += 1
        matches = np.where(ranked == gold)[0]
        if len(matches) > 0:
            mrr10 += 1.0 / (int(matches[0]) + 1)

    n = len(queries)
    return {
        "recall_at_1": round(r1 / n, 4),
        "recall_at_5": round(r5 / n, 4),
        "mrr_at_10": round(mrr10 / n, 4),
        "encode_ms_per_query": round(q_ms, 3),
    }


def run_vcache_pass(
    stream: List[Dict[str, Any]],
    lookup_engine: PrecomputedLookupEngine,
    policy_type: str,
    threshold_or_delta: float,
    use_two_stage: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    inf_engine = PassthroughInferenceEngine()
    evaluator = BenchmarkComparisonSimilarityEvaluator()

    if policy_type == "static":
        policy = BenchmarkStaticDecisionPolicy(threshold=threshold_or_delta)
    elif policy_type == "verified":
        policy = VerifiedDecisionPolicy(delta=threshold_or_delta, sync_updates=True)
    else:
        raise ValueError(policy_type)

    cfg = VCacheConfig(
        inference_engine=inf_engine,
        embedding_engine=lookup_engine,
        vector_db=HNSWLibVectorDB(max_capacity=20000),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        eviction_policy=NoEvictionPolicy(),
        similarity_evaluator=evaluator,
        use_two_stage_cache=use_two_stage,
        coarse_k=16,
        theta1=0.35,
    )
    vc = VCache(config=cfg, policy=policy)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    phenom_stats: Dict[str, Dict[str, int]] = {}
    nn_sims_pos: List[float] = []
    nn_sims_neg: List[float] = []
    roc_labels: List[int] = []
    roc_scores: List[float] = []

    lookup_elapsed_s = 0.0
    for req in stream:
        packed = pack_dialogue(query=req["query"], context=req["context"], max_turns=6)
        inf_engine.set_next_response(req["response"], cost=0.5)

        # Diagnostic ROC-AUC probe (excluded from cache lookup latency timer)
        if not vc.vcache_policy.cache.is_empty():
            knn = vc.vcache_policy.cache.get_knn(packed, k=1)
            if knn:
                top_sim, top_id = knn[0]
                top_meta = vc.vcache_policy.cache.get_metadata(top_id)
                same_cluster = int(top_meta.id_set == req["id_set"])
                if req["phenomenon"] != "seed_cache":
                    roc_labels.append(same_cluster)
                    roc_scores.append(float(top_sim))
                    if same_cluster == 1:
                        nn_sims_pos.append(float(top_sim))
                    else:
                        nn_sims_neg.append(float(top_sim))

        t_req = time.perf_counter()
        is_hit, resp, _, nn_meta = vc.infer_with_cache_info(
            prompt=packed,
            id_set=req["id_set"],
        )
        lookup_elapsed_s += time.perf_counter() - t_req

        ph = req["phenomenon"]
        if ph == "seed_cache":
            continue

        st = phenom_stats.setdefault(ph, {"total": 0, "hit": 0, "correct_hit": 0, "false_hit": 0})
        st["total"] += 1

        if is_hit:
            st["hit"] += 1
            hit_correct = bool(nn_meta is not None and nn_meta.id_set == req["id_set"])
            if hit_correct:
                tp += 1
                st["correct_hit"] += 1
            else:
                fp += 1
                st["false_hit"] += 1
        else:
            if req["should_hit"]:
                fn += 1
            else:
                tn += 1

    cache_overhead_ms = lookup_elapsed_s * 1000.0 / max(1, len(stream))
    if hasattr(policy, "shutdown"):
        policy.shutdown()

    eval_total = tp + fp + fn + tn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    hit_rate = (tp + fp) / max(1, eval_total)
    error_rate = fp / max(1, eval_total)

    def rate(ph_name: str, num_key: str) -> float:
        d = phenom_stats.get(ph_name, {})
        tot = d.get("total", 0)
        return round(d.get(num_key, 0) / tot, 4) if tot > 0 else 0.0

    auc = (
        round(float(roc_auc_score(roc_labels, roc_scores)), 4)
        if len(set(roc_labels)) > 1
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hit_rate": round(hit_rate, 4),
        "error_rate": round(error_rate, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "roc_auc": auc,
        "mean_pos_nn_sim": round(float(np.mean(nn_sims_pos)) if nn_sims_pos else 0.0, 4),
        "mean_neg_nn_sim": round(float(np.mean(nn_sims_neg)) if nn_sims_neg else 0.0, 4),
        "coref_hit_rate": rate("coref_paraphrase_pos", "correct_hit"),
        "standalone_hit_rate": rate("cross_turn_standalone_pos", "correct_hit"),
        "topic_switch_hit_rate": rate("topic_switch_pos", "correct_hit"),
        "cross_ctx_trap_fpr": rate("cross_context_same_query_neg", "false_hit"),
        "same_ctx_trap_fpr": rate("same_context_diff_aspect_neg", "false_hit"),
        "cache_lookup_ms": round(cache_overhead_ms, 3),
    }


def generate_plots(results_by_method: Dict[str, Dict[str, Any]], out_png: str) -> None:
    labels_map = {
        "query_only": "Query-Only",
        "concat": "History\nConcat",
        "turn_mean": "Turn-Mean",
        "turn_adapter": "ContextCache",
        "qwen_lora": "ContextCache+",
    }
    keys = [k for k in labels_map if k in results_by_method]
    x_labels = [labels_map[k] for k in keys]
    base_positions = [0.0, 1.05, 2.10, 3.28, 4.52]
    x = np.array(base_positions[: len(keys)], dtype=np.float64)
    ours_set = {"turn_adapter", "qwen_lora"}

    rc = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#1E293B",
        "xtick.color": "#1E293B",
        "ytick.color": "#334155",
    }

    with plt.rc_context(rc):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.55), dpi=320)
        fig.patch.set_facecolor("white")

        ours_pos = [x[i] for i, k in enumerate(keys) if k in ours_set]
        for ax in (ax0, ax1):
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#CBD5E1", alpha=0.65, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(x[0] - 0.62, x[-1] + 0.65)
            if ours_pos:
                ax.axvspan(
                    min(ours_pos) - 0.56,
                    max(ours_pos) + 0.56,
                    color="#EFF6FF",
                    alpha=0.80,
                    zorder=0,
                )

        # Panel (a): Multi-Turn Retrieval (CANARD Recall@1) & Cache F1
        r1_vals = [results_by_method[k]["canard"]["recall_at_1"] * 100.0 for k in keys]
        f1_vals = [results_by_method[k]["vcache_best_static"]["f1"] * 100.0 for k in keys]
        w = 0.36
        sep = 0.20

        for idx, k in enumerate(keys):
            is_ours = k in ours_set
            alpha_v = 0.96 if is_ours else 0.78
            lw_v = 0.9 if is_ours else 0.6
            b1 = ax0.bar(
                x[idx] - sep,
                r1_vals[idx],
                width=w,
                color="#2B5C8F",
                alpha=alpha_v,
                edgecolor="#1E293B",
                linewidth=lw_v,
                zorder=3,
                label="CANARD Recall@1 (%)" if idx == len(keys) - 1 else None,
            )
            b2 = ax0.bar(
                x[idx] + sep,
                f1_vals[idx],
                width=w,
                color="#167C65",
                alpha=alpha_v,
                edgecolor="#1E293B",
                linewidth=lw_v,
                zorder=3,
                label="Multi-Turn Cache F1 (%)" if idx == len(keys) - 1 else None,
            )
            fw = "bold" if is_ours else "normal"
            col = "#0F172A" if is_ours else "#475569"
            ax0.text(
                b1[0].get_x() + b1[0].get_width() / 2,
                r1_vals[idx] + 1.4,
                f"{r1_vals[idx]:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight=fw,
                color=col,
            )
            ax0.text(
                b2[0].get_x() + b2[0].get_width() / 2,
                f1_vals[idx] + 1.4,
                f"{f1_vals[idx]:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight=fw,
                color=col,
            )

        ax0.set_title("(a) Multi-Turn Retrieval & Cache Accuracy (↑)", fontsize=11.8, fontweight="bold", pad=10, color="#0F172A")
        ax0.set_ylabel("Accuracy / F1 Score (%)", fontsize=10.5)
        ax0.set_xticks(x)
        ax0.set_xticklabels(x_labels, fontsize=9.4)
        ax0.set_ylim(0, 113)
        ax0.legend(fontsize=8.8, loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95)

        # Panel (b): False-Hit Error Rate (Lower is Better ↓)
        err_vals = [results_by_method[k]["vcache_best_static"]["error_rate"] * 100.0 for k in keys]
        err_colors = {
            "query_only": "#C96A6A",
            "concat": "#D99178",
            "turn_mean": "#94A3B8",
            "turn_adapter": "#2B5C8F",
            "qwen_lora": "#167C65",
        }
        w_err = 0.48
        for idx, k in enumerate(keys):
            is_ours = k in ours_set
            bar = ax1.bar(
                x[idx],
                max(err_vals[idx], 0.45),
                width=w_err,
                color=err_colors.get(k, "#94A3B8"),
                alpha=0.96 if is_ours else 0.84,
                edgecolor="#1E293B",
                linewidth=0.9 if is_ours else 0.6,
                zorder=3,
            )
            lbl_str = f"{err_vals[idx]:.2f}%" if err_vals[idx] < 2.0 else f"{err_vals[idx]:.1f}%"
            ax1.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                max(err_vals[idx], 0.45) + 1.5,
                lbl_str,
                ha="center",
                va="bottom",
                fontsize=9.2 if is_ours else 8.8,
                fontweight="bold" if is_ours else "normal",
                color="#0F172A" if is_ours else "#334155",
            )

        ax1.set_title("(b) Multi-Turn False-Hit Error Rate (↓)", fontsize=11.8, fontweight="bold", pad=10, color="#0F172A")
        ax1.set_ylabel("False-Hit Error Rate (%)", fontsize=10.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, fontsize=9.4)
        ax1.set_ylim(0, max(err_vals, default=75.0) * 1.18)

        if "query_only" in results_by_method and "turn_adapter" in results_by_method and "qwen_lora" in results_by_method:
            ax1.text(
                0.97,
                0.88,
                "Error Reduction vs. Single-Turn:\n"
                f"• ContextCache:  {results_by_method['query_only']['vcache_best_static']['error_rate']*100:.1f}% → {results_by_method['turn_adapter']['vcache_best_static']['error_rate']*100:.2f}%\n"
                f"• ContextCache+: {results_by_method['query_only']['vcache_best_static']['error_rate']*100:.1f}% → {results_by_method['qwen_lora']['vcache_best_static']['error_rate']*100:.2f}%",
                transform=ax1.transAxes,
                ha="right",
                va="top",
                fontsize=8.6,
                color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.8),
            )

        for ax in (ax0, ax1):
            for idx, tick in enumerate(ax.get_xticklabels()):
                if keys[idx] in ours_set:
                    tick.set_fontweight("bold")
                    tick.set_color("#0F172A")

        plt.tight_layout(w_pad=2.4)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--turn_adapter_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "turn_adapter"))
    parser.add_argument("--lora_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "qwen3_context_lora"))
    parser.add_argument("--results_dir", type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--config_out", type=str, default=os.path.join(REPO_ROOT, "configs", "default_model.json"))
    parser.add_argument("--methods", type=str, default="query_only,concat,turn_mean,turn_adapter,qwen_lora")
    parser.add_argument("--partial_out", type=str, default="")
    parser.add_argument("--merge_partials", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    all_methods_order = ["query_only", "concat", "turn_mean", "turn_adapter", "qwen_lora"]
    all_results: Dict[str, Dict[str, Any]] = {}

    if args.merge_partials:
        for p_file in [x.strip() for x in args.merge_partials.split(",") if x.strip()]:
            with open(p_file, "r", encoding="utf-8") as f:
                all_results.update(json.load(f))
        methods = [m for m in all_methods_order if m in all_results]
    else:
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
        device = str(select_torch_device())

        canard_queries = load_jsonl(os.path.join(args.data_dir, "canard_test_queries.jsonl"))
        canard_pool = load_jsonl(os.path.join(args.data_dir, "canard_test_pool.jsonl"))
        stream_reqs = load_jsonl(os.path.join(args.data_dir, "vcache_stream_test.jsonl"))

        print(
            f"[eval] Loaded {len(canard_queries)} CANARD test queries (pool={len(canard_pool)}) "
            f"and {len(stream_reqs)} streaming vCache requests on {device} for {methods}"
        )

        need_base = any(m in {"query_only", "concat", "turn_mean", "turn_adapter"} for m in methods)
        base_engine = (
            Qwen3EmbeddingEngine(
                model_name_or_path=args.base_model,
                mode="query_only",
                device=device,
            )
            if need_base
            else None
        )
        turn_adapter_engine = (
            GatedTurnCrossAttentionEngine(
                adapter_path=args.turn_adapter_dir,
                base_engine=base_engine,
                device=device,
            )
            if "turn_adapter" in methods
            else None
        )
        lora_engine = (
            Qwen3ContextLoRAEngine(
                lora_adapter_path=args.lora_dir,
                base_model_name_or_path=args.base_model,
                device=device,
                merge_lora=True,
            )
            if "qwen_lora" in methods
            else None
        )

        thresholds = [round(t, 2) for t in np.arange(0.55, 0.96, 0.03)]
        stream_dialogues = [(r["context"], r["query"]) for r in stream_reqs]
        packed_keys = [pack_dialogue(query=r["query"], context=r["context"], max_turns=6) for r in stream_reqs]

        for m in methods:
            print(f"\n[eval] === Evaluating method: {m} ===")
            canard_res = evaluate_canard_retrieval(
                m, canard_queries, canard_pool, base_engine, turn_adapter_engine, lora_engine
            )
            print(f"[eval] {m} CANARD: {json.dumps(canard_res)}")

            s_vecs, stream_enc_ms = batch_encode_method(
                m, stream_dialogues, base_engine, turn_adapter_engine, lora_engine
            )
            mapping = {pk: vec.tolist() for pk, vec in zip(packed_keys, s_vecs)}
            lookup = PrecomputedLookupEngine(mapping)

            best_static = None
            best_thr = 0.80
            sweep_records = []
            for thr in thresholds:
                res = run_vcache_pass(
                    stream_reqs, lookup, policy_type="static", threshold_or_delta=thr, use_two_stage=True
                )
                res["threshold"] = thr
                sweep_records.append(res)
                if best_static is None or res["f1"] > best_static["f1"]:
                    best_static = res
                    best_thr = thr

            verified_res = run_vcache_pass(
                stream_reqs,
                lookup,
                policy_type="verified",
                threshold_or_delta=0.05,
                use_two_stage=True,
            )
            verified_strict = run_vcache_pass(
                stream_reqs,
                lookup,
                policy_type="verified",
                threshold_or_delta=0.02,
                use_two_stage=True,
            )

            online_lat = measure_online_single_request_latency(
                m, stream_reqs, best_thr, base_engine, turn_adapter_engine, lora_engine
            )

            best_static["best_threshold"] = best_thr
            best_static["total_latency_ms"] = round(stream_enc_ms + best_static["cache_lookup_ms"], 3)
            verified_res["total_latency_ms"] = round(stream_enc_ms + verified_res["cache_lookup_ms"], 3)

            print(f"[eval] {m} Best Static (t={best_thr}): {json.dumps(best_static)}")
            print(f"[eval] {m} Online B=1 Latency: {json.dumps(online_lat)}")
            print(f"[eval] {m} vCache Verified (delta=0.05): {json.dumps(verified_res)}")

            all_results[m] = {
                "canard": canard_res,
                "stream_encode_ms": round(stream_enc_ms, 3),
                "online_latency": online_lat,
                "vcache_best_static": best_static,
                "vcache_verified_d005": verified_res,
                "vcache_verified_d002": verified_strict,
                "threshold_sweep": sweep_records,
            }

        if args.partial_out:
            os.makedirs(os.path.dirname(args.partial_out), exist_ok=True)
            with open(args.partial_out, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"[eval] Saved partial results ({list(all_results.keys())}) to {args.partial_out}")
            return

    def composite_score(m_key: str) -> float:
        r = all_results[m_key]
        return (
            0.45 * r["vcache_best_static"]["f1"]
            + 0.30 * r["canard"]["mrr_at_10"]
            + 0.25 * r["vcache_verified_d005"]["f1"]
        )

    candidate_scores: Dict[str, float] = {}
    for cand_key in ("turn_adapter", "qwen_lora"):
        if cand_key in all_results:
            candidate_scores[cand_key] = round(composite_score(cand_key), 4)

    if candidate_scores:
        winner = max(candidate_scores.items(), key=lambda kv: kv[1])[0]
    else:
        winner = "turn_adapter"

    winner_rel_ckpt = (
        "checkpoints/qwen3_context_lora"
        if winner == "qwen_lora"
        else "checkpoints/turn_adapter"
    )

    default_cfg: Dict[str, Any] = {
        "selected_backend": winner,
        "checkpoint_path": winner_rel_ckpt,
        "base_model": "Qwen/Qwen3-Embedding-0.6B",
        "composite_scores": candidate_scores,
        "selection_rule": "0.45 * vcache_static_f1 + 0.30 * canard_mrr10 + 0.25 * vcache_verified_f1",
    }
    os.makedirs(os.path.dirname(args.config_out), exist_ok=True)
    with open(args.config_out, "w", encoding="utf-8") as f:
        json.dump(default_cfg, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.results_dir, "contextcache_benchmark_table.csv")
    fieldnames = [
        "method",
        "canard_r1",
        "canard_r5",
        "canard_mrr10",
        "cache_precision",
        "cache_recall",
        "cache_f1",
        "false_hit_error",
    ]
    method_export_names = {
        "turn_adapter": "contextcache",
        "qwen_lora": "contextcache+",
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in methods:
            r = all_results[m]
            st = r["vcache_best_static"]
            writer.writerow(
                {
                    "method": method_export_names.get(m, m),
                    "canard_r1": r["canard"]["recall_at_1"],
                    "canard_r5": r["canard"]["recall_at_5"],
                    "canard_mrr10": r["canard"]["mrr_at_10"],
                    "cache_precision": st["precision"],
                    "cache_recall": st["recall"],
                    "cache_f1": st["f1"],
                    "false_hit_error": st["error_rate"],
                }
            )

    generate_plots(all_results, os.path.join(args.results_dir, "contextcache_benchmark.png"))
    print(f"\n[eval] Selected backend: {winner} (scores={candidate_scores})")
    print(f"[eval] Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()

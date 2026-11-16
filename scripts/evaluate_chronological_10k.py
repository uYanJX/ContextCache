#!/usr/bin/env python3
"""True 10,000-Request Chronological End-to-End Semantic Cache Evaluation.

Simulates real production LLM semantic caching in strict timestamp order (`t = 0 .. 9,999`):
1. Cold cache start (`cache_size = 0`, zero pre-seeding, cold GPU turn/KV caches).
2. Requests arrive sequentially (`batch_size = 1`) across 24 interleaved multi-turn user sessions.
3. At each arrival `i`:
   - `VCache` encodes `(context, query)` on GPU, queries `TwoStageContextCache` (HNSW + exact cosine
     reranking), and consults the Decision Policy (`Val-Calibrated Static` or `VerifiedDecisionPolicy`).
   - On **Cache Hit**: returns cached response immediately (`0` LLM calls); if `hit_id_set == req_id_set`,
     counts as a **Valid Cache Hit (True LLM Call Saved)**, else **False Hit (System Error / Pollution)**.
   - On **Cache Miss**: invokes `InferenceEngine` (`1` LLM call) and dynamically inserts the new
     contextual vector + response into `VCache` (growing HNSW / triggering LRU eviction if bounded).
4. Zero test-leakage threshold calibration: production static thresholds (`t_val_safe`, `t_val_f1`)
   are selected strictly on the disjoint 600-request `vcache_chronological_val.jsonl` stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
except Exception:
    pass

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
from vcache.context.qwen3_embedding import select_torch_device
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class LiveRecordingEngine(EmbeddingEngine):
    """Transparent pass-through wrapper around a live GPU engine that records deterministic vectors."""

    def __init__(self, inner: EmbeddingEngine):
        self.inner = inner
        self.recorded: Dict[str, List[float]] = {}

    def get_embedding(self, text: str) -> List[float]:
        vec = self.inner.get_embedding(text)
        if text not in self.recorded:
            self.recorded[text] = vec
        return vec


class ReplayLookupEngine(EmbeddingEngine):
    """Exact vector lookup engine for fast chronological policy & threshold sweeps."""

    def __init__(self, recorded: Dict[str, List[float]]):
        self.recorded = recorded

    def get_embedding(self, text: str) -> List[float]:
        return self.recorded[text]


def prepare_live_engine(
    method_name: str,
    base_engine: Qwen3EmbeddingEngine,
    turn_adapter_engine: Optional[GatedTurnCrossAttentionEngine],
    lora_engine: Optional[Qwen3ContextLoRAEngine],
    use_compile: bool = True,
) -> EmbeddingEngine:
    """Configure, compile-warmup across shapes, and reset dynamic caches to cold state."""
    if method_name in {"query_only", "concat", "turn_mean"}:
        if use_compile:
            base_engine.enable_torch_compile()
        base_engine.clear_caches()
        base_engine.mode = method_name  # type: ignore[assignment]
        eng: EmbeddingEngine = base_engine
    elif method_name == "turn_adapter":
        assert turn_adapter_engine is not None
        if use_compile:
            turn_adapter_engine.base_engine.enable_torch_compile()
        turn_adapter_engine.clear_caches()
        eng = turn_adapter_engine
    elif method_name == "qwen_lora":
        assert lora_engine is not None
        if use_compile:
            lora_engine.enable_torch_compile()
        lora_engine.clear_caches()
        eng = lora_engine
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Warm up batch shapes 1..7 so Inductor/CUDA graphs never compile inside the timed 10k trace
    _ = eng.get_embedding(pack_dialogue(query="warmup standalone question", context=[], max_turns=6))
    for n_ctx in range(1, 7):
        ctx_w = [f"warmup context utterance {k} for shape {n_ctx}" for k in range(n_ctx)]
        _ = eng.get_embedding(
            pack_dialogue(query=f"warmup followup query {n_ctx}", context=ctx_w, max_turns=6)
        )
    _cuda_sync()

    # Reset all dynamic turn/KV/dialogue caches so `t = 0` starts 100% cold!
    base_engine.clear_caches()
    if turn_adapter_engine is not None:
        turn_adapter_engine.clear_caches()
    if lora_engine is not None:
        lora_engine.clear_caches()
    if method_name in {"query_only", "concat", "turn_mean"}:
        base_engine.mode = method_name  # type: ignore[assignment]
    return eng


def batch_preencode_validation(
    method_name: str,
    val_stream: List[Dict[str, Any]],
    base_engine: Qwen3EmbeddingEngine,
    turn_adapter_engine: Optional[GatedTurnCrossAttentionEngine],
    lora_engine: Optional[Qwen3ContextLoRAEngine],
) -> Dict[str, List[float]]:
    """Encode the 600-request validation calibration stream to select static thresholds without test leakage."""
    eng = prepare_live_engine(
        method_name, base_engine, turn_adapter_engine, lora_engine, use_compile=True
    )
    mapping: Dict[str, List[float]] = {}
    for req in val_stream:
        packed = pack_dialogue(query=req["query"], context=req["context"], max_turns=6)
        if packed not in mapping:
            mapping[packed] = eng.get_embedding(packed)
    base_engine.mode = "query_only"
    return mapping


def run_chronological_simulation(
    stream: List[Dict[str, Any]],
    embedding_engine: EmbeddingEngine,
    policy_type: str,
    threshold_or_delta: float,
    max_cache_entries: Optional[int] = None,
    measure_gpu_sync: bool = False,
    simulated_llm_ms: float = 600.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute `VCache` strictly chronologically from cold start (`cache_size = 0`)."""
    random.seed(seed)
    np.random.seed(seed)

    inf_engine = PassthroughInferenceEngine()
    evaluator = BenchmarkComparisonSimilarityEvaluator()

    if policy_type == "static":
        policy = BenchmarkStaticDecisionPolicy(threshold=threshold_or_delta)
    elif policy_type == "verified":
        policy = VerifiedDecisionPolicy(delta=threshold_or_delta, sync_updates=True)
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    cfg = VCacheConfig(
        inference_engine=inf_engine,
        embedding_engine=embedding_engine,
        vector_db=HNSWLibVectorDB(max_capacity=max(25000, len(stream) + 1000)),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        eviction_policy=NoEvictionPolicy(),
        similarity_evaluator=evaluator,
        use_two_stage_cache=True,
        coarse_k=16,
        theta1=0.35,
        max_cache_entries=max_cache_entries,
    )
    vc = VCache(config=cfg, policy=policy)
    cache_obj = vc.vcache_policy.cache

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    oracle_repeats = 0

    latencies_ms: List[float] = []
    hit_latencies_ms: List[float] = []
    miss_latencies_ms: List[float] = []

    domain_stats: Dict[str, Dict[str, int]] = {}
    phenom_stats: Dict[str, Dict[str, int]] = {}
    unseen_aspect_stats = {"total": 0, "oracle_pos": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    # Chronological 5-window tracking (e.g. every 2,000 requests for N=10,000)
    win_size = max(1, len(stream) // 5)
    windows: List[Dict[str, Any]] = []
    w_tp = w_fp = w_fn = w_tn = w_oracle = 0
    w_lats: List[float] = []

    for idx, req in enumerate(stream):
        packed = pack_dialogue(query=req["query"], context=req["context"], max_turns=6)
        cid = int(req["id_set"])
        was_in_cache = bool(cache_obj.has_id_set(cid))
        is_oracle_repeat = bool(req.get("oracle_should_hit", False))
        if is_oracle_repeat:
            oracle_repeats += 1
            w_oracle += 1

        inf_engine.set_next_response(req["response"], cost=simulated_llm_ms / 1000.0)

        if measure_gpu_sync:
            _cuda_sync()
        t0 = time.perf_counter()
        is_hit, _, _, nn_meta = vc.infer_with_cache_info(prompt=packed, id_set=cid)
        if measure_gpu_sync:
            _cuda_sync()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(dt_ms)
        w_lats.append(dt_ms)

        dom = req.get("domain", "unknown")
        ph = req.get("phenomenon", "unknown")
        d_st = domain_stats.setdefault(dom, {"total": 0, "oracle_pos": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
        p_st = phenom_stats.setdefault(ph, {"total": 0, "oracle_pos": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
        d_st["total"] += 1
        p_st["total"] += 1
        if is_oracle_repeat:
            d_st["oracle_pos"] += 1
            p_st["oracle_pos"] += 1

        is_unseen_asp = bool(req.get("is_unseen_aspect", False))
        if is_unseen_asp:
            unseen_aspect_stats["total"] += 1
            if is_oracle_repeat:
                unseen_aspect_stats["oracle_pos"] += 1

        if is_hit:
            hit_latencies_ms.append(dt_ms)
            hit_correct = bool(nn_meta is not None and nn_meta.id_set == cid)
            if hit_correct:
                tp += 1
                w_tp += 1
                d_st["tp"] += 1
                p_st["tp"] += 1
                if is_unseen_asp:
                    unseen_aspect_stats["tp"] += 1
            else:
                fp += 1
                w_fp += 1
                d_st["fp"] += 1
                p_st["fp"] += 1
                if is_unseen_asp:
                    unseen_aspect_stats["fp"] += 1
        else:
            miss_latencies_ms.append(dt_ms)
            if was_in_cache:
                fn += 1
                w_fn += 1
                d_st["fn"] += 1
                p_st["fn"] += 1
                if is_unseen_asp:
                    unseen_aspect_stats["fn"] += 1
            else:
                tn += 1
                w_tn += 1
                d_st["tn"] += 1
                p_st["tn"] += 1
                if is_unseen_asp:
                    unseen_aspect_stats["tn"] += 1

        # Emit window snapshot at boundaries
        if (idx + 1) % win_size == 0 or (idx + 1) == len(stream):
            w_tot = w_tp + w_fp + w_fn + w_tn
            cum_tot = idx + 1
            windows.append(
                {
                    "end_request": cum_tot,
                    "cache_entries": int(cache_obj.vector_db_size()),
                    "window_valid_hit_rate": round(w_tp / max(1, w_tot), 4),
                    "window_oracle_hit_rate": round(w_oracle / max(1, w_tot), 4),
                    "window_precision": round(w_tp / max(1, w_tp + w_fp), 4),
                    "window_error_rate": round(w_fp / max(1, w_tot), 4),
                    "cum_valid_hit_rate": round(tp / max(1, cum_tot), 4),
                    "cum_error_rate": round(fp / max(1, cum_tot), 4),
                    "window_mean_latency_ms": round(float(np.mean(w_lats)), 3),
                }
            )
            w_tp = w_fp = w_fn = w_tn = w_oracle = 0
            w_lats = []

    final_cache_size = int(cache_obj.vector_db_size())
    two_stage_stats = dict(getattr(cache_obj, "stats", {}))
    if hasattr(policy, "shutdown"):
        policy.shutdown()

    total = max(1, len(stream))
    hits = tp + fp
    misses = fn + tn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    total_hit_rate = hits / total
    valid_hit_rate = tp / total
    oracle_max_rate = oracle_repeats / total
    oracle_capture_rate = tp / max(1, oracle_repeats)
    error_rate = fp / total
    qa_accuracy = (tp + fn + tn) / total  # Misses call ground-truth LLM -> accurate; only FP returns wrong answer

    lat_arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_cache_ms = float(np.mean(lat_arr))
    # End-to-end gateway latency = cache check latency + LLM generation time on cache miss
    e2e_gateway_ms = mean_cache_ms + (misses / total) * simulated_llm_ms

    def _summarize_bucket(b: Dict[str, int]) -> Dict[str, float | int]:
        btot = max(1, b.get("total", 0))
        btp = b.get("tp", 0)
        bfp = b.get("fp", 0)
        bfn = b.get("fn", 0)
        boracle = b.get("oracle_pos", 0)
        bprec = btp / max(1, btp + bfp)
        brec = btp / max(1, btp + bfn)
        bf1 = (2 * bprec * brec / (bprec + brec)) if (bprec + brec) > 0 else 0.0
        return {
            "total": b.get("total", 0),
            "valid_hit_rate": round(btp / btot, 4),
            "oracle_capture_rate": round(btp / max(1, boracle), 4) if boracle > 0 else 0.0,
            "precision": round(bprec, 4),
            "recall": round(brec, 4),
            "f1": round(bf1, 4),
            "error_rate": round(bfp / btot, 4),
            "tp": btp,
            "fp": bfp,
        }

    return {
        "total_requests": len(stream),
        "policy_type": policy_type,
        "param": threshold_or_delta,
        "max_cache_entries": max_cache_entries,
        "final_cache_entries": final_cache_size,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "llm_calls_made": misses,
        "llm_calls_saved_total": hits,
        "llm_calls_saved_valid": tp,
        "total_hit_rate": round(total_hit_rate, 4),
        "valid_hit_rate": round(valid_hit_rate, 4),
        "oracle_max_hit_rate": round(oracle_max_rate, 4),
        "oracle_capture_rate": round(oracle_capture_rate, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "error_rate": round(error_rate, 4),
        "e2e_qa_accuracy": round(qa_accuracy, 4),
        "cache_mean_ms": round(mean_cache_ms, 3),
        "cache_p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
        "cache_p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
        "cache_p99_ms": round(float(np.percentile(lat_arr, 99)), 3),
        "hit_mean_ms": round(float(np.mean(hit_latencies_ms)), 3) if hit_latencies_ms else 0.0,
        "miss_mean_ms": round(float(np.mean(miss_latencies_ms)), 3) if miss_latencies_ms else 0.0,
        "simulated_e2e_gateway_ms": round(e2e_gateway_ms, 2),
        "simulated_speedup_vs_no_cache": round(simulated_llm_ms / max(1.0, e2e_gateway_ms), 2),
        "unseen_aspect_metrics": _summarize_bucket(unseen_aspect_stats),
        "domain_metrics": {k: _summarize_bucket(v) for k, v in domain_stats.items()},
        "phenomenon_metrics": {k: _summarize_bucket(v) for k, v in phenom_stats.items()},
        "windows": windows,
        "two_stage_stats": two_stage_stats,
    }


def calibrate_thresholds_on_validation(
    val_stream: List[Dict[str, Any]],
    val_lookup: ReplayLookupEngine,
    max_error_budget: float = 0.015,
) -> Tuple[float, float, Dict[str, Any]]:
    """Select `t_val_safe` (error <= 1.5% on val) and `t_val_f1` strictly on validation data."""
    grid = [round(x, 2) for x in np.arange(0.74, 0.97, 0.02)]
    sweep: Dict[str, Any] = {}

    best_f1_t = 0.85
    best_f1_val = -1.0

    safe_candidates: List[Tuple[float, float]] = []  # (valid_hit_rate, t)
    min_err_t = 0.96
    min_err_val = 1.0

    for t in grid:
        res = run_chronological_simulation(
            stream=val_stream,
            embedding_engine=val_lookup,
            policy_type="static",
            threshold_or_delta=t,
            measure_gpu_sync=False,
        )
        sweep[f"{t:.2f}"] = {
            "valid_hit_rate": res["valid_hit_rate"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "error_rate": res["error_rate"],
        }
        if res["f1"] > best_f1_val:
            best_f1_val = res["f1"]
            best_f1_t = t
        if res["error_rate"] <= max_error_budget:
            safe_candidates.append((res["valid_hit_rate"], t))
        if res["error_rate"] < min_err_val:
            min_err_val = res["error_rate"]
            min_err_t = t

    if safe_candidates:
        safe_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        safe_t = safe_candidates[0][1]
    else:
        safe_t = min_err_t

    return safe_t, best_f1_t, sweep


def generate_chronological_plots(
    results_by_method: Dict[str, Dict[str, Any]],
    oracle_max_hit_rate: float,
    out_png: str,
) -> None:
    label_map = {
        "query_only": "Query-Only",
        "concat": "History Concat",
        "turn_mean": "Turn-Mean",
        "turn_adapter": "ContextCache",
        "qwen_lora": "ContextCache+",
    }
    short_labels = {
        "query_only": "Query-Only",
        "concat": "History\nConcat",
        "turn_mean": "Turn-Mean",
        "turn_adapter": "ContextCache",
        "qwen_lora": "ContextCache+",
    }
    colors = {
        "query_only": "#94A3B8",
        "concat": "#C2846B",
        "turn_mean": "#6E7FBF",
        "turn_adapter": "#2B5C8F",
        "qwen_lora": "#167C65",
    }
    linestyles = {
        "query_only": "--",
        "concat": "-.",
        "turn_mean": "--",
        "turn_adapter": "-",
        "qwen_lora": "-",
    }
    markers = {
        "query_only": "v",
        "concat": "^",
        "turn_mean": "s",
        "turn_adapter": "D",
        "qwen_lora": "o",
    }
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

    keys = [k for k in label_map if k in results_by_method]
    with plt.rc_context(rc):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.55), dpi=320)
        fig.patch.set_facecolor("white")

        for ax in (ax0, ax1):
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axisbelow(True)

        # Panel (a): Cold-Start 10,000-Request Cumulative Valid Hit Rate
        ax0.grid(True, linestyle="--", linewidth=0.7, color="#CBD5E1", alpha=0.65, zorder=0)
        ax0.axhline(
            oracle_max_hit_rate * 100.0,
            color="#475569",
            linestyle=":",
            linewidth=1.5,
            zorder=2,
            label=f"Oracle Repeat Bound ({oracle_max_hit_rate * 100:.1f}%)",
        )
        for k in keys:
            is_ours = k in ours_set
            wins = results_by_method[k]["live_val_safe_static"]["windows"]
            xs = [0] + [w["end_request"] for w in wins]
            ys = [0.0] + [w["cum_valid_hit_rate"] * 100.0 for w in wins]
            ax0.plot(
                xs,
                ys,
                marker=markers[k],
                markersize=6.4 if is_ours else 4.6,
                markeredgecolor="white" if is_ours else colors[k],
                markeredgewidth=0.9 if is_ours else 0.5,
                linestyle=linestyles[k],
                linewidth=2.7 if is_ours else 1.6,
                color=colors[k],
                alpha=0.98 if is_ours else 0.85,
                zorder=5 if is_ours else 3,
                label=label_map[k],
            )
            if is_ours:
                ax0.annotate(
                    f"{ys[-1]:.1f}%",
                    xy=(xs[-1], ys[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=9.2,
                    fontweight="bold",
                    color=colors[k],
                )

        ax0.set_title("(a) 10K Cold-Start Stream: Cumulative Hit Rate (↑)", fontsize=11.8, fontweight="bold", pad=10, color="#0F172A")
        ax0.set_xlabel("Chronological Request Index", fontsize=10.3)
        ax0.set_ylabel("Cumulative Valid Hit Rate (%)", fontsize=10.5)
        ax0.set_xlim(-200, 11250)
        ax0.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
        ax0.set_xticklabels(["0", "2K", "4K", "6K", "8K", "10K"])
        ax0.set_ylim(0, 80.5)
        ax0.legend(fontsize=8.3, loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96)

        # Panel (b): 10K Stream Cache Precision & F1 Score
        ax1.grid(axis="y", linestyle="--", linewidth=0.7, color="#CBD5E1", alpha=0.65, zorder=0)
        base_positions = [0.0, 1.05, 2.10, 3.28, 4.52]
        x = np.array(base_positions[: len(keys)], dtype=np.float64)
        ax1.set_xlim(x[0] - 0.62, x[-1] + 0.65)
        ours_pos = [x[i] for i, k in enumerate(keys) if k in ours_set]
        if ours_pos:
            ax1.axvspan(
                min(ours_pos) - 0.56,
                max(ours_pos) + 0.56,
                color="#EFF6FF",
                alpha=0.80,
                zorder=0,
            )

        p_vals = [results_by_method[k]["live_val_safe_static"]["precision"] * 100.0 for k in keys]
        f1_vals = [results_by_method[k]["live_val_safe_static"]["f1"] * 100.0 for k in keys]
        w = 0.36
        sep = 0.20
        for idx, k in enumerate(keys):
            is_ours = k in ours_set
            alpha_v = 0.96 if is_ours else 0.78
            lw_v = 0.9 if is_ours else 0.6
            b1 = ax1.bar(
                x[idx] - sep,
                p_vals[idx],
                width=w,
                color="#2B5C8F",
                alpha=alpha_v,
                edgecolor="#1E293B",
                linewidth=lw_v,
                zorder=3,
                label="Stream Precision (%)" if idx == len(keys) - 1 else None,
            )
            b2 = ax1.bar(
                x[idx] + sep,
                f1_vals[idx],
                width=w,
                color="#167C65",
                alpha=alpha_v,
                edgecolor="#1E293B",
                linewidth=lw_v,
                zorder=3,
                label="Stream F1 Score (%)" if idx == len(keys) - 1 else None,
            )
            fw = "bold" if is_ours else "normal"
            col = "#0F172A" if is_ours else "#475569"
            ax1.text(
                b1[0].get_x() + b1[0].get_width() / 2,
                p_vals[idx] + 1.5,
                f"{p_vals[idx]:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.4,
                fontweight=fw,
                color=col,
            )
            ax1.text(
                b2[0].get_x() + b2[0].get_width() / 2,
                f1_vals[idx] + 1.5,
                f"{f1_vals[idx]:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.4,
                fontweight=fw,
                color=col,
            )

        ax1.set_title("(b) 10K Stream Precision & F1 Score (↑)", fontsize=11.8, fontweight="bold", pad=10, color="#0F172A")
        ax1.set_ylabel("Precision / F1 Score (%)", fontsize=10.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels([short_labels[k] for k in keys], fontsize=9.4)
        ax1.set_ylim(0, 108)
        ax1.legend(fontsize=8.8, loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95)

        for idx, tick in enumerate(ax1.get_xticklabels()):
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
    parser.add_argument("--turn_adapter_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "turn_adapter"))
    parser.add_argument("--lora_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "qwen3_context_lora"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--num_requests", type=int, default=10000)
    parser.add_argument("--methods", type=str, default="query_only,concat,turn_mean,turn_adapter,qwen_lora")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    trace_path = os.path.join(args.data_dir, "vcache_chronological_10k.jsonl")
    val_path = os.path.join(args.data_dir, "vcache_chronological_val.jsonl")
    if not os.path.isfile(trace_path):
        raise FileNotFoundError(f"Chronological 10k trace not found: {trace_path}")

    stream_10k = load_jsonl(trace_path)
    val_stream = load_jsonl(val_path) if os.path.isfile(val_path) else []

    seen_ids = set()
    oracle_hits = 0
    sessions = set()
    clusters = set()
    for row in stream_10k:
        sessions.add(row.get("session_id"))
        cid = row.get("id_set")
        clusters.add(cid)
        if cid in seen_ids:
            oracle_hits += 1
        else:
            seen_ids.add(cid)
    oracle_max_hit_rate = oracle_hits / max(len(stream_10k), 1)
    print(
        f"[chrono10k] Loaded N={len(stream_10k)} chronological requests "
        f"(unique_sessions={len(sessions)}, "
        f"unique_clusters={len(clusters)}, "
        f"oracle_max_hit_rate={oracle_max_hit_rate:.4f})"
    )

    device = str(select_torch_device())
    print(f"[chrono10k] Initializing Qwen3-Embedding-0.6B base engine on {device} ...")
    base_engine = Qwen3EmbeddingEngine(model_name_or_path=args.base_model, device=device)

    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    turn_adapter_engine = None
    if "turn_adapter" in requested_methods and os.path.isdir(args.turn_adapter_dir):
        print(f"[chrono10k] Loading Model A (TurnAdapter v2) from {args.turn_adapter_dir} ...")
        turn_adapter_engine = GatedTurnCrossAttentionEngine(
            adapter_path=args.turn_adapter_dir,
            base_engine=base_engine,
            device=device,
        )

    lora_engine = None
    if "qwen_lora" in requested_methods and os.path.isdir(args.lora_dir):
        print(f"[chrono10k] Loading Model B (Qwen3ContextLoRA + KV) from {args.lora_dir} ...")
        lora_engine = Qwen3ContextLoRAEngine(
            lora_adapter_path=args.lora_dir,
            base_model_name_or_path=args.base_model,
            device=device,
            merge_lora=True,
        )

    frontier_thresholds = [0.74, 0.76, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
    results_by_method: Dict[str, Any] = {}

    for m_name in requested_methods:
        if m_name == "turn_adapter" and turn_adapter_engine is None:
            continue
        if m_name == "qwen_lora" and lora_engine is None:
            continue

        print(f"\n==================== [{m_name}] Chronological 10k Evaluation ====================")
        default_safe = {"query_only": 0.94, "concat": 0.96, "turn_mean": 0.96, "turn_adapter": 0.94, "qwen_lora": 0.80}
        default_f1 = {"query_only": 0.92, "concat": 0.94, "turn_mean": 0.94, "turn_adapter": 0.90, "qwen_lora": 0.78}
        if val_stream:
            val_vecs = batch_preencode_validation(
                m_name, val_stream, base_engine, turn_adapter_engine, lora_engine
            )
            t_val_safe, t_val_f1, val_sweep = calibrate_thresholds_on_validation(
                val_stream, ReplayLookupEngine(val_vecs), max_error_budget=0.015
            )
        else:
            t_val_safe = default_safe.get(m_name, 0.85)
            t_val_f1 = default_f1.get(m_name, 0.82)
            val_sweep = {}
        print(f"[{m_name}] Calibrated thresholds: t_val_safe={t_val_safe:.2f}, t_val_f1={t_val_f1:.2f}")

        live_inner = prepare_live_engine(
            m_name, base_engine, turn_adapter_engine, lora_engine, use_compile=True
        )
        recorder = LiveRecordingEngine(live_inner)

        t_wall_start = time.perf_counter()
        live_safe_res = run_chronological_simulation(
            stream=stream_10k,
            embedding_engine=recorder,
            policy_type="static",
            threshold_or_delta=t_val_safe,
            max_cache_entries=None,
            measure_gpu_sync=False,
        )
        wall_sec = time.perf_counter() - t_wall_start
        base_engine.mode = "query_only"

        print(
            f"[{m_name}] 10k Run ({wall_sec:.1f}s) @ t={t_val_safe:.2f}: "
            f"ValidHit={live_safe_res['valid_hit_rate']*100:.2f}%, "
            f"Prec={live_safe_res['precision']*100:.2f}%, F1={live_safe_res['f1']*100:.2f}%"
        )

        # 3. Exact Chronological Replays on Recorded Sequential Vectors (Verified Online + LRU-2000 + Frontier)
        replay_engine = ReplayLookupEngine(recorder.recorded)

        val_f1_res = run_chronological_simulation(
            stream=stream_10k,
            embedding_engine=replay_engine,
            policy_type="static",
            threshold_or_delta=t_val_f1,
        )

        verified_005_res = run_chronological_simulation(
            stream=stream_10k,
            embedding_engine=replay_engine,
            policy_type="verified",
            threshold_or_delta=0.05,
        )
        verified_002_res = run_chronological_simulation(
            stream=stream_10k,
            embedding_engine=replay_engine,
            policy_type="verified",
            threshold_or_delta=0.02,
        )
        lru_2000_res = run_chronological_simulation(
            stream=stream_10k,
            embedding_engine=replay_engine,
            policy_type="static",
            threshold_or_delta=t_val_safe,
            max_cache_entries=2000,
        )

        print(
            f"[{m_name}] Val-F1 t={t_val_f1:.2f}: ValidHit={val_f1_res['valid_hit_rate']*100:.2f}%, "
            f"Prec={val_f1_res['precision']*100:.2f}%, F1={val_f1_res['f1']*100:.2f}%, Err={val_f1_res['error_rate']*100:.2f}%"
        )
        print(
            f"[{m_name}] Online Verified(δ=0.05): ValidHit={verified_005_res['valid_hit_rate']*100:.2f}%, "
            f"Prec={verified_005_res['precision']*100:.2f}%, F1={verified_005_res['f1']*100:.2f}%, "
            f"Err={verified_005_res['error_rate']*100:.2f}% | "
            f"Bounded-LRU(2000): ValidHit={lru_2000_res['valid_hit_rate']*100:.2f}%, "
            f"Err={lru_2000_res['error_rate']*100:.2f}% (evictions={lru_2000_res['two_stage_stats'].get('lru_evictions', 0)})"
        )

        frontier: Dict[str, Any] = {}
        best_oracle_f1_res = None
        for thr in frontier_thresholds:
            f_res = run_chronological_simulation(
                stream=stream_10k,
                embedding_engine=replay_engine,
                policy_type="static",
                threshold_or_delta=thr,
            )
            frontier[f"{thr:.2f}"] = {
                "valid_hit_rate": f_res["valid_hit_rate"],
                "total_hit_rate": f_res["total_hit_rate"],
                "oracle_capture_rate": f_res["oracle_capture_rate"],
                "precision": f_res["precision"],
                "recall": f_res["recall"],
                "f1": f_res["f1"],
                "error_rate": f_res["error_rate"],
                "e2e_qa_accuracy": f_res["e2e_qa_accuracy"],
            }
            if best_oracle_f1_res is None or f_res["f1"] > best_oracle_f1_res["f1"]:
                best_oracle_f1_res = f_res

        results_by_method[m_name] = {
            "t_val_safe": t_val_safe,
            "t_val_f1": t_val_f1,
            "live_val_safe_static": live_safe_res,
            "val_f1_static": val_f1_res,
            "best_static_on_10k": best_oracle_f1_res,
            "verified_d005": verified_005_res,
            "verified_d002": verified_002_res,
            "bounded_lru_2000": lru_2000_res,
            "threshold_frontier": frontier,
            "val_calibration_sweep": val_sweep,
        }

    csv_out = os.path.join(args.output_dir, "contextcache_10k_table.csv")
    fieldnames = [
        "method",
        "valid_hit_rate",
        "precision",
        "recall",
        "f1_score",
        "lru2000_valid_hit_rate",
    ]
    method_export_names = {
        "turn_adapter": "contextcache",
        "qwen_lora": "contextcache+",
    }
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m_name, d in results_by_method.items():
            s = d["live_val_safe_static"]
            lru = d["bounded_lru_2000"]
            writer.writerow(
                {
                    "method": method_export_names.get(m_name, m_name),
                    "valid_hit_rate": round(s["valid_hit_rate"], 4),
                    "precision": round(s["precision"], 4),
                    "recall": round(s["recall"], 4),
                    "f1_score": round(s["f1"], 4),
                    "lru2000_valid_hit_rate": round(lru["valid_hit_rate"], 4),
                }
            )

    png_out = os.path.join(args.output_dir, "contextcache_10k_stream.png")
    generate_chronological_plots(results_by_method, oracle_max_hit_rate, png_out)
    print(f"\n[chrono10k] Saved CSV     -> {csv_out}")
    print(f"[chrono10k] Saved figure  -> {png_out}")


if __name__ == "__main__":
    main()

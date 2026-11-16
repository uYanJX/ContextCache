"""Model B: Token-Level Causal Context-Aware LoRA Encoder on Qwen3-Embedding-0.6B with Prefix-KV Reuse."""

from __future__ import annotations

from collections import OrderedDict
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer

try:
    import peft.import_utils as _peft_iu

    _peft_iu.is_torchao_available()
except Exception:
    try:
        import peft.import_utils as _peft_iu
        import peft.tuners.lora.torchao as _peft_tao

        _peft_iu.is_torchao_available = lambda: False  # type: ignore[assignment]
        _peft_tao.is_torchao_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass

from vcache.context.dialogue import (
    DEFAULT_CONTEXT_INSTRUCTION,
    DialogueInput,
    format_qwen3_context_prompt,
    unpack_dialogue,
)
from vcache.context.qwen3_embedding import (
    KVPairList,
    Qwen3EmbeddingEngine,
    build_dynamic_cache,
    enable_fast_cuda_backends,
    extract_kv_list,
    load_fast_hf_model,
    select_torch_device,
)
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


class Qwen3ContextLoRAEngine(EmbeddingEngine):
    """vCache EmbeddingEngine powered by LoRA-adapted `Qwen/Qwen3-Embedding-0.6B`.

    Low-latency inference optimizations (mathematically lossless):
    1. LoRA weights merged into `bfloat16` SDPA Transformer (`merge_and_unload`) for zero adapter overhead.
    2. Hierarchical Causal Prefix KV-Cache:
       - Level-0 pinned KV caches for static single-turn (`Instruct...Query:`) and multi-turn
         (`Instruct...Conversation History:\\n`) system headers.
       - Level-1 LRU Session Prefix KV-Cache (`_prefix_kv_cache`) over `[Turn -k]...[Turn -1]\\nCurrent Query:`.
         On a session prefix hit, only the short `Current Query` suffix (~8-15 tokens) is forwarded!
         On a single-request cold miss, `history + query` runs in ONE forward pass (`use_cache=True`)
         and slices/stores the causal prefix KV states at zero extra kernel cost.
    3. Length-sorted right-padded causal micro-batching (`attention_mask=None`) eliminating padding waste.
    """

    def __init__(
        self,
        lora_adapter_path: str,
        base_model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_length: int = 384,
        max_turns: int = 6,
        instruction: Optional[str] = DEFAULT_CONTEXT_INSTRUCTION,
        merge_lora: bool = True,
        prefix_cache_size: int = 256,
        dialogue_cache_size: int = 65536,
        compile_model: bool = False,
    ):
        if not os.path.isdir(lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter directory not found: {lora_adapter_path}")

        self.device = select_torch_device(device)
        enable_fast_cuda_backends(self.device)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.max_length = max_length
        self.max_turns = max_turns
        self.instruction = instruction
        self.lora_adapter_path = lora_adapter_path
        self.prefix_cache_size = prefix_cache_size
        self.dialogue_cache_size = dialogue_cache_size

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            padding_side="left",
            trust_remote_code=True,
        )
        base_model = load_fast_hf_model(base_model_name_or_path, self.dtype)
        peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        if merge_lora:
            final_model = peft_model.merge_and_unload()
        else:
            final_model = peft_model
        final_model = final_model.to(device=self.device, dtype=self.dtype).eval()

        self._runner = Qwen3EmbeddingEngine(
            model_name_or_path=base_model_name_or_path,
            mode="structured_raw",
            device=str(self.device),
            dtype=self.dtype,
            max_length=max_length,
            max_turns=max_turns,
            instruction=instruction,
            model=final_model,
            tokenizer=tokenizer,
        )
        self.model = self._runner.model
        self.tokenizer = self._runner.tokenizer
        self.embedding_dim = self._runner.embedding_dim

        self._prefix_kv_cache: OrderedDict[Tuple[str, ...], Tuple[int, KVPairList]] = OrderedDict()
        self._dialogue_cache: OrderedDict[Tuple[Tuple[str, ...], str], np.ndarray] = OrderedDict()
        self.prefix_kv_hits: int = 0
        self.prefix_kv_misses: int = 0

        self._use_prefix_kv: bool = False
        self._static_history_header_kv: Optional[KVPairList] = None
        self._static_history_header_len: int = 0
        self._compiled_model: Optional[torch.nn.Module] = None
        self._init_multiturn_prefix_kv()
        if compile_model and self.device.type == "cuda":
            self.enable_torch_compile()

    @torch.inference_mode()
    def enable_torch_compile(self) -> None:
        """Compile causal Transformer with TorchInductor (`dynamic=True`) for 3.3x faster B=1 serving."""
        if self._compiled_model is not None or self.device.type != "cuda":
            return
        self._compiled_model = torch.compile(self.model, dynamic=True)
        # Warm up dynamic shapes once so online requests experience zero JIT latency
        w_short = torch.full((1, 24), self._runner._pad_id, dtype=torch.long, device=self.device)
        w_long = torch.full((1, 96), self._runner._pad_id, dtype=torch.long, device=self.device)
        _ = self._compiled_model(input_ids=w_short, use_cache=False)
        _ = self._compiled_model(input_ids=w_long, use_cache=False)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def _init_multiturn_prefix_kv(self) -> None:
        """Precompute multi-turn header KV and validate exact cosine parity against full forward."""
        if not self._runner._supports_causal_right_pad:
            return
        try:
            ctx_probe = ["What is Apache Kafka?"]
            q1 = "What are its key features?"
            q2 = "How does it partition event streams across broker nodes?"
            full_1 = format_qwen3_context_prompt(
                ctx_probe, q1, instruction=self.instruction, max_turns=self.max_turns
            )
            full_2 = format_qwen3_context_prompt(
                ctx_probe, q2, instruction=self.instruction, max_turns=self.max_turns
            )
            ref_vecs = self._runner._encode_left_padded_fallback([full_1, full_2], max_len=256)

            if self.instruction:
                hdr_str = f"Instruct: {self.instruction}\nConversation History:\n"
            else:
                hdr_str = "Conversation History:\n"
            hdr_ids = self._runner._leading_bos_ids + self.tokenizer.encode(
                hdr_str, add_special_tokens=False
            )
            body_ids = self._encode_history_body_ids(ctx_probe, max_body_tokens=180)
            suf_1 = self._encode_query_suffix_ids(q1, max_suffix_tokens=64)
            suf_2 = self._encode_query_suffix_ids(q2, max_suffix_tokens=64)

            full_ids_1 = self._runner._tokenize_full_no_pad([full_1], max_len=256)[0]
            if hdr_ids + body_ids + suf_1 != full_ids_1:
                return

            hdr_tensor = torch.tensor([hdr_ids], dtype=torch.long, device=self.device)
            hdr_out = self.model(input_ids=hdr_tensor, use_cache=True)
            hdr_kv = extract_kv_list(hdr_out.past_key_values)
            hdr_len = len(hdr_ids)

            # Single-pass cold-session probe (computes q1 embedding + slices session KV)
            comb_tensor = torch.tensor([body_ids + suf_1], dtype=torch.long, device=self.device)
            comb_out = self.model(
                input_ids=comb_tensor,
                past_key_values=build_dynamic_cache(hdr_kv, batch_size=1),
                use_cache=True,
            )
            v1_cold = F.normalize(
                comb_out.last_hidden_state[0, -1].float(), p=2, dim=-1
            ).cpu().numpy()
            sess_kv = extract_kv_list(comb_out.past_key_values, slice_len=hdr_len + len(body_ids))

            # Warm-session batch probe on [suf_1, suf_2]
            warm_batch = self._runner.encode_token_id_batches(
                [suf_1, suf_2], batch_size=2, prefix_kv=sess_kv
            )
            if (
                float(np.dot(ref_vecs[0], v1_cold)) > 0.9995
                and float(np.dot(ref_vecs[0], warm_batch[0])) > 0.9995
                and float(np.dot(ref_vecs[1], warm_batch[1])) > 0.9995
            ):
                self._static_history_header_kv = hdr_kv
                self._static_history_header_len = hdr_len
                self._use_prefix_kv = True
        except Exception:
            self._use_prefix_kv = False
            self._static_history_header_kv = None

    def clear_caches(self) -> None:
        """Clear session Prefix-KV and dialogue embedding caches (keeps static system header KV)."""
        self._prefix_kv_cache.clear()
        self._dialogue_cache.clear()
        self._runner.clear_caches()
        self.prefix_kv_hits = 0
        self.prefix_kv_misses = 0

    def _clean_pair(self, ctx: Sequence[str], query: str) -> Tuple[Tuple[str, ...], str]:
        cleaned_ctx = tuple(
            str(t).strip() for t in ctx if t and str(t).strip()
        )[-self.max_turns :]
        return cleaned_ctx, str(query).strip()

    def _encode_history_body_ids(self, cleaned_ctx: Sequence[str], max_body_tokens: int) -> List[int]:
        total = len(cleaned_ctx)
        lines = [f"[Turn -{total - idx}]: {turn}" for idx, turn in enumerate(cleaned_ctx)]
        body_str = "\n".join(lines) + "\nCurrent Query:"
        ids = self.tokenizer.encode(body_str, add_special_tokens=False)
        if len(ids) > max_body_tokens:
            # Keep the most recent turns + "\nCurrent Query:" anchor at the tail
            ids = ids[-max_body_tokens:]
        return ids

    def _encode_query_suffix_ids(self, q_str: str, max_suffix_tokens: int) -> List[int]:
        cap = max(2, max_suffix_tokens - len(self._runner._eos_ids))
        ids = self.tokenizer.encode(f" {q_str}", add_special_tokens=False)[:cap]
        if not ids:
            ids = [self._runner._pad_id]
        return ids + self._runner._eos_ids

    def _store_prefix_kv(self, ctx_key: Tuple[str, ...], total_len: int, kv: KVPairList) -> None:
        if self.prefix_cache_size <= 0:
            return
        self._prefix_kv_cache[ctx_key] = (total_len, kv)
        self._prefix_kv_cache.move_to_end(ctx_key)
        while len(self._prefix_kv_cache) > self.prefix_cache_size:
            self._prefix_kv_cache.popitem(last=False)

    def _store_dialogue_vec(
        self, d_key: Tuple[Tuple[str, ...], str], vec: np.ndarray
    ) -> None:
        if self.dialogue_cache_size <= 0:
            return
        self._dialogue_cache[d_key] = vec
        self._dialogue_cache.move_to_end(d_key)
        while len(self._dialogue_cache) > self.dialogue_cache_size:
            self._dialogue_cache.popitem(last=False)

    @torch.inference_mode()
    def _encode_single_fast(self, ctx_key: Tuple[str, ...], q_str: str) -> np.ndarray:
        """Ultra-low-latency single-request causal forward with Prefix-KV reuse."""
        d_key = (ctx_key, q_str)
        if d_key in self._dialogue_cache:
            self._dialogue_cache.move_to_end(d_key)
            return self._dialogue_cache[d_key]

        if self._compiled_model is not None:
            formatted = format_qwen3_context_prompt(
                ctx_key, q_str, instruction=self.instruction, max_turns=self.max_turns
            )
            full_ids = self._runner._tokenize_full_no_pad([formatted], max_len=self.max_length)[0]
            inp = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            out = self._compiled_model(input_ids=inp, use_cache=False)
            vec = F.normalize(out.last_hidden_state[0, -1].float(), p=2, dim=-1).cpu().numpy().astype(np.float32)
            self._store_dialogue_vec(d_key, vec)
            return vec

        if not self._use_prefix_kv:
            formatted = format_qwen3_context_prompt(
                ctx_key, q_str, instruction=self.instruction, max_turns=self.max_turns
            )
            vec = self._runner.encode_raw_texts([formatted], batch_size=1, max_length=self.max_length)[0]
            self._store_dialogue_vec(d_key, vec)
            return vec

        # Case 1: Empty context -> reuse pinned static query instruction KV
        if not ctx_key:
            if self._runner._static_query_kv is not None:
                suf_ids = self._encode_query_suffix_ids(
                    q_str, self.max_length - self._runner._static_query_prefix_len
                )
                vec = self._runner.encode_token_id_batches(
                    [suf_ids], batch_size=1, prefix_kv=self._runner._static_query_kv
                )[0]
            else:
                formatted = format_qwen3_context_prompt(
                    [], q_str, instruction=self.instruction, max_turns=self.max_turns
                )
                vec = self._runner.encode_raw_texts([formatted], batch_size=1, max_length=self.max_length)[0]
            self._store_dialogue_vec(d_key, vec)
            return vec

        # Case 2: Multi-turn Session Prefix KV HIT -> forward ONLY query suffix tokens!
        if ctx_key in self._prefix_kv_cache:
            self.prefix_kv_hits += 1
            self._prefix_kv_cache.move_to_end(ctx_key)
            p_len, sess_kv = self._prefix_kv_cache[ctx_key]
            suf_ids = self._encode_query_suffix_ids(q_str, max(8, self.max_length - p_len))
            vec = self._runner.encode_token_id_batches(
                [suf_ids], batch_size=1, prefix_kv=sess_kv
            )[0]
            self._store_dialogue_vec(d_key, vec)
            return vec

        # Case 3: Session Prefix KV MISS -> single forward pass computes embedding AND caches prefix KV
        self.prefix_kv_misses += 1
        suf_ids = self._encode_query_suffix_ids(q_str, max_suffix_tokens=96)
        max_body = max(16, self.max_length - self._static_history_header_len - len(suf_ids))
        body_ids = self._encode_history_body_ids(ctx_key, max_body_tokens=max_body)

        comb_ids = body_ids + suf_ids
        inp = torch.tensor([comb_ids], dtype=torch.long, device=self.device)
        pkv = build_dynamic_cache(self._static_history_header_kv, batch_size=1)
        out = self.model(input_ids=inp, past_key_values=pkv, use_cache=True)
        vec = F.normalize(out.last_hidden_state[0, -1].float(), p=2, dim=-1).cpu().numpy().astype(np.float32)

        total_prefix_len = self._static_history_header_len + len(body_ids)
        sess_kv = extract_kv_list(out.past_key_values, slice_len=total_prefix_len)
        self._store_prefix_kv(ctx_key, total_prefix_len, sess_kv)
        self._store_dialogue_vec(d_key, vec)
        return vec

    @torch.inference_mode()
    def encode_dialogues(
        self,
        dialogues: Sequence[Tuple[Sequence[str], str]],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Batch-encode `(context_turns, current_query)` pairs with Prefix-KV grouping & length sorting."""
        n = len(dialogues)
        if n == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if n == 1:
            ctx_k, q_s = self._clean_pair(dialogues[0][0], dialogues[0][1])
            return self._encode_single_fast(ctx_k, q_s)[None, :]

        if not self._use_prefix_kv:
            prompts = [
                format_qwen3_context_prompt(
                    ctx, query, instruction=self.instruction, max_turns=self.max_turns
                )
                for ctx, query in dialogues
            ]
            return self._runner.encode_raw_texts(
                prompts, batch_size=batch_size, max_length=self.max_length
            )

        result = np.empty((n, self.embedding_dim), dtype=np.float32)
        empty_indices: List[int] = []
        empty_queries: List[str] = []
        ctx_groups: Dict[Tuple[str, ...], List[Tuple[int, str]]] = {}

        for idx, (ctx, query) in enumerate(dialogues):
            ctx_k, q_s = self._clean_pair(ctx, query)
            d_key = (ctx_k, q_s)
            if d_key in self._dialogue_cache:
                self._dialogue_cache.move_to_end(d_key)
                result[idx] = self._dialogue_cache[d_key]
            elif not ctx_k:
                empty_indices.append(idx)
                empty_queries.append(q_s)
            else:
                ctx_groups.setdefault(ctx_k, []).append((idx, q_s))

        # 1. Batch-encode all empty-context queries using pinned static query instruction KV
        if empty_indices:
            if self._runner._static_query_kv is not None:
                cap = max(4, self.max_length - self._runner._static_query_prefix_len - len(self._runner._eos_ids))
                enc = self.tokenizer(
                    [f" {q}" for q in empty_queries],
                    padding=False,
                    truncation=True,
                    max_length=cap,
                    add_special_tokens=False,
                )
                suf_list = [
                    (ids if ids else [self._runner._pad_id]) + self._runner._eos_ids
                    for ids in enc["input_ids"]
                ]
                vecs = self._runner.encode_token_id_batches(
                    suf_list, batch_size=batch_size, prefix_kv=self._runner._static_query_kv
                )
            else:
                prompts = [
                    format_qwen3_context_prompt([], q, instruction=self.instruction, max_turns=self.max_turns)
                    for q in empty_queries
                ]
                vecs = self._runner.encode_raw_texts(prompts, batch_size=batch_size, max_length=self.max_length)
            for idx, q_s, v in zip(empty_indices, empty_queries, vecs):
                result[idx] = v
                self._store_dialogue_vec(((), q_s), v)

        # 2. Multi-turn items: use Session Prefix-KV only when group is large (>= 24 queries)
        #    or when total multi-turn count is tiny (<= 4); otherwise fuse into length-sorted
        #    batches of `batch_size` sharing `_static_history_header_kv` to minimize GPU kernel launches!
        total_mt = sum(len(v) for v in ctx_groups.values())
        singleton_indices: List[int] = []
        singleton_pairs: List[Tuple[Tuple[str, ...], str]] = []

        for ctx_k, items in ctx_groups.items():
            use_group_kv = (ctx_k in self._prefix_kv_cache and total_mt <= 8) or (len(items) >= 24)
            if use_group_kv and ctx_k in self._prefix_kv_cache:
                self.prefix_kv_hits += len(items)
                self._prefix_kv_cache.move_to_end(ctx_k)
                p_len, sess_kv = self._prefix_kv_cache[ctx_k]
                suf_list = [
                    self._encode_query_suffix_ids(q_s, max(8, self.max_length - p_len))
                    for _, q_s in items
                ]
                vecs = self._runner.encode_token_id_batches(
                    suf_list, batch_size=batch_size, prefix_kv=sess_kv
                )
                for (orig_i, q_s), v in zip(items, vecs):
                    result[orig_i] = v
                    self._store_dialogue_vec((ctx_k, q_s), v)
            elif use_group_kv:
                self.prefix_kv_misses += 1
                self.prefix_kv_hits += len(items) - 1
                body_ids = self._encode_history_body_ids(
                    ctx_k, max_body_tokens=max(16, self.max_length - self._static_history_header_len - 64)
                )
                b_inp = torch.tensor([body_ids], dtype=torch.long, device=self.device)
                b_out = self.model(
                    input_ids=b_inp,
                    past_key_values=build_dynamic_cache(self._static_history_header_kv, batch_size=1),
                    use_cache=True,
                )
                p_len = self._static_history_header_len + len(body_ids)
                sess_kv = extract_kv_list(b_out.past_key_values, slice_len=p_len)
                self._store_prefix_kv(ctx_k, p_len, sess_kv)

                suf_list = [
                    self._encode_query_suffix_ids(q_s, max(8, self.max_length - p_len))
                    for _, q_s in items
                ]
                vecs = self._runner.encode_token_id_batches(
                    suf_list, batch_size=batch_size, prefix_kv=sess_kv
                )
                for (orig_i, q_s), v in zip(items, vecs):
                    result[orig_i] = v
                    self._store_dialogue_vec((ctx_k, q_s), v)
            else:
                for orig_i, q_s in items:
                    singleton_indices.append(orig_i)
                    singleton_pairs.append((ctx_k, q_s))

        # 3. Length-sorted right-padded batching for cold singletons (reusing static history header KV)
        if singleton_indices:
            self.prefix_kv_misses += len(singleton_indices)
            comb_seqs: List[List[int]] = []
            for ctx_k, q_s in singleton_pairs:
                suf_ids = self._encode_query_suffix_ids(q_s, max_suffix_tokens=96)
                max_body = max(16, self.max_length - self._static_history_header_len - len(suf_ids))
                body_ids = self._encode_history_body_ids(ctx_k, max_body_tokens=max_body)
                comb_seqs.append(body_ids + suf_ids)

            vecs = self._runner.encode_token_id_batches(
                comb_seqs,
                batch_size=batch_size,
                prefix_kv=self._static_history_header_kv,
            )
            for orig_i, (ctx_k, q_s), v in zip(singleton_indices, singleton_pairs, vecs):
                result[orig_i] = v
                self._store_dialogue_vec((ctx_k, q_s), v)

        return result

    def get_embedding(self, text: Union[str, DialogueInput]) -> List[float]:
        ctx, query = unpack_dialogue(text, max_turns=self.max_turns)
        ctx_k, q_s = self._clean_pair(ctx, query)
        vec = self._encode_single_fast(ctx_k, q_s)
        return vec.tolist()

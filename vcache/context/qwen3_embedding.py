"""Qwen3-Embedding-0.6B embedding engine with causal Flash/SDPA batching and Prefix-KV reuse."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from vcache.context.dialogue import (
    DEFAULT_CONTEXT_INSTRUCTION,
    DialogueInput,
    format_qwen3_context_prompt,
    format_qwen3_query_prompt,
    unpack_dialogue,
)
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine

QwenBaselineMode = Literal["query_only", "concat", "turn_mean", "sim_turn_mean", "structured_raw"]
KVPairList = List[Tuple[torch.Tensor, torch.Tensor]]


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract the final non-padding token representation (official Qwen3-Embedding pooling)."""
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def select_torch_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def enable_fast_cuda_backends(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True


def load_fast_hf_model(
    model_name_or_path: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Load HuggingFace Transformer preferring fused PyTorch SDPA attention."""
    try:
        return AutoModel.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    except TypeError:
        return AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    except Exception:
        return AutoModel.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            trust_remote_code=True,
        )


def extract_kv_list(past_key_values, slice_len: Optional[int] = None) -> KVPairList:
    """Extract detached `(K, V)` per-layer tensors from HF DynamicCache or tuple."""
    pairs: KVPairList = []
    if (
        hasattr(past_key_values, "layers")
        and len(past_key_values.layers) > 0
        and hasattr(past_key_values.layers[0], "keys")
    ):
        for lyr in past_key_values.layers:
            k = lyr.keys.detach()
            v = lyr.values.detach()
            if slice_len is not None:
                k = k[:, :, :slice_len, :].clone()
                v = v[:, :, :slice_len, :].clone()
            pairs.append((k, v))
    elif hasattr(past_key_values, "key_cache"):
        for k_t, v_t in zip(past_key_values.key_cache, past_key_values.value_cache):
            k = k_t.detach()
            v = v_t.detach()
            if slice_len is not None:
                k = k[:, :, :slice_len, :].clone()
                v = v[:, :, :slice_len, :].clone()
            pairs.append((k, v))
    else:
        for k_t, v_t in past_key_values:
            k = k_t.detach()
            v = v_t.detach()
            if slice_len is not None:
                k = k[:, :, :slice_len, :].clone()
                v = v[:, :, :slice_len, :].clone()
            pairs.append((k, v))
    return pairs


def build_dynamic_cache(kv_list: KVPairList, batch_size: int = 1):
    """Materialize a fresh HF DynamicCache expanded to `batch_size` without mutating cached tensors."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_list):
        if batch_size == 1:
            k_b, v_b = k, v
        else:
            k_b = k.expand(batch_size, -1, -1, -1).contiguous()
            v_b = v.expand(batch_size, -1, -1, -1).contiguous()
        cache.update(k_b, v_b, layer_idx)
    return cache


class Qwen3EmbeddingEngine(EmbeddingEngine):
    """vCache EmbeddingEngine powered by `Qwen/Qwen3-Embedding-0.6B` (1024-d).

    Supports:
    - `mode="query_only"`: embeds only the current query (vanilla single-turn vCache baseline).
    - `mode="concat"`: concatenates historical turns and current query as flat text.
    - `mode="turn_mean"`: embeds each turn individually (with LRU caching) and computes
      exponentially recency-weighted mean pooling.
    - `mode="structured_raw"`: uses structured multi-turn formatting without fine-tuning.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        mode: QwenBaselineMode = "query_only",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_length: int = 512,
        max_turns: int = 6,
        instruction: Optional[str] = DEFAULT_CONTEXT_INSTRUCTION,
        turn_cache_size: int = 65536,
        recency_decay: float = 0.7,
        sim_temperature: float = 0.5,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.mode: QwenBaselineMode = mode
        self.device = select_torch_device(device)
        enable_fast_cuda_backends(self.device)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.max_length = max_length
        self.max_turns = max_turns
        self.instruction = instruction
        self.turn_cache_size = turn_cache_size
        self.recency_decay = recency_decay
        self.sim_temperature = sim_temperature

        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(
                model_name_or_path,
                padding_side="left",
                trust_remote_code=True,
            )
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if model is not None:
            self.model = model
        else:
            self.model = load_fast_hf_model(model_name_or_path, self.dtype).to(self.device)
        self.model.eval()

        # Detect special token behavior (BOS / trailing EOS or EOT added by fast tokenizer)
        probe_special = self.tokenizer.encode("probe", add_special_tokens=True)
        probe_plain = self.tokenizer.encode("probe", add_special_tokens=False)
        has_trailing_special = bool(
            len(probe_special) > len(probe_plain)
            and probe_plain
            and probe_special[-1] != probe_plain[-1]
        )
        self._auto_adds_eos = has_trailing_special
        self._leading_bos_ids: List[int] = (
            [probe_special[0]]
            if (
                len(probe_special) > len(probe_plain)
                and probe_plain
                and probe_special[0] != probe_plain[0]
            )
            else []
        )
        if has_trailing_special:
            self._eos_ids: List[int] = [int(probe_special[-1])]
        elif self.tokenizer.eos_token_id is not None:
            self._eos_ids = [int(self.tokenizer.eos_token_id)]
        else:
            self._eos_ids = []
        self._pad_id: int = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id or 0)
        )

        self.embedding_dim: int = int(getattr(self.model.config, "hidden_size", 1024))
        self._turn_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Verify causal right-padding + static query instruction KV cache equivalence
        self._supports_causal_right_pad: bool = False
        self._static_query_kv: Optional[KVPairList] = None
        self._static_query_prefix_len: int = 0
        self._compiled_model: Optional[torch.nn.Module] = None
        self._init_fast_causal_paths()

    @torch.inference_mode()
    def enable_torch_compile(self) -> None:
        """Compile causal Transformer with TorchInductor (`dynamic=True`) for fast B=1 serving."""
        if self._compiled_model is not None or self.device.type != "cuda":
            return
        self._compiled_model = torch.compile(self.model, dynamic=True)
        w_short = torch.full((1, 24), self._pad_id, dtype=torch.long, device=self.device)
        w_long = torch.full((1, 96), self._pad_id, dtype=torch.long, device=self.device)
        _ = self._compiled_model(input_ids=w_short, use_cache=False)
        _ = self._compiled_model(input_ids=w_long, use_cache=False)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def _init_fast_causal_paths(self) -> None:
        """Probe causal right-pad parity and precompute static instruction KV cache."""
        try:
            t_short = format_qwen3_query_prompt("What is Kafka?", instruction=self.instruction)
            t_long = format_qwen3_query_prompt(
                "Can you explain how multi-version concurrency control works in PostgreSQL?",
                instruction=self.instruction,
            )
            ref_vecs = self._encode_left_padded_fallback([t_short, t_long], max_len=128)
            ref_vec, ref_long = ref_vecs[0], ref_vecs[1]

            ids_short = self._tokenize_full_no_pad([t_short], max_len=128)[0]
            ids_long = self._tokenize_full_no_pad([t_long], max_len=128)[0]
            test_batch = self.encode_token_id_batches([ids_short, ids_long], batch_size=2, prefix_kv=None)
            if (
                float(np.dot(ref_vec, test_batch[0])) > 0.9995
                and float(np.dot(ref_long, test_batch[1])) > 0.9995
            ):
                self._supports_causal_right_pad = True

            if self._supports_causal_right_pad and self.instruction:
                prefix_str = f"Instruct: {self.instruction}\n"
                p_ids = self._leading_bos_ids + self.tokenizer.encode(
                    prefix_str, add_special_tokens=False
                )
                s_short = (
                    self.tokenizer.encode("Query: What is Kafka?", add_special_tokens=False)
                    + self._eos_ids
                )
                s_long = (
                    self.tokenizer.encode(
                        "Query: Can you explain how multi-version concurrency control works in PostgreSQL?",
                        add_special_tokens=False,
                    )
                    + self._eos_ids
                )
                if p_ids + s_short == ids_short and p_ids + s_long == ids_long:
                    p_tensor = torch.tensor([p_ids], dtype=torch.long, device=self.device)
                    p_out = self.model(input_ids=p_tensor, use_cache=True)
                    kv = extract_kv_list(p_out.past_key_values)
                    kv_batch = self.encode_token_id_batches(
                        [s_short, s_long], batch_size=2, prefix_kv=kv
                    )
                    kv_single = self.encode_token_id_batches(
                        [s_short], batch_size=1, prefix_kv=kv
                    )[0]
                    if (
                        float(np.dot(ref_vec, kv_batch[0])) > 0.9995
                        and float(np.dot(ref_long, kv_batch[1])) > 0.9995
                        and float(np.dot(ref_vec, kv_single)) > 0.9995
                    ):
                        self._static_query_kv = kv
                        self._static_query_prefix_len = len(p_ids)
        except Exception:
            self._supports_causal_right_pad = False
            self._static_query_kv = None

    def clear_caches(self) -> None:
        """Clear dynamic utterance LRU cache (retains static instruction prefix KV)."""
        self._turn_cache.clear()

    def _prepare_texts_with_eos(self, texts: Sequence[str]) -> List[str]:
        if self._auto_adds_eos or not self.tokenizer.eos_token:
            return list(texts)
        eos = self.tokenizer.eos_token
        return [t if t.endswith(eos) else f"{t}{eos}" for t in texts]

    def _tokenize_full_no_pad(self, texts: Sequence[str], max_len: int) -> List[List[int]]:
        cap = max(2, max_len - len(self._eos_ids) - len(self._leading_bos_ids))
        enc = self.tokenizer(
            list(texts),
            padding=False,
            truncation=True,
            max_length=cap,
            add_special_tokens=False,
        )
        return [self._leading_bos_ids + ids + self._eos_ids for ids in enc["input_ids"]]

    @torch.inference_mode()
    def _encode_left_padded_fallback(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        max_len: int = 512,
    ) -> np.ndarray:
        prepared = self._prepare_texts_with_eos(texts)
        outputs_list: List[np.ndarray] = []
        for start in range(0, len(prepared), batch_size):
            batch_texts = prepared[start : start + batch_size]
            if self._auto_adds_eos:
                batch_dict = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
            else:
                eos_tok = self.tokenizer.eos_token or ""
                stripped = [
                    t[: -len(eos_tok)] if eos_tok and t.endswith(eos_tok) else t
                    for t in batch_texts
                ]
                encoded = self.tokenizer(
                    stripped,
                    padding=False,
                    truncation=True,
                    max_length=max(2, max_len - 1),
                    add_special_tokens=True,
                )
                eos_id = self.tokenizer.eos_token_id
                seqs = [ids + ([eos_id] if eos_id is not None else []) for ids in encoded["input_ids"]]
                batch_dict = self.tokenizer.pad(
                    {"input_ids": seqs},
                    padding=True,
                    return_tensors="pt",
                )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            model_out = self.model(**batch_dict)
            pooled = last_token_pool(model_out.last_hidden_state, batch_dict["attention_mask"])
            normed = F.normalize(pooled.float(), p=2, dim=-1)
            outputs_list.append(normed.cpu().numpy())
        return np.vstack(outputs_list).astype(np.float32)

    @torch.inference_mode()
    def encode_token_id_batches(
        self,
        seq_ids_list: Sequence[List[int]],
        batch_size: int = 64,
        prefix_kv: Optional[KVPairList] = None,
    ) -> np.ndarray:
        """Length-sorted right-padded causal encoding with optional shared `prefix_kv`."""
        n = len(seq_ids_list)
        if n == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if n == 1:
            seq = seq_ids_list[0]
            inp = torch.tensor([seq], dtype=torch.long, device=self.device)
            pkv = build_dynamic_cache(prefix_kv, batch_size=1) if prefix_kv is not None else None
            out = self.model(input_ids=inp, past_key_values=pkv, use_cache=False)
            vec = F.normalize(out.last_hidden_state[0, -1].float(), p=2, dim=-1)
            return vec.unsqueeze(0).cpu().numpy().astype(np.float32)

        order = np.argsort([len(s) for s in seq_ids_list])
        result = np.empty((n, self.embedding_dim), dtype=np.float32)

        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            bsz = len(batch_idx)
            lens = [len(seq_ids_list[i]) for i in batch_idx]
            max_l = max(lens)

            inp_cpu = torch.full((bsz, max_l), self._pad_id, dtype=torch.long)
            for row_i, orig_i in enumerate(batch_idx):
                s = seq_ids_list[orig_i]
                inp_cpu[row_i, : len(s)] = torch.tensor(s, dtype=torch.long)
            inp = inp_cpu.to(self.device, non_blocking=True)

            pkv = build_dynamic_cache(prefix_kv, batch_size=bsz) if prefix_kv is not None else None
            out = self.model(input_ids=inp, past_key_values=pkv, use_cache=False)
            last_idx = torch.tensor([l - 1 for l in lens], dtype=torch.long, device=self.device)
            pooled = out.last_hidden_state[torch.arange(bsz, device=self.device), last_idx]
            normed = F.normalize(pooled.float(), p=2, dim=-1).cpu().numpy()
            result[batch_idx] = normed

        return result

    @torch.inference_mode()
    def encode_raw_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        """Encode already-formatted text strings into L2-normalized `(N, D)` float32 vectors."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        max_len = max_length or self.max_length
        if not self._supports_causal_right_pad:
            return self._encode_left_padded_fallback(texts, batch_size=batch_size, max_len=max_len)

        if len(texts) == 1 and self._compiled_model is not None:
            seq = self._tokenize_full_no_pad(texts, max_len=max_len)[0]
            inp = torch.tensor([seq], dtype=torch.long, device=self.device)
            out = self._compiled_model(input_ids=inp, use_cache=False)
            vec = F.normalize(out.last_hidden_state[0, -1].float(), p=2, dim=-1)
            return vec.unsqueeze(0).cpu().numpy().astype(np.float32)

        # Fast path when batched prompts share the pinned single-turn instruction header
        if len(texts) >= 8 and self._static_query_kv is not None and self.instruction:
            header = f"Instruct: {self.instruction}\n"
            if all(t.startswith(header) for t in texts):
                suffixes = [t[len(header):] for t in texts]
                cap = max(2, max_len - self._static_query_prefix_len - len(self._eos_ids))
                enc = self.tokenizer(
                    suffixes,
                    padding=False,
                    truncation=True,
                    max_length=cap,
                    add_special_tokens=False,
                )
                suffix_ids = [
                    (ids if ids else [self._pad_id]) + self._eos_ids for ids in enc["input_ids"]
                ]
                return self.encode_token_id_batches(
                    suffix_ids, batch_size=batch_size, prefix_kv=self._static_query_kv
                )

        seq_ids = self._tokenize_full_no_pad(texts, max_len=max_len)
        return self.encode_token_id_batches(seq_ids, batch_size=batch_size, prefix_kv=None)

    def encode_single_turns_cached(
        self,
        turns: Sequence[str],
        use_instruction: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode standalone utterances using an internal LRU cache to avoid redundant forward passes."""
        if not turns:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        keys = [
            format_qwen3_query_prompt(t, instruction=self.instruction if use_instruction else None)
            for t in turns
        ]
        local_lookup = {}
        missing_keys: List[str] = []
        seen_missing = set()
        for k in keys:
            if k in self._turn_cache:
                local_lookup[k] = self._turn_cache[k]
            elif k not in seen_missing:
                missing_keys.append(k)
                seen_missing.add(k)

        if missing_keys:
            computed = self.encode_raw_texts(missing_keys, batch_size=batch_size)
            for k, vec in zip(missing_keys, computed):
                local_lookup[k] = vec
                self._turn_cache[k] = vec
                self._turn_cache.move_to_end(k)
                while len(self._turn_cache) > self.turn_cache_size:
                    self._turn_cache.popitem(last=False)

        result = np.stack([local_lookup[k] for k in keys], axis=0)
        for k in keys:
            if k in self._turn_cache:
                self._turn_cache.move_to_end(k)
        return result.astype(np.float32)

    def get_embedding(self, text: Union[str, DialogueInput]) -> List[float]:
        """Compute a single L2-normalized embedding according to `self.mode`."""
        ctx, query = unpack_dialogue(text, max_turns=self.max_turns)

        if self.mode == "query_only" or not ctx:
            formatted = format_qwen3_query_prompt(query, instruction=self.instruction)
            vec = self.encode_Single_or_Cached(formatted)
            return vec.tolist()

        if self.mode == "concat":
            joined = " | ".join(ctx + [query])
            formatted = format_qwen3_query_prompt(joined, instruction=self.instruction)
            vec = self.encode_Single_or_Cached(formatted)
            return vec.tolist()

        if self.mode == "structured_raw":
            formatted = format_qwen3_context_prompt(
                ctx, query, instruction=self.instruction, max_turns=self.max_turns
            )
            vec = self.encode_Single_or_Cached(formatted)
            return vec.tolist()

        if self.mode == "turn_mean":
            all_turns = ctx + [query]
            turn_vecs = self.encode_single_turns_cached(all_turns, use_instruction=True)
            n = len(all_turns)
            weights = np.array(
                [self.recency_decay ** (n - 1 - i) for i in range(n)],
                dtype=np.float32,
            )[:, None]
            pooled = (turn_vecs * weights).sum(axis=0)
            norm = np.linalg.norm(pooled) + 1e-12
            return (pooled / norm).astype(np.float32).tolist()

        if self.mode == "sim_turn_mean":
            all_turns = ctx + [query]
            turn_vecs = self.encode_single_turns_cached(all_turns, use_instruction=True)
            q_vec = turn_vecs[-1]
            sims = np.dot(turn_vecs, q_vec).astype(np.float32)
            tau = max(float(self.sim_temperature), 1e-6)
            logits = (sims - float(np.max(sims))) / tau
            weights = np.exp(logits).astype(np.float32)
            weights = (weights / (weights.sum() + 1e-12))[:, None]
            pooled = (turn_vecs * weights).sum(axis=0)
            norm = np.linalg.norm(pooled) + 1e-12
            return (pooled / norm).astype(np.float32).tolist()

        raise ValueError(f"Unsupported Qwen3EmbeddingEngine mode: {self.mode}")

    def encode_Single_or_Cached(self, formatted_text: str) -> np.ndarray:
        if formatted_text in self._turn_cache:
            self._turn_cache.move_to_end(formatted_text)
            return self._turn_cache[formatted_text]
        vec = self.encode_raw_texts([formatted_text], batch_size=1)[0]
        self._turn_cache[formatted_text] = vec
        while len(self._turn_cache) > self.turn_cache_size:
            self._turn_cache.popitem(last=False)
        return vec

"""ContextCache: Attention-based turn context integration module over pretrained turn embeddings."""

from __future__ import annotations

from collections import OrderedDict
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from vcache.context.dialogue import DialogueInput, unpack_dialogue
from vcache.context.qwen3_embedding import Qwen3EmbeddingEngine, select_torch_device
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


class SelfAttentionBlock(nn.Module):
    """Pre-LN Multi-Turn Context Self-Attention block for multi-hop entity fusion across history turns."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        ctx_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure rows with all-False ctx_mask never see all-True key_padding_mask (avoids softmax NaN)
        safe_pad_mask = ~ctx_mask.clone()
        empty_rows = ~ctx_mask.any(dim=1)
        if bool(empty_rows.any().item()):
            safe_pad_mask[empty_rows, 0] = False

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=safe_pad_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x * ctx_mask.unsqueeze(-1).to(x.dtype)


class CrossAttentionBlock(nn.Module):
    """Pre-LN Query-Centric Multi-Head Cross-Attention + FFN block."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, attn_weights = self.attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        q = q + attn_out
        q = q + self.ffn(self.norm_ffn(q))
        return q, attn_weights


class GatedTurnCrossAttentionNet(nn.Module):
    """Contextual Representation Network operating on turn-level Qwen3 embeddings.

    Key design features (v2 Dual Erase-and-Write Highway Architecture):
    1. Multi-hop Context Self-Attention fuses entity & assistant turns across `[Turn -k]..[Turn -1]`.
    2. Relative recency embeddings (`1..max_turns`) bind coreferences preferentially to recent turns.
    3. Learnable `null_memory_token` acts as an attention sink when the user switches topics.
    4. Dual Erase-and-Write Highway Gate:
       `z = Normalize((1 - g_erase) * v_q + g_write * delta_c)`
       erases generic pronoun/ellipsis template directions in `v_q` while writing resolved entity
       semantics from `delta_c` (eliminating cross-entity pronoun collisions and anchoring
       contextualized vectors directly onto the standalone Qwen3 manifold).
    5. Exact empty-context identity (`has_ctx == 0 -> F.normalize(v_q)`).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 3,
        max_turns: int = 8,
        dropout: float = 0.1,
        use_isotropic_head: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_turns = max_turns
        self.dropout = dropout
        self.use_isotropic_head = use_isotropic_head

        proj_q_lin = nn.Linear(embed_dim, hidden_dim)
        proj_ctx_lin = nn.Linear(embed_dim, hidden_dim)
        if hidden_dim == embed_dim:
            with torch.no_grad():
                nn.init.eye_(proj_q_lin.weight)
                proj_q_lin.weight.add_(torch.randn_like(proj_q_lin.weight) * 0.02)
                nn.init.zeros_(proj_q_lin.bias)
                nn.init.eye_(proj_ctx_lin.weight)
                proj_ctx_lin.weight.add_(torch.randn_like(proj_ctx_lin.weight) * 0.02)
                nn.init.zeros_(proj_ctx_lin.bias)

        self.in_proj_q = nn.Sequential(proj_q_lin, nn.LayerNorm(hidden_dim))
        self.in_proj_ctx = nn.Sequential(proj_ctx_lin, nn.LayerNorm(hidden_dim))

        self.recency_emb = nn.Embedding(max_turns + 1, hidden_dim, padding_idx=0)
        self.ctx_self_attn = SelfAttentionBlock(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        self.null_memory_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.ctx_out_proj = nn.Linear(hidden_dim, embed_dim)
        self.copy_scale = nn.Parameter(torch.tensor(0.35))

        # Dual gate MLP taking [q_vec, delta_dir, q_vec * delta_dir, |q_vec - delta_dir|] -> (erase, write)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim * 2),
        )
        with torch.no_grad():
            self.gate_net[-1].bias[:embed_dim].fill_(-1.5)
            self.gate_net[-1].bias[embed_dim:].fill_(-0.5)
            nn.init.xavier_uniform_(self.ctx_out_proj.weight, gain=0.15)
            nn.init.zeros_(self.ctx_out_proj.bias)

        if use_isotropic_head:
            self.iso_head: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, embed_dim),
            )
            self.iso_scale = nn.Parameter(torch.tensor(0.5))
            with torch.no_grad():
                nn.init.xavier_uniform_(self.iso_head[-1].weight, gain=0.2)
                nn.init.zeros_(self.iso_head[-1].bias)
        else:
            self.iso_head = None

    def _apply_iso_head(self, vec: torch.Tensor) -> torch.Tensor:
        if self.iso_head is None:
            return vec
        return F.normalize(vec + self.iso_scale * self.iso_head(vec), p=2, dim=-1)

    def forward(
        self,
        q_vec: torch.Tensor,
        ctx_vecs: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute L2-normalized contextualized query embeddings."""
        q_vec = F.normalize(q_vec, p=2, dim=-1)
        bsz = q_vec.shape[0]

        if ctx_vecs is None or ctx_vecs.shape[1] == 0:
            out_q = self._apply_iso_head(q_vec)
            if return_diagnostics:
                zeros_gate = torch.zeros_like(q_vec)
                return out_q, {
                    "gate": zeros_gate,
                    "erase_gate": zeros_gate,
                    "write_gate": zeros_gate,
                    "null_attn": torch.ones(bsz, device=q_vec.device),
                }
            return out_q

        max_m = ctx_vecs.shape[1]
        if ctx_mask is None:
            ctx_mask = torch.ones((bsz, max_m), dtype=torch.bool, device=q_vec.device)

        has_ctx = ctx_mask.any(dim=1, keepdim=True).to(q_vec.dtype)  # (B, 1)
        if not bool(ctx_mask.any().item()):
            out_q = self._apply_iso_head(q_vec)
            if return_diagnostics:
                zeros_gate = torch.zeros_like(q_vec)
                return out_q, {
                    "gate": zeros_gate,
                    "erase_gate": zeros_gate,
                    "write_gate": zeros_gate,
                    "null_attn": torch.ones(bsz, device=q_vec.device),
                }
            return out_q

        # Compute relative distance 1..max_turns from current query for valid left-aligned turns
        lengths = ctx_mask.sum(dim=1, keepdim=True)  # (B, 1)
        positions = torch.arange(max_m, device=q_vec.device).unsqueeze(0)  # (1, M)
        rel_dist = (lengths - positions).clamp(min=1, max=self.max_turns)
        rel_dist = torch.where(ctx_mask, rel_dist, torch.zeros_like(rel_dist))

        ctx_norm = F.normalize(ctx_vecs, p=2, dim=-1)  # (B, M, D)
        q_h = self.in_proj_q(q_vec).unsqueeze(1)  # (B, 1, H)
        ctx_h = self.in_proj_ctx(ctx_norm) + self.recency_emb(rel_dist)
        ctx_h = self.ctx_self_attn(ctx_h, ctx_mask)

        null_tok = self.null_memory_token.expand(bsz, 1, -1)  # (B, 1, H)
        memory = torch.cat([null_tok, ctx_h], dim=1)  # (B, 1 + M, H)

        null_unmasked = torch.zeros((bsz, 1), dtype=torch.bool, device=q_vec.device)
        key_padding_mask = torch.cat([null_unmasked, ~ctx_mask], dim=1)  # (B, 1 + M)

        last_attn = None
        for block in self.blocks:
            q_h, last_attn = block(q_h, memory, key_padding_mask=key_padding_mask)

        assert last_attn is not None
        copy_ctx = torch.bmm(last_attn[:, :, 1:], ctx_norm).squeeze(1)  # (B, D)
        # Project out shared prompt-template cone along q_vec to isolate pure unseen entity semantics
        copy_ctx_orth = copy_ctx - (copy_ctx * q_vec).sum(dim=-1, keepdim=True) * q_vec
        delta_c = self.copy_scale * copy_ctx_orth + self.ctx_out_proj(q_h.squeeze(1))  # (B, D)
        delta_dir = F.normalize(delta_c, p=2, dim=-1)
        gate_in = torch.cat(
            [q_vec, delta_dir, q_vec * delta_dir, (q_vec - delta_dir).abs()],
            dim=-1,
        )
        erase_logits, write_logits = self.gate_net(gate_in).chunk(2, dim=-1)
        erase_gate = torch.sigmoid(erase_logits) * has_ctx
        write_gate = torch.sigmoid(write_logits) * has_ctx

        fused = F.normalize((1.0 - erase_gate) * q_vec + write_gate * delta_c, p=2, dim=-1)
        out = self._apply_iso_head(fused)
        if return_diagnostics:
            null_attn = (
                last_attn[:, 0, 0]
                if last_attn is not None
                else torch.zeros(bsz, device=q_vec.device)
            )
            gate_mean = 0.5 * (erase_gate + write_gate)
            return out, {
                "gate": gate_mean,
                "erase_gate": erase_gate,
                "write_gate": write_gate,
                "fused": fused,
                "null_attn": null_attn,
                "attn_weights": last_attn,
            }
        return out

    def get_config(self) -> Dict[str, int | float | bool]:
        return {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_turns": self.max_turns,
            "dropout": self.dropout,
            "use_isotropic_head": self.use_isotropic_head,
        }

    def save_pretrained(self, save_dir: str, max_shard_bytes: int = 45 * 1024 * 1024) -> None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_config(), f, indent=2)
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        total_bytes = sum(t.numel() * t.element_size() for t in state.values())
        if max_shard_bytes <= 0 or total_bytes <= max_shard_bytes:
            save_file(state, os.path.join(save_dir, "model.safetensors"))
            return

        shards: List[Dict[str, torch.Tensor]] = []
        cur_shard: Dict[str, torch.Tensor] = {}
        cur_bytes = 0
        for k, tensor in state.items():
            t_bytes = tensor.numel() * tensor.element_size()
            if cur_shard and cur_bytes + t_bytes > max_shard_bytes:
                shards.append(cur_shard)
                cur_shard = {}
                cur_bytes = 0
            cur_shard[k] = tensor
            cur_bytes += t_bytes
        if cur_shard:
            shards.append(cur_shard)

        num_shards = len(shards)
        for idx, shard_dict in enumerate(shards, start=1):
            shard_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
            save_file(shard_dict, os.path.join(save_dir, shard_name))

    @staticmethod
    def _has_weights(load_dir: str) -> bool:
        if os.path.isfile(os.path.join(load_dir, "model.safetensors")):
            return True
        return bool(glob.glob(os.path.join(load_dir, "model-*-of-*.safetensors")))

    @classmethod
    def from_pretrained(
        cls,
        load_dir: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "GatedTurnCrossAttentionNet":
        cfg_path = os.path.join(load_dir, "adapter_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        net = cls(**cfg)
        weights_path = os.path.join(load_dir, "model.safetensors")
        if os.path.isfile(weights_path):
            state = load_file(weights_path)
        else:
            shard_paths = sorted(glob.glob(os.path.join(load_dir, "model-*-of-*.safetensors")))
            if not shard_paths:
                raise FileNotFoundError(
                    f"No model.safetensors or sharded safetensors found in {load_dir}"
                )
            state = {}
            for sp in shard_paths:
                state.update(load_file(sp))
        net.load_state_dict(state)
        if device is not None:
            net = net.to(device)
        net.eval()
        return net


class GatedTurnCrossAttentionEngine(EmbeddingEngine):
    """vCache EmbeddingEngine combining cached Qwen3-0.6B turn embeddings with `GatedTurnCrossAttentionNet`."""

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        base_engine: Optional[Qwen3EmbeddingEngine] = None,
        model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        max_turns: int = 6,
        adapter_net: Optional[GatedTurnCrossAttentionNet] = None,
        dialogue_cache_size: int = 65536,
    ):
        self.device = select_torch_device(device)
        self.base_engine = (
            base_engine
            if base_engine is not None
            else Qwen3EmbeddingEngine(
                model_name_or_path=model_name_or_path,
                mode="query_only",
                device=str(self.device),
                max_turns=max_turns,
            )
        )
        self.max_turns = max_turns
        self.dialogue_cache_size = dialogue_cache_size
        self._dialogue_cache: OrderedDict[Tuple[Tuple[str, ...], str], np.ndarray] = OrderedDict()

        if adapter_net is not None:
            self.adapter = adapter_net.to(self.device).eval()
        elif (
            adapter_path is not None
            and os.path.isdir(adapter_path)
            and GatedTurnCrossAttentionNet._has_weights(adapter_path)
        ):
            self.adapter = GatedTurnCrossAttentionNet.from_pretrained(
                adapter_path, device=self.device
            )
        else:
            self.adapter = GatedTurnCrossAttentionNet(
                embed_dim=self.base_engine.embedding_dim,
                max_turns=max(8, max_turns),
            ).to(self.device).eval()

    def clear_caches(self) -> None:
        self._dialogue_cache.clear()
        self.base_engine.clear_caches()

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
    def encode_dialogues(
        self,
        dialogues: Sequence[Tuple[Sequence[str], str]],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Batch-encode `(context_turns, current_query)` pairs with turn-embedding reuse."""
        if not dialogues:
            return np.zeros((0, self.base_engine.embedding_dim), dtype=np.float32)

        unique_utts: List[str] = []
        seen = set()
        clipped_items: List[Tuple[List[str], str]] = []
        for ctx, q in dialogues:
            c_list = [str(t).strip() for t in ctx if t and str(t).strip()][-self.max_turns :]
            q_str = str(q).strip()
            clipped_items.append((c_list, q_str))
            for u in c_list + [q_str]:
                if u not in seen:
                    seen.add(u)
                    unique_utts.append(u)

        utt_vecs = self.base_engine.encode_single_turns_cached(
            unique_utts, use_instruction=True, batch_size=max(batch_size, 128)
        )
        utt_map = {u: vec for u, vec in zip(unique_utts, utt_vecs)}

        dim = self.base_engine.embedding_dim
        results: List[np.ndarray] = []

        for start in range(0, len(clipped_items), batch_size):
            chunk = clipped_items[start : start + batch_size]
            bsz = len(chunk)
            max_m = max((len(c) for c, _ in chunk), default=0)

            q_arr = np.stack([utt_map[q] for _, q in chunk], axis=0)
            q_tensor = torch.from_numpy(q_arr).to(self.device, dtype=torch.float32)

            if max_m == 0:
                out = self.adapter(q_tensor, None, None)
            else:
                ctx_arr = np.zeros((bsz, max_m, dim), dtype=np.float32)
                mask_arr = np.zeros((bsz, max_m), dtype=bool)
                for idx, (c_list, _) in enumerate(chunk):
                    for j, turn_text in enumerate(c_list):
                        ctx_arr[idx, j] = utt_map[turn_text]
                        mask_arr[idx, j] = True
                ctx_tensor = torch.from_numpy(ctx_arr).to(self.device, dtype=torch.float32)
                mask_tensor = torch.from_numpy(mask_arr).to(self.device)
                out = self.adapter(q_tensor, ctx_tensor, mask_tensor)

            out_np = out.cpu().numpy().astype(np.float32)
            for (c_list, q_str), v in zip(chunk, out_np):
                self._store_dialogue_vec((tuple(c_list), q_str), v)
            results.append(out_np)

        return np.vstack(results)

    @torch.inference_mode()
    def get_embedding(self, text: Union[str, DialogueInput]) -> List[float]:
        ctx, query = unpack_dialogue(text, max_turns=self.max_turns)
        c_list = [str(t).strip() for t in ctx if t and str(t).strip()][-self.max_turns :]
        q_str = str(query).strip()
        d_key = (tuple(c_list), q_str)
        if d_key in self._dialogue_cache:
            self._dialogue_cache.move_to_end(d_key)
            return self._dialogue_cache[d_key].tolist()

        if not c_list:
            q_vec = self.base_engine.encode_single_turns_cached([q_str], use_instruction=True, batch_size=1)[0]
            if self.adapter.iso_head is not None:
                q_t = torch.from_numpy(q_vec[None, :]).to(self.device, dtype=torch.float32)
                q_vec = self.adapter(q_t, None, None)[0].cpu().numpy().astype(np.float32)
            self._store_dialogue_vec(d_key, q_vec)
            return q_vec.tolist()

        vec = self.encode_dialogues([(c_list, q_str)], batch_size=16)[0]
        return vec.tolist()

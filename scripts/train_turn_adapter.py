#!/usr/bin/env python3
"""Train Model A (v2): Dual Erase-and-Write + Isotropic Contrastive Turn Adapter over Qwen3-Embedding-0.6B."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vcache.context.qwen3_embedding import Qwen3EmbeddingEngine, select_torch_device
from vcache.context.turn_adapter import GatedTurnCrossAttentionNet


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def collect_unique_utterances(records: Sequence[Dict[str, Any]], max_turns: int) -> List[str]:
    seen = set()
    utts: List[str] = []

    def add_turn(t: str) -> None:
        s = str(t).strip()
        if s and s not in seen:
            seen.add(s)
            utts.append(s)

    for r in records:
        for t in r["anchor_context"][-max_turns:]:
            add_turn(t)
        add_turn(r["anchor_query"])
        for t in r["pos_context"][-max_turns:]:
            add_turn(t)
        add_turn(r["pos_query"])
        for t in r["same_ctx_neg_context"][-max_turns:]:
            add_turn(t)
        add_turn(r["same_ctx_neg_query"])
        for t in r["cross_ctx_neg_context"][-max_turns:]:
            add_turn(t)
        add_turn(r["cross_ctx_neg_query"])
        add_turn(r.get("same_ctx_neg_standalone") or r["same_ctx_neg_query"])
        add_turn(r.get("cross_ctx_neg_standalone") or r["cross_ctx_neg_query"])
    return utts


class TurnTripletDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], utt2idx: Dict[str, int], max_turns: int = 6):
        self.records = records
        self.utt2idx = utt2idx
        self.max_turns = max_turns
        # Build integer semantic cluster IDs so ONLY genuinely identical positive targets are masked
        cluster_map: Dict[str, int] = {}
        self.cluster_ids: List[int] = []
        for r in records:
            if not r["pos_context"]:
                key = str(r["pos_query"]).strip().lower()
            else:
                first_turn = str(r["anchor_context"][0]).strip().lower() if r["anchor_context"] else ""
                key = f"{first_turn}||{str(r['pos_query']).strip().lower()}"
            if key not in cluster_map:
                cluster_map[key] = len(cluster_map)
            self.cluster_ids.append(cluster_map[key])

    def __len__(self) -> int:
        return len(self.records)

    def _encode_pair(self, ctx: Sequence[str], q: str) -> Tuple[List[int], int]:
        c_ids = [
            self.utt2idx[str(t).strip()]
            for t in ctx[-self.max_turns :]
            if str(t).strip() in self.utt2idx
        ]
        q_id = self.utt2idx[str(q).strip()]
        return c_ids, q_id

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        a_ctx, a_q = self._encode_pair(r["anchor_context"], r["anchor_query"])
        p_ctx, p_q = self._encode_pair(r["pos_context"], r["pos_query"])
        n1_ctx, n1_q = self._encode_pair(r["same_ctx_neg_context"], r["same_ctx_neg_query"])
        n2_ctx, n2_q = self._encode_pair(r["cross_ctx_neg_context"], r["cross_ctx_neg_query"])
        n1_sa_str = str(r.get("same_ctx_neg_standalone") or r["same_ctx_neg_query"]).strip()
        n2_sa_str = str(r.get("cross_ctx_neg_standalone") or r["cross_ctx_neg_query"]).strip()
        return {
            "a": (a_ctx, a_q),
            "p": (p_ctx, p_q),
            "n1": (n1_ctx, n1_q),
            "n2": (n2_ctx, n2_q),
            "n1_sa": self.utt2idx[n1_sa_str],
            "n2_sa": self.utt2idx[n2_sa_str],
            "cid": self.cluster_ids[idx],
            "is_switch": bool(r.get("is_topic_switch", False)),
        }


def make_collate_fn(emb_matrix: np.ndarray, max_turns: int):
    dim = emb_matrix.shape[1]

    def pack_group(pairs: List[Tuple[List[int], int]]):
        bsz = len(pairs)
        q_vecs = np.stack([emb_matrix[q_id] for _, q_id in pairs], axis=0)
        max_m = max((len(c_ids) for c_ids, _ in pairs), default=0)
        max_m = max(max_m, 1)
        ctx_vecs = np.zeros((bsz, max_m, dim), dtype=np.float32)
        ctx_mask = np.zeros((bsz, max_m), dtype=bool)
        for i, (c_ids, _) in enumerate(pairs):
            for j, cid in enumerate(c_ids[:max_turns]):
                ctx_vecs[i, j] = emb_matrix[cid]
                ctx_mask[i, j] = True
        return (
            torch.from_numpy(q_vecs),
            torch.from_numpy(ctx_vecs),
            torch.from_numpy(ctx_mask),
        )

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        n1_sa = np.stack([emb_matrix[b["n1_sa"]] for b in batch], axis=0)
        n2_sa = np.stack([emb_matrix[b["n2_sa"]] for b in batch], axis=0)
        return {
            "a": pack_group([b["a"] for b in batch]),
            "p": pack_group([b["p"] for b in batch]),
            "n1": pack_group([b["n1"] for b in batch]),
            "n2": pack_group([b["n2"] for b in batch]),
            "n1_sa": torch.from_numpy(n1_sa),
            "n2_sa": torch.from_numpy(n2_sa),
            "cid": torch.tensor([b["cid"] for b in batch], dtype=torch.long),
            "is_switch": torch.tensor([b["is_switch"] for b in batch], dtype=torch.bool),
        }

    return collate


@torch.no_grad()
def evaluate_val(
    net: GatedTurnCrossAttentionNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    net.eval()
    pos_sims: List[float] = []
    same_neg_sims: List[float] = []
    cross_neg_sims: List[float] = []
    sa_neg_sims: List[float] = []
    wins = 0
    total = 0

    for batch in loader:
        aq, actx, amask = [x.to(device) for x in batch["a"]]
        pq, pctx, pmask = [x.to(device) for x in batch["p"]]
        n1q, n1ctx, n1mask = [x.to(device) for x in batch["n1"]]
        n2q, n2ctx, n2mask = [x.to(device) for x in batch["n2"]]
        n1_sa_q = batch["n1_sa"].to(device)
        n2_sa_q = batch["n2_sa"].to(device)

        za = net(aq, actx, amask)
        zp = net(pq, pctx, pmask)
        zn1 = net(n1q, n1ctx, n1mask)
        zn2 = net(n2q, n2ctx, n2mask)
        zn1_sa = net(n1_sa_q, None, None)
        zn2_sa = net(n2_sa_q, None, None)

        sp = (za * zp).sum(dim=-1)
        sn1 = (za * zn1).sum(dim=-1)
        sn2 = (za * zn2).sum(dim=-1)
        sn1_sa = (za * zn1_sa).sum(dim=-1)
        sn2_sa = (za * zn2_sa).sum(dim=-1)
        sp_n1 = (zp * zn1).sum(dim=-1)
        sp_n2 = (zp * zn2).sum(dim=-1)
        s_sa = torch.stack([sn1_sa, sn2_sa, sp_n1, sp_n2], dim=-1).max(dim=-1).values

        wins += int(((sp > sn1) & (sp > sn2) & (sp > s_sa)).sum().item())
        total += za.shape[0]
        pos_sims.extend(sp.cpu().tolist())
        same_neg_sims.extend(sn1.cpu().tolist())
        cross_neg_sims.extend(sn2.cpu().tolist())
        sa_neg_sims.extend(s_sa.cpu().tolist())

    mean_pos = float(np.mean(pos_sims))
    mean_n1 = float(np.mean(same_neg_sims))
    mean_n2 = float(np.mean(cross_neg_sims))
    mean_sa = float(np.mean(sa_neg_sims))
    return {
        "triplet_acc": round(wins / max(1, total), 4),
        "pos_sim": round(mean_pos, 4),
        "same_ctx_neg_sim": round(mean_n1, 4),
        "cross_ctx_neg_sim": round(mean_n2, 4),
        "standalone_neg_sim": round(mean_sa, 4),
        "margin": round(mean_pos - max(mean_n1, mean_n2, mean_sa), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "turn_adapter"))
    parser.add_argument("--emb_cache_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_torch_device()
    os.makedirs(args.output_dir, exist_ok=True)

    train_recs = load_jsonl(os.path.join(args.data_dir, "train_triplets.jsonl"))
    val_recs = load_jsonl(os.path.join(args.data_dir, "val_triplets.jsonl"))
    print(f"[turn_adapter] Loaded {len(train_recs)} train and {len(val_recs)} val triplets on {device}")

    all_utts = collect_unique_utterances(train_recs + val_recs, max_turns=args.max_turns)
    utt2idx = {u: i for i, u in enumerate(all_utts)}
    print(f"[turn_adapter] Unique single-turn utterances to embed: {len(all_utts)}")

    t0 = time.time()
    cache_loaded = False
    if args.emb_cache_path and os.path.isfile(args.emb_cache_path):
        cached = np.load(args.emb_cache_path, allow_pickle=False)
        if cached["embeddings"].shape[0] == len(all_utts):
            emb_matrix = cached["embeddings"]
            cache_loaded = True
            print(f"[turn_adapter] Loaded precomputed turn embeddings from {args.emb_cache_path}")

    if not cache_loaded:
        base_engine = Qwen3EmbeddingEngine(
            model_name_or_path=args.base_model,
            mode="query_only",
            device=str(device),
            max_length=256,
        )
        formatted = [
            f"Instruct: {base_engine.instruction}\nQuery: {u}" for u in all_utts
        ]
        emb_matrix = base_engine.encode_raw_texts(formatted, batch_size=128, max_length=256)
        if args.emb_cache_path:
            os.makedirs(os.path.dirname(args.emb_cache_path), exist_ok=True)
            np.savez_compressed(args.emb_cache_path, embeddings=emb_matrix)
        del base_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[turn_adapter] Turn embeddings ready ({emb_matrix.shape}) in {time.time() - t0:.1f}s")

    collate_fn = make_collate_fn(emb_matrix, max_turns=args.max_turns)
    train_loader = DataLoader(
        TurnTripletDataset(train_recs, utt2idx, max_turns=args.max_turns),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TurnTripletDataset(val_recs, utt2idx, max_turns=args.max_turns),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    net = GatedTurnCrossAttentionNet(
        embed_dim=int(emb_matrix.shape[1]),
        hidden_dim=args.hidden_dim,
        num_heads=8,
        num_layers=args.num_layers,
        max_turns=max(8, args.max_turns),
        dropout=0.1,
        use_isotropic_head=False,
    ).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    history: List[Dict[str, Any]] = []
    best_score = -1.0
    t_train = time.time()

    init_metrics = evaluate_val(net, val_loader, device)
    print(f"[turn_adapter] Epoch 0 (untrained): {json.dumps(init_metrics)}")

    for epoch in range(1, args.epochs + 1):
        net.train()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            aq, actx, amask = [x.to(device) for x in batch["a"]]
            pq, pctx, pmask = [x.to(device) for x in batch["p"]]
            n1q, n1ctx, n1mask = [x.to(device) for x in batch["n1"]]
            n2q, n2ctx, n2mask = [x.to(device) for x in batch["n2"]]
            n1_sa_q = batch["n1_sa"].to(device)
            n2_sa_q = batch["n2_sa"].to(device)
            cids = batch["cid"].to(device)
            is_switch = batch["is_switch"].to(device)

            za, diag_a = net(aq, actx, amask, return_diagnostics=True)
            zp = net(pq, pctx, pmask)
            zn1 = net(n1q, n1ctx, n1mask)
            zn2 = net(n2q, n2ctx, n2mask)
            zn1_sa = net(n1_sa_q, None, None)
            zn2_sa = net(n2_sa_q, None, None)

            bsz = za.shape[0]
            labels = torch.arange(bsz, device=device)

            # Exact cluster equality mask (only masks genuine identical target clusters off-diagonal)
            same_cid = cids.unsqueeze(1) == cids.unsqueeze(0)
            self_eye = torch.eye(bsz, dtype=torch.bool, device=device)
            dup_mask = same_cid & (~self_eye)

            # 1. Forward InfoNCE: za -> [zp, za, zn1, zn2, zn1_sa, zn2_sa]
            cand_a = torch.cat([zp, za, zn1, zn2, zn1_sa, zn2_sa], dim=0)  # (6B, D)
            logits_a = (za @ cand_a.T) / args.temperature
            logits_a[:, :bsz] = logits_a[:, :bsz].masked_fill(dup_mask, -1e4)
            logits_a[:, bsz : 2 * bsz] = logits_a[:, bsz : 2 * bsz].masked_fill(same_cid, -1e4)
            loss_a2p = F.cross_entropy(logits_a, labels)

            # 2. Symmetric Reverse InfoNCE: zp -> [za, zp, zn1, zn2, zn1_sa, zn2_sa]
            # Trains isotropic head + adapter so standalone queries cleanly separate cross-entity distractors!
            cand_p = torch.cat([za, zp, zn1, zn2, zn1_sa, zn2_sa], dim=0)  # (6B, D)
            logits_p = (zp @ cand_p.T) / args.temperature
            logits_p[:, :bsz] = logits_p[:, :bsz].masked_fill(dup_mask, -1e4)
            logits_p[:, bsz : 2 * bsz] = logits_p[:, bsz : 2 * bsz].masked_fill(same_cid, -1e4)
            loss_p2a = F.cross_entropy(logits_p, labels)

            # 3. Cross-Modality & Hard-Negative Margin + Ceiling (on active trainable similarities)
            sp = (za * zp).sum(dim=-1)
            sn1 = (za * zn1).sum(dim=-1)
            sn2 = (za * zn2).sum(dim=-1)
            sn1_sa = (za * zn1_sa).sum(dim=-1)
            sn2_sa = (za * zn2_sa).sum(dim=-1)
            sp_n1 = (zp * zn1).sum(dim=-1)
            sp_n2 = (zp * zn2).sum(dim=-1)
            s_frozen_rival = torch.maximum(
                (zp * zn1_sa).sum(dim=-1), (zp * zn2_sa).sum(dim=-1)
            ).detach()

            active_hardest_neg = torch.stack(
                [sn1, sn2, sn1_sa, sn2_sa, sp_n1, sp_n2], dim=-1
            ).max(dim=-1).values
            loss_margin = (
                F.relu(0.38 - (sp - active_hardest_neg)).mean()
                + 0.6 * F.relu(0.18 - (sp - s_frozen_rival)).mean()
            )
            loss_ceiling = F.relu(active_hardest_neg - 0.54).pow(2).mean()

            # 4. Topic-Switch Dual-Gate + Null-Sink Suppression + Standalone Identity Alignment
            if bool(is_switch.any().item()):
                za_standalone = net(aq[is_switch], None, None)
                gate_pen = (
                    diag_a["erase_gate"][is_switch].mean()
                    + diag_a["write_gate"][is_switch].mean()
                )
                null_pen = (1.0 - diag_a["null_attn"][is_switch]).mean()
                id_pen = (1.0 - (za[is_switch] * za_standalone).sum(dim=-1)).mean()
                switch_loss = 0.5 * gate_pen + 0.5 * null_pen + id_pen
            else:
                switch_loss = torch.tensor(0.0, device=device)

            loss = (
                loss_a2p
                + 0.7 * loss_p2a
                + 0.85 * loss_margin
                + 1.0 * loss_ceiling
                + 0.25 * switch_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            steps += 1

        val_metrics = evaluate_val(net, val_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = round(running_loss / max(1, steps), 4)
        history.append(val_metrics)
        print(f"[turn_adapter] Epoch {epoch:02d}: {json.dumps(val_metrics)}")

        score = val_metrics["triplet_acc"] + 0.40 * val_metrics["margin"]
        if score > best_score:
            best_score = score
            net.save_pretrained(args.output_dir)

    elapsed = time.time() - t_train
    summary = {
        "model_type": "GatedTurnCrossAttentionNet_v2_PointerEraseWrite",
        "base_model": args.base_model,
        "trainable_params": sum(p.numel() for p in net.parameters() if p.requires_grad),
        "train_seconds": round(elapsed, 2),
        "init_val": init_metrics,
        "best_val": max(history, key=lambda m: m["triplet_acc"] + 0.40 * m["margin"]),
        "history": history,
    }
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[turn_adapter] Saved best checkpoint to {args.output_dir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

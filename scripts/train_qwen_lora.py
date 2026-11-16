#!/usr/bin/env python3
"""Train Model B: Token-Level Causal Context-Aware LoRA on Qwen3-Embedding-0.6B (Fast Fused BF16)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vcache.context.dialogue import format_qwen3_context_prompt
from vcache.context.qwen3_embedding import last_token_pool, select_torch_device


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class LoRATripletDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        max_turns: int = 6,
        frozen_pos_embs: np.ndarray | None = None,
    ):
        self.records = records
        self.max_turns = max_turns
        self.frozen_pos_embs = frozen_pos_embs

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        item: Dict[str, Any] = {
            "anchor": format_qwen3_context_prompt(
                r["anchor_context"], r["anchor_query"], max_turns=self.max_turns
            ),
            "pos": format_qwen3_context_prompt(
                r["pos_context"], r["pos_query"], max_turns=self.max_turns
            ),
            "same_neg": format_qwen3_context_prompt(
                r["same_ctx_neg_context"], r["same_ctx_neg_query"], max_turns=self.max_turns
            ),
            "cross_neg": format_qwen3_context_prompt(
                r["cross_ctx_neg_context"], r["cross_ctx_neg_query"], max_turns=self.max_turns
            ),
            "pos_is_standalone": len(r["pos_context"]) == 0,
        }
        if self.frozen_pos_embs is not None:
            item["zp_frozen"] = self.frozen_pos_embs[idx]
        return item


def make_tokenizer_fn(tokenizer: AutoTokenizer, max_length: int, auto_adds_eos: bool):
    eos_id = tokenizer.eos_token_id

    def tokenize_texts(texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        if auto_adds_eos:
            return tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        encoded = tokenizer(
            list(texts),
            padding=False,
            truncation=True,
            max_length=max(2, max_length - 1),
            add_special_tokens=True,
        )
        seqs = [ids + ([eos_id] if eos_id is not None else []) for ids in encoded["input_ids"]]
        return tokenizer.pad({"input_ids": seqs}, padding=True, return_tensors="pt")

    return tokenize_texts


def forward_embed(model, batch_tokens: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch_tokens.items()}
    outputs = model(**inputs)
    pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return F.normalize(pooled.float(), p=2, dim=-1)


@torch.no_grad()
def precompute_frozen_pos_embeddings(
    base_model,
    pos_prompts: Sequence[str],
    tokenize_fn,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    base_model.eval()
    chunks: List[np.ndarray] = []
    for start in range(0, len(pos_prompts), batch_size):
        tok = tokenize_fn(pos_prompts[start : start + batch_size])
        embs = forward_embed(base_model, tok, device)
        chunks.append(embs.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def evaluate_lora_val(
    model,
    val_loader: DataLoader,
    tokenize_fn,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    pos_sims: List[float] = []
    n1_sims: List[float] = []
    n2_sims: List[float] = []
    wins = 0
    total = 0

    for batch in val_loader:
        bsz = len(batch["anchor"])
        all_texts = (
            list(batch["anchor"])
            + list(batch["pos"])
            + list(batch["same_neg"])
            + list(batch["cross_neg"])
        )
        tok = tokenize_fn(all_texts)
        embs = forward_embed(model, tok, device)
        za, zp, zn1, zn2 = embs.chunk(4, dim=0)

        sp = (za * zp).sum(dim=-1)
        sn1 = (za * zn1).sum(dim=-1)
        sn2 = (za * zn2).sum(dim=-1)

        wins += int(((sp > sn1) & (sp > sn2)).sum().item())
        total += bsz
        pos_sims.extend(sp.cpu().tolist())
        n1_sims.extend(sn1.cpu().tolist())
        n2_sims.extend(sn2.cpu().tolist())

    mean_pos = float(np.mean(pos_sims))
    mean_n1 = float(np.mean(n1_sims))
    mean_n2 = float(np.mean(n2_sims))
    return {
        "triplet_acc": wins / max(1, total),
        "pos_sim": mean_pos,
        "same_ctx_neg_sim": mean_n1,
        "cross_ctx_neg_sim": mean_n2,
        "margin": mean_pos - max(mean_n1, mean_n2),
    }


def select_balanced_subset(records: List[Dict[str, Any]], max_samples: int, seed: int) -> List[Dict[str, Any]]:
    if len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    canard = [r for r in records if str(r.get("source", "")).startswith("canard")]
    bilingual = [r for r in records if not str(r.get("source", "")).startswith("canard")]
    rng.shuffle(canard)
    rng.shuffle(bilingual)
    half = max_samples // 2
    take_b = min(len(bilingual), half)
    take_c = min(len(canard), max_samples - take_b)
    subset = canard[:take_c] + bilingual[:take_b]
    rng.shuffle(subset)
    return subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "qwen3_context_lora"))
    parser.add_argument("--max_train_samples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = select_torch_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    train_recs = select_balanced_subset(
        load_jsonl(os.path.join(args.data_dir, "train_triplets.jsonl")),
        args.max_train_samples,
        args.seed,
    )
    val_recs = load_jsonl(os.path.join(args.data_dir, "val_triplets.jsonl"))[:500]
    print(f"[qwen_lora] Training on {len(train_recs)} balanced triplets, validating on {len(val_recs)} ({device}, {dtype})")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    probe_ids = tokenizer.encode("probe", add_special_tokens=True)
    auto_adds_eos = bool(
        probe_ids and tokenizer.eos_token_id is not None and probe_ids[-1] == tokenizer.eos_token_id
    )
    tokenize_fn = make_tokenizer_fn(tokenizer, args.max_length, auto_adds_eos)

    base_model = AutoModel.from_pretrained(
        args.base_model,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    base_model.config.use_cache = False

    t_pre = time.time()
    pos_prompts = [
        format_qwen3_context_prompt(r["pos_context"], r["pos_query"], max_turns=args.max_turns)
        for r in train_recs
    ]
    frozen_pos_embs = precompute_frozen_pos_embeddings(
        base_model, pos_prompts, tokenize_fn, device, batch_size=128
    )
    print(f"[qwen_lora] Precomputed frozen standalone anchors in {time.time() - t_pre:.1f}s")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_cfg).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[qwen_lora] Trainable params: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.2f}%)")

    train_loader = DataLoader(
        LoRATripletDataset(train_recs, max_turns=args.max_turns, frozen_pos_embs=frozen_pos_embs),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        LoRATripletDataset(val_recs, max_turns=args.max_turns),
        batch_size=32,
        shuffle=False,
    )

    init_val = evaluate_lora_val(model, val_loader, tokenize_fn, device)
    print(f"[qwen_lora] Step 0 (base structured_raw): {json.dumps(init_val)}")

    total_opt_steps = max(1, (args.epochs * len(train_loader)) // args.grad_accum)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps)

    history: List[Dict[str, Any]] = []
    best_score = -1.0
    global_step = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        log_steps = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            all_texts = (
                list(batch["anchor"])
                + list(batch["pos"])
                + list(batch["same_neg"])
                + list(batch["cross_neg"])
            )
            tok_all = tokenize_fn(all_texts)
            embs = forward_embed(model, tok_all, device)
            za, zp, zn1, zn2 = embs.chunk(4, dim=0)

            candidates = torch.cat([zp, zn1, zn2], dim=0)
            logits = (za @ candidates.T) / args.temperature
            labels = torch.arange(za.shape[0], device=device)
            loss_infonce = F.cross_entropy(logits, labels)

            sp = (za * zp).sum(dim=-1)
            sn1 = (za * zn1).sum(dim=-1)
            sn2 = (za * zn2).sum(dim=-1)
            loss_margin = F.relu(0.28 - (sp - torch.maximum(sn1, sn2))).mean()

            standalone_mask = batch["pos_is_standalone"].to(device)
            if bool(standalone_mask.any().item()):
                zp_frozen = batch["zp_frozen"].to(device, non_blocking=True)
                loss_anchor_space = (
                    1.0 - (zp[standalone_mask] * zp_frozen[standalone_mask]).sum(dim=-1)
                ).mean()
            else:
                loss_anchor_space = torch.tensor(0.0, device=device)

            loss = (loss_infonce + 0.5 * loss_margin + 0.20 * loss_anchor_space) / args.grad_accum
            loss.backward()

            running_loss += float(loss.item()) * args.grad_accum
            log_steps += 1

            if step_idx % args.grad_accum == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    print(
                        f"[qwen_lora] Epoch {epoch} | opt_step {global_step}/{total_opt_steps} "
                        f"| loss={running_loss / max(1, log_steps):.4f} | elapsed={time.time() - t_start:.1f}s"
                    )
                    running_loss = 0.0
                    log_steps = 0

        val_metrics = evaluate_lora_val(model, val_loader, tokenize_fn, device)
        val_metrics["epoch"] = epoch
        val_metrics["opt_step"] = global_step
        history.append(val_metrics)
        print(f"[qwen_lora] Epoch {epoch} validation: {json.dumps(val_metrics)}")

        score = val_metrics["triplet_acc"] + 0.25 * val_metrics["margin"]
        if score > best_score:
            best_score = score
            model.save_pretrained(args.output_dir)

    elapsed = time.time() - t_start
    summary = {
        "model_type": "Qwen3ContextLoRA",
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "trainable_params": trainable_params,
        "train_seconds": round(elapsed, 2),
        "init_val": init_val,
        "best_val": max(history, key=lambda m: m["triplet_acc"] + 0.25 * m["margin"]),
        "history": history,
    }
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[qwen_lora] Saved best LoRA checkpoint to {args.output_dir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

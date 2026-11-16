---
base_model: Qwen/Qwen3-Embedding-0.6B
library_name: peft
tags:
- base_model:adapter:Qwen/Qwen3-Embedding-0.6B
- lora
- transformers
- semantic-caching
---

# ContextCache Model B: Qwen3ContextLoRA (`r=16, alpha=32`)

Token-level causal context-aware LoRA adapter trained on top of `Qwen/Qwen3-Embedding-0.6B` for multi-turn semantic prompt caching (`ContextCache`).

- **Trainable Parameters**: `10,092,544` (`~1.67%` of base model)
- **Attention & FFN Target Modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Inference Acceleration**: Merged into BF16 weights (`merge_and_unload()`) with Hierarchical Causal Prefix KV-Cache reuse (`vcache/context/qwen_context_lora.py`).
- **Validation Convergence**: Triplet accuracy `68.40% -> 100.00%`, pos/neg cosine margin `+0.3632`.

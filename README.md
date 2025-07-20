# ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries in Large Language Models


## Demo
https://youtu.be/R3NByaQS7Ws

## Install
```pip install gptcache```

## 🚀 What is ContextCache?
Large language models (LLMs) like ChatGPT enable powerful applications, but scaling them introduces significant costs and latency from repeated API calls. While solutions like GPTCache have emerged to cache LLM responses through semantic matching, existing approaches primarily focus on individual queries without considering conversational context.

This limitation becomes particularly problematic in multi-turn dialogues, where similar queries in different contexts may trigger incorrect cache hits. To solve this, we present ContextCache, a context-aware semantic caching system specifically designed for conversational AI.

ContextCache features:
- A two-stage retrieval architecture combining vector search with dialogue-aware matching
- Self-attention mechanisms to integrate current and historical conversation context
- Demonstrated 10x lower latency than direct LLM calls
- Improvements in precision and recall over existing methods

Our evaluations show ContextCache delivers both performance gains and cost reductions for production-scale LLM applications while maintaining contextual accuracy.


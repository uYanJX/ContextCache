from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.cache.embedding_engine.strategies.benchmark import (
    BenchmarkEmbeddingEngine,
)

try:
    from vcache.vcache_core.cache.embedding_engine.strategies.open_ai import (
        OpenAIEmbeddingEngine,
    )
except ImportError:
    OpenAIEmbeddingEngine = None  # type: ignore[assignment,misc]

try:
    from vcache.vcache_core.cache.embedding_engine.strategies.lang_chain import (
        LangChainEmbeddingEngine,
    )
except ImportError:
    LangChainEmbeddingEngine = None  # type: ignore[assignment,misc]

__all__ = [
    "EmbeddingEngine",
    "BenchmarkEmbeddingEngine",
    "OpenAIEmbeddingEngine",
    "LangChainEmbeddingEngine",
]

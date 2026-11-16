from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.benchmark_comparison import (
    BenchmarkComparisonSimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.embedding_comparison import (
    EmbeddingComparisonSimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)

try:
    from vcache.vcache_core.similarity_evaluator.strategies.llm_comparison import (
        LLMComparisonSimilarityEvaluator,
    )
except ImportError:
    LLMComparisonSimilarityEvaluator = None  # type: ignore[assignment,misc]

__all__ = [
    "SimilarityEvaluator",
    "StringComparisonSimilarityEvaluator",
    "EmbeddingComparisonSimilarityEvaluator",
    "BenchmarkComparisonSimilarityEvaluator",
    "LLMComparisonSimilarityEvaluator",
]

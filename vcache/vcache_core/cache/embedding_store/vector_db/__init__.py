from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
    HNSWLibVectorDB,
)
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib_persistent import (
    PersistentHNSWLibVectorDB,
)
from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)

try:
    from vcache.vcache_core.cache.embedding_store.vector_db.strategies.faiss import (
        FAISSVectorDB,
    )
except ImportError:
    FAISSVectorDB = None  # type: ignore[assignment,misc]

try:
    from vcache.vcache_core.cache.embedding_store.vector_db.strategies.chroma import (
        ChromaVectorDB,
    )
except ImportError:
    ChromaVectorDB = None  # type: ignore[assignment,misc]

__all__ = [
    "VectorDB",
    "SimilarityMetricType",
    "HNSWLibVectorDB",
    "PersistentHNSWLibVectorDB",
    "FAISSVectorDB",
    "ChromaVectorDB",
]

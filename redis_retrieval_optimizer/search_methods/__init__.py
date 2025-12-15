"""
Search methods package for optimizing retrieval systems.
Contains various search algorithms and strategies.
"""

# Import submodules to make them available when importing the package
from redis_retrieval_optimizer.search_methods.bm25 import gather_bm25_results
from redis_retrieval_optimizer.search_methods.hybrid import gather_hybrid_results
from redis_retrieval_optimizer.search_methods.hybrid_8_4 import (
    gather_hybrid_8_4_results,
)
from redis_retrieval_optimizer.search_methods.rerank import gather_rerank_results
from redis_retrieval_optimizer.search_methods.rrf_8_4 import gather_rrf_8_4_results
from redis_retrieval_optimizer.search_methods.vector import gather_vector_results
from redis_retrieval_optimizer.search_methods.weighted_rrf import gather_weighted_rrf

SEARCH_METHOD_MAP = {
    "bm25": gather_bm25_results,
    "rerank": gather_rerank_results,
    "hybrid": gather_hybrid_results,
    "vector": gather_vector_results,
    "weighted_rrf": gather_weighted_rrf,
    "hybrid_8_4": gather_hybrid_8_4_results,
    "rrf_8_4": gather_rrf_8_4_results,
}

# Define __all__ to control what's imported with "from search_methods import *"
__all__ = [
    "bm25",
    "hybrid",
    "vector",
    "rerank",
    "weighted_rrf",
    "hybrid_8_4",
    "rrf_8_4",
    "SEARCH_METHOD_MAP",
]

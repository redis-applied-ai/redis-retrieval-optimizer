"""
Search methods package for optimizing retrieval systems.
Contains various search algorithms and strategies.
"""

# Import submodules to make them available when importing the package
from . import bm25, lin_combo, rerank, vector, weighted_rrf

SEARCH_METHOD_MAP = {
    "bm25": bm25.gather_bm25_results,
    "rerank": rerank.gather_rerank_results,
    "lin_combo": lin_combo.gather_lin_combo_results,
    "vector": vector.gather_vector_results,
    "weighted_rrf": weighted_rrf.gather_weighted_rrf,
}

# Define __all__ to control what's imported with "from search_methods import *"
__all__ = [
    "bm25",
    "lin_combo",
    "vector",
    "rerank",
    "weighted_rrf",
    "SEARCH_METHOD_MAP",
]

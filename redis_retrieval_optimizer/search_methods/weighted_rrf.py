import logging
import time
from typing import Any, Dict, List

from ranx import Run

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.bm25 import bm25_query_optional
from redis_retrieval_optimizer.search_methods.hybrid import vector_query_filter

logger = logging.getLogger(__name__)


def fuse_rankings_rrf(*ranked_lists, weights=None, k=60):
    """
    Perform Weighted Reciprocal Rank Fusion on N number of ordered lists.
    """
    item_scores = {}

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    else:
        assert len(weights) == len(
            ranked_lists
        ), "Number of weights must match number of ranked lists"
        assert all(0 <= w <= 1 for w in weights), "Weights must be between 0 and 1"

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked_list, start=1):
            if item not in item_scores:
                item_scores[item] = 0
            item_scores[item] += weight * (1 / (rank + k))

    # Sort items by their weighted RRF scores in descending order
    return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)


def weighted_rrf(
    index,
    emb_model,
    user_query: str,
    alpha: float = 0.5,
    num_results: int = 4,
    k: int = 20,
    id_field_name: str = "_id",
) -> List[Dict[str, Any]]:
    """Implemented client-side RRF after querying from Redis."""

    # Create the vector query
    vector_query = vector_query_filter(emb_model, user_query, num_results=k)

    # Create the full-text bm25 query
    full_text_query = bm25_query_optional(
        "text", id_field_name, user_query, num_results=k
    )

    # Run queries individually
    vector_query_results = index.query(vector_query)
    full_text_query_results = index.query(full_text_query)

    # Extract _id from results
    vector_ids = [res[id_field_name] for res in vector_query_results]
    full_text_ids = [res[id_field_name] for res in full_text_query_results]

    # Perform weighted RRF
    return fuse_rankings_rrf(
        vector_ids, full_text_ids, weights=[alpha, 1 - alpha], k=k
    )[:num_results]


def make_score_dict_w_rff(res):
    return {rec[0]: rec[1] for rec in res}


def gather_weighted_rrf(search_method_input: SearchMethodInput) -> SearchMethodOutput:
    redis_res_w_rrf = {}

    for key in search_method_input.raw_queries:
        text_query = search_method_input.raw_queries[key]
        start = time.time()
        try:
            w_rff = weighted_rrf(
                search_method_input.index,
                search_method_input.emb_model,
                text_query,
                num_results=search_method_input.ret_k,
                k=20,  # TODO make this configurable
                id_field_name=search_method_input.id_field_name,
            )
            scores_dict = make_score_dict_w_rff(w_rff)
        except Exception as e:
            logger.exception(
                f"Weighted RRF search failed for {key=}, {text_query=} \n {e=}"
            )
            scores_dict = {"no_match": 0}
        finally:
            query_time = time.time() - start
            search_method_input.query_metrics.query_times.append(query_time)

        redis_res_w_rrf[key] = scores_dict

    return SearchMethodOutput(
        run=Run(redis_res_w_rrf),
        query_metrics=search_method_input.query_metrics,
    )

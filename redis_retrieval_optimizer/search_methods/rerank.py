import logging
import time
from typing import Any, Dict, List

from ranx import Run

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.bm25 import bm25_query_optional
from redis_retrieval_optimizer.search_methods.hybrid import vector_query_filter

logger = logging.getLogger(__name__)


def rerank(
    index,
    reranker,
    emb_model,
    user_query: str,
    num_results: int = 20,
    text_field_name: str = "text",
    id_field_name: str = "_id",
) -> List[Dict[str, Any]]:
    """Rerank the candidates based on the user query with an external model/module."""

    # Create the vector query default larger set that is pruned down to ret_k
    vector_query = vector_query_filter(emb_model, user_query, num_results=20)

    # Create the full-text query default larger set that is pruned down to ret_k
    full_text_query = bm25_query_optional(
        text_field_name, id_field_name, user_query, num_results=20
    )

    # Run queries individually
    vector_query_results = index.query(vector_query)
    full_text_query_results = index.query(full_text_query)

    # Assemble list of potential candidates with their IDs
    candidate_map = {}
    for res in vector_query_results + full_text_query_results:
        candidate = f"Id: {res[id_field_name]}. Text: {res[text_field_name]}"
        if candidate not in candidate_map:
            candidate_map[candidate] = res

    # Rerank candidates
    reranked, scores = reranker.rank(
        query=user_query,
        docs=list(candidate_map.keys()),
        limit=num_results,
        return_score=True,
    )

    # Fetch full objects for the reranked results
    return [
        (candidate_map[rr["content"]][id_field_name], score)
        for rr, score in zip(reranked, scores)
    ]


def make_score_dict_rerank(res):
    if not res:
        return {"no_match": 0}
    return {rec[0]: rec[1] for rec in res}


def gather_rerank_results(search_method_input: SearchMethodInput):
    # lazy load the reranker
    from redisvl.utils.rerank import HFCrossEncoderReranker

    # Load the ms marco MiniLM cross encoder model from huggingface
    reranker = HFCrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    redis_res_rerank = {}

    for key in search_method_input.raw_queries:
        text_query = search_method_input.raw_queries[key]
        start = time.time()
        try:
            rerank_res = rerank(
                search_method_input.index,
                reranker,
                search_method_input.emb_model,
                text_query,
                num_results=search_method_input.ret_k,
                text_field_name=search_method_input.text_field_name,
                id_field_name=search_method_input.id_field_name,
            )
            scores_dict = make_score_dict_rerank(rerank_res)
        except Exception as e:
            logger.exception(f"Rerank search failed for {key=}, {text_query=} \n {e=}")
            scores_dict = {"no_match": 0}
        finally:
            query_time = time.time() - start
            search_method_input.query_metrics.query_times.append(query_time)

        redis_res_rerank[key] = scores_dict

    return SearchMethodOutput(
        run=Run(redis_res_rerank),
        query_metrics=search_method_input.query_metrics,
    )

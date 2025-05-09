import time
from typing import Any, Dict, List

from ranx import Run
from redisvl.utils.rerank import HFCrossEncoderReranker

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.bm25 import bm25_query_optional
from redis_retrieval_optimizer.search_methods.lin_combo import vector_query_filter

# Load the ms marco MiniLM cross encoder model from huggingface
reranker = HFCrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(
    index,
    emb_model,
    user_query: str,
    num_results: int = 10,
) -> List[Dict[str, Any]]:
    """Rerank the candidates based on the user query with an external model/module."""
    # Create the vector query
    vector_query = vector_query_filter(emb_model, user_query, num_results=num_results)

    # Create the full-text query
    full_text_query = bm25_query_optional("text", user_query, num_results=num_results)

    # Run queries individually
    vector_query_results = index.query(vector_query)
    full_text_query_results = index.query(full_text_query)

    # Assemble list of potential candidates with their IDs
    candidate_map = {}
    for res in vector_query_results + full_text_query_results:
        candidate = f"Id: {res['_id']}. Text: {res['text']}"
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
        (candidate_map[rr["content"]]["_id"], score)
        for rr, score in zip(reranked, scores)
    ]


def make_score_dict_rerank(res):
    return {rec[0]: rec[1] for rec in res}


def gather_rerank_results(search_method_input: SearchMethodInput):
    redis_res_rerank = {}

    for key in search_method_input.raw_queries:
        text_query = search_method_input.raw_queries[key]
        try:
            start = time.time()
            rerank_res = rerank(
                search_method_input.index,
                search_method_input.emb_model,
                text_query,
                num_results=10,
            )
            query_time = time.time() - start
            search_method_input.query_metrics.query_times.append(query_time)

            scores_dict = make_score_dict_rerank(rerank_res)
        except:
            print(f"failed for {key}, {text_query}")
            scores_dict = {}

        redis_res_rerank[key] = scores_dict

    return SearchMethodOutput(
        run=Run(redis_res_rerank),
        query_metrics=search_method_input.query_metrics,
    )

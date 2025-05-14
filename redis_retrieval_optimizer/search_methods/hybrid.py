import os

from ranx import Run
from redisvl.query import HybridQuery, VectorQuery

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.base import run_search_w_time


def vector_query_filter(
    emb_model, user_query: str, num_results: int, filters=None
) -> VectorQuery:
    """Generate a Redis vector query given user query string."""
    vector = emb_model.embed(user_query, as_buffer=True, dtype="float32")
    query = VectorQuery(
        vector=vector,
        vector_field_name="vector",
        num_results=num_results,
        return_fields=["_id", "text"],
    )
    if filters:
        query.set_filter(filters)

    return query


def gen_hybrid_query(emb_model, user_query: str, num_results: int) -> HybridQuery:
    """Generate a Redis vector query given user query string."""
    VECTOR_FIELD_NAME = os.environ.get("VECTOR_FIELD_NAME", "vector")
    TEXT_FIELD_NAME = os.environ.get("TEXT_FIELD_NAME", "text")
    ID_FIELD_NAME = os.environ.get("ID_FIELD_NAME", "_id")

    vector = emb_model.embed(user_query, as_buffer=True, dtype="float32")

    query = HybridQuery(
        text=user_query,
        text_field_name=TEXT_FIELD_NAME,
        vector=vector,
        vector_field_name=VECTOR_FIELD_NAME,
        alpha=0.7,
        num_results=num_results,
        return_fields=[ID_FIELD_NAME, TEXT_FIELD_NAME],
    )

    return query


def hybrid_scores_dict(res):
    ID_FIELD_NAME = os.environ.get("ID_FIELD_NAME", "_id")
    if res:
        scores_dict = {}

        for rec in res:
            if ID_FIELD_NAME in rec:
                scores_dict[rec[ID_FIELD_NAME]] = float(rec["hybrid_score"])
            else:
                scores_dict["no_match"] = 1
        return scores_dict
    else:
        return {"no_match": 0}


def gather_hybrid_results(
    search_method_input: SearchMethodInput,
) -> SearchMethodOutput:
    redis_res_hybrid = {}

    for key in search_method_input.raw_queries:
        text_query = search_method_input.raw_queries[key]
        try:
            hybrid_query = gen_hybrid_query(
                search_method_input.emb_model, text_query, 10
            )
            res = run_search_w_time(
                search_method_input.index,
                hybrid_query,
                search_method_input.query_metrics,
            )
            score_dict = hybrid_scores_dict(res)
        except Exception as e:
            print(f"failed for {key}, {text_query}")
            score_dict = {"no_match": 0}
        redis_res_hybrid[key] = score_dict

    return SearchMethodOutput(
        run=Run(redis_res_hybrid),
        query_metrics=search_method_input.query_metrics,
    )

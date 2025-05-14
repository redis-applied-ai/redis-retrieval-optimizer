import os

from ranx import Run
from redisvl.query import VectorQuery

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.base import run_search_w_time


def vector_query(query: str, num_results: int, emb_model) -> VectorQuery:
    vector = emb_model.embed(query, as_buffer=True)
    VECTOR_FIELD_NAME = os.environ.get("VECTOR_FIELD_NAME", "vector")
    ID_FIELD_NAME = os.environ.get("ID_FIELD_NAME", "_id")
    TEXT_FIELD_NAME = os.environ.get("TEXT_FIELD_NAME", "text")

    return VectorQuery(
        vector=vector,
        vector_field_name=VECTOR_FIELD_NAME,
        num_results=num_results,
        return_fields=[
            ID_FIELD_NAME,
            TEXT_FIELD_NAME,
        ],
    )


def make_score_dict_vec(res):
    ID_FIELD_NAME = os.environ.get("ID_FIELD_NAME", "_id")

    scores_dict = {}
    if not res:
        return {"no_match": 0}
    for rec in res:
        if ID_FIELD_NAME in rec:
            scores_dict[rec[ID_FIELD_NAME]] = 2 - float(rec["vector_distance"]) / 2
        else:
            scores_dict["no_match"] = 0

    return scores_dict


def gather_vector_results(
    search_method_input: SearchMethodInput,
) -> SearchMethodOutput:
    redis_res_vector = {}

    for key in search_method_input.raw_queries:
        text_query = search_method_input.raw_queries[key]
        vec_query = vector_query(text_query, 10, search_method_input.emb_model)
        res = run_search_w_time(
            search_method_input.index, vec_query, search_method_input.query_metrics
        )

        score_dict = make_score_dict_vec(res)
        redis_res_vector[key] = score_dict

    return SearchMethodOutput(
        run=Run(redis_res_vector),
        query_metrics=search_method_input.query_metrics,
    )

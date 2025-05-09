from ranx import Run
from redisvl.query import VectorQuery

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.base import run_search_w_time


def vector_query(query: str, num_results: int, emb_model) -> VectorQuery:
    vector = emb_model.embed(query, as_buffer=True)

    return VectorQuery(
        vector=vector,
        vector_field_name="vector",
        num_results=num_results,
        return_fields=["_id", "text", "title"],  # update to read from env maybe?
    )


def make_score_dict_vec(res):
    scores_dict = {}
    for rec in res:
        if "_id" in rec:
            scores_dict[rec["_id"]] = 2 - float(rec["vector_distance"]) / 2
        else:
            scores_dict["no_match"] = 1

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

import logging

from ranx import Run
from redisvl.query import VectorQuery

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.base import run_search_w_time

logger = logging.getLogger(__name__)


def vector_query(
    query: str,
    num_results: int,
    emb_model,
    vector_field_name: str = "vector",
    id_field_name: str = "_id",
    text_field_name: str = "text",
) -> VectorQuery:
    vector = emb_model.embed(query, as_buffer=True)

    return VectorQuery(
        vector=vector,
        vector_field_name=vector_field_name,
        num_results=num_results,
        return_fields=[
            id_field_name,
            text_field_name,
        ],
    )


def make_score_dict_vec(res, id_field_name: str) -> dict:
    scores_dict = {}
    if not res:
        return {"no_match": 0}
    for rec in res:
        if id_field_name in rec:
            scores_dict[rec[id_field_name]] = 2 - float(rec["vector_distance"]) / 2
        else:
            scores_dict["no_match"] = 0

    return scores_dict


def gather_vector_results(
    search_method_input: SearchMethodInput,
) -> SearchMethodOutput:
    """
    Execute vector search using the improved SearchMethodInput interface.

    This method handles both dictionary and list query formats and demonstrates
    how the enhanced classes make the code cleaner and more maintainable.
    """
    redis_res_vector = {}

    for key in search_method_input.raw_queries:
        text_query = search_method_input.raw_queries[key]
        try:
            vec_query = vector_query(
                query=text_query,
                num_results=search_method_input.ret_k,
                emb_model=search_method_input.emb_model,
                vector_field_name=search_method_input.vector_field_name,
                id_field_name=search_method_input.id_field_name,
                text_field_name=search_method_input.text_field_name,
            )
            res = run_search_w_time(
                search_method_input.index, vec_query, search_method_input.query_metrics
            )
            score_dict = make_score_dict_vec(res, search_method_input.id_field_name)
        except Exception as e:
            logger.exception(f"Vector search failed for {key=}, {text_query=} \n {e=}")
            score_dict = {"no_match": 0}
        redis_res_vector[key] = score_dict

    return SearchMethodOutput(
        run=Run(redis_res_vector),
        query_metrics=search_method_input.query_metrics,
    )

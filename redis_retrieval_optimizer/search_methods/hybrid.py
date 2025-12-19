import logging

from ranx import Run
from redisvl.query import VectorQuery
from redisvl.query.aggregate import AggregateHybridQuery

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.base import run_search_w_time

logger = logging.getLogger(__name__)


def vector_query_filter(
    emb_model, user_query: str, num_results: int, filters=None
) -> VectorQuery:
    """Generate a Redis vector query given user query string."""
    vector = emb_model.embed(user_query, as_buffer=True)
    query = VectorQuery(
        vector=vector,
        vector_field_name="vector",
        num_results=num_results,
        return_fields=["_id", "text"],
    )
    if filters:
        query.set_filter(filters)

    return query


# TODO is this needed as a separate function?
def gen_hybrid_query(
    emb_model,
    user_query: str,
    num_results: int,
    vector_field_name: str = "vector",
    text_field_name: str = "text",
    id_field_name: str = "_id",
) -> AggregateHybridQuery:
    """Generate a Redis vector query given user query string."""

    vector = emb_model.embed(user_query, as_buffer=True)

    query = AggregateHybridQuery(
        text=user_query,
        text_field_name=text_field_name,
        vector=vector,
        vector_field_name=vector_field_name,
        alpha=0.7,  # TODO make this configurable
        num_results=num_results,
        return_fields=[id_field_name, text_field_name],
    )

    return query


def hybrid_scores_dict(res, id_field_name: str) -> dict:
    if res:
        scores_dict = {}

        for rec in res:
            if id_field_name in rec:
                scores_dict[rec[id_field_name]] = float(rec["hybrid_score"])
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
                emb_model=search_method_input.emb_model,
                user_query=text_query,
                num_results=search_method_input.ret_k,
                vector_field_name=search_method_input.vector_field_name,
                text_field_name=search_method_input.text_field_name,
                id_field_name=search_method_input.id_field_name,
            )
            res = run_search_w_time(
                search_method_input.index,
                hybrid_query,
                search_method_input.query_metrics,
            )
            score_dict = hybrid_scores_dict(res, search_method_input.id_field_name)
        except Exception as e:
            logger.exception(f"Hybrid search failed for {key=}, {text_query=} \n {e=}")
            score_dict = {"no_match": 0}
        redis_res_hybrid[key] = score_dict

    return SearchMethodOutput(
        run=Run(redis_res_hybrid),
        query_metrics=search_method_input.query_metrics,
    )

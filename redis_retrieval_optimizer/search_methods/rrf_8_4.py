import logging

from ranx import Run
from redisvl.query.hybrid import HybridQuery

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.base import run_search_w_time

logger = logging.getLogger(__name__)


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


def gather_rrf_8_4_results(
    search_method_input: SearchMethodInput,
) -> SearchMethodOutput:
    redis_res_hybrid = {}

    for key in search_method_input.raw_queries:
        text_query = search_method_input.raw_queries[key]
        try:
            vector = search_method_input.emb_model.embed(text_query, as_buffer=True)

            hybrid_query = HybridQuery(
                text=text_query,
                text_field_name=search_method_input.text_field_name,
                vector=vector,
                vector_field_name=search_method_input.vector_field_name,
                text_scorer="BM25STD",
                combination_method="RRF",
                return_fields=[
                    search_method_input.id_field_name,
                    search_method_input.text_field_name,
                ],
                yield_combined_score_as="hybrid_score",
            )

            res = run_search_w_time(
                search_method_input.index,
                hybrid_query,
                search_method_input.query_metrics,
            )
            score_dict = hybrid_scores_dict(res, search_method_input.id_field_name)
        except Exception as e:
            logger.exception(f"RRF 8_4 search failed for {key=}, {text_query=} \n {e=}")
            score_dict = {"no_match": 0}
        redis_res_hybrid[key] = score_dict

    return SearchMethodOutput(
        run=Run(redis_res_hybrid),
        query_metrics=search_method_input.query_metrics,
    )

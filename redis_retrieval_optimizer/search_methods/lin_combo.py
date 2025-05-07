from redis.commands.search.aggregation import AggregateRequest, Desc
from redisvl.query import VectorQuery
from redisvl.query.filter import Text
from redisvl.redis.utils import convert_bytes, make_dict

from redis_retrieval_optimizer.search_methods.bm25 import tokenize_and_escape_query


def vector_query_filter(
    emb_model, user_query: str, num_results: int, filters=None
) -> VectorQuery:
    """Generate a Redis vector query given user query string."""
    vector = emb_model.embed(user_query, as_buffer=True, dtype="float32")
    query = VectorQuery(
        vector=vector,
        vector_field_name="vector",
        num_results=num_results,
        return_fields=["_id", "title", "text"],
    )
    if filters:
        query.set_filter(filters)

    return query


def linear_combo(
    emb_model, user_query: str, alpha: float, num_results: int = 3
) -> AggregateRequest:
    # Add the optional flag, "~", so that this doesn't also act as a strict text filter
    text = f"(~{Text('text') % tokenize_and_escape_query(user_query)})"

    # Build vector query
    query = vector_query_filter(
        emb_model, user_query, num_results=num_results, filters=text
    )

    # Build aggregation request
    req = (
        AggregateRequest(query.query_string())
        .scorer("BM25STD")
        .add_scores()
        .apply(cosine_similarity="(2 - @vector_distance)/2", bm25_score="@__score")
        .apply(hybrid_score=f"{1-alpha}*@bm25_score + {alpha}*@cosine_similarity")
        .sort_by(Desc("@hybrid_score"), max=num_results)
        .load("_id", "title", "text", "cosine_similarity", "bm25_score", "hybrid_score")
        .dialect(2)
    )

    query_params = {"vector": query._vector}

    return req, query_params


def gather_lin_combo_results(queries, index, emb_model, alpha=0.7):
    redis_res_lin_combo = {}

    def agg_scores_dict(res):
        if res:
            results = [make_dict(row) for row in convert_bytes(res.rows)]
            return {rec["_id"]: float(rec["hybrid_score"]) for rec in results}

    for key in queries:
        text_query = queries[key]
        alpha = 0.8  # weight for cosine similarity vs bm25
        agg_req, query_params = linear_combo(emb_model, text_query, alpha, 10)
        try:
            res = index.aggregate(agg_req, query_params=query_params)
            score_dict = agg_scores_dict(res)
        except Exception as e:
            print(f"failed for {key}, {text_query}")
            score_dict = {}
        redis_res_lin_combo[key] = score_dict

    return redis_res_lin_combo

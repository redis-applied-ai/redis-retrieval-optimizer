import json
from typing import Any, Dict, List

from redis.commands.search.aggregation import AggregateRequest, Desc
from redisvl.query import FilterQuery
from redisvl.query.filter import Text
from redisvl.utils.token_escaper import TokenEscaper

from examples.beir.helpers.stopwords import STOPWORDS_EN

escaper = TokenEscaper()


def tokenize_and_escape_query(user_query: str) -> str:
    """Convert a raw user query to a redis full text query joined by ORs"""
    tokens = [
        escaper.escape(
            token.strip().strip(",").replace("“", "").replace("”", "").lower()
        )
        for token in user_query.split()
    ]
    return " | ".join(
        [token for token in tokens if token and token not in STOPWORDS_EN]
    )


def bm25_query(
    text_field: str, user_query: str, num_results: int, scorer="BM25STD"
) -> FilterQuery:
    """Generate a Redis full-text query given a user query string."""
    return (
        FilterQuery(
            filter_expression=f"~({Text(text_field) % tokenize_and_escape_query(user_query)})",
            num_results=num_results,
            return_fields=["_id", "text", "title"],
            dialect=2,
        )
        .scorer(scorer)
        .with_scores()
    )


def make_score_dict(res):
    return {rec["_id"]: rec["score"] for rec in res.docs}


def gather_bm25_results(queries, index):
    redis_res_bm25 = {}
    batch_queries = []

    for key in queries:
        text_query = queries[key]
        ft_query = bm25_query("text", text_query, 10)
        batch_queries.append(ft_query)
        # try:
        #     res = index.search(ft_query)
        #     score_dict = make_score_dict(res)
        # except Exception as e:
        #     print(f"failed for {key}, {text_query}")
        #     score_dict = {}
        # redis_res_bm25[key] = score_dict

    batch_res = index.batch_search(batch_queries)

    return redis_res_bm25

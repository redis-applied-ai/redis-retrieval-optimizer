import json
from typing import Any, Dict, List

from redis.commands.search.aggregation import AggregateRequest, Desc
from redisvl.query import FilterQuery
from redisvl.query.filter import Text
from redisvl.utils.token_escaper import TokenEscaper

STOPWORDS_EN = set(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "ain",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "couldn",
        "couldn't",
        "d",
        "did",
        "didn",
        "didn't",
        "do",
        "does",
        "doesn",
        "doesn't",
        "doing",
        "don",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "hadn't",
        "has",
        "hasn",
        "hasn't",
        "have",
        "haven",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "just",
        "ll",
        "m",
        "ma",
        "me",
        "mightn",
        "mightn't",
        "more",
        "most",
        "mustn",
        "mustn't",
        "my",
        "myself",
        "needn",
        "needn't",
        "no",
        "nor",
        "not",
        "now",
        "o",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "s",
        "same",
        "shan",
        "shan't",
        "she",
        "she's",
        "should",
        "should've",
        "shouldn",
        "shouldn't",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "that'll",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "ve",
        "very",
        "was",
        "wasn",
        "wasn't",
        "we",
        "were",
        "weren",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
        "y",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)

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
            filter_expression=f"({Text(text_field) % tokenize_and_escape_query(user_query)})",
            num_results=num_results,
            return_fields=["_id", "text", "title"],
            dialect=2,
        )
        .scorer(scorer)
        .with_scores()
    )


def bm25_query_optional(
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
    return {rec["_id"]: rec["score"] for rec in res}


def gather_bm25_results(queries, index, emb_model):
    redis_res_bm25 = {}

    for key in queries:
        text_query = queries[key]
        ft_query = bm25_query("text", text_query, 10)
        try:
            res = index.query(ft_query)
            score_dict = make_score_dict(res)
        except Exception as e:
            print(f"failed for {key}, {text_query}")
            score_dict = {}
        redis_res_bm25[key] = score_dict

    return redis_res_bm25

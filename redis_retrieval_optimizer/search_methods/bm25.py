from ranx import Run
from redisvl.query import FilterQuery
from redisvl.query.filter import Text
from redisvl.utils.token_escaper import TokenEscaper
from redisvl.utils.vectorize.base import BaseVectorizer

from redis_retrieval_optimizer.schema import (
    QueryMetrics,
    SearchMethodInput,
    SearchMethodOutput,
)
from redis_retrieval_optimizer.search_methods.base import run_search_w_time

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


def make_score_dict_bm25(res):
    scores_dict = {}
    if not res:
        return {"no_match": 1}

    for rec in res:
        if "_id" in rec:
            scores_dict[rec["_id"]] = rec["score"]
        else:
            scores_dict["no_match"] = 1

    return scores_dict


def gather_bm25_results(search_method_input: SearchMethodInput) -> SearchMethodOutput:
    redis_res_bm25 = {}

    for key in search_method_input.raw_queries:
        text_query = search_method_input.raw_queries[key]
        ft_query = bm25_query("text", text_query, 10)
        try:
            res = run_search_w_time(
                search_method_input.index, ft_query, search_method_input.query_metrics
            )
            score_dict = make_score_dict_bm25(res)
        except Exception as e:
            print(f"failed for {key}, {text_query}: error: {e}")
            score_dict = {}
        redis_res_bm25[key] = score_dict

    return SearchMethodOutput(
        run=Run(redis_res_bm25),
        query_metrics=search_method_input.query_metrics,
    )

"""Integration coverage for #1 — configured ret_k is actually used by the study.

Before the fix the study built SearchMethodInput without ret_k, so every search
retrieved the default 6 docs regardless of config. With a 5-doc corpus and
queries that have 2 relevant docs each, recall at ret_k=1 is mathematically
capped well below recall at ret_k=5 — so if ret_k is honored the two runs must
differ. On the buggy code both runs retrieve the same 6->5 docs and recall is
identical.
"""

import os

from redisvl.index import SearchIndex

import redis_retrieval_optimizer.utils as utils
from redis_retrieval_optimizer.corpus_processors import eval_beir
from redis_retrieval_optimizer.grid_study import run_grid_study

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = f"{TEST_DIR}/grid_data"


def _config(ret_k: int) -> dict:
    return {
        "corpus": f"{DATA}/corpus.json",
        "queries": f"{DATA}/queries.json",
        "qrels": f"{DATA}/qrels.json",
        "embedding_models": [
            {
                "dim": 384,
                "embedding_cache_name": "vec-cache",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "type": "hf",
            }
        ],
        "index_settings": {
            "from_existing": False,
            "name": "test-retk",
            "vector_dim": 384,
            "additional_fields": [{"name": "title", "type": "text"}],
        },
        "search_methods": ["vector"],
        "vector_data_types": ["float32"],
        "ret_k": ret_k,
    }


def _cleanup(redis_url):
    last_schema = utils.get_last_index_settings(redis_url)
    if last_schema:
        index = SearchIndex.from_existing(last_schema["name"], redis_url=redis_url)
        index.client.json().delete("ret-opt:last_schema")
        index.client.json().delete("ret-opt:last_indexing_time")
        index.delete(drop=True)


def test_ret_k_is_respected(redis_url):
    try:
        recall_k1 = run_grid_study(
            config=_config(1),
            redis_url=redis_url,
            corpus_processor=eval_beir.process_corpus,
        )["recall"].iloc[0]

        recall_k5 = run_grid_study(
            config=_config(5),
            redis_url=redis_url,
            corpus_processor=eval_beir.process_corpus,
        )["recall"].iloc[0]

        # ret_k=5 retrieves the whole 5-doc corpus -> every relevant doc found.
        assert recall_k5 > recall_k1
        assert recall_k1 < 1.0
    finally:
        _cleanup(redis_url)

"""Integration coverage for #9 — the from_existing dim-mismatch error path.

Pre-fix the error message interpolated `.attrs['dims']` (subscript) on a
pydantic attrs model, which raised TypeError while *building* the message, so
the caller saw a TypeError instead of the intended ValueError.
"""

import pytest
from redisvl.index import SearchIndex

from redis_retrieval_optimizer.grid_study import init_index_from_grid_settings
from redis_retrieval_optimizer.utils import load_grid_study_config

IDX = "test-dim-mismatch"


def test_from_existing_dim_mismatch_raises_valueerror(redis_url):
    schema = {
        "index": {"name": IDX, "prefix": IDX},
        "fields": [
            {"name": "_id", "type": "tag"},
            {"name": "text", "type": "text"},
            {
                "name": "vector",
                "type": "vector",
                "attrs": {
                    "dims": 384,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                },
            },
        ],
    }
    index = SearchIndex.from_dict(schema, redis_url=redis_url)
    try:
        index.delete(drop=True)
    except Exception:
        pass
    index.create()

    try:
        # config declares a 128-dim embedding against the existing 384-dim index
        config = load_grid_study_config(
            config={
                "index_settings": {
                    "name": IDX,
                    "from_existing": True,
                    "vector_dim": 128,
                },
                "embedding_models": [{"type": "hf", "model": "m", "dim": 128}],
                "search_methods": ["vector"],
                "qrels": "unused.json",
                "queries": "unused.json",
            }
        )

        with pytest.raises(ValueError, match="does not match index dimension"):
            init_index_from_grid_settings(
                config, redis_url, corpus_processor=lambda *a, **k: []
            )
    finally:
        index.delete(drop=True)

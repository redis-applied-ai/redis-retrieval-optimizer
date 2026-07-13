"""Integration coverage for #12 — get_index_memory_stats must not count the
internal bookkeeping keys (ret-opt:last_schema / ret-opt:last_indexing_time)
as indexed objects. They share the default "ret-opt" prefix, so the old
KEYS("ret-opt*") scan swept them in.
"""

import numpy as np
from redisvl.index import SearchIndex

import redis_retrieval_optimizer.utils as utils
from redis_retrieval_optimizer.utils import get_index_memory_stats

PREFIX = "ret-opt"


def test_bookkeeping_keys_excluded_from_object_memory(redis_url):
    schema = {
        "index": {"name": PREFIX, "prefix": PREFIX},
        "fields": [
            {"name": "_id", "type": "tag"},
            {"name": "text", "type": "text"},
            {
                "name": "vector",
                "type": "vector",
                "attrs": {
                    "dims": 3,
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

    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    index.load([{"_id": f"d{i}", "text": "t", "vector": vec} for i in range(3)])

    client = index.client
    try:
        # measure with no bookkeeping keys present
        client.delete(utils.LAST_SCHEMA_KEY, utils.LAST_INDEXING_TIME_KEY)
        absent = get_index_memory_stats(PREFIX, PREFIX, redis_url)[
            "total_object_memory_mb"
        ]

        # add the bookkeeping keys under the same prefix
        utils.set_last_index_settings(redis_url, {"name": PREFIX})
        utils.set_last_indexing_time(redis_url, 1.23)
        present = get_index_memory_stats(PREFIX, PREFIX, redis_url)[
            "total_object_memory_mb"
        ]

        assert absent > 0  # the 3 real docs are counted
        # excluding bookkeeping keys means the count is unchanged by their presence
        assert present == absent
    finally:
        client.delete(utils.LAST_SCHEMA_KEY, utils.LAST_INDEXING_TIME_KEY)
        index.delete(drop=True)

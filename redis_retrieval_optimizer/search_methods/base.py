import time

from redisvl.index import SearchIndex
from redisvl.query import BaseQuery

from redis_retrieval_optimizer.schema import QueryMetrics


def run_search_w_time(
    index: SearchIndex, query: BaseQuery, query_metrics: QueryMetrics
) -> tuple:
    start_time = time.time()
    res = index.query(query)
    query_time = time.time() - start_time
    query_metrics.query_times.append(query_time)
    return res

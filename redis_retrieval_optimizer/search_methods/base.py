import time
from functools import wraps

from redisvl.index import SearchIndex
from redisvl.query import BaseQuery

from redis_retrieval_optimizer.schema import QueryMetrics


def time_query(func):
    """Decorator to measure query execution time and record it in QueryMetrics.

    Ensures a timing entry is added even if the wrapped function raises.
    """

    @wraps(func)
    def wrapper(
        index: SearchIndex,
        query: BaseQuery,
        query_metrics: QueryMetrics,
        *args,
        **kwargs,
    ) -> tuple:
        start_time = time.time()
        try:
            return func(index, query, query_metrics, *args, **kwargs)
        finally:
            elapsed = time.time() - start_time
            query_metrics.query_times.append(elapsed)

    return wrapper


@time_query
def run_search_w_time(
    index: SearchIndex, query: BaseQuery, query_metrics: QueryMetrics
) -> tuple:
    return index.query(query)

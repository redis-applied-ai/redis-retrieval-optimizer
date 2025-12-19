import logging

import pandas as pd
from ranx import Qrels
from redis import Redis
from redis.commands.json.path import Path
from redisvl.index import SearchIndex

from redis_retrieval_optimizer import utils as utils
from redis_retrieval_optimizer.schema import SearchMethodInput
from redis_retrieval_optimizer.search_methods import SEARCH_METHOD_MAP

logger = logging.getLogger(__name__)


def update_search_metric_row(
    metrics,
    search_study_config,
    search_method,
    trial_metrics: dict,
):
    metrics["search_method"].append(search_method)
    metrics["ret_k"].append(search_study_config.ret_k)
    metrics["recall"].append(trial_metrics["recall"])
    metrics["ndcg"].append(trial_metrics["ndcg"])
    metrics["precision"].append(trial_metrics["precision"])
    metrics["f1"].append(trial_metrics["f1"])
    metrics["total_indexing_time"].append(trial_metrics["total_indexing_time"])
    metrics["avg_query_time"].append(trial_metrics["query_stats"]["avg_query_time"])
    metrics["total_memory_mb"].append(trial_metrics["total_memory_mb"])
    return metrics


def persist_search_metrics(metrics, redis_url, study_id):
    client = Redis.from_url(redis_url)
    client.json().set(f"study:{study_id}", Path.root_path(), metrics)


def run_search_study(
    config_path: str,
    redis_url: str,
    search_method_map=SEARCH_METHOD_MAP,
):
    search_study_config = utils.load_search_study_config(config_path)

    # load queries and qrels
    queries = utils.load_json(search_study_config.queries)
    qrels = Qrels(utils.load_json(search_study_config.qrels))

    # connect to existing index
    logger.info("Connecting to existing index: %s", search_study_config.index_name)
    index = SearchIndex.from_existing(
        name=search_study_config.index_name,
        redis_url=redis_url,
    )
    logger.info(
        "Connected to index: %s with %s objects",
        index.name,
        index.info()["num_docs"],
    )

    # Get index info for metrics
    index_info = index.info()

    # Get embedding model from config
    emb_model = utils.get_embedding_model(
        search_study_config.embedding_model, redis_url
    )

    metrics: dict = {
        "search_method": [],
        "total_indexing_time": [],
        "total_memory_mb": [],
        "avg_query_time": [],
        "recall": [],
        "ndcg": [],
        "f1": [],
        "precision": [],
        "ret_k": [],
    }

    for search_method in search_study_config.search_methods:
        logger.info("Running search method: %s", search_method)

        # get search method to try
        search_fn = search_method_map[search_method]
        search_input = SearchMethodInput(
            index=index,
            raw_queries=queries,
            emb_model=emb_model,
            id_field_name=search_study_config.id_field_name,
            vector_field_name=search_study_config.vector_field_name,
            text_field_name=search_study_config.text_field_name,
            ret_k=search_study_config.ret_k,
        )

        search_method_output = search_fn(search_input)

        trial_metrics = utils.eval_trial_metrics(qrels, search_method_output.run)

        last_indexing_time = utils.get_last_indexing_time(redis_url)
        trial_metrics["total_indexing_time"] = (
            last_indexing_time if last_indexing_time is not None else 0.0
        )

        memory_stats = utils.get_index_memory_stats(
            search_study_config.index_name,
            index_info.get("prefix", "ret-opt"),  # Default prefix
            redis_url,
        )

        trial_metrics["total_memory_mb"] = (
            memory_stats["total_index_memory_sz_mb"]
            + memory_stats["total_object_memory_mb"]
        )

        trial_metrics["query_stats"] = utils.get_query_time_stats(
            search_method_output.query_metrics.query_times
        )

        metrics = update_search_metric_row(
            metrics,
            search_study_config,
            search_method,
            trial_metrics,
        )

        persist_search_metrics(metrics, redis_url, search_study_config.study_id)

    return pd.DataFrame(metrics)

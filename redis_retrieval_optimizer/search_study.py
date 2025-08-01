import pandas as pd
from ranx import Qrels
from redis import Redis
from redis.commands.json.path import Path
from redisvl.index import SearchIndex

import redis_retrieval_optimizer.utils as utils
from redis_retrieval_optimizer.schema import SearchMethodInput, SearchStudyConfig
from redis_retrieval_optimizer.search_methods import SEARCH_METHOD_MAP


def update_search_metric_row(
    metrics,
    search_study_config,
    search_method,
    trial_metrics: dict,
    index_info: dict,
):
    metrics["search_method"].append(search_method)
    metrics["ret_k"].append(search_study_config.ret_k)
    metrics["algorithm"].append(index_info.get("algorithm", "unknown"))
    metrics["ef_construction"].append(index_info.get("ef_construction", 0))
    metrics["ef_runtime"].append(index_info.get("ef_runtime", 0))
    metrics["m"].append(index_info.get("m", 0))
    metrics["distance_metric"].append(index_info.get("distance_metric", "unknown"))
    metrics["vector_data_type"].append(index_info.get("vector_data_type", "unknown"))
    metrics["recall"].append(trial_metrics["recall"])
    metrics["ndcg"].append(trial_metrics["ndcg"])
    metrics["precision"].append(trial_metrics["precision"])
    metrics["f1"].append(trial_metrics["f1"])
    metrics["total_indexing_time"].append(trial_metrics["total_indexing_time"])
    metrics["avg_query_time"].append(trial_metrics["query_stats"]["avg_query_time"])
    metrics["total_index_memory_sz_mb"].append(
        trial_metrics["memory_stats"]["total_index_memory_sz_mb"]
    )
    metrics["total_object_memory_mb"].append(
        trial_metrics["memory_stats"]["total_object_memory_mb"]
    )
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
    print(f"Connecting to existing index: {search_study_config.existing_index_name}")
    index = SearchIndex.from_existing(
        name=search_study_config.existing_index_name,
        redis_url=redis_url,
    )
    print(f"Connected to index: {index.name} with {index.info()['num_docs']} objects")

    # Get index info for metrics
    index_info = index.info()

    # Get embedding model from config
    emb_model = utils.get_embedding_model(
        search_study_config.embedding_model, redis_url
    )

    metrics: dict = {
        "search_method": [],
        "total_indexing_time": [],
        "total_index_memory_sz_mb": [],
        "total_object_memory_mb": [],
        "avg_query_time": [],
        "recall": [],
        "ndcg": [],
        "f1": [],
        "precision": [],
        "ret_k": [],
        "algorithm": [],
        "ef_construction": [],
        "ef_runtime": [],
        "m": [],
        "distance_metric": [],
        "vector_data_type": [],
    }

    for search_method in search_study_config.search_methods:
        print(f"Running search method: {search_method}")

        # get search method to try
        search_fn = search_method_map[search_method]
        search_input = SearchMethodInput(
            index=index,
            raw_queries=queries,
            emb_model=emb_model,  # Provide embedding model for vector-based methods
            id_field_name="_id",  # Default field names
            vector_field_name="vector",
            text_field_name="text",
        )

        search_method_output = search_fn(search_input)

        trial_metrics = utils.eval_trial_metrics(qrels, search_method_output.run)

        trial_metrics["total_indexing_time"] = index_info["total_indexing_time"]

        trial_metrics["memory_stats"] = utils.get_index_memory_stats(
            search_study_config.existing_index_name,
            index_info.get("prefix", "ret-opt"),  # Default prefix
            redis_url,
        )
        trial_metrics["query_stats"] = utils.get_query_time_stats(
            search_method_output.query_metrics.query_times
        )

        metrics = update_search_metric_row(
            metrics,
            search_study_config,
            search_method,
            trial_metrics,
            index_info,
        )

        persist_search_metrics(metrics, redis_url, search_study_config.study_id)

    return pd.DataFrame(metrics)

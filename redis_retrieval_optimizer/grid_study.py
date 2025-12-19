import logging
import os
import time
from typing import Callable

import pandas as pd
from ranx import Qrels
from redis import Redis
from redis.commands.json.path import Path
from redisvl.index import SearchIndex

import redis_retrieval_optimizer.utils as utils
from redis_retrieval_optimizer.schema import GridStudyConfig, SearchMethodInput
from redis_retrieval_optimizer.search_methods import SEARCH_METHOD_MAP

logger = logging.getLogger(__name__)


def update_metric_row(
    metrics,
    grid_study_config,
    search_method,
    embedding_settings,
    trial_metrics: dict,
    vector_data_type: str = None,
):
    metrics["search_method"].append(search_method)
    metrics["ret_k"].append(grid_study_config.ret_k)
    metrics["algorithm"].append(grid_study_config.index_settings.algorithm)
    metrics["ef_construction"].append(grid_study_config.index_settings.ef_construction)
    metrics["ef_runtime"].append(grid_study_config.index_settings.ef_runtime)
    metrics["m"].append(grid_study_config.index_settings.m)
    metrics["distance_metric"].append(grid_study_config.index_settings.distance_metric)
    metrics["vector_data_type"].append(
        vector_data_type or grid_study_config.index_settings.vector_data_type
    )
    metrics["model"].append(embedding_settings.model)
    metrics["model_dim"].append(embedding_settings.dim)
    metrics["recall"].append(trial_metrics["recall"])
    metrics["ndcg"].append(trial_metrics["ndcg"])
    metrics["precision"].append(trial_metrics["precision"])
    metrics["f1"].append(trial_metrics["f1"])
    metrics["total_indexing_time"].append(trial_metrics["total_indexing_time"])
    metrics["avg_query_time"].append(trial_metrics["query_stats"]["avg_query_time"])
    metrics["total_memory_mb"].append(trial_metrics["total_memory_mb"])
    return metrics


def persist_metrics(metrics, redis_url, study_id):
    client = Redis.from_url(redis_url)
    client.json().set(f"study:{study_id}", Path.root_path(), metrics)


def init_index_from_grid_settings(
    grid_study_config: GridStudyConfig,
    redis_url: str,
    corpus_processor: Callable,
    dtype: str = None,
) -> SearchIndex:
    index_settings = grid_study_config.index_settings.model_dump()
    embed_settings = grid_study_config.embedding_models[0]
    index_settings["embedding"] = embed_settings.model_dump()

    # Use provided dtype or default from embedding model
    if dtype:
        index_settings["vector_data_type"] = dtype
        embed_settings.dtype = dtype

    if grid_study_config.index_settings.from_existing:
        print(f"Connecting to existing index: {grid_study_config.index_settings.name}")

        index = SearchIndex.from_existing(
            name=grid_study_config.index_settings.name,
            redis_url=redis_url,
        )
        print(
            f"Connected to index: {index.name} with {index.info()['num_docs']} objects"
        )
        print(
            f"From existing, assuming {grid_study_config.embedding_models[0].model} embedding model"
        )
        if (
            embed_settings.dim
            != index.schema.fields[
                grid_study_config.index_settings.vector_field_name
            ].attrs.dims
        ):
            raise ValueError(
                f"Embedding model dimension {embed_settings.dim} does not match index dimension {index.schema.fields[grid_study_config.index_settings.vector_field_name].attrs['dims']}"
            )
        utils.set_last_index_settings(redis_url, index_settings)
    else:
        last_index_settings = utils.get_last_index_settings(redis_url)
        recreate_index, recreate_data = utils.check_recreate(
            index_settings, last_index_settings
        )

        # Create a copy of index settings with current dtype
        current_index_settings = grid_study_config.index_settings.model_copy()
        if dtype:
            current_index_settings.vector_data_type = dtype

        schema = utils.schema_from_settings(current_index_settings)

        index = SearchIndex.from_dict(schema, redis_url=redis_url)
        index.create(overwrite=recreate_index, drop=recreate_data)

        if recreate_data:
            emb_model = utils.get_embedding_model(
                grid_study_config.embedding_models[0], redis_url, dtype=dtype
            )
            logger.info("Recreating: loading corpus from file")
            corpus = utils.load_json(grid_study_config.corpus)
            # corpus processing functions should be user defined
            corpus_data = corpus_processor(corpus, emb_model)

            indexing_start_time = time.time()
            index.load(corpus_data)

            while float(index.info()["percent_indexed"]) < 1:
                time.sleep(1)
                logging.info(f"Indexing progress: {index.info()['percent_indexed']}")

            total_indexing_time = time.time() - indexing_start_time
            utils.set_last_indexing_time(redis_url, total_indexing_time)

        index_settings["embedding"] = embed_settings.model_dump()
        utils.set_last_index_settings(redis_url, index_settings)

    return index


def run_grid_study(
    config_path: str,
    redis_url: str,
    corpus_processor: Callable,
    search_method_map=SEARCH_METHOD_MAP,
):
    grid_study_config = utils.load_grid_study_config(config_path)

    # load queries and qrels
    queries = utils.load_json(grid_study_config.queries)
    qrels = Qrels(utils.load_json(grid_study_config.qrels))

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
        "algorithm": [],
        "ef_construction": [],
        "ef_runtime": [],
        "m": [],
        "distance_metric": [],
        "vector_data_type": [],
        "model": [],
        "model_dim": [],
    }

    for i, embedding_model in enumerate(grid_study_config.embedding_models):
        for dtype in grid_study_config.vector_data_types:
            # Update index settings for current dtype
            current_index_settings = grid_study_config.index_settings.model_copy()
            current_index_settings.vector_data_type = dtype

            # Create or get index for current settings
            if i == 0 and dtype == grid_study_config.vector_data_types[0]:
                # First iteration - initialize index
                index = init_index_from_grid_settings(
                    grid_study_config, redis_url, corpus_processor, dtype=dtype
                )
            else:
                # Recreate index with new settings
                logger.info("Recreating index with dtype: %s", dtype)

                # Update index settings for current embedding model and dtype
                current_index_settings.vector_dim = embedding_model.dim

                schema = utils.schema_from_settings(current_index_settings)
                index = utils.index_from_schema(
                    schema, redis_url, recreate_index=True, recreate_data=True
                )

                logger.info("Recreating: loading corpus from file")
                emb_model = utils.get_embedding_model(
                    embedding_model, redis_url, dtype=dtype
                )
                corpus = utils.load_json(grid_study_config.corpus)

                # corpus processing functions should be user defined
                corpus_data = corpus_processor(corpus, emb_model)
                indexing_start_time = time.time()
                index.load(corpus_data)

                while float(index.info()["percent_indexed"]) < 1:
                    time.sleep(1)
                    logging.info(
                        f"Indexing progress: {index.info()['percent_indexed']}"
                    )

                total_indexing_time = time.time() - indexing_start_time
                utils.set_last_indexing_time(redis_url, total_indexing_time)

            # Get embedding model with current dtype
            emb_model = utils.get_embedding_model(
                embedding_model, redis_url, dtype=dtype
            )

            for search_method in grid_study_config.search_methods:
                logger.info(
                    "Running search method: %s with dtype: %s", search_method, dtype
                )
                # get search method to try
                search_fn = search_method_map[search_method]
                search_input = SearchMethodInput(
                    index=index,
                    raw_queries=queries,
                    emb_model=emb_model,
                    id_field_name=grid_study_config.index_settings.id_field_name,
                    vector_field_name=grid_study_config.index_settings.vector_field_name,
                    text_field_name=grid_study_config.index_settings.text_field_name,
                )

                search_method_output = search_fn(search_input)

                trial_metrics = utils.eval_trial_metrics(
                    qrels, search_method_output.run
                )

                last_indexing_time = utils.get_last_indexing_time(redis_url)

                trial_metrics["total_indexing_time"] = (
                    last_indexing_time if last_indexing_time is not None else 0.0
                )

                memory_stats = utils.get_index_memory_stats(
                    grid_study_config.index_settings.name,
                    grid_study_config.index_settings.prefix,
                    redis_url,
                )

                trial_metrics["total_memory_mb"] = (
                    memory_stats["total_index_memory_sz_mb"]
                    + memory_stats["total_object_memory_mb"]
                )

                trial_metrics["query_stats"] = utils.get_query_time_stats(
                    search_method_output.query_metrics.query_times
                )

                # Create embedding settings with current dtype for metrics
                embedding_settings_with_dtype = embedding_model.model_copy()
                embedding_settings_with_dtype.dtype = dtype

                metrics = update_metric_row(
                    metrics,
                    grid_study_config,
                    search_method,
                    embedding_settings_with_dtype,
                    trial_metrics,
                    vector_data_type=dtype,
                )

                persist_metrics(metrics, redis_url, grid_study_config.study_id)
    return pd.DataFrame(metrics)

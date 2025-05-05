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


def update_metric_row(
    metrics, grid_study_config, search_method, embedding_settings, trial_metrics: dict
):
    metrics["search_method"].append(search_method)
    metrics["ret_k"].append(grid_study_config.ret_k)
    metrics["algorithm"].append(grid_study_config.index_settings.algorithm)
    metrics["ef_construction"].append(grid_study_config.index_settings.ef_construction)
    metrics["ef_runtime"].append(grid_study_config.index_settings.ef_runtime)
    metrics["m"].append(grid_study_config.index_settings.m)
    metrics["distance_metric"].append(grid_study_config.index_settings.distance_metric)
    metrics["vector_data_type"].append(
        grid_study_config.index_settings.vector_data_type
    )
    metrics["model"].append(embedding_settings.model)
    metrics["model_dim"].append(embedding_settings.dim)
    metrics["recall@k"].append(trial_metrics["recall"])
    metrics["ndcg@k"].append(trial_metrics["ndcg"])
    metrics["precision"].append(trial_metrics["precision"])
    metrics["f1@k"].append(trial_metrics["f1"])
    metrics["total_indexing_time"].append(trial_metrics["total_indexing_time"])
    metrics["avg_query_time"].append(trial_metrics["query_stats"]["avg_query_time"])
    return metrics


def persist_metrics(metrics, redis_url, study_id):
    client = Redis.from_url(redis_url)
    client.json().set(f"study:{study_id}", Path.root_path(), metrics)


def init_index_from_grid_settings(
    grid_study_config: GridStudyConfig, redis_url: str, corpus_processor: Callable
) -> SearchIndex:
    index_settings = grid_study_config.index_settings.model_dump()
    embed_settings = grid_study_config.embedding_models[0]
    index_settings["embedding"] = embed_settings.model_dump()

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
            != index.schema.fields[grid_study_config.vector_field_name].attrs.dims
        ):
            raise ValueError(
                f"Embedding model dimension {emb_model.dims} does not match index dimension {index.schema.fields[grid_study_config.vector_field_name].attrs['dims']}"
            )
        utils.set_last_index_settings(redis_url, index_settings)
    else:
        last_index_settings = utils.get_last_index_settings(redis_url)
        recreate_index, recreate_data = utils.check_recreate(
            index_settings, last_index_settings
        )

        schema = utils.schema_from_settings(
            grid_study_config.index_settings,
        )

        index = SearchIndex.from_dict(schema, redis_url=redis_url)
        index.create(overwrite=recreate_index, drop=recreate_data)

        if recreate_data:
            emb_model = utils.get_embedding_model(
                grid_study_config.embedding_models[0], redis_url
            )
            print("Recreating: loading corpus from file")
            corpus = utils.load_json(grid_study_config.corpus)
            # corpus processing functions should be user defined
            corpus_data = corpus_processor(corpus, emb_model)

            index.load(corpus_data)

            while float(index.info()["percent_indexed"]) < 1:
                time.sleep(1)
                logging.info(f"Indexing progress: {index.info()['percent_indexed']}")

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

    index = init_index_from_grid_settings(
        grid_study_config, redis_url, corpus_processor
    )

    metrics: dict = {
        "search_method": [],
        "total_indexing_time": [],
        "avg_query_time": [],
        "recall@k": [],
        "ndcg@k": [],
        "f1@k": [],
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
        if i > 0:
            # assuming that you didn't pass the same embedding model twice like a fool
            print("Recreating index with new embedding model")

            # delete old index and data with embedding cache it's not expensive to recreate
            # consider potential of pre-fixing studies with study_id for separation
            index_settings = grid_study_config.index_settings

            # assign new vector info to index_settings
            index_settings.vector_data_type = embedding_model.dtype
            index_settings.vector_dim = embedding_model.dim

            schema = utils.schema_from_settings(index_settings)
            index = utils.index_from_schema(
                schema, redis_url, recreate_index=True, recreate_data=True
            )

            # TODO: be able to dump existing index corpus to file automatically which shouldn't be too hard
            print(
                "If using multiple embedding models assuming there is a json version of corpus available."
            )
            print("Recreating: loading corpus from file")
            emb_model = utils.get_embedding_model(embedding_model, redis_url)
            corpus = utils.load_json(grid_study_config.corpus)

            # corpus processing functions should be user defined
            corpus_data = corpus_processor(corpus, emb_model)
            index.load(corpus_data)

            while float(index.info()["percent_indexed"]) < 1:
                time.sleep(1)
                logging.info(f"Indexing progress: {index.info()['percent_indexed']}")

        # check if matches with last index settings
        emb_model = utils.get_embedding_model(embedding_model, redis_url)

        for search_method in grid_study_config.search_methods:
            print(f"Running search method: {search_method}")
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

            trial_metrics = utils.eval_trial_metrics(qrels, search_method_output.run)
            trial_metrics["total_indexing_time"] = round(
                float(index.info()["total_indexing_time"]) / 1000, 5
            )
            trial_metrics["query_stats"] = utils.get_query_time_stats(
                search_method_output.query_metrics.query_times
            )

            metrics = update_metric_row(
                metrics,
                grid_study_config,
                search_method,
                embedding_model,
                trial_metrics,
            )

            persist_metrics(metrics, redis_url, grid_study_config.study_id)

    return pd.DataFrame(metrics)

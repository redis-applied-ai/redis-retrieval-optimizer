from typing import Callable

import pandas as pd
from ranx import Qrels, Run
from redis import Redis
from redis.commands.json.path import Path
from redisvl.index import SearchIndex

import redis_retrieval_optimizer.utils as utils
from redis_retrieval_optimizer.schema import GridStudyConfig
from redis_retrieval_optimizer.search_methods import SEARCH_METHOD_MAP


# TODO: generalize metric update functions
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
    # metrics["embedding_latency"].append(trial_metrics["embedding_latency"])
    # metrics["avg_query_latency"].append(eval_obj.avg_query_latency)
    # metrics["obj_val"].append(eval_obj.obj_val)
    # metrics["retriever"].append(str(eval_obj.retriever.__name__))
    return metrics


def persist_metrics(metrics, redis_url, study_id):
    # update_metric_row(trial_settings, trial_metrics)
    # logging.info(f"Saving metrics for study: {study_id}, {METRICS=}")

    client = Redis.from_url(redis_url)
    client.json().set(f"study:{study_id}", Path.root_path(), metrics)


# TODO: generalize with other study configs
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
        recreate = utils.check_recreate_schema(index_settings, last_index_settings)

        schema = utils.schema_from_settings(
            grid_study_config.index_settings,
        )

        index = SearchIndex.from_dict(schema, redis_url=redis_url)
        index.create(overwrite=False, drop=False)

        if recreate:
            emb_model = utils.get_embedding_model(
                grid_study_config.embedding_models[0], redis_url
            )
            print("Recreating: loading corpus from file")
            corpus = utils.load_json(grid_study_config.corpus)
            # corpus processing functions should be user defined
            corpus_data = corpus_processor(corpus, emb_model)

            index.load(corpus_data)

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
        "ret_k": [],
        "algorithm": [],
        "ef_construction": [],
        "ef_runtime": [],
        "m": [],
        "distance_metric": [],
        "vector_data_type": [],
        "model": [],
        "model_dim": [],
        "recall@k": [],
        "ndcg@k": [],
        "f1@k": [],
        "total_indexing_time": [],
        "precision": [],
        # "indexing_time": [],
        # "avg_query_latency": [],
        # "obj_val": [],
        # "retriever": [],
    }

    for i, embedding_model in enumerate(grid_study_config.embedding_models):
        if i > 0:
            # assuming that you didn't pass the same embedding model twice like a fool
            print("Recreating index with new embedding model")
            index_settings = grid_study_config.index_settings.model_dump()
            index_settings["embedding"] = embedding_model.model_dump()

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

        # check if matches with last index settings
        emb_model = utils.get_embedding_model(embedding_model, redis_url)

        for search_method in grid_study_config.search_methods:
            print(f"Running search method: {search_method}")
            # get search method to try
            search_fn = search_method_map[search_method]
            trial_results = search_fn(queries, index, emb_model)

            run = Run(trial_results)

            trial_metrics = utils.eval_trial_metrics(qrels, run)
            # TODO: real values
            trial_metrics["total_indexing_time"] = 0

            metrics = update_metric_row(
                metrics,
                grid_study_config,
                search_method,
                embedding_model,
                trial_metrics,
            )

            persist_metrics(metrics, redis_url, grid_study_config.study_id)
            pd.DataFrame(metrics).to_csv(
                f"data/{grid_study_config.study_id}_metrics.csv", index=False
            )

    return metrics

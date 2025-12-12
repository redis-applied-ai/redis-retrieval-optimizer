import logging
import time
import warnings
from functools import partial
from typing import Callable

import optuna
import pandas as pd
from ranx import Qrels
from redis import Redis
from redis.commands.json.path import Path

# import search_methods.bm25
import redis_retrieval_optimizer.utils as utils
from redis_retrieval_optimizer.schema import (
    SearchMethodInput,
    TrialSettings,
    get_trial_settings,
)
from redis_retrieval_optimizer.search_methods import SEARCH_METHOD_MAP

warnings.filterwarnings("ignore")

METRICS: dict = {
    "search_method": [],
    "total_indexing_time": [],
    "avg_query_time": [],
    "model": [],
    "model_dim": [],
    "ret_k": [],
    "recall": [],
    "ndcg": [],
    "f1": [],
    "precision": [],
    "algorithm": [],
    "ef_construction": [],
    "ef_runtime": [],
    "m": [],
    "distance_metric": [],
    "vector_data_type": [],
    "objective_value": [],
    "total_memory_mb": [],
}


def update_metric_row(trial_settings: TrialSettings, trial_metrics: dict):
    METRICS["search_method"].append(trial_settings.search_method)
    METRICS["ret_k"].append(trial_settings.ret_k)
    METRICS["algorithm"].append(trial_settings.index_settings.algorithm)
    METRICS["ef_construction"].append(trial_settings.index_settings.ef_construction)
    METRICS["ef_runtime"].append(trial_settings.index_settings.ef_runtime)
    METRICS["m"].append(trial_settings.index_settings.m)
    METRICS["distance_metric"].append(trial_settings.index_settings.distance_metric)
    METRICS["vector_data_type"].append(trial_settings.index_settings.vector_data_type)
    METRICS["model"].append(trial_settings.embedding.model)
    METRICS["model_dim"].append(trial_settings.embedding.dim)
    METRICS["recall"].append(trial_metrics["recall"])
    METRICS["ndcg"].append(trial_metrics["ndcg"])
    METRICS["precision"].append(trial_metrics["precision"])
    METRICS["f1"].append(trial_metrics["f1"])
    METRICS["total_indexing_time"].append(trial_metrics["total_indexing_time"])
    METRICS["avg_query_time"].append(trial_metrics["avg_query_time"])
    METRICS["objective_value"].append(trial_metrics["objective_value"])
    METRICS["total_memory_mb"].append(trial_metrics["total_memory_mb"])


def persist_metrics(
    redis_url, trial_settings: TrialSettings, trial_metrics: dict, study_id
):
    update_metric_row(trial_settings, trial_metrics)
    logging.info(f"Saving metrics for study: {study_id}, {METRICS=}")

    client = Redis.from_url(redis_url)
    client.json().set(f"study:{study_id}", Path.root_path(), METRICS)


def norm_metric(value: float):
    """Normalize a metric value using 1/(1+value) formula.

    Handles edge cases:
    - When value is -1, returns a large positive number (infinity equivalent)
    - When value is very negative, returns a large positive number
    - When value is very positive, returns a small positive number
    """
    if value == -1:
        # Return a large positive number to represent "infinity" for optimization
        return 1000.0
    return 1 / (1 + value)


def cost_fn(metrics: dict, weights: dict):
    objective = 0
    for key in metrics:
        if key == "avg_query_time" or key == "total_indexing_time":
            objective += weights.get(key, 0) * -norm_metric(metrics[key])
        else:
            objective += weights.get(key, 0) * metrics[key]
    return objective


def objective(trial, study_config, redis_url, corpus_processor, search_method_map):

    # optimizer will select hyperparameters from available option in study_config
    trial_settings = get_trial_settings(trial, study_config)

    index_settings = trial_settings.index_settings.model_dump()
    index_settings["embedding"] = trial_settings.embedding.model_dump()

    last_index_settings = utils.get_last_index_settings(redis_url)
    recreate_index, recreate_data = utils.check_recreate(
        index_settings, last_index_settings
    )

    schema_dict = utils.schema_from_settings(trial_settings.index_settings)
    trial_index = utils.index_from_schema(
        schema_dict, redis_url, recreate_index, recreate_data
    )

    emb_model = utils.get_embedding_model(
        trial_settings.embedding,
        redis_url,
        dtype=trial_settings.index_settings.vector_data_type,
    )


    if recreate_data:
        logging.info("Recreating index...")
        corpus = utils.load_json(study_config.corpus)
        corpus_data = corpus_processor(corpus, emb_model)
        corpus_size = len(corpus_data)
        logging.info(f"Corpus size: {corpus_size}")

        # reload data and measure wall-clock time until indexing completes
        indexing_start_time = time.time()
        trial_index.load(corpus_data)

        while float(trial_index.info()["percent_indexed"]) < 1:
            time.sleep(1)
            logging.info(f"Indexing progress: {trial_index.info()['percent_indexed']}")
    else:
        # Only wait if index is not fully indexed
        if float(trial_index.info()["percent_indexed"]) < 1:
            while float(trial_index.info()["percent_indexed"]) < 1:
                time.sleep(1)
                logging.info(f"Indexing progress: {trial_index.info()['percent_indexed']}")

    if recreate_data:
        assert indexing_start_time is not None
        total_indexing_time = time.time() - indexing_start_time
        utils.set_last_indexing_time(redis_url, total_indexing_time)
    else:
        last_indexing_time = utils.get_last_indexing_time(redis_url)
        total_indexing_time = (
            last_indexing_time if last_indexing_time is not None else 0.0
        )

    num_docs = trial_index.info()["num_docs"]

    logging.info(f"Data indexed {total_indexing_time=}s, {num_docs=}")

    if num_docs == 0:
        raise ValueError("No documents indexed, check corpus and index settings")

    # save config since it loaded
    index_settings["embedding"] = trial_settings.embedding.model_dump()
    utils.set_last_index_settings(redis_url, index_settings)

    # get search method to try
    search_fn = search_method_map[trial_settings.search_method]

    # run search method
    queries = utils.load_json(study_config.queries)
    qrels = Qrels(utils.load_json(study_config.qrels))

    search_input = SearchMethodInput(
        index=trial_index,
        raw_queries=queries,
        emb_model=emb_model,
        vector_field_name=index_settings["vector_field_name"],
        text_field_name=index_settings["text_field_name"],
    )

    search_method_output = search_fn(search_input)

    trial_metrics = utils.eval_trial_metrics(qrels, search_method_output.run)
    trial_metrics["total_indexing_time"] = total_indexing_time
    trial_metrics["avg_query_time"] = utils.get_query_time_stats(
        search_method_output.query_metrics.query_times
    )["avg_query_time"]

    memory_stats = utils.get_index_memory_stats(
        trial_index.name, trial_index.prefix, redis_url
    )

    trial_metrics["total_memory_mb"] = (
        memory_stats["total_index_memory_sz_mb"]
        + memory_stats["total_object_memory_mb"]
    )

    trial_metrics["objective_value"] = cost_fn(
        trial_metrics, study_config.optimization_settings.metric_weights.model_dump()
    )

    # save results as we go in case of failure
    persist_metrics(redis_url, trial_settings, trial_metrics, study_config.study_id)

    return trial_metrics["objective_value"]


def run_bayes_study(
    config_path: str,
    redis_url: str,
    corpus_processor: Callable,
    search_method_map=SEARCH_METHOD_MAP,
):

    study_config = utils.load_bayes_study_config(config_path)

    study = optuna.create_study(
        study_name="test",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    obj = partial(
        objective,
        study_config=study_config,
        redis_url=redis_url,
        corpus_processor=corpus_processor,
        search_method_map=search_method_map,
    )

    study.optimize(
        obj,
        n_trials=study_config.optimization_settings.n_trials,
        n_jobs=study_config.optimization_settings.n_jobs,
    )

    logging.info(f"Completed Bayesian optimization... \n\n")

    best_trial = study.best_trial
    logging.info(f"Best Configuration: {best_trial.number}: {best_trial.params}:\n\n")
    logging.info(f"Best Score: {best_trial.values}\n\n")

    return pd.DataFrame(METRICS)

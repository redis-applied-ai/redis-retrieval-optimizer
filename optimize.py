import argparse
import logging
import time
import warnings
from functools import partial

import numpy as np
import optuna
import pandas as pd
import yaml
from beir.retrieval.evaluation import EvaluateRetrieval
from ranx import Qrels, Run, evaluate
from redis import Redis
from redis.commands.json.path import Path

import eval_beir
import search_methods

# import search_methods.bm25
import utils
from schema import StudyConfig, TrialSettings, get_trial_settings

SEARCH_METHOD_MAP = {
    "bm25": search_methods.bm25.gather_bm25_results,
    "rerank": search_methods.rerank.gather_rerank_results,
    "lin_combo": search_methods.lin_combo.gather_lin_combo_results,
    "vector": search_methods.vector.gather_vector_results,
    "weighted_rrf": search_methods.weighted_rrf.gather_weighted_rrf,
}

warnings.filterwarnings("ignore")

METRICS: dict = {
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


def load_config(config_path: str) -> StudyConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return StudyConfig(**config)


def update_metric_row(trial_settings: TrialSettings, trial_metrics: dict):
    METRICS["search_method"].append(trial_settings.search_method)
    METRICS["ret_k"].append(trial_settings.ret_k)
    METRICS["algorithm"].append(trial_settings.index.algorithm)
    METRICS["ef_construction"].append(trial_settings.index.ef_construction)
    METRICS["ef_runtime"].append(trial_settings.index.ef_runtime)
    METRICS["m"].append(trial_settings.index.m)
    METRICS["distance_metric"].append(trial_settings.index.distance_metric)
    METRICS["vector_data_type"].append(trial_settings.index.vector_data_type)
    METRICS["model"].append(trial_settings.embedding.model)
    METRICS["model_dim"].append(trial_settings.embedding.dim)
    METRICS["recall@k"].append(trial_metrics["recall"])
    METRICS["ndcg@k"].append(trial_metrics["ndcg"])
    METRICS["precision"].append(trial_metrics["precision"])
    METRICS["f1@k"].append(trial_metrics["f1"])
    METRICS["total_indexing_time"].append(trial_metrics["total_indexing_time"])
    # METRICS["embedding_latency"].append(trial_metrics["embedding_latency"])
    # METRICS["avg_query_latency"].append(eval_obj.avg_query_latency)
    # METRICS["obj_val"].append(eval_obj.obj_val)
    # METRICS["retriever"].append(str(eval_obj.retriever.__name__))


def get_last_index_settings(redis_url):
    client = Redis.from_url(redis_url)
    return client.json().get("ret-opt:last_schema")


def set_last_index_settings(redis_url, index_settings):
    client = Redis.from_url(redis_url)
    client.json().set("ret-opt:last_schema", Path.root_path(), index_settings)


def check_recreate_schema(index_settings, last_index_settings):
    if not last_index_settings:
        return True
    if last_index_settings and index_settings != last_index_settings:
        return True
    return False


def persist_metrics(
    redis_url, trial_settings: TrialSettings, trial_metrics: dict, study_id
):
    update_metric_row(trial_settings, trial_metrics)
    logging.info(f"Saving metrics for study: {study_id}, {METRICS=}")

    client = Redis.from_url(redis_url)
    client.json().set(f"study:{study_id}", Path.root_path(), METRICS)


def cost_fn(metrics: dict, weights: dict):
    objective = 0
    for key in metrics:
        objective += weights.get(key, 0) * metrics[key]
    return objective


def norm_metric(value: float):
    return 1 / (1 + value)


def objective(trial, study_config, custom_retrievers):

    # optimizer will select hyperparameters from available option in study_config
    trial_settings = get_trial_settings(
        trial, study_config, custom_retrievers=custom_retrievers
    )

    index_settings = trial_settings.index.model_dump()
    index_settings["embedding"] = trial_settings.embedding.model_dump()
    last_index_settings = get_last_index_settings(study_config.redis_url)
    recreate = check_recreate_schema(index_settings, last_index_settings)

    schema_dict = utils.schema_from_settings(trial_settings)
    trial_index = utils.index_from_schema(schema_dict, study_config.redis_url, recreate)

    emb_model = utils.get_embedding_model(trial_settings.embedding)

    if recreate:
        print("Recreating index...")
        corpus = utils.load_json(study_config.corpus)
        corpus_data = eval_beir.process_corpus(corpus, emb_model)

        trial_index.load(corpus_data)
    else:
        print("Skip recreate")

    while float(trial_index.info()["percent_indexed"]) < 1:
        time.sleep(1)
        logging.info(f"Indexing progress: {trial_index.info()['percent_indexed']}")

    # save config since it loaded
    set_last_index_settings(study_config.redis_url, index_settings)

    # capture index metrics
    total_indexing_time = round(
        float(trial_index.info()["total_indexing_time"]) / 1000, 3
    )
    num_docs = trial_index.info()["num_docs"]
    logging.info(f"Data indexed {total_indexing_time=}s, {num_docs=}")

    # get search method to try
    search_fn = SEARCH_METHOD_MAP[trial_settings.search_method]

    # run search method
    queries = utils.load_json(study_config.queries)
    trial_results = search_fn(queries, trial_index, emb_model)

    qrels = Qrels(utils.load_json(study_config.qrels))
    run = Run(trial_results)

    ndcg = evaluate(qrels, run, metrics=["ndcg"])
    recall = evaluate(qrels, run, metrics=["recall"])
    f1 = evaluate(qrels, run, metrics=["f1"])
    precision = evaluate(qrels, run, metrics=["precision"])

    trial_metrics = {
        "ndcg": ndcg,
        "recall": recall,
        "f1": f1,
        "precision": precision,
        "total_indexing_time": total_indexing_time,
    }

    # save results as we go in case of failure
    persist_metrics(
        study_config.redis_url, trial_settings, trial_metrics, study_config.study_id
    )

    return cost_fn(trial_metrics, study_config.metric_weights.dict())


def run_study(study_config: StudyConfig, custom_retrievers=None, save_pandas=False):

    study = optuna.create_study(
        study_name="test",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    obj = partial(
        objective,
        study_config=study_config,
        custom_retrievers=custom_retrievers,
    )

    study.optimize(
        obj,
        n_trials=study_config.n_trials,
        n_jobs=study_config.n_jobs,
    )

    print(f"Completed Bayesian optimization... \n\n")

    best_trial = study.best_trial
    print(f"Best Configuration: {best_trial.number}: {best_trial.params}:\n\n")
    print(f"Best Score: {best_trial.values}\n\n")

    if save_pandas:
        pd.DataFrame(METRICS).to_csv(
            f"data/{study_config.study_id}_metrics.csv", index=False
        )


def run_study_cli():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for Redis Vector Store given config file"
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    study_config = load_config(args.config)
    run_study(study_config)


if __name__ == "__main__":
    time_start = time.time()
    study_config = utils.load_study_config("study_config.yaml")
    run_study(study_config, save_pandas=True)
    print(f"Total time taken: {round((time.time() - time_start) / 60, 2)} minutes")

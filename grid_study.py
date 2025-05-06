import os
from uuid import uuid4

import pandas as pd
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from ranx import Qrels, Run, evaluate
from redis import Redis
from redis.commands.json.path import Path
from redisvl.index import SearchIndex

import eval_beir
import search_methods
import utils

load_dotenv()


# move to schema
class AdditionalField(BaseModel):
    name: str
    type: str


class IndexSettings(BaseModel):
    name: str = "ret-opt"
    from_existing: bool = False
    algorithm: str = "flat"
    distance_metric: str = "cosine"
    vector_data_type: str = "float32"
    vector_dim: int = 368
    ef_construction: int = 0
    ef_runtime: int = 0
    m: int = 0
    additional_fields: list[AdditionalField] = []


class EmbeddingModel(BaseModel):
    type: str
    model: str
    dim: int
    embedding_cache_name: str = ""
    embedding_cache_redis_url: str = "redis://localhost:6379/0"


class GridStudyConfig(BaseModel):
    study_id: str = str(uuid4())
    # index settings
    index_settings: IndexSettings

    # data
    corpus: str = ""
    qrels: str
    queries: str

    vector_field_name: str = "vector"
    text_field_name: str = "text"
    primary_id_field_name: str = "_id"  # this is what links corpus to qrels

    index_settings: IndexSettings

    embedding_models: list[EmbeddingModel]
    search_methods: list[str]
    ret_k: int = 6


# move to utils


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


def persist_metrics(metrics, redis_url, study_id):
    # update_metric_row(trial_settings, trial_metrics)
    # logging.info(f"Saving metrics for study: {study_id}, {METRICS=}")

    client = Redis.from_url(redis_url)
    client.json().set(f"study:{study_id}", Path.root_path(), metrics)


def load_grid_study_config(config_path: str) -> GridStudyConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return GridStudyConfig(**config)


def schema_from_settings(index_settings: IndexSettings, additional_schema_fields=None):
    schema = {
        "index": {"name": "optimize", "prefix": "ret-opt"},
        "fields": [
            {"name": "_id", "type": "tag"},
            {"name": "text", "type": "text"},
            {"name": "title", "type": "text"},
            {
                "name": "vector",
                "type": "vector",
                "attrs": {
                    "dims": index_settings.vector_dim,
                    "distance_metric": index_settings.distance_metric,
                    "algorithm": index_settings.algorithm,
                    "datatype": index_settings.vector_data_type,
                    "ef_construction": index_settings.ef_construction,
                    "ef_runtime": index_settings.ef_runtime,
                    "m": index_settings.m,
                },
            },
        ],
    }

    # define a custom search method to do pre-filtering etc.
    if additional_schema_fields:
        for field in additional_schema_fields:
            schema["fields"].append(field)  # type: ignore

    return schema


# TODO: generalize with other study configs
def init_index_from_grid_settings(grid_study_config: GridStudyConfig) -> SearchIndex:
    redis_url = os.environ.get("REDIS_URL")

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
        set_last_index_settings(redis_url, index_settings)
    else:
        last_index_settings = get_last_index_settings(grid_study_config.redis_url)
        recreate = check_recreate_schema(index_settings, last_index_settings)

        schema = schema_from_settings(
            grid_study_config.index_settings,
            additional_schema_fields=grid_study_config.index_settings.additional_fields,
        )

        index = SearchIndex.from_dict(schema, redis_url=redis_url)

        if recreate:
            emb_model = utils.get_embedding_model(grid_study_config.embedding_models[0])
            print("Recreating: loading corpus from file")
            corpus = utils.load_json(grid_study_config.corpus)
            # corpus processing functions should be user defined
            corpus_data = eval_beir.process_corpus(corpus, emb_model)

            index.load(corpus_data)

    return index


def run_grid_study(config_path: str):
    grid_study_config = load_grid_study_config(config_path)
    redis_url = os.environ.get("REDIS_URL")

    # load queries and qrels
    queries = utils.load_json(grid_study_config.queries)
    qrels = Qrels(utils.load_json(grid_study_config.qrels))

    index = init_index_from_grid_settings(grid_study_config)

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
            emb_model = utils.get_embedding_model(embedding_model)
            corpus = utils.load_json(grid_study_config.corpus)
            # corpus processing functions should be user defined
            corpus_data = eval_beir.process_corpus(corpus, emb_model)
            index.load(corpus_data)

        # check if matches with last index settings
        emb_model = utils.get_embedding_model(embedding_model)

        for search_method in grid_study_config.search_methods:
            print(f"Running search method: {search_method}")
            # get search method to try
            search_fn = search_methods.SEARCH_METHOD_MAP[search_method]
            trial_results = search_fn(queries, index, emb_model)

            run = Run(trial_results)

            # TODO: generalize with bayesian optimization
            ndcg = evaluate(qrels, run, metrics=["ndcg"])
            recall = evaluate(qrels, run, metrics=["recall"])
            f1 = evaluate(qrels, run, metrics=["f1"])
            precision = evaluate(qrels, run, metrics=["precision"])

            trial_metrics = {
                "ndcg": ndcg,
                "recall": recall,
                "f1": f1,
                "precision": precision,
                "total_indexing_time": 0,
            }

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


if __name__ == "__main__":
    config_path = "grid_study_config.yaml"
    metrics = run_grid_study(config_path)

import json
import os

import yaml
from ranx import Qrels, Run, evaluate
from redis import Redis
from redis.commands.json.path import Path
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.index import SearchIndex

# from redisvl.extensions.cache.embeddings import EmbeddingsCache
# from redisvl.utils.vectorize import vectorizer_from_dict TODO: update actual function in redisvl
from redisvl.utils.vectorize.base import BaseVectorizer, Vectorizers
from redisvl.utils.vectorize.text.azureopenai import AzureOpenAITextVectorizer
from redisvl.utils.vectorize.text.cohere import CohereTextVectorizer
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer
from redisvl.utils.vectorize.text.mistral import MistralAITextVectorizer
from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer
from redisvl.utils.vectorize.text.vertexai import VertexAITextVectorizer
from redisvl.utils.vectorize.text.voyageai import VoyageAITextVectorizer

from redis_retrieval_optimizer.schema import (
    BayesStudyConfig,
    EmbeddingModel,
    GridStudyConfig,
    IndexSettings,
)


def vectorizer_from_dict(
    vectorizer: dict,
    cache: dict = {},
) -> BaseVectorizer:
    vectorizer_type = Vectorizers(vectorizer["type"])
    model = vectorizer["model"]
    dtype = vectorizer.get("dtype", "float32")

    args = {"model": model, "dtype": dtype}
    if cache:
        emb_cache = EmbeddingsCache(**cache)
        args["cache"] = emb_cache

    if vectorizer_type == Vectorizers.cohere:
        return CohereTextVectorizer(**args)
    elif vectorizer_type == Vectorizers.openai:
        return OpenAITextVectorizer(**args)
    elif vectorizer_type == Vectorizers.azure_openai:
        return AzureOpenAITextVectorizer(**args)
    elif vectorizer_type == Vectorizers.hf:
        return HFTextVectorizer(**args)
    elif vectorizer_type == Vectorizers.mistral:
        return MistralAITextVectorizer(**args)
    elif vectorizer_type == Vectorizers.vertexai:
        return VertexAITextVectorizer(**args)
    elif vectorizer_type == Vectorizers.voyageai:
        return VoyageAITextVectorizer(**args)
    else:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")


def get_last_index_settings(redis_url):
    client = Redis.from_url(redis_url)
    return client.json().get("ret-opt:last_schema")


def set_last_index_settings(redis_url, index_settings):
    client = Redis.from_url(redis_url)
    client.json().set("ret-opt:last_schema", Path.root_path(), index_settings)


def check_recreate(index_settings, last_index_settings):
    embedding_settings = index_settings.pop("embedding") if index_settings else None
    last_embedding_settings = (
        last_index_settings.pop("embedding") if last_index_settings else None
    )

    if not last_index_settings:
        recreate_index = True
        recreate_data = True
    elif index_settings != last_index_settings:
        recreate_index = True

        if embedding_settings != last_embedding_settings:
            recreate_data = True
        else:
            recreate_data = False
    else:
        recreate_index = False
        recreate_data = False

    return recreate_index, recreate_data


def get_embedding_model(
    embedding_model: EmbeddingModel, redis_url: str, dtype=None
) -> BaseVectorizer:
    vectorizer = {"type": embedding_model.type, "model": embedding_model.model}
    if dtype:
        # optimization can override input dtype
        vectorizer["dtype"] = dtype
    else:
        vectorizer["dtype"] = embedding_model.dtype

    return vectorizer_from_dict(
        vectorizer=vectorizer,
        cache={
            "name": embedding_model.embedding_cache_name,
            "redis_url": redis_url,
        },
    )


def load_bayes_study_config(config_path: str) -> BayesStudyConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return BayesStudyConfig(**config)


def load_grid_study_config(config_path: str) -> GridStudyConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return GridStudyConfig(**config)


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def schema_from_settings(index_settings: IndexSettings):
    schema = {
        "index": {"name": index_settings.name, "prefix": index_settings.prefix},
        "fields": [
            {"name": index_settings.id_field_name, "type": "tag"},
            {"name": index_settings.text_field_name, "type": "text"},
            {
                "name": index_settings.vector_field_name,
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
    if index_settings.additional_fields:
        for field in index_settings.additional_fields:
            schema["fields"].append({"name": field.name, "type": field.type})  # type: ignore

    return schema


def index_from_schema(schema, redis_url, recreate_index, recreate_data):
    index = SearchIndex.from_dict(schema, redis_url=redis_url)

    if recreate_index:
        if recreate_data:
            index.delete(drop=True)
        index.create()

    return index


def eval_trial_metrics(qrels: Qrels, run: Run):
    ndcg = evaluate(qrels, run, metrics=["ndcg"])
    recall = evaluate(qrels, run, metrics=["recall"])
    f1 = evaluate(qrels, run, metrics=["f1"])
    precision = evaluate(qrels, run, metrics=["precision"])

    return {"ndcg": ndcg, "recall": recall, "f1": f1, "precision": precision}


def get_query_time_stats(query_times: list[float]):
    """
    Calculate the average and standard deviation of query times.
    """
    import numpy as np

    avg_query_time = np.mean(query_times)
    std_query_time = np.std(query_times)
    min_query_time = np.min(query_times)
    max_query_time = np.max(query_times)

    return {
        "avg_query_time": avg_query_time,
        "std_query_time": std_query_time,
        "min_query_time": min_query_time,
        "max_query_time": max_query_time,
    }

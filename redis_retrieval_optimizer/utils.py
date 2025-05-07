import json
import os

import yaml
from redis import Redis
from redis.commands.json.path import Path
from redisvl.index import SearchIndex

# from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize import vectorizer_from_dict
from redisvl.utils.vectorize.base import BaseVectorizer, Vectorizers

from redis_retrieval_optimizer.schema import EmbeddingModel, StudyConfig, TrialSettings

# from redisvl.utils.vectorize.text.azureopenai import AzureOpenAITextVectorizer
# from redisvl.utils.vectorize.text.bedrock import BedrockTextVectorizer
# from redisvl.utils.vectorize.text.cohere import CohereTextVectorizer
# from redisvl.utils.vectorize.text.custom import CustomTextVectorizer
# from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer
# from redisvl.utils.vectorize.text.mistral import MistralAITextVectorizer
# from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer
# from redisvl.utils.vectorize.text.vertexai import VertexAITextVectorizer
# from redisvl.utils.vectorize.text.voyageai import VoyageAITextVectorizer


# __all__ = [
#     "BaseVectorizer",
#     "CohereTextVectorizer",
#     "HFTextVectorizer",
#     "OpenAITextVectorizer",
#     "VertexAITextVectorizer",
#     "AzureOpenAITextVectorizer",
#     "MistralAITextVectorizer",
#     "CustomTextVectorizer",
#     "BedrockTextVectorizer",
#     "VoyageAITextVectorizer",
# ]


# def vectorizer_from_dict(
#     vectorizer: dict,
#     cache: dict = {},
# ) -> BaseVectorizer:
#     vectorizer_type = Vectorizers(vectorizer["type"])

#     args = {"model": vectorizer["model"]}
#     if cache:
#         emb_cache = EmbeddingsCache(**cache)
#         args["cache"] = emb_cache

#     if vectorizer_type == Vectorizers.cohere:
#         return CohereTextVectorizer(**args)
#     elif vectorizer_type == Vectorizers.openai:
#         return OpenAITextVectorizer(**args)
#     elif vectorizer_type == Vectorizers.azure_openai:
#         return AzureOpenAITextVectorizer(**args)
#     elif vectorizer_type == Vectorizers.hf:
#         return HFTextVectorizer(**args)
#     elif vectorizer_type == Vectorizers.mistral:
#         return MistralAITextVectorizer(**args)
#     elif vectorizer_type == Vectorizers.vertexai:
#         return VertexAITextVectorizer(**args)
#     elif vectorizer_type == Vectorizers.voyageai:
#         return VoyageAITextVectorizer(**args)
#     else:
#         raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")


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


def get_embedding_model(
    embedding_model: EmbeddingModel, redis_url: str
) -> BaseVectorizer:
    return vectorizer_from_dict(
        vectorizer={"type": embedding_model.type, "model": embedding_model.model},
        cache={
            "name": embedding_model.embedding_cache_name,
            "redis_url": redis_url,
        },
    )


def load_study_config(config_path: str) -> StudyConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return StudyConfig(**config)


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def schema_from_settings(settings: TrialSettings, additional_schema_fields=None):
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
                    "dims": settings.embedding.dim,
                    "distance_metric": settings.index.distance_metric,
                    "algorithm": settings.index.algorithm,
                    "datatype": settings.index.vector_data_type,
                    "ef_construction": settings.index.ef_construction,
                    "ef_runtime": settings.index.ef_runtime,
                    "m": settings.index.m,
                },
            },
        ],
    }

    # define a custom search method to do pre-filtering etc.
    if additional_schema_fields:
        for field in additional_schema_fields:
            schema["fields"].append(field)  # type: ignore

    return schema


def index_from_schema(schema, redis_url, recreate=True):
    index = SearchIndex.from_dict(schema, redis_url=redis_url)

    if recreate:
        index.create(overwrite=True, drop=True)
    return index

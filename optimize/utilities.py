import json
import os
import time
from typing import List

from pydantic import TypeAdapter
from redis.asyncio import Redis as AsyncRedis
from redisvl.index import AsyncSearchIndex
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer

from optimize.models import EmbeddingSettings, LabeledItem, Settings

cache_folder = os.getenv("MODEL_CACHE", "models")


def schema_from_settings(settings: Settings, additional_schema_fields=None):
    schema = {
        "index": {"name": f"{settings.test_id}", "prefix": "rvl"},
        "fields": [
            {"name": "text", "type": "text"},
            {"name": "id", "type": "tag"},
            {"name": "file_name", "type": "tag"},
            {"name": "item_id", "type": "tag"},
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

    if additional_schema_fields:
        for field in additional_schema_fields:
            schema["fields"].append(field)  # type: ignore
    return schema


def load_labeled_items(settings: Settings):
    with open(settings.data.labeled_data_path, "r") as f:
        labeled_items = json.load(f)

    labeled_items_ta = TypeAdapter(List[LabeledItem])
    return labeled_items_ta.validate_python(labeled_items)


async def connect_to_index(settings: Settings, schema):
    aclient = AsyncRedis.from_url(settings.redis_url)
    index = AsyncSearchIndex.from_dict(schema)
    await index.set_client(aclient)
    return index


def embed_chunks(chunks, model: BaseVectorizer, dtype: str):
    start = time.time()
    embeddings = model.embed_many(chunks, as_buffer=True, dtype=dtype)
    embedding_latency = time.time() - start
    return embeddings, embedding_latency


def get_embedding_model(embedding: EmbeddingSettings):

    if embedding.provider == "hf":
        try:
            if cache_folder:
                return HFTextVectorizer(embedding.model, cache_folder=cache_folder)
            return HFTextVectorizer(embedding.model)
        except ValueError as e:
            raise ValueError(f"Error loading HuggingFace model: {e}")
    else:
        raise ValueError(f"Unknown embedding provider: {embedding.provider}")

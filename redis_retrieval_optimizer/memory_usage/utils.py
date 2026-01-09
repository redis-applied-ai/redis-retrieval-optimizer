import logging
import time

from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema

from redis_retrieval_optimizer.utils import vectorizer_from_dict

logger = logging.getLogger(__name__)


def clean_memory_info_output(output):
    """Parse raw Redis ``INFO MEMORY`` bytes output into a dictionary.

    The returned mapping contains key/value pairs as strings, and lines that
    start with ``#`` (comments) are ignored.
    """

    text = output.decode("utf-8")
    memory_stats = {}
    for line in text.split("\r\n"):
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            memory_stats[key] = value
    return memory_stats


def get_insight_calc(
    redis_client: Redis,
) -> dict:
    """Return parsed Redis memory statistics for the given client.

    This helper executes the ``INFO MEMORY`` command on ``redis_client`` and
    parses its output using :func:`clean_memory_info_output`.
    """

    output = redis_client.execute_command("INFO MEMORY")
    return clean_memory_info_output(output)


def estimate_index_size(
    sample_object: dict,
    num_objects: int,
    schema: IndexSchema,
    embedding_model_info: dict,
    redis_url: str = "redis://localhost:6379",
    vector_field_name: str = "vector",
) -> dict:
    """Estimate Redis memory usage for a sample index and its payload.

    The function creates a temporary index defined by ``schema``, embeds and
    loads ``num_objects`` copies of ``sample_object`` into Redis, waits for
    indexing to complete, and then computes memory usage statistics for both
    the index structures and the stored objects.

    You can clean up after yourself - I'm not your mom.
    """

    emb_model = vectorizer_from_dict(embedding_model_info)

    as_buffer = (
        schema.index.storage_type.value
        if schema.index.storage_type.value == "hash"
        else False
    )
    sample_vec = emb_model.embed("test embedding", as_buffer=as_buffer)
    sample_object[vector_field_name] = sample_vec

    sample_objects = [sample_object] * num_objects

    logger.info(f"Creating index {schema.index.name=} with {num_objects} objects")
    index = SearchIndex.from_dict(schema, redis_url=redis_url)
    index.create(overwrite=True, drop=True)

    logger.info("Loading sample objects")
    keys = index.load(sample_objects)

    while float(index.info()["percent_indexed"]) < 1:
        time.sleep(0.5)
        logger.info(f"Indexing progress: {index.info()['percent_indexed']}")

    logger.info(f"Indexing complete {index.info()['num_docs']=}")

    assert index.info()["num_docs"] == num_objects

    memory_usage_keys_bytes = 0

    logger.info("Calculating memory usage per key")
    for key in keys:
        key_memory = index.client.memory_usage(key)
        memory_usage_keys_bytes += key_memory

    info = index.info()

    if len(index.client.execute_command("FT._LIST")) > 1:
        logger.info(
            "WARNING: More than one index exists in Redis. Memory usage may be underestimated."
        )

    index_memory_mb = float(
        info["total_index_memory_sz_mb"]
    )  # total memory used by all indexes
    object_memory_mb = memory_usage_keys_bytes / (1024**2)
    total_memory_mb = index_memory_mb + (object_memory_mb)

    index_memory_gb = index_memory_mb / 1024
    object_memory_gb = object_memory_mb / 1024
    total_memory_gb = index_memory_gb + object_memory_gb

    memory_info_output = get_insight_calc(index.client)
    info_used_memory_bytes = int(memory_info_output["used_memory"])

    info_used_memory_mb = info_used_memory_bytes / (1024**2)
    info_used_memory_gb = info_used_memory_bytes / (1024**3)

    logger.info(f"Index memory: {index_memory_gb:.4f}GB")
    logger.info(f"Object memory: {object_memory_gb:.4f}GB")
    logger.info(f"Total memory: {total_memory_gb:.4f}GB")

    return {
        "info_used_memory_mb": info_used_memory_mb,
        "index_memory_mb": index_memory_mb,
        "object_memory_mb": object_memory_mb,
        "total_memory_mb": total_memory_mb,
        "index_memory_gb": index_memory_gb,
        "object_memory_gb": object_memory_gb,
        "total_memory_gb": total_memory_gb,
        "info_used_memory_gb": info_used_memory_gb,
        "single_key_memory_mb": key_memory / (1024**2),
    }

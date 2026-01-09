import pytest

from redis_retrieval_optimizer.memory_usage import utils as mem_utils


class DummyRedisClient:
    def __init__(self, output: bytes):
        self.output = output
        self.commands = []

    def execute_command(self, command: str):
        self.commands.append(command)
        assert command == "INFO MEMORY"
        return self.output


def test_clean_memory_info_output_parses_lines_and_ignores_comments():
    raw = b"# Memory\r\nused_memory:1024\r\nmaxmemory:0\r\n# End\r\n"

    result = mem_utils.clean_memory_info_output(raw)

    assert result == {"used_memory": "1024", "maxmemory": "0"}


def test_get_insight_calc_calls_info_memory_and_parses_output():
    raw = b"used_memory:2048\r\n"
    client = DummyRedisClient(raw)

    result = mem_utils.get_insight_calc(client)

    assert "INFO MEMORY" in client.commands
    assert result == {"used_memory": "2048"}


class DummyVectorizer:
    def embed(self, text: str, as_buffer=False):  # pragma: no cover - trivial
        return b"VECTOR" if as_buffer else [0.0]


class DummyStorageType:
    def __init__(self, value: str = "hash"):
        self.value = value


class DummyIndex:
    def __init__(self, storage_type_value: str = "hash"):
        self.storage_type = DummyStorageType(storage_type_value)
        self.name = "test-index"


class DummySchema:
    def __init__(self):
        self.index = DummyIndex()


class FakeRedisClientForIndex:
    def __init__(self, per_key_bytes: int = 1_048_576):
        self.per_key_bytes = per_key_bytes

    def memory_usage(self, key):  # pragma: no cover - simple helper
        return self.per_key_bytes

    def execute_command(self, command: str):
        if command == "FT._LIST":
            return ["test-index"]
        if command == "INFO MEMORY":
            # 3 MiB used
            return b"used_memory:3145728\r\n"
        raise ValueError(f"Unexpected command: {command}")


class FakeSearchIndex:
    def __init__(self, schema, redis_url: str):
        self.schema = schema
        self.redis_url = redis_url
        self._percent_indexed = "0"
        self._num_docs = 0
        self.client = FakeRedisClientForIndex()

    @classmethod
    def from_dict(cls, schema, redis_url: str, **kwargs):  # pragma: no cover - trivial
        return cls(schema, redis_url)

    def create(self, overwrite: bool = False, drop: bool = False):  # pragma: no cover
        # No-op for tests
        pass

    def load(self, docs):
        self._num_docs = len(docs)
        self._percent_indexed = "1"
        return [f"doc:{i}" for i in range(len(docs))]

    def info(self):
        return {
            "percent_indexed": self._percent_indexed,
            "num_docs": self._num_docs,
            # 10 MiB of index memory
            "total_index_memory_sz_mb": "10.0",
        }


def test_estimate_index_size_computes_expected_memory_stats(monkeypatch):
    # Patch heavy external dependencies with lightweight fakes
    monkeypatch.setattr(
        mem_utils, "vectorizer_from_dict", lambda info: DummyVectorizer()
    )
    monkeypatch.setattr(mem_utils, "SearchIndex", FakeSearchIndex)

    sample_object = {"id": "1"}
    num_objects = 2
    schema = DummySchema()
    embedding_model_info = {"model": "dummy"}

    result = mem_utils.estimate_index_size(
        sample_object=sample_object,
        num_objects=num_objects,
        schema=schema,
        embedding_model_info=embedding_model_info,
        redis_url="redis://localhost:6379",
        vector_field_name="vector",
    )

    # Index memory is fixed at 10 MiB in FakeSearchIndex.info
    assert result["index_memory_mb"] == pytest.approx(10.0)

    # Each key uses 1 MiB and we load two objects
    assert result["object_memory_mb"] == pytest.approx(2.0)
    assert result["total_memory_mb"] == pytest.approx(12.0)

    # Used memory reported by INFO MEMORY is 3 MiB
    assert result["info_used_memory_mb"] == pytest.approx(3.0)

    # Single key memory should be 1 MiB
    assert result["single_key_memory_mb"] == pytest.approx(1.0)

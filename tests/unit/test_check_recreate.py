"""Unit coverage for #11 — check_recreate must not mutate its inputs (no Redis)."""

from redis_retrieval_optimizer.utils import check_recreate


def _settings():
    return {
        "name": "idx",
        "vector_data_type": "float32",
        "embedding": {"model": "m", "dim": 384},
    }


def test_no_last_settings_recreates_everything():
    assert check_recreate(_settings(), None) == (True, True)


def test_identical_settings_recreate_nothing():
    assert check_recreate(_settings(), _settings()) == (False, False)


def test_does_not_mutate_inputs():
    current, last = _settings(), _settings()
    check_recreate(current, last)
    # the old implementation .pop()'d "embedding" off both dicts
    assert "embedding" in current
    assert "embedding" in last


def test_index_field_change_recreates_index_but_keeps_data():
    current = _settings()
    current["name"] = "other"
    assert check_recreate(current, _settings()) == (True, False)


def test_dtype_change_recreates_data():
    current = _settings()
    current["vector_data_type"] = "float16"
    assert check_recreate(current, _settings()) == (True, True)


def test_embedding_change_alongside_index_change_recreates_data():
    current = _settings()
    current["name"] = "other"
    current["embedding"] = {"model": "different", "dim": 384}
    assert check_recreate(current, _settings()) == (True, True)

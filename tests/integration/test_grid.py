import os

import yaml
from redisvl.index import SearchIndex

import redis_retrieval_optimizer.utils as utils
from redis_retrieval_optimizer.corpus_processors import eval_beir
from redis_retrieval_optimizer.grid_study import run_grid_study

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_run_grid_study(redis_url):
    config_path = f"{TEST_DIR}/grid_data/test_grid_study_config.yaml"

    with open(config_path, "r") as f:
        study_config = yaml.safe_load(f)

    study_config["corpus"] = f"{TEST_DIR}/grid_data/corpus.json"
    study_config["queries"] = f"{TEST_DIR}/grid_data/queries.json"
    study_config["qrels"] = f"{TEST_DIR}/grid_data/qrels.json"

    # Add vector_data_types to test the new dtype functionality
    study_config["vector_data_types"] = ["float32"]

    with open(config_path, "w") as f:
        yaml.dump(study_config, f)

    metrics = run_grid_study(
        config_path=config_path,
        redis_url=redis_url,
        corpus_processor=eval_beir.process_corpus,
    )

    # Calculate expected number of trials: embedding_models * vector_data_types * search_methods
    expected_trials = (
        len(study_config["embedding_models"])
        * len(study_config["vector_data_types"])
        * len(study_config["search_methods"])
    )

    assert metrics.shape[0] == expected_trials

    for score in metrics["f1"].tolist():
        assert score > 0.0

    last_schema = utils.get_last_index_settings(redis_url)
    assert last_schema is not None

    index = SearchIndex.from_existing(last_schema["name"], redis_url=redis_url)

    assert index.info()["num_docs"] == 5

    # clean up
    index.client.json().delete("ret-opt:last_schema")
    index.delete(drop=True)


def test_run_grid_study_with_multiple_dtypes(redis_url):
    """Test grid study with multiple vector data types."""
    config_path = f"{TEST_DIR}/grid_data/test_grid_study_config.yaml"

    with open(config_path, "r") as f:
        study_config = yaml.safe_load(f)

    study_config["corpus"] = f"{TEST_DIR}/grid_data/corpus.json"
    study_config["queries"] = f"{TEST_DIR}/grid_data/queries.json"
    study_config["qrels"] = f"{TEST_DIR}/grid_data/qrels.json"

    # Test with multiple dtypes
    study_config["vector_data_types"] = ["float16", "float32"]

    with open(config_path, "w") as f:
        yaml.dump(study_config, f)

    metrics = run_grid_study(
        config_path=config_path,
        redis_url=redis_url,
        corpus_processor=eval_beir.process_corpus,
    )

    # Calculate expected number of trials: embedding_models * vector_data_types * search_methods
    expected_trials = (
        len(study_config["embedding_models"])
        * len(study_config["vector_data_types"])
        * len(study_config["search_methods"])
    )

    assert metrics.shape[0] == expected_trials

    # Verify that both dtypes are present in the results
    unique_dtypes = metrics["vector_data_type"].unique()
    assert "float16" in unique_dtypes
    assert "float32" in unique_dtypes

    for score in metrics["f1"].tolist():
        assert score > 0.0

    last_schema = utils.get_last_index_settings(redis_url)
    assert last_schema is not None

    index = SearchIndex.from_existing(last_schema["name"], redis_url=redis_url)

    assert index.info()["num_docs"] == 5

    # clean up
    index.client.json().delete("ret-opt:last_schema")
    index.delete(drop=True)

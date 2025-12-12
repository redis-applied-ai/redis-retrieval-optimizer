import os
import time

import pytest
import yaml
from redisvl.index import SearchIndex
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

import redis_retrieval_optimizer.utils as utils
from redis_retrieval_optimizer.corpus_processors import eval_beir
from redis_retrieval_optimizer.search_study import run_search_study

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_run_search_study(redis_url):
    # First, create an index to test against
    search_config_path = f"{TEST_DIR}/search_data/test_search_study_config.yaml"
    index_config_path = f"{TEST_DIR}/search_data/index_config.yaml"

    with open(search_config_path, "r") as f:
        search_study_config = yaml.safe_load(f)

    # Update paths to use relative paths
    search_study_config["qrels"] = f"{TEST_DIR}/search_data/qrels.json"
    search_study_config["queries"] = f"{TEST_DIR}/search_data/queries.json"

    with open(search_config_path, "w") as f:
        yaml.dump(search_study_config, f)

    # Create index from config
    index = SearchIndex.from_yaml(index_config_path, redis_url=redis_url)
    index.create()

    # Create a simple embedding model for testing
    emb_model = HFTextVectorizer(
        model="sentence-transformers/all-MiniLM-L6-v2", dtype="float32"
    )

    # Load corpus data
    corpus = utils.load_json(f"{TEST_DIR}/search_data/corpus.json")
    corpus_data = eval_beir.process_corpus(corpus, emb_model)
    indexing_start_time = time.time()
    index.load(corpus_data)

    # Wait for indexing to complete
    while float(index.info()["percent_indexed"]) < 1:
        time.sleep(1)

    total_indexing_time = time.time() - indexing_start_time
    # Sanity check: indexing time should be positive for a small test corpus.
    assert total_indexing_time > 0.0

    # Persist the measured indexing time so search_study can reuse it.
    utils.set_last_indexing_time(redis_url, total_indexing_time)

    # Run search study
    metrics = run_search_study(
        config_path=search_config_path,
        redis_url=redis_url,
    )

    # Calculate expected number of trials: search_methods
    expected_trials = len(search_study_config["search_methods"])

    assert metrics.shape[0] == expected_trials

    # total_indexing_time should be present and match the value we measured.
    assert "total_indexing_time" in metrics.columns

    unique_indexing_times = metrics["total_indexing_time"].unique()
    assert len(unique_indexing_times) == 1
    assert unique_indexing_times[0] == pytest.approx(total_indexing_time)

    for score in metrics["f1"].tolist():
        assert score > 0.0

    # Verify that all search methods were tested
    unique_methods = metrics["search_method"].unique()
    assert len(unique_methods) == expected_trials
    for method in search_study_config["search_methods"]:
        assert method in unique_methods

    # Clean up
    index.client.json().delete("ret-opt:last_indexing_time")
    index.delete(drop=True)


def test_search_study_requires_embedding_model(redis_url):
    """Test that search study requires embedding_model in config."""
    # Create a config without embedding_model
    config_path = f"{TEST_DIR}/search_data/test_search_study_config_no_embedding.yaml"

    config = {
        "study_id": "test-search-study-no-embedding",
        "index_name": "test-search-index",
        "qrels": f"{TEST_DIR}/search_data/qrels.json",
        "queries": f"{TEST_DIR}/search_data/queries.json",
        "search_methods": ["bm25", "vector"],
        "ret_k": 6,
        # Note: embedding_model is intentionally missing
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Should raise validation error for missing embedding_model
    with pytest.raises(Exception) as exc_info:
        run_search_study(
            config_path=config_path,
            redis_url=redis_url,
        )

    # Clean up test file
    os.remove(config_path)

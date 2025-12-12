import os

import pytest
import yaml
from redisvl.index import SearchIndex

import redis_retrieval_optimizer.utils as utils
from redis_retrieval_optimizer.bayes_study import run_bayes_study
from redis_retrieval_optimizer.corpus_processors import eval_beir

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_run_bayes_study(redis_url):
    config_path = f"{TEST_DIR}/bayes_data/test_bayes_study_config.yaml"

    with open(config_path, "r") as f:
        study_config = yaml.safe_load(f)

    study_config["corpus"] = f"{TEST_DIR}/bayes_data/corpus.json"
    study_config["queries"] = f"{TEST_DIR}/bayes_data/queries.json"
    study_config["qrels"] = f"{TEST_DIR}/bayes_data/qrels.json"

    with open(config_path, "w") as f:
        yaml.dump(study_config, f)

    metrics = run_bayes_study(
        config_path=config_path,
        redis_url=redis_url,
        corpus_processor=eval_beir.process_corpus,
    )

    assert metrics.shape[0] == study_config["optimization_settings"]["n_trials"]

    # total_indexing_time should be recorded for each trial and persisted
    assert "total_indexing_time" in metrics.columns

    last_indexing_time = utils.get_last_indexing_time(redis_url)
    assert last_indexing_time is not None
    assert last_indexing_time > 0.0

    # The last trial's recorded indexing time should match the persisted value
    assert metrics["total_indexing_time"].iloc[-1] == pytest.approx(last_indexing_time)

    for score in metrics["f1"].tolist():
        assert score > 0.0

    last_schema = utils.get_last_index_settings(redis_url)
    assert last_schema is not None

    index = SearchIndex.from_existing(last_schema["name"], redis_url=redis_url)

    assert index.info()["num_docs"] == 5

    # clean up
    index.client.json().delete("ret-opt:last_schema")
    index.client.json().delete("ret-opt:last_indexing_time")
    index.delete(drop=True)

import os

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

    for score in metrics["f1"].tolist():
        assert score > 0.0

    last_schema = utils.get_last_index_settings(redis_url)
    assert last_schema is not None

    index = SearchIndex.from_existing(last_schema["name"], redis_url=redis_url)

    assert index.info()["num_docs"] == 5

    # clean up
    index.client.json().delete("ret-opt:last_schema")
    index.delete(drop=True)

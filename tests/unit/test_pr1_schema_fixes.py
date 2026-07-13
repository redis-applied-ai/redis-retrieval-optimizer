"""Unit coverage for PR 1 correctness fixes (no Redis required).

- #2 per-instance uuid defaults
- #5 MetricWeights rejects unknown keys and all-zero weights
- #6 get_trial_settings resets HNSW params when a non-hnsw algorithm is chosen
"""

import pytest
from optuna.trial import FixedTrial
from pydantic import ValidationError

from redis_retrieval_optimizer.schema import (
    BayesStudyConfig,
    DataSettings,
    EmbeddingModel,
    GridStudyConfig,
    IndexSettings,
    MetricWeights,
    OptimizationSettings,
    SearchStudyConfig,
    TrialSettings,
    get_trial_settings,
)


def _embedding():
    return EmbeddingModel(type="hf", model="some/model", dim=384)


def _trial_settings():
    return TrialSettings(
        index_settings=IndexSettings(vector_dim=384),
        embedding=_embedding(),
        data=DataSettings(corpus="c.json", queries="q.json", qrels="r.json"),
    )


class TestUniqueIdDefaults:
    """#2 — ids must be generated per instance, not once at class definition."""

    def test_trial_ids_are_unique(self):
        assert _trial_settings().trial_id != _trial_settings().trial_id

    def test_bayes_study_ids_are_unique(self):
        opt = OptimizationSettings(
            algorithms=["flat"],
            vector_data_types=["float32"],
            distance_metrics=["cosine"],
            n_trials=1,
            n_jobs=1,
            metric_weights=MetricWeights(f1=1),
        )
        kwargs = dict(
            corpus="c.json",
            qrels="r.json",
            queries="q.json",
            index_settings=IndexSettings(vector_dim=384),
            optimization_settings=opt,
            embedding_models=[_embedding()],
            search_methods=["vector"],
        )
        assert (
            BayesStudyConfig(**kwargs).study_id != BayesStudyConfig(**kwargs).study_id
        )

    def test_grid_and_search_study_ids_are_unique(self):
        grid_kwargs = dict(
            index_settings=IndexSettings(vector_dim=384),
            qrels="r.json",
            queries="q.json",
            embedding_models=[_embedding()],
            search_methods=["vector"],
        )
        assert (
            GridStudyConfig(**grid_kwargs).study_id
            != GridStudyConfig(**grid_kwargs).study_id
        )

        search_kwargs = dict(
            index_name="idx",
            qrels="r.json",
            queries="q.json",
            search_methods=["vector"],
            embedding_model=_embedding(),
        )
        assert (
            SearchStudyConfig(**search_kwargs).study_id
            != SearchStudyConfig(**search_kwargs).study_id
        )


class TestMetricWeightsValidation:
    """#5 — bad weight configs must fail loudly rather than silently no-op."""

    def test_valid_weights_ok(self):
        assert MetricWeights(f1=1).f1 == 1

    def test_all_zero_weights_rejected(self):
        with pytest.raises(ValidationError, match="non-zero"):
            MetricWeights()

    def test_unknown_weight_key_rejected(self):
        # the exact typo that was silently dropped before (f1_at_k -> f1)
        with pytest.raises(ValidationError):
            MetricWeights(f1_at_k=1)


class TestHnswParamReset:
    """#6 — a flat trial must not inherit a prior hnsw trial's ef/m values."""

    def _config(self):
        opt = OptimizationSettings(
            algorithms=["hnsw", "flat"],
            vector_data_types=["float32"],
            distance_metrics=["cosine"],
            n_trials=2,
            n_jobs=1,
            metric_weights=MetricWeights(f1=1),
        )
        return BayesStudyConfig(
            corpus="c.json",
            qrels="r.json",
            queries="q.json",
            index_settings=IndexSettings(vector_dim=384),
            optimization_settings=opt,
            embedding_models=[_embedding()],
            search_methods=["vector"],
        )

    def test_flat_trial_resets_hnsw_params(self):
        config = self._config()
        model_info = config.embedding_models[0].model_dump()
        common = dict(
            model_info=model_info,
            search_method="vector",
            var_dtype="float32",
            distance_metric="cosine",
            ret_k=5,
        )

        # hnsw trial sets ef/m on the shared index_settings
        hnsw = get_trial_settings(
            FixedTrial(
                {
                    **common,
                    "algorithm": "hnsw",
                    "ef_runtime": 50,
                    "ef_construction": 300,
                    "m": 64,
                }
            ),
            config,
        )
        assert (hnsw.index_settings.m, hnsw.index_settings.ef_construction) == (64, 300)

        # subsequent flat trial must reset them (bug left them at 64/300/50)
        flat = get_trial_settings(FixedTrial({**common, "algorithm": "flat"}), config)
        assert flat.index_settings.m == 0
        assert flat.index_settings.ef_construction == 0
        assert flat.index_settings.ef_runtime == 0

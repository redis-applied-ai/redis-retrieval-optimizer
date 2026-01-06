import tempfile

import pytest
import yaml

from redis_retrieval_optimizer.schema import (
    BayesStudyConfig,
    GridStudyConfig,
    SearchStudyConfig,
)
from redis_retrieval_optimizer.utils import (
    load_bayes_study_config,
    load_grid_study_config,
    load_search_study_config,
)


class TestLoadGridStudyConfig:
    """Test cases for load_grid_study_config function"""

    def get_invalid_config(self):
        """Return an invalid config dict for testing validation errors"""
        return {
            "corpus": "/path/to/corpus.json",
            # Missing required fields like embedding_models, index_settings, etc.
        }

    def get_sample_grid_config(self):
        """Return a sample grid study config dict"""
        return {
            "corpus": "/path/to/corpus.json",
            "embedding_models": [
                {
                    "dim": 384,
                    "embedding_cache_name": "vec-cache",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "type": "hf",
                }
            ],
            "index_settings": {
                "from_existing": False,
                "name": "test",
                "vector_dim": 384,
            },
            "qrels": "/path/to/qrels.json",
            "queries": "/path/to/queries.json",
            "search_methods": ["bm25", "vector"],
        }

    def test_load_from_config_dict(self):
        """Test loading config from a dictionary"""
        config = self.get_sample_grid_config()

        result = load_grid_study_config(config=config)

        assert isinstance(result, GridStudyConfig)
        assert result.corpus == "/path/to/corpus.json"
        assert len(result.embedding_models) == 1
        assert (
            result.embedding_models[0].model == "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_load_from_config_path(self):
        """Test loading config from a file path"""
        config = self.get_sample_grid_config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        result = load_grid_study_config(config_path=temp_path)

        assert isinstance(result, GridStudyConfig)
        assert result.corpus == "/path/to/corpus.json"

    def test_raises_error_when_neither_provided(self):
        """Test that ValueError is raised when neither config_path nor config is provided"""
        with pytest.raises(
            ValueError, match="Either config_path or config must be provided"
        ):
            load_grid_study_config()

    def test_config_dict_takes_precedence(self):
        """Test that config dict is used when both are provided"""
        config_dict = self.get_sample_grid_config()
        config_dict["corpus"] = "/dict/path/corpus.json"

        other_config = self.get_sample_grid_config()
        other_config["corpus"] = "/file/path/corpus.json"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(other_config, f)
            temp_path = f.name

        result = load_grid_study_config(config_path=temp_path, config=config_dict)

        # Config dict should be used, not the file
        assert result.corpus == "/dict/path/corpus.json"

    def test_raises_user_friendly_error_on_invalid_config(self):
        """Test that a user-friendly error is raised when config validation fails"""
        invalid_config = self.get_invalid_config()

        with pytest.raises(ValueError) as exc_info:
            load_grid_study_config(config=invalid_config)

        error_message = str(exc_info.value)
        assert "Invalid GridStudyConfig configuration" in error_message
        assert "Please check your config dict or YAML file" in error_message


class TestLoadBayesStudyConfig:
    """Test cases for load_bayes_study_config function"""

    def get_sample_bayes_config(self):
        """Return a sample bayes study config dict"""
        return {
            "corpus": "/path/to/corpus.json",
            "embedding_models": [
                {
                    "dim": 384,
                    "dtype": "float32",
                    "embedding_cache_name": "vec-cache",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "type": "hf",
                }
            ],
            "index_settings": {
                "from_existing": False,
                "name": "test",
                "vector_dim": 384,
            },
            "optimization_settings": {
                "algorithms": ["hnsw"],
                "distance_metrics": ["cosine"],
                "ef_construction": [100, 200],
                "ef_runtime": [10, 20],
                "m": [8, 16],
                "n_jobs": 1,
                "n_trials": 5,
                "ret_k": [1, 10],
                "vector_data_types": ["float32"],
            },
            "qrels": "/path/to/qrels.json",
            "queries": "/path/to/queries.json",
            "search_methods": ["vector"],
        }

    def test_load_from_config_dict(self):
        """Test loading config from a dictionary"""
        config = self.get_sample_bayes_config()

        result = load_bayes_study_config(config=config)

        assert isinstance(result, BayesStudyConfig)
        assert result.corpus == "/path/to/corpus.json"
        assert result.optimization_settings.n_trials == 5

    def test_raises_error_when_neither_provided(self):
        """Test that ValueError is raised when neither config_path nor config is provided"""
        with pytest.raises(
            ValueError, match="Either config_path or config must be provided"
        ):
            load_bayes_study_config()


class TestLoadSearchStudyConfig:
    """Test cases for load_search_study_config function"""

    def get_sample_search_config(self):
        """Return a sample search study config dict"""
        return {
            "embedding_model": {
                "dim": 384,
                "dtype": "float32",
                "embedding_cache_name": "vec-cache",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "type": "hf",
            },
            "index_name": "test-search-index",
            "qrels": "/path/to/qrels.json",
            "queries": "/path/to/queries.json",
            "ret_k": 6,
            "search_methods": ["bm25", "vector"],
            "study_id": "test-search-study",
        }

    def test_load_from_config_dict(self):
        """Test loading config from a dictionary"""
        config = self.get_sample_search_config()

        result = load_search_study_config(config=config)

        assert isinstance(result, SearchStudyConfig)
        assert result.index_name == "test-search-index"
        assert result.ret_k == 6

    def test_raises_error_when_neither_provided(self):
        """Test that ValueError is raised when neither config_path nor config is provided"""
        with pytest.raises(
            ValueError, match="Either config_path or config must be provided"
        ):
            load_search_study_config()

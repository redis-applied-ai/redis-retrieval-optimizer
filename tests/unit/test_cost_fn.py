import pytest

from redis_retrieval_optimizer.bayes_study import cost_fn, norm_metric


class TestCostFunction:
    """Test cases for the cost_fn function in bayes_study.py"""

    def test_norm_metric_function(self):
        """Test that norm_metric correctly normalizes values"""
        # Test with zero
        assert norm_metric(0) == 1.0

        # Test with positive values
        assert norm_metric(1) == 0.5
        assert norm_metric(2) == 1 / 3
        assert norm_metric(9) == 0.1

        # Test that higher values result in lower normalized values
        assert norm_metric(1) > norm_metric(2) > norm_metric(10)

        # Test edge case: -1 should return 1000.0 (infinity equivalent)
        assert norm_metric(-1) == 1000.0

    def test_time_metrics_are_subtracted_and_normalized(self):
        """Test that avg_query_time and total_indexing_time are subtracted and normalized"""
        metrics = {
            "avg_query_time": 2.0,
            "total_indexing_time": 5.0,
            "recall": 0.8,
            "ndcg": 0.7,
        }
        weights = {
            "avg_query_time": 1.0,
            "total_indexing_time": 1.0,
            "recall": 1.0,
            "ndcg": 1.0,
        }

        result = cost_fn(metrics, weights)

        # Calculate expected values
        expected_avg_query_contribution = -norm_metric(2.0)  # Should be subtracted
        expected_total_indexing_contribution = -norm_metric(5.0)  # Should be subtracted
        expected_recall_contribution = 0.8  # Should be added directly
        expected_ndcg_contribution = 0.7  # Should be added directly

        expected_total = (
            expected_avg_query_contribution
            + expected_total_indexing_contribution
            + expected_recall_contribution
            + expected_ndcg_contribution
        )

        assert result == pytest.approx(expected_total, rel=1e-10)

        # Verify that time metrics are indeed subtracted (negative contribution)
        assert expected_avg_query_contribution < 0
        assert expected_total_indexing_contribution < 0

    def test_non_time_metrics_are_added_directly(self):
        """Test that non-time metrics are added directly without normalization"""
        metrics = {"recall": 0.8, "ndcg": 0.7, "precision": 0.9, "f1": 0.85}
        weights = {"recall": 1.0, "ndcg": 1.0, "precision": 1.0, "f1": 1.0}

        result = cost_fn(metrics, weights)

        # All metrics should be added directly
        expected = 0.8 + 0.7 + 0.9 + 0.85
        assert result == pytest.approx(expected, rel=1e-10)

    def test_mixed_metrics_with_weights(self):
        """Test cost function with mixed metrics and different weights"""
        metrics = {
            "avg_query_time": 3.0,
            "total_indexing_time": 10.0,
            "recall": 0.8,
            "ndcg": 0.7,
        }
        weights = {
            "avg_query_time": 2.0,  # Higher weight for query time
            "total_indexing_time": 1.0,
            "recall": 3.0,  # Higher weight for recall
            "ndcg": 1.0,
        }

        result = cost_fn(metrics, weights)

        # Calculate expected values with weights
        expected_avg_query_contribution = 2.0 * -norm_metric(3.0)
        expected_total_indexing_contribution = 1.0 * -norm_metric(10.0)
        expected_recall_contribution = 3.0 * 0.8
        expected_ndcg_contribution = 1.0 * 0.7

        expected_total = (
            expected_avg_query_contribution
            + expected_total_indexing_contribution
            + expected_recall_contribution
            + expected_ndcg_contribution
        )

        assert result == pytest.approx(expected_total, rel=1e-10)

    def test_missing_weights_default_to_zero(self):
        """Test that metrics without weights default to zero contribution"""
        metrics = {"avg_query_time": 2.0, "recall": 0.8, "ndcg": 0.7}
        weights = {
            "avg_query_time": 1.0,
            # recall and ndcg weights are missing
        }

        result = cost_fn(metrics, weights)

        # Only avg_query_time should contribute
        expected = -norm_metric(2.0)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_empty_metrics_returns_zero(self):
        """Test that empty metrics returns zero"""
        metrics = {}
        weights = {"recall": 1.0}

        result = cost_fn(metrics, weights)
        assert result == 0.0

    def test_empty_weights_returns_zero(self):
        """Test that empty weights returns zero"""
        metrics = {"recall": 0.8, "ndcg": 0.7}
        weights = {}

        result = cost_fn(metrics, weights)
        assert result == 0.0

    def test_negative_time_values_handled_correctly(self):
        """Test that negative time values are handled correctly by norm_metric"""
        # Test with -1.0 (edge case that should be handled)
        metrics = {"avg_query_time": -1.0, "recall": 0.8}
        weights = {"avg_query_time": 1.0, "recall": 1.0}

        result = cost_fn(metrics, weights)

        # norm_metric(-1) should return 1000.0 (our infinity equivalent)
        expected_avg_query_contribution = -norm_metric(-1.0)  # Should be -1000.0
        expected_recall_contribution = 0.8
        expected_total = expected_avg_query_contribution + expected_recall_contribution

        assert result == pytest.approx(expected_total, rel=1e-10)

        # Test with -0.5 (normal case)
        metrics["avg_query_time"] = -0.5
        result = cost_fn(metrics, weights)

        expected_avg_query_contribution = -norm_metric(-0.5)  # Should be -2.0
        expected_recall_contribution = 0.8
        expected_total = expected_avg_query_contribution + expected_recall_contribution

        assert result == pytest.approx(expected_total, rel=1e-10)

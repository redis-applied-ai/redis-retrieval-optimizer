import pytest
from ranx import Run

from redis_retrieval_optimizer.schema import (
    QueryMetrics,
    SearchMethodInput,
    SearchMethodOutput,
)


class TestSearchMethodInputRetK:
    """Test cases for the ret_k parameter in SearchMethodInput"""

    def test_ret_k_default_value(self):
        """Test that ret_k has the correct default value"""
        # Create SearchMethodInput with minimal required fields
        search_input = SearchMethodInput(raw_queries={"q1": "test query"}, index=None)

        # Check default value
        assert search_input.ret_k == 6

    def test_ret_k_custom_value(self):
        """Test that ret_k can be set to custom values"""
        # Test with different custom values
        for custom_ret_k in [1, 5, 10, 100]:
            search_input = SearchMethodInput(
                raw_queries={"q1": "test query"}, index=None, ret_k=custom_ret_k
            )
            assert search_input.ret_k == custom_ret_k

    def test_ret_k_validation(self):
        """Test that ret_k validation works correctly"""
        # Test that ret_k must be at least 1
        with pytest.raises(ValueError, match="ret_k must be at least 1"):
            SearchMethodInput(raw_queries={"q1": "test query"}, index=None, ret_k=0)

        with pytest.raises(ValueError, match="ret_k must be at least 1"):
            SearchMethodInput(raw_queries={"q1": "test query"}, index=None, ret_k=-5)

    def test_ret_k_with_dict_queries(self):
        """Test that ret_k works with dictionary format queries"""
        search_input = SearchMethodInput(
            raw_queries={
                "q1": {"query": "test query 1", "query_metadata": {"category": "test"}},
                "q2": {"query": "test query 2", "query_metadata": {"category": "test"}},
            },
            index=None,
            ret_k=15,
        )

        assert search_input.ret_k == 15
        assert len(search_input.raw_queries) == 2

    def test_ret_k_with_list_queries(self):
        """Test that ret_k works with list format queries"""
        search_input = SearchMethodInput(
            raw_queries=["query1", "query2", "query3"], index=None, ret_k=8
        )

        assert search_input.ret_k == 8
        assert len(search_input.raw_queries) == 3

    def test_ret_k_accessibility_in_search_methods(self):
        """Test that ret_k is accessible and usable in search method context"""
        # Simulate what a search method would do
        search_input = SearchMethodInput(
            raw_queries={"q1": "test query"}, index=None, ret_k=12
        )

        # Simulate accessing ret_k in a search method
        def mock_search_method(input_data: SearchMethodInput):
            # This simulates how search methods use ret_k
            return input_data.ret_k

        result = mock_search_method(search_input)
        assert result == 12

    def test_ret_k_with_other_fields(self):
        """Test that ret_k works correctly with other SearchMethodInput fields"""
        search_input = SearchMethodInput(
            raw_queries={"q1": "test query"},
            index=None,
            ret_k=25,
            id_field_name="custom_id",
            text_field_name="custom_text",
            vector_field_name="custom_vector",
        )

        # Check all fields are set correctly
        assert search_input.ret_k == 25
        assert search_input.id_field_name == "custom_id"
        assert search_input.text_field_name == "custom_text"
        assert search_input.vector_field_name == "custom_vector"

    def test_ret_k_edge_cases(self):
        """Test ret_k with edge case values"""
        # Test with very large values
        large_ret_k = 10000
        search_input = SearchMethodInput(
            raw_queries={"q1": "test query"}, index=None, ret_k=large_ret_k
        )
        assert search_input.ret_k == large_ret_k

        # Test with minimum valid value
        min_ret_k = 1
        search_input = SearchMethodInput(
            raw_queries={"q1": "test query"}, index=None, ret_k=min_ret_k
        )
        assert search_input.ret_k == min_ret_k


class TestSearchMethodOutput:
    """Test cases for SearchMethodOutput"""

    def test_search_method_output_creation(self):
        """Test that SearchMethodOutput can be created correctly"""
        query_metrics = QueryMetrics()
        run = Run({"q1": {"doc1": 0.9, "doc2": 0.8}})

        output = SearchMethodOutput(run=run, query_metrics=query_metrics)

        assert output.run == run
        assert output.query_metrics == query_metrics

    def test_query_metrics_timing(self):
        """Test QueryMetrics timing functionality"""
        metrics = QueryMetrics()

        # Add some query times
        metrics.query_times = [0.1, 0.2, 0.3]

        # Test convenience methods
        assert metrics.query_times == [0.1, 0.2, 0.3]

        # Test that we can access the timing data
        assert len(metrics.query_times) == 3
        assert sum(metrics.query_times) == 0.6

"""Unit coverage for PR 2 search-method input handling (no Redis required).

- #13 SearchMethodInput normalizes the documented query shapes to {id: text}
- #14 vector similarity score maps cosine distance [0, 2] -> [0, 1]
"""

from redis_retrieval_optimizer.schema import SearchMethodInput
from redis_retrieval_optimizer.search_methods.vector import make_score_dict_vec


class TestRawQueryNormalization:
    """#13 — every documented query shape becomes {query_id: text}."""

    def test_flat_dict_unchanged(self):
        si = SearchMethodInput(raw_queries={"q1": "hello"}, index=None)
        assert si.raw_queries == {"q1": "hello"}

    def test_nested_dict_extracts_query_and_drops_metadata(self):
        si = SearchMethodInput(
            raw_queries={"q1": {"query": "hello", "query_metadata": {"make": "x"}}},
            index=None,
        )
        assert si.raw_queries == {"q1": "hello"}

    def test_list_is_keyed_by_position(self):
        si = SearchMethodInput(raw_queries=["a", "b", "c"], index=None)
        assert si.raw_queries == {"0": "a", "1": "b", "2": "c"}


class TestVectorScoreBounds:
    """#14 — cosine distance is mapped to a [0, 1] similarity."""

    def test_perfect_match_scores_one(self):
        assert make_score_dict_vec([{"_id": "d", "vector_distance": 0.0}], "_id") == {
            "d": 1.0
        }

    def test_orthogonal_scores_half(self):
        assert make_score_dict_vec([{"_id": "d", "vector_distance": 1.0}], "_id") == {
            "d": 0.5
        }

    def test_opposite_scores_zero(self):
        assert make_score_dict_vec([{"_id": "d", "vector_distance": 2.0}], "_id") == {
            "d": 0.0
        }

    def test_empty_results(self):
        assert make_score_dict_vec([], "_id") == {"no_match": 0}

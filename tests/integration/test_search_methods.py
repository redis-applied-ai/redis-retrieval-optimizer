import json
import os

import pytest
from ranx import Qrels, Run, evaluate
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

from redis_retrieval_optimizer.schema import QueryMetrics, SearchMethodInput
from redis_retrieval_optimizer.search_methods.bm25 import gather_bm25_results
from redis_retrieval_optimizer.search_methods.vector import (
    gather_vector_results,
    make_score_dict_vec,
    vector_query,
)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def test_data():
    """Load test corpus, queries, and qrels."""
    corpus_path = f"{TEST_DIR}/vector_data/corpus.json"
    queries_path = f"{TEST_DIR}/vector_data/queries.json"
    qrels_path = f"{TEST_DIR}/vector_data/qrels.json"

    # Create the test data directory if it doesn't exist
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

    # Create test corpus if it doesn't exist
    if not os.path.exists(corpus_path):
        corpus = {
            "doc1": {
                "text": "This document is about machine learning.",
                "title": "ML Intro",
            },
            "doc2": {
                "text": "Natural language processing is fascinating.",
                "title": "NLP",
            },
            "doc3": {
                "text": "Vector databases use embeddings for search.",
                "title": "Vector DBs",
            },
            "doc4": {
                "text": "Redis is an in-memory data structure store.",
                "title": "Redis",
            },
            "doc5": {
                "text": "Deep learning has transformed AI research.",
                "title": "Deep Learning",
            },
        }
        with open(corpus_path, "w") as f:
            json.dump(corpus, f)
    else:
        with open(corpus_path, "r") as f:
            corpus = json.load(f)

    # Create test queries if they don't exist
    if not os.path.exists(queries_path):
        queries = {
            "q1": "How does machine learning work?",
            "q2": "What is NLP?",
            "q3": "Tell me about vector search",
        }
        with open(queries_path, "w") as f:
            json.dump(queries, f)
    else:
        with open(queries_path, "r") as f:
            queries = json.load(f)

    # Create test qrels if they don't exist
    if not os.path.exists(qrels_path):
        qrels_dict = {
            "q1": {"doc1": 2, "doc5": 1},
            "q2": {"doc2": 2},
            "q3": {"doc3": 2, "doc4": 1},
        }
        with open(qrels_path, "w") as f:
            json.dump(qrels_dict, f)
        qrels = Qrels(qrels_dict)
    else:
        with open(qrels_path, "r") as f:
            qrels = Qrels(json.load(f))

    return corpus, queries, qrels


@pytest.fixture
def vector_index(redis_url, test_data):
    """Create a test vector index with documents."""
    corpus, _, _ = test_data
    emb_model = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

    # Create index schema
    schema = {
        "index": {"name": "test", "prefix": "test"},
        "fields": [
            {"name": "_id", "type": "tag"},
            {"name": "text", "type": "text"},
            {
                "name": "vector",
                "type": "vector",
                "attrs": {
                    "dims": 384,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                },
            },
        ],
    }

    # Initialize index
    index = SearchIndex.from_dict(schema, redis_url=redis_url)

    # Clean up any existing data
    try:
        index.delete(drop=True)
    except:
        pass

    # Create the index
    index.create()

    # Process and index corpus
    docs = []
    for doc_id, doc in corpus.items():
        vector = emb_model.embed(doc["text"], as_buffer=True)
        docs.append(
            {
                "_id": doc_id,
                "text": doc["text"],
                "title": doc["title"],
                "vector": vector,
            }
        )

    # Index the documents
    index.load(docs)

    yield index, emb_model

    index.delete(drop=True)


def test_vector_query(vector_index):
    """Test the vector_query function."""
    index, emb_model = vector_index
    query_text = "How does machine learning work?"

    # Test the function
    query_obj = vector_query(query_text, num_results=3, emb_model=emb_model)

    # Verify it's a VectorQuery object
    assert query_obj is not None


def test_make_score_dict_vec():
    """Test the make_score_dict_vec function."""
    # Create some mock results
    results = [
        {"_id": "doc1", "vector_distance": 0.2},
        {"_id": "doc2", "vector_distance": 0.5},
    ]

    scores = make_score_dict_vec(results, "_id")

    # Verify scores are calculated correctly
    assert "doc1" in scores
    assert "doc2" in scores
    # Score should be 2 - distance/2
    assert scores["doc1"] == 2 - 0.2 / 2
    assert scores["doc2"] == 2 - 0.5 / 2

    # Test with empty results
    empty_scores = make_score_dict_vec([], "_id")
    assert "no_match" in empty_scores
    assert empty_scores["no_match"] == 0


def test_gather_vector_results(vector_index, test_data):
    """Test the gather_vector_results function with real Redis index."""
    index, emb_model = vector_index
    _, queries, qrels = test_data

    # Create input for gather_vector_results
    query_metrics = QueryMetrics()
    search_input = SearchMethodInput(
        index=index,
        raw_queries=queries,
        emb_model=emb_model,
        query_metrics=query_metrics,
    )

    # Execute search
    result = gather_vector_results(search_input)

    # Verify results
    assert result is not None
    assert isinstance(result.run, Run)

    # Check that query metrics were collected
    assert len(result.query_metrics.query_times) == len(queries)

    f1 = evaluate(qrels, result.run, metrics=["f1"])

    assert f1 > 0


def test_gather_bm25_results(vector_index, test_data):
    """Test the gather_bm25_results function with real Redis index."""
    index, _ = vector_index  # We don't need the embedding model for BM25
    _, queries, qrels = test_data

    # Create input for gather_bm25_results
    query_metrics = QueryMetrics()
    search_input = SearchMethodInput(
        index=index,
        raw_queries=queries,
        query_metrics=query_metrics,
    )

    # Execute search
    result = gather_bm25_results(search_input)

    # Verify results
    assert result is not None
    assert isinstance(result.run, Run)

    # Check that query metrics were collected
    assert len(result.query_metrics.query_times) == len(queries)

    f1 = evaluate(qrels, result.run, metrics=["f1"])

    assert f1 > 0


def test_gather_rerank_results(vector_index, test_data):
    """Test the gather_rerank_results function with real Redis index."""
    index, emb_model = vector_index  # We need the embedding model for rerank
    _, queries, qrels = test_data

    # Create input for gather_rerank_results
    query_metrics = QueryMetrics()
    search_input = SearchMethodInput(
        index=index,
        raw_queries=queries,
        emb_model=emb_model,  # Rerank uses both BM25 and embeddings
        query_metrics=query_metrics,
    )

    # Import the rerank function
    from redis_retrieval_optimizer.search_methods.rerank import gather_rerank_results

    # Execute search
    result = gather_rerank_results(search_input)

    # Verify results
    assert result is not None
    assert isinstance(result.run, Run)

    # Check that query metrics were collected
    assert len(result.query_metrics.query_times) == len(queries)

    f1 = evaluate(qrels, result.run, metrics=["f1"])

    assert f1 > 0


def test_gather_hybrid_results(vector_index, test_data):
    """Test the gather_hybrid_results function with real Redis index."""
    index, emb_model = vector_index
    _, queries, qrels = test_data

    # Create input for gather_hybrid_results
    query_metrics = QueryMetrics()
    search_input = SearchMethodInput(
        index=index,
        raw_queries=queries,
        emb_model=emb_model,  # Lin combo uses both text and embeddings
        query_metrics=query_metrics,
    )

    # Import the hybrid function
    from redis_retrieval_optimizer.search_methods.hybrid import gather_hybrid_results

    # Execute search
    result = gather_hybrid_results(search_input)

    # Verify results
    assert result is not None
    assert isinstance(result.run, Run)

    # Check that query metrics were collected
    assert len(result.query_metrics.query_times) == len(queries)

    f1 = evaluate(qrels, result.run, metrics=["f1"])

    assert f1 > 0


def test_gather_weighted_rrf_results(vector_index, test_data):
    """Test the gather_weighted_rrf function with real Redis index."""
    index, emb_model = vector_index
    _, queries, qrels = test_data

    # Create input for gather_weighted_rrf
    query_metrics = QueryMetrics()
    search_input = SearchMethodInput(
        index=index,
        raw_queries=queries,
        emb_model=emb_model,  # Weighted RRF uses both BM25 and embeddings
        query_metrics=query_metrics,
    )

    # Import the weighted_rrf function
    from redis_retrieval_optimizer.search_methods.weighted_rrf import (
        gather_weighted_rrf,
    )

    # Execute search
    result = gather_weighted_rrf(search_input)

    # Verify results
    assert result is not None
    assert isinstance(result.run, Run)

    # Check that query metrics were collected
    assert len(result.query_metrics.query_times) == len(queries)

    f1 = evaluate(qrels, result.run, metrics=["f1"])

    assert f1 > 0


def test_gather_hybrid_8_4_results(vector_index, test_data):
    """Test the gather_hybrid_results function with real Redis index."""
    index, emb_model = vector_index
    _, queries, qrels = test_data

    # Create input for gather_hybrid_results
    query_metrics = QueryMetrics()
    search_input = SearchMethodInput(
        index=index,
        raw_queries=queries,
        emb_model=emb_model,  # Lin combo uses both text and embeddings
        query_metrics=query_metrics,
    )

    # Import the hybrid function
    from redis_retrieval_optimizer.search_methods.hybrid_8_4 import (
        gather_hybrid_8_4_results,
    )

    # Execute search
    result = gather_hybrid_8_4_results(search_input)

    # Verify results
    assert result is not None
    assert isinstance(result.run, Run)

    # Check that query metrics were collected
    assert len(result.query_metrics.query_times) == len(queries)

    f1 = evaluate(qrels, result.run, metrics=["f1"])

    assert f1 > 0


def test_gather_rrf_8_4_results(vector_index, test_data):
    """Test the gather_hybrid_results function with real Redis index."""
    index, emb_model = vector_index
    _, queries, qrels = test_data

    # Create input for gather_hybrid_results
    query_metrics = QueryMetrics()
    search_input = SearchMethodInput(
        index=index,
        raw_queries=queries,
        emb_model=emb_model,  # Lin combo uses both text and embeddings
        query_metrics=query_metrics,
    )

    # Import the hybrid function
    from redis_retrieval_optimizer.search_methods.rrf_8_4 import gather_rrf_8_4_results

    # Execute search
    result = gather_rrf_8_4_results(search_input)

    # Verify results
    assert result is not None
    assert isinstance(result.run, Run)

    # Check that query metrics were collected
    assert len(result.query_metrics.query_times) == len(queries)

    f1 = evaluate(qrels, result.run, metrics=["f1"])

    assert f1 > 0

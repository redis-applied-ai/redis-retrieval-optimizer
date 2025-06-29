import sys

import pytest

if sys.version_info.major == 3 and sys.version_info.minor < 10:
    pytest.skip("Test requires Python 3.10 or higher", allow_module_level=True)

from redis import Redis
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.extensions.router import Route, SemanticRouter
from redisvl.extensions.router.schema import RoutingConfig

from redis_retrieval_optimizer.threshold_optimization import (
    CacheThresholdOptimizer,
    EvalMetric,
    RouterThresholdOptimizer,
)


def skip_if_redis_version_below(client, version):
    """Skip test if Redis version is below the specified version."""
    try:
        server_info = client.info()
        server_version = server_info.get("redis_version", "0.0.0")
        if server_version < version:
            pytest.skip(f"Redis version {server_version} < {version}")
    except Exception:
        pytest.skip("Cannot determine Redis version")


@pytest.fixture
def routes():
    return [
        Route(
            name="greeting",
            references=["hello", "hi"],
            metadata={"type": "greeting"},
            distance_threshold=0.3,
        ),
        Route(
            name="farewell",
            references=["bye", "goodbye"],
            metadata={"type": "farewell"},
            distance_threshold=0.2,
        ),
    ]


@pytest.fixture
def redis_url():
    return "redis://localhost:6379"


@pytest.fixture
def hf_vectorizer():
    """Mock HF vectorizer for testing"""
    try:
        from redisvl.utils.vectorize import HFTextVectorizer

        return HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        pytest.skip("HuggingFace transformers not available")


@pytest.fixture
def semantic_router(redis_url, routes, hf_vectorizer):
    router = SemanticRouter(
        name="test-router",
        routes=routes,
        vectorizer=hf_vectorizer,
        routing_config=RoutingConfig(max_k=2),
        redis_url=redis_url,
        overwrite=False,
    )
    yield router
    router.delete()


@pytest.fixture
def test_data_optimization():
    return [
        # Greetings
        {"query": "hello", "query_match": "greeting"},  # English
        {"query": "hola", "query_match": "greeting"},  # Spanish
        {"query": "bonjour", "query_match": "greeting"},  # French
        {"query": "ciao", "query_match": "greeting"},  # Italian
        {"query": "hallo", "query_match": "greeting"},  # German
        {"query": "こんにちは", "query_match": "greeting"},  # Japanese
        {"query": "안녕하세요", "query_match": "greeting"},  # Korean
        {"query": "你好", "query_match": "greeting"},  # Chinese
        {"query": "مرحبا", "query_match": "greeting"},  # Arabic
        {"query": "привет", "query_match": "greeting"},  # Russian
        {"query": "γεια σας", "query_match": "greeting"},  # Greek
        {"query": "namaste", "query_match": "greeting"},  # Hindi
        {"query": "olá", "query_match": "greeting"},  # Portuguese
        {"query": "salut", "query_match": "greeting"},  # French informal
        {"query": "cześć", "query_match": "greeting"},  # Polish
        # Farewells
        {"query": "goodbye", "query_match": "farewell"},  # English
        {"query": "adiós", "query_match": "farewell"},  # Spanish
        {"query": "au revoir", "query_match": "farewell"},  # French
        {"query": "arrivederci", "query_match": "farewell"},  # Italian
        {"query": "auf wiedersehen", "query_match": "farewell"},  # German
        {"query": "さようなら", "query_match": "farewell"},  # Japanese
        {"query": "안녕히 가세요", "query_match": "farewell"},  # Korean
        {"query": "再见", "query_match": "farewell"},  # Chinese
        {"query": "مع السلامة", "query_match": "farewell"},  # Arabic
        {"query": "до свидания", "query_match": "farewell"},  # Russian
        {"query": "αντίο", "query_match": "farewell"},  # Greek
        {"query": "अलविदा", "query_match": "farewell"},  # Hindi
        {"query": "adeus", "query_match": "farewell"},  # Portuguese
        {"query": "tchau", "query_match": "farewell"},  # Portuguese informal
        {"query": "do widzenia", "query_match": "farewell"},  # Polish
    ]


@pytest.fixture
def client(redis_url):
    return Redis.from_url(redis_url)


def test_routes_different_distance_thresholds_optimizer_default(
    routes, redis_url, test_data_optimization, hf_vectorizer
):
    redis = Redis.from_url(redis_url)
    skip_if_redis_version_below(redis, "7.0.0")

    zero_threshold = 0.0

    # Test that it updates the thresholds
    routes[0].distance_threshold = zero_threshold
    routes[1].distance_threshold = zero_threshold

    router = SemanticRouter(
        name="test_routes_different_distance_optimizer",
        routes=routes,
        vectorizer=hf_vectorizer,
        redis_url=redis_url,
        overwrite=True,
    )

    # szia is hello in hungarian and not in our test data
    matches = router.route_many("Szia", max_k=2)
    assert len(matches) == 0

    # now run optimizer
    router_optimizer = RouterThresholdOptimizer(router, test_data_optimization)
    router_optimizer.optimize(max_iterations=20, search_step=0.5)

    # test that it updated thresholds beyond the null case
    for route in routes:
        assert route.distance_threshold > zero_threshold


def test_routes_different_distance_thresholds_optimizer_precision(
    routes, redis_url, test_data_optimization, hf_vectorizer
):
    redis = Redis.from_url(redis_url)
    skip_if_redis_version_below(redis, "7.0.0")

    zero_threshold = 0.0

    # Test that it updates the thresholds
    routes[0].distance_threshold = zero_threshold
    routes[1].distance_threshold = zero_threshold

    router = SemanticRouter(
        name="test_routes_different_distance_optimizer",
        routes=routes,
        vectorizer=hf_vectorizer,
        redis_url=redis_url,
        overwrite=True,
    )

    # szia is hello in hungarian and not in our test data
    matches = router.route_many("Szia", max_k=2)
    assert len(matches) == 0

    # now run optimizer
    router_optimizer = RouterThresholdOptimizer(
        router, test_data_optimization, eval_metric="precision"
    )
    router_optimizer.optimize(max_iterations=20, search_step=0.5)

    # test that it updated thresholds beyond the null case
    for route in routes:
        assert route.distance_threshold > zero_threshold


def test_routes_different_distance_thresholds_optimizer_recall(
    routes, redis_url, test_data_optimization, hf_vectorizer, client
):
    redis = Redis.from_url(redis_url)
    skip_if_redis_version_below(redis, "7.0.0")

    zero_threshold = 0.0

    # Test that it updates the thresholds
    routes[0].distance_threshold = zero_threshold
    routes[1].distance_threshold = zero_threshold

    router = SemanticRouter(
        name="test_routes_different_distance_optimizer",
        routes=routes,
        vectorizer=hf_vectorizer,
        redis_url=redis_url,
        overwrite=True,
    )

    # szia is hello in hungarian and not in our test data
    matches = router.route_many("Szia", max_k=2)
    assert len(matches) == 0

    # now run optimizer
    router_optimizer = RouterThresholdOptimizer(
        router, test_data_optimization, eval_metric="recall"
    )
    router_optimizer.optimize(max_iterations=20, search_step=0.5)

    # test that it updated thresholds beyond the null case
    for route in routes:
        assert route.distance_threshold > zero_threshold


def test_optimize_threshold_cache_default(redis_url):
    null_threshold = 0.0
    cache = SemanticCache(
        name="test_optimize_threshold_cache",
        redis_url=redis_url,
        distance_threshold=null_threshold,
    )

    skip_if_redis_version_below(cache._index.client, "7.0.0")

    paris_key = cache.store(prompt="what is the capital of france?", response="paris")
    rabat_key = cache.store(prompt="what is the capital of morocco?", response="rabat")

    test_dict = [
        {"query": "what actually is the capital of france?", "query_match": paris_key},
        {"query": "what actually is the capital of morocco?", "query_match": rabat_key},
        {"query": "What is the state bird of virginia?", "query_match": ""},
    ]

    cache_optimizer = CacheThresholdOptimizer(cache, test_dict)

    cache_optimizer.optimize()

    assert cache.distance_threshold > null_threshold


def test_optimize_threshold_cache_precision(client, redis_url):
    skip_if_redis_version_below(client, "7.0.0")

    null_threshold = 0.0
    cache = SemanticCache(
        name="test_optimize_threshold_cache",
        redis_url=redis_url,
        distance_threshold=null_threshold,
    )

    paris_key = cache.store(prompt="what is the capital of france?", response="paris")
    rabat_key = cache.store(prompt="what is the capital of morocco?", response="rabat")

    test_dict = [
        {"query": "what actually is the capital of france?", "query_match": paris_key},
        {"query": "what actually is the capital of morocco?", "query_match": rabat_key},
        {"query": "What is the state bird of virginia?", "query_match": ""},
    ]

    cache_optimizer = CacheThresholdOptimizer(cache, test_dict, eval_metric="precision")

    cache_optimizer.optimize()

    assert cache.distance_threshold > null_threshold


def test_optimize_threshold_cache_recall(client, redis_url):
    skip_if_redis_version_below(client, "7.0.0")

    null_threshold = 0.0
    cache = SemanticCache(
        name="test_optimize_threshold_cache",
        redis_url=redis_url,
        distance_threshold=null_threshold,
    )

    paris_key = cache.store(prompt="what is the capital of france?", response="paris")
    rabat_key = cache.store(prompt="what is the capital of morocco?", response="rabat")

    test_dict = [
        {"query": "what actually is the capital of france?", "query_match": paris_key},
        {"query": "what actually is the capital of morocco?", "query_match": rabat_key},
        {"query": "What is the state bird of virginia?", "query_match": ""},
    ]

    cache_optimizer = CacheThresholdOptimizer(cache, test_dict, eval_metric="recall")

    cache_optimizer.optimize()

    assert cache.distance_threshold > null_threshold


def test_eval_metric_from_string():
    """Test that EvalMetric.from_string works for valid metrics."""
    assert EvalMetric("f1") == EvalMetric.F1
    assert EvalMetric("precision") == EvalMetric.PRECISION
    assert EvalMetric("recall") == EvalMetric.RECALL


def test_eval_metric_invalid():
    """Test that EvalMetric.from_string raises ValueError for invalid metrics."""
    with pytest.raises(ValueError):
        EvalMetric("invalid_metric")


def test_optimizer_with_invalid_metric(redis_url):
    """Test that optimizers raise ValueError when initialized with invalid metric."""
    cache = SemanticCache(
        name="test_invalid_metric",
        redis_url=redis_url,
    )

    test_dict = [{"query": "test", "query_match": ""}]

    with pytest.raises(ValueError):
        CacheThresholdOptimizer(cache, test_dict, eval_metric="invalid_metric")

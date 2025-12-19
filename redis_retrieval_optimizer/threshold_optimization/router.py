import logging
import random
from typing import Any, Callable, Dict, List

import numpy as np
from ranx import Qrels, Run, evaluate
from redisvl.extensions.router.semantic import SemanticRouter

from redis_retrieval_optimizer.threshold_optimization.base import (
    BaseThresholdOptimizer,
    EvalMetric,
)
from redis_retrieval_optimizer.threshold_optimization.schema import LabeledData
from redis_retrieval_optimizer.threshold_optimization.utils import (
    NULL_RESPONSE_KEY,
    _format_qrels,
)

logger = logging.getLogger(__name__)


def _generate_run_router(test_data: List[LabeledData], router: SemanticRouter) -> "Run":
    """Format router results into format for ranx Run"""
    if Run is None:
        raise ImportError("ranx is required for threshold optimization")
    if np is None:
        raise ImportError("numpy is required for threshold optimization")

    run_dict: Dict[Any, Any] = {}

    for td in test_data:
        run_dict[td.id] = {}
        route_match = router(td.query)
        if route_match and route_match.name == td.query_match:
            run_dict[td.id][td.query_match] = np.int64(1)
        else:
            run_dict[td.id][NULL_RESPONSE_KEY] = np.int64(1)

    return Run(run_dict)


def _eval_router(
    router: SemanticRouter,
    test_data: List[LabeledData],
    qrels: "Qrels",
    eval_metric: str,
) -> float:
    """Evaluate acceptable metric given run and qrels data"""
    if evaluate is None:
        raise ImportError("ranx is required for threshold optimization")

    run = _generate_run_router(test_data, router)
    return evaluate(qrels, run, eval_metric, make_comparable=True)


def _router_random_search(
    route_names: List[str], route_thresholds: dict, search_step=0.10
):
    """Performs random search for many thresholds to many routes"""
    if np is None:
        raise ImportError("numpy is required for threshold optimization")

    score_threshold_values = []
    for route in route_names:
        score_threshold_values.append(
            np.linspace(
                start=max(route_thresholds[route] - search_step, 0),
                stop=route_thresholds[route] + search_step,
                num=100,
            )
        )

    return {
        route: float(random.choice(score_threshold_values[i]))
        for i, route in enumerate(route_names)
    }


def _random_search_opt_router(
    router: SemanticRouter,
    test_data: List[LabeledData],
    qrels: "Qrels",
    eval_metric: EvalMetric,
    **kwargs: Any,
):
    """Performs complete optimization for router cases provide acceptable metric"""

    start_score = _eval_router(router, test_data, qrels, eval_metric.value)
    best_score = start_score
    best_thresholds = router.route_thresholds

    max_iterations = kwargs.get("max_iterations", 20)
    search_step = kwargs.get("search_step", 0.10)

    for _ in range(max_iterations):
        route_names = router.route_names
        route_thresholds = router.route_thresholds
        thresholds = _router_random_search(
            route_names=route_names,
            route_thresholds=route_thresholds,
            search_step=search_step,
        )
        router.update_route_thresholds(thresholds)
        score = _eval_router(router, test_data, qrels, eval_metric.value)
        if score > best_score:
            best_score = score
            best_thresholds = thresholds

    logger.info(
        "Eval metric %s: start %.3f, end %.3f. Ending thresholds: %s",
        eval_metric.value.upper(),
        round(start_score, 3),
        round(best_score, 3),
        router.route_thresholds,
    )
    router.update_route_thresholds(best_thresholds)


class RouterThresholdOptimizer(BaseThresholdOptimizer):
    """
    Class for optimizing thresholds for a SemanticRouter.

    .. code-block:: python

        from redisvl.extensions.router import Route, SemanticRouter
        from redisvl.utils.vectorize import HFTextVectorizer
        from redis_retrieval_optimizer.threshold_optimization import RouterThresholdOptimizer

        routes = [
                Route(
                    name="greeting",
                    references=["hello", "hi"],
                    metadata={"type": "greeting"},
                    distance_threshold=0.5,
                ),
                Route(
                    name="farewell",
                    references=["bye", "goodbye"],
                    metadata={"type": "farewell"},
                    distance_threshold=0.5,
                ),
            ]

        router = SemanticRouter(
            name="greeting-router",
            vectorizer=HFTextVectorizer(),
            routes=routes,
            redis_url="redis://localhost:6379",
            overwrite=True # Blow away any other routing index with this name
        )

        test_data = [
            {"query": "hello", "query_match": "greeting"},
            {"query": "goodbye", "query_match": "farewell"},
            ...
        ]

        optimizer = RouterThresholdOptimizer(router, test_data)
        optimizer.optimize()
    """

    def __init__(
        self,
        router: SemanticRouter,
        test_dict: List[Dict[str, Any]],
        opt_fn: Callable = _random_search_opt_router,
        eval_metric: str = "f1",
    ):
        """Initialize the router optimizer.

        Args:
            router (SemanticRouter): The RedisVL SemanticRouter instance to optimize.
            test_dict (List[Dict[str, Any]]): List of test cases.
            opt_fn (Callable): Function to perform optimization. Defaults to
                grid search.
            eval_metric (str): Evaluation metric for threshold optimization.
                Defaults to "f1" score.
        Raises:
            ValueError: If the test_dict not in LabeledData format.
        """
        super().__init__(router, test_dict, opt_fn, eval_metric)

    def optimize(self, **kwargs: Any):
        """Optimize kicks off the optimization process for router"""
        qrels = _format_qrels(self.test_data)
        self.opt_fn(self.optimizable, self.test_data, qrels, self.eval_metric, **kwargs)

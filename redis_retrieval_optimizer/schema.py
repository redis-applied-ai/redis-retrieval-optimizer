from typing import Any, Dict, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from ranx import Run
from redisvl.index import SearchIndex
from redisvl.utils.vectorize.base import BaseVectorizer


class QueryMetrics(BaseModel):
    """
    Tracks query execution performance metrics.

    This class stores timing information for each query executed during
    a search operation, allowing for performance analysis and optimization.

    Timing Format:
    - query_times: List of execution times in seconds
    - Index corresponds to query order: query_times[0] = time for first query
    - Example: [0.125, 0.098, 0.156] means 3 queries took 125ms, 98ms, and 156ms
    """

    query_times: list[float] = Field(
        default_factory=list, description="List of query execution times in seconds"
    )


class SearchMethodInput(BaseModel):
    """
    Input parameters for search method execution.

    This class provides all necessary information for a search method to execute
    queries against a Redis index, including the queries, index configuration,
    and search parameters.

    Query Format Options:
    - Dictionary format: {query_id: {"query": "text", "query_metadata": {...}}}
      Example: {"q1": {"query": "car search", "query_metadata": {"make": "Toyota"}}}
    - List format: ["query1", "query2", "query3"]
      Example: ["car search", "truck search", "bike search"]
    """

    model_config = {"arbitrary_types_allowed": True}

    # Core search components
    raw_queries: Union[Dict[str, Union[str, Dict[str, Any]]], list[str]] = Field(
        description="Queries to execute - either a dict with query_id as key and query info as value, or a list of query strings"
    )
    index: Optional[SearchIndex] = Field(description="Redis index to search against")

    # Search configuration
    ret_k: int = Field(default=6, description="Number of results to retrieve per query")
    id_field_name: str = Field(
        default="_id",
        description="Field name containing document IDs in the Redis index",
    )
    text_field_name: str = Field(
        default="text",
        description="Field name containing document text content in the Redis index",
    )
    vector_field_name: str = Field(
        default="vector",
        description="Field name containing document vector embeddings in the Redis index",
    )

    # Optional components
    emb_model: Optional[BaseVectorizer] = Field(
        default=None, description="Embedding model for vector operations"
    )
    query_metrics: QueryMetrics = Field(
        default_factory=QueryMetrics, description="Performance metrics tracker"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional search parameters"
    )

    @field_validator("index")
    @classmethod
    def validate_index(cls, v):
        # Allow None for testing purposes, but validate actual instances
        if v is not None and not isinstance(v, SearchIndex):
            raise ValueError("Must be a SearchIndex instance")
        return v


class SearchMethodOutput(BaseModel):
    """
    Output from search method execution.

    Contains the search results in ranx Run format and associated
    performance metrics for evaluation.

    Run Format (ranx.Run):
    The run field contains search results in the standard ranx format:
    {query_id: {doc_id: score, doc_id: score, ...}}

    Example:
    {
        "q1": {"doc1": 0.95, "doc2": 0.87, "doc3": 0.76},
        "q2": {"doc4": 0.92, "doc1": 0.85, "doc5": 0.71}
    }

    Where:
    - query_id: matches the input query identifier
    - doc_id: document identifier from the search index
    - score: relevance score (higher = more relevant, typically 0.0 to 1.0)

    Evaluation:
    This Run format is designed to work with ranx.evaluate() alongside Qrels
    (query relevance labels) to compute metrics like NDCG, recall, precision, and F1.

    Qrels Format (ranx.Qrels):
    Qrels define ground truth relevance and follow the same structure:
    {query_id: {doc_id: relevance_score, doc_id: relevance_score, ...}}

    Where relevance_score is typically:
    - 0: Not relevant
    - 1: Relevant
    - 2: Highly relevant
    - etc. (depending on your relevance scale)
    """

    model_config = {"arbitrary_types_allowed": True}

    run: Run = Field(description="Search results in ranx Run format")
    query_metrics: QueryMetrics = Field(
        description="Performance metrics from query execution"
    )

    @field_validator("run")
    @classmethod
    def validate_run(cls, v):
        if not isinstance(v, Run):
            raise ValueError("Must be a ranx Run instance")
        return v

    def get_avg_query_time(self) -> float:
        """Get the average query execution time."""
        times = self.query_metrics.query_times
        return sum(times) / len(times) if times else 0.0

    def get_total_query_time(self) -> float:
        """Get the total query execution time."""
        return sum(self.query_metrics.query_times)


class AdditionalField(BaseModel):
    name: str
    type: str


class DataSettings(BaseModel):
    corpus: str
    queries: str
    qrels: str


class IndexSettings(BaseModel):
    name: str = "ret-opt"
    prefix: str = "ret-opt"
    from_existing: bool = False
    vector_dim: int
    algorithm: str = "flat"
    vector_field_name: str = "vector"
    id_field_name: str = "_id"
    text_field_name: str = "text"
    distance_metric: str = "cosine"
    vector_data_type: str = "float32"
    ef_construction: int = 0
    ef_runtime: int = 0
    m: int = 0
    additional_fields: list[AdditionalField] = []


class LabeledItem(BaseModel):
    query: str
    query_metadata: dict = {}
    relevant_item_ids: list[str]


class EmbeddingModel(BaseModel):
    type: str
    model: str
    dim: int
    embedding_cache_name: str = "vec_cache"
    dtype: str = "float32"
    # redis_url should be set in the env


class MetricWeights(BaseModel):
    f1: float = 0
    recall: float = 0
    ndcg: float = 0
    precision: float = 0
    total_indexing_time: float = 0
    avg_query_time: float = 0


class TrialSettings(BaseModel):
    trial_id: str = str(uuid4())
    index_settings: IndexSettings
    embedding: EmbeddingModel
    data: DataSettings
    ret_k: int = 6
    search_method: str = "vector"


class OptimizationSettings(BaseModel):
    algorithms: list[str]
    vector_data_types: list[str]
    distance_metrics: list[str]
    n_trials: int
    n_jobs: int
    metric_weights: MetricWeights = MetricWeights()
    ret_k: tuple[int, int] = [1, 10]  # type: ignore # pydantic vs mypy
    ef_runtime: list = [10, 50]
    ef_construction: list = [100, 300]
    m: list = [8, 64]


class BayesStudyConfig(BaseModel):
    study_id: str = str(uuid4())
    corpus: str
    qrels: str
    queries: str
    index_settings: IndexSettings
    optimization_settings: OptimizationSettings
    embedding_models: list[EmbeddingModel]
    search_methods: list[str]


class GridStudyConfig(BaseModel):
    study_id: str = str(uuid4())
    # index settings
    index_settings: IndexSettings

    # data
    corpus: str = ""
    qrels: str
    queries: str

    embedding_models: list[EmbeddingModel]
    search_methods: list[str]
    ret_k: int = 6
    vector_data_types: list[str] = ["float32"]  # data types to be included in the study


class SearchStudyConfig(BaseModel):
    study_id: str = str(uuid4())
    index_name: str

    qrels: str
    queries: str

    search_methods: list[str]
    ret_k: int = 6
    id_field_name: str = "_id"
    vector_field_name: str = "vector"
    text_field_name: str = "text"

    # embedding model for vector-based search methods (must match index's model)
    embedding_model: EmbeddingModel


def get_trial_settings(trial, study_config):

    model_info = trial.suggest_categorical(
        "model_info",
        [m.model_dump() for m in study_config.embedding_models],
    )

    search_method = trial.suggest_categorical(
        "search_method", study_config.search_methods
    )

    study_config.index_settings.algorithm = trial.suggest_categorical(
        "algorithm", study_config.optimization_settings.algorithms
    )
    study_config.index_settings.vector_data_type = trial.suggest_categorical(
        "var_dtype", study_config.optimization_settings.vector_data_types
    )
    study_config.index_settings.distance_metric = trial.suggest_categorical(
        "distance_metric", study_config.optimization_settings.distance_metrics
    )

    ret_k = trial.suggest_int(
        "ret_k",
        study_config.optimization_settings.ret_k[0],
        study_config.optimization_settings.ret_k[1],
    )

    if study_config.index_settings.algorithm == "hnsw":
        ef_runtime = trial.suggest_categorical(
            "ef_runtime", study_config.optimization_settings.ef_runtime
        )
        ef_construction = trial.suggest_categorical(
            "ef_construction", study_config.optimization_settings.ef_construction
        )
        m = trial.suggest_categorical("m", study_config.optimization_settings.m)

        study_config.index_settings.ef_construction = ef_construction
        study_config.index_settings.ef_runtime = ef_runtime
        study_config.index_settings.m = m

    embedding_settings = EmbeddingModel(
        model=model_info["model"],
        dim=model_info["dim"],
        type=model_info["type"],
        embedding_cache_name=model_info["embedding_cache_name"],
        dtype=study_config.index_settings.vector_data_type,
    )

    data_settings = DataSettings(
        corpus=study_config.corpus,
        queries=study_config.queries,
        qrels=study_config.qrels,
    )

    return TrialSettings(
        index_settings=study_config.index_settings,
        embedding=embedding_settings,
        ret_k=ret_k,
        data=data_settings,
        search_method=search_method,
    )

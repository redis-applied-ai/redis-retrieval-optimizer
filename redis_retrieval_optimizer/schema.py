from typing import Any
from uuid import uuid4

from pydantic import BaseModel, field_validator
from ranx import Run
from redisvl.index import SearchIndex
from redisvl.utils.vectorize.base import BaseVectorizer


class QueryMetrics(BaseModel):
    query_times: list[float] = []


class SearchMethodInput(BaseModel):
    raw_queries: Any
    index: Any
    query_metrics: QueryMetrics = QueryMetrics()
    emb_model: BaseVectorizer = None
    id_field_name: str = "_id"
    text_field_name: str = "text"
    vector_field_name: str = "vector"
    kwargs: dict = {}

    @field_validator("index")
    @classmethod
    def validate_index(cls, v):
        if not isinstance(v, SearchIndex):
            raise ValueError("Must be a SearchIndex instance")
        return v


class SearchMethodOutput(BaseModel):
    run: Any
    query_metrics: QueryMetrics

    @field_validator("run")
    @classmethod
    def validate_run(cls, v):
        if not isinstance(v, Run):
            raise ValueError("Must be a ranx Run instance")
        return v


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

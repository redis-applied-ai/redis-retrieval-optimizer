from typing import Optional
from uuid import uuid4

from pydantic import BaseModel


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
    id_field_name: str = "_id"
    text_field_name: str = "text"
    vector_field_name: str = "vector"
    from_existing: bool = False
    vector_dim: int
    algorithm: str = "flat"
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
    embedding_cache_name: str = ""
    # redis_url should be set in the env


class MetricWeights(BaseModel):
    f1_at_k: int = 1
    embedding_latency: int = 1
    total_indexing_time: int = 1


class TrialSettings(BaseModel):
    trial_id: str = str(uuid4())
    index_settings: IndexSettings
    embedding: EmbeddingModel
    data: DataSettings
    ret_k: int = 6
    search_method: str = "vector"


class StudyConfig(BaseModel):
    study_id: str = str(uuid4())
    redis_url: str = "redis://localhost:6379/0"
    algorithms: list[str]
    vector_data_types: list[str]
    distance_metrics: list[str]
    corpus: str
    qrels: str
    queries: str
    embedding_models: list[EmbeddingModel]
    n_trials: int
    n_jobs: int
    metric_weights: MetricWeights = MetricWeights()
    search_methods: list[str]
    ret_k: tuple[int, int] = [1, 10]  # type: ignore # pydantic vs mypy
    ef_runtime: list = [10, 50]
    ef_construction: list = [100, 300]
    m: list = [8, 64]


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


def get_trial_settings(trial, study_config, custom_retrievers=None):

    model_info = trial.suggest_categorical(
        "model_info",
        [m.model_dump() for m in study_config.embedding_models],
    )

    search_method = trial.suggest_categorical(
        "search_method", study_config.search_methods
    )

    # if custom_retrievers:
    #     retriever_name = trial.suggest_categorical(
    #         "retriever", list(custom_retrievers.keys())
    #     )
    #     obj = custom_retrievers[retriever_name]
    #     retriever = obj["retriever"]
    #     additional_schema_fields = custom_retrievers[retriever_name].get(
    #         "additional_schema_fields", None
    #     )
    # else:
    #     retriever = DefaultQueryRetriever
    #     additional_schema_fields = None

    # logging.info(
    #     f"Running for Retriever: {retriever.__name__} with {additional_schema_fields=}"
    # )

    algorithm = trial.suggest_categorical("algorithm", study_config.algorithms)
    vec_dtype = trial.suggest_categorical("var_dtype", study_config.vector_data_types)
    dist_metric = trial.suggest_categorical(
        "distance_metric", study_config.distance_metrics
    )

    ret_k = trial.suggest_int("ret_k", study_config.ret_k[0], study_config.ret_k[1])

    index_settings = IndexSettings(
        algorithm=algorithm,
        distance_metric=dist_metric,
        vector_data_type=vec_dtype,
    )

    if algorithm == "hnsw":
        ef_runtime = trial.suggest_categorical("ef_runtime", study_config.ef_runtime)
        ef_construction = trial.suggest_categorical(
            "ef_construction", study_config.ef_construction
        )
        m = trial.suggest_categorical("m", study_config.m)

        index_settings.ef_construction = ef_construction
        index_settings.ef_runtime = ef_runtime
        index_settings.m = m

    embedding_settings = EmbeddingModel(
        model=model_info["model"],
        dim=model_info["dim"],
        type=model_info["type"],
        embedding_cache_name=model_info["embedding_cache_name"],
    )

    data_settings = DataSettings(
        corpus=study_config.corpus,
        queries=study_config.queries,
        qrels=study_config.qrels,
    )

    return TrialSettings(
        index=index_settings,
        embedding=embedding_settings,
        ret_k=ret_k,
        data=data_settings,
        search_method=search_method,
    )

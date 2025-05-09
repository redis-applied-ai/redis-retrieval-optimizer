<div align="center">
<div><img src="https://raw.githubusercontent.com/redis/redis-vl-python/main/docs/_static/Redis_Logo_Red_RGB.svg" style="width: 130px"> </div>
<h1>Retrieval Optimizer</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/github/languages/top/redis-applied-ai/retrieval-optimizer)
![GitHub last commit](https://img.shields.io/github/last-commit/redis-applied-ai/retrieval-optimizer)

</div>

Search and information retrieval is a challenging problem. With the proliferation of vector search tools in the market, focus has heavily shifted towards SEO and marketing wins, rather than fundamental quality.

The **Retrieval Optimizer** from Redis focuses on measuring and improving retrieval quality. This framework helps determine optimal **embedding models**, **retrieval strategies**, and **index configurations** for your specific data and use case. It implements all redis indexing and embedding caching for you to quickly iterate different scenarios.


# Quick start

## Grid study
Tries all options specified good for studies where the number of configurations you're testing is small or you want to be exact.

Define study config
```yaml
# paths to necessary data files
corpus: "data/nfcorpus_corpus.json" # optional if from_existing
queries: "data/nfcorpus_queries.json"
qrels: "data/nfcorpus_qrels.json"

# vector field names
index_settings:
  name: "optimize"
  vector_field_name: "vector" # name of the vector field to search on
  text_field_name: "text" # name of the text field for lexical search
  from_existing: false
  additional_fields:
    - name: "title"
      type: "text"
  vector_dim: 384 # should match first embedding model or from_existing

# will run all search methods for each embedding model and then iterate
embedding_models: # embedding cache would be awesome here.
# if from_existing is true, first record is assumed to be the one used to create the index
  - type: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dim: 384
    embedding_cache_name: "vec-cache" # avoid names with including 'ret-opt' as this can cause collisions

search_methods: ["bm25", "vector", "lin_combo", "rerank", "weighted_rrf"] # must match what is passed in search_method_map
```

Code
```python
import os
from redis_retrieval_optimizer.grid_study import run_grid_study
from redis_retrieval_optimizer.corpus_processors import eval_beir
from dotenv import load_dotenv

# load environment variables containing necessary credentials
load_dotenv()

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

metrics = run_grid_study(
    config_path="grid_study_config.yaml",
    redis_url="redis://localhost:6379/0",
    corpus_processor=eval_beir.process_corpus
)
```

Example output
![grid study output](/reference/grid_output.png)

## Bayesian Optimization
Selects the next best configuration to try based on a heuristic. This is good when it would take a very long time to test all possible configurations.

Study config:
```yaml
# path to data files for easy read
corpus: "data/nfcorpus_corpus.json"
queries: "data/nfcorpus_queries.json"
qrels: "data/nfcorpus_qrels.json"

index_settings:
  name: "optimize"
  vector_field_name: "vector" # name of the vector field to search on
  text_field_name: "text" # name of the text field for lexical search
  from_existing: false
  vector_dim: 384 # should match first embedding model or from_existing
  additional_fields:
      - name: "title"
        type: "text"

optimization_settings:
  # defines the options optimization can take
  metric_weights:
    f1_at_k: 1
    embedding_latency: 1
    total_indexing_time: 1
  algorithms: ["hnsw"]
  vector_data_types: ["float16", "float32"]
  distance_metrics: ["cosine"]
  n_trials: 10
  n_jobs: 1
  ret_k: [1, 10] # potential range of value to be sampled during study
  ef_runtime: [10, 20, 30, 50]
  ef_construction: [100, 150, 200, 250, 300]
  m: [8, 16, 64]


search_methods: ["vector", "lin_combo"]
embedding_models:
  - type: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dim: 384
    embedding_cache_name: "vec-cache" # avoid names with including 'ret-opt' as this can cause collisions
    dtype: "float32"
```

```python
import os
from redis_retrieval_optimizer.bayes_study import run_bayes_study
from redis_retrieval_optimizer.corpus_processors import eval_beir
from dotenv import load_dotenv

# load environment variables containing necessary credentials
load_dotenv()

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

metrics = run_bayes_study(
    config_path="bayes_study_config.yaml",
    redis_url=redis_url,
    corpus_processor=eval_beir.process_corpus
)
```

Example output
![bayes study output](/reference/bayes_output.png)

## Custom processors/search methods

With the retrieval optimizer you can specify your own search methods and corpus processor functions.

Study config
```yaml
# paths to necessary data files
corpus: "data/car_corpus.json" # optional if from_existing
queries: "data/car_queries.json"
qrels: "data/car_qrels.json"

# vector field names
index_settings:
  name: "car"
  prefix: "car" # prefix for index name
  vector_field_name: "vector" # name of the vector field to search on
  text_field_name: "text" # name of the text field for lexical search
  from_existing: false
  additional_fields:
    - name: "make"
      type: "tag"
    - name: "model"
      type: "tag"
  vector_dim: 384 # should match first embedding model or from_existing

# will run all search methods for each embedding model and then iterate
embedding_models: # embedding cache would be awesome here.
# if from_existing is true, first record is assumed to be the one used to create the index
  - type: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dim: 384
    embedding_cache_name: "vec-cache" # avoid names with including 'ret-opt' as this can cause collisions

search_methods: ["basic_vector", "pre_filter_vector"] # must match what is passed in search_method_map
```

Custom search methods:

Note: A search method can be whatever you want it just has to take a `SearchMethodInput` and return a `SearchMethodOutput`. Between taking the input and formatting the output - go wild!

```python
from ranx import Run
from redis_retrieval_optimizer.search_methods.base import run_search_w_time
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

from redis_retrieval_optimizer.schema import SearchMethodInput, SearchMethodOutput
from redis_retrieval_optimizer.search_methods.vector import make_score_dict_vec

def vector_query(query_info, num_results: int, emb_model) -> VectorQuery:
    vector = emb_model.embed(query_info["query"], as_buffer=True)

    return VectorQuery(
        vector=vector,
        vector_field_name="vector",
        num_results=num_results,
        return_fields=["_id", "make", "model", "text"],  # update to read from env maybe?
    )

def pre_filter_query(query_info, num_results, emb_model) -> VectorQuery:
    vec = emb_model.embed(query_info["query"])
    make = query_info["query_metadata"]["make"]
    model = query_info["query_metadata"]["model"]

    filter = (Tag("make") == make) & (Tag("model") == model)

    # Create a vector query
    query = VectorQuery(
        vector=vec,
        vector_field_name="vector",
        num_results=num_results,
        filter_expression=filter,
        return_fields=["_id", "make", "model", "text"]
    )

    return query

def gather_pre_filter_results(search_method_input: SearchMethodInput) -> SearchMethodOutput:
    redis_res_vector = {}

    for key in search_method_input.raw_queries:
        query_info = search_method_input.raw_queries[key]
        query = pre_filter_query(query_info, 10, search_method_input.emb_model)
        res = run_search_w_time(
            search_method_input.index, query, search_method_input.query_metrics
        )
        score_dict = make_score_dict_vec(res)

        redis_res_vector[key] = score_dict

    return SearchMethodOutput(
        run=Run(redis_res_vector),
        query_metrics=search_method_input.query_metrics,
    )


def gather_vector_results(search_method_input: SearchMethodInput) -> SearchMethodOutput:
    redis_res_vector = {}

    for key in search_method_input.raw_queries:
        text_query = search_method_input.raw_queries[key]
        vec_query = vector_query(text_query, 10, search_method_input.emb_model)
        res = run_search_w_time(
            search_method_input.index, vec_query, search_method_input.query_metrics
        )
        score_dict = make_score_dict_vec(res)
        redis_res_vector[key] = score_dict

    return SearchMethodOutput(
        run=Run(redis_res_vector),
        query_metrics=search_method_input.query_metrics,
    )

```

Custom corpus processor:

The corpus you provide can be of any format as long as the corpus processing function provide returns an object that can be indexed according to the index_settings in the study config.

```python
def process_car_corpus(
    corpus, emb_model
):
    corpus_data = []
    corpus_texts = [c["text"] for c in corpus]

    text_embeddings = emb_model.embed_many(corpus_texts, as_buffer=True)

    for i, c in enumerate(corpus):
        corpus_data.append(
            {
                "_id": c["item_id"],
                "text": c["text"],
                "make": c["query_metadata"]["make"],
                "model": c["query_metadata"]["model"],
                "vector": text_embeddings[i],
            }
        )

    return corpus_data
```

Run custom study:

```python
import os
from redis_retrieval_optimizer.grid_study import run_grid_study
from dotenv import load_dotenv

CUSTOM_SEARCH_METHOD_MAP = {
    "basic_vector": gather_vector_results,
    "pre_filter_vector": gather_pre_filter_results,
}

# load environment variables containing necessary credentials
load_dotenv()

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

metrics = run_grid_study(
    config_path="custom_grid_study_config.yaml",
    redis_url="redis://localhost:6379/0",
    corpus_processor=process_car_corpus,
    search_method_map=CUSTOM_SEARCH_METHOD_MAP,
)
```

# Data requirements

### Queries

Note: can be any form if you define a custom search methods that know how to unpack the objects correctly.

General form:
```json
{
    "corpus_id": {
        "text": "test to be searched on or vectorized",
        "title": "associated title",
    }
}
```

Concrete example:
```json
{
    "MED-10": {
        "text": "Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear. ...",
        "title": "Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland"
    },
    ...
}
```

### Corpus

Note: can be of any form if custom corpus processor provided. Also, can be read **directly from pre-existing index** if you already have a redis instance live.

### Qrels

Qrels MUST be of the following form in order for the metrics to be calculated correctly.

General form:
```json
{
    "query_id": {
        "corpus_id": "score",
        ...
    },
    ...
}
```

Concrete example:
```json
{
    "PLAIN-2": {
        "MED-2427": 2,
        "MED-2440": 1,
        "MED-2434": 1,
        "MED-2435": 1,
        "MED-2436": 1,
    },
    "PLAIN-12": {
        "MED-2513": 2,
        "MED-5237": 2,
    },
}
```

# Contributing
We love contributors if you have an addition follow this process:
- Fork the repo
- Make contribution
- Add tests for contribution to test folder
- Make a PR
- Get reviewed
- Merged!

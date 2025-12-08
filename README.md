<div align="center">

<img src="https://raw.githubusercontent.com/redis/redis-vl-python/main/docs/_static/Redis_Logo_Red_RGB.svg" width="130">

# 🚀 Redis Retrieval Optimizer


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/github/languages/top/redis-applied-ai/retrieval-optimizer)
![GitHub last commit](https://img.shields.io/github/last-commit/redis-applied-ai/retrieval-optimizer)

---

**Stop guessing. Start measuring.** 📊

Transform your retrieval system from *"looks good to me"* to *"proven to perform"* with systematic benchmarking and Bayesian optimization.

</div>

## ✨ What is Redis Retrieval Optimizer?

The **Retrieval Optimizer** is a powerful framework that takes the guesswork out of building high-performance search systems. Whether you're working with vector search, hybrid retrieval, or traditional text search, this tool helps you **scientifically measure and optimize** your search performance.

### 🎯 Why You Need This

- **🔬 Evidence-Based Decisions**: Replace assumptions with hard data about what actually works for your use case
- **⚡ Performance Optimization**: Find the perfect balance between speed, accuracy, and cost
- **🤖 Smart Automation**: Use Bayesian optimization to intelligently explore the configuration space
- **📈 Comprehensive Metrics**: Track everything from latency to relevance with industry-standard evaluation metrics
- **🔧 Easy Integration**: Works with your existing Redis setup and popular embedding models

### 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **🔄 Grid Search** | Systematically test different embedding models and retrieval strategies |
| **🧠 Bayesian Optimization** | Intelligently fine-tune index configurations for optimal performance |
| **📊 Rich Metrics** | Track recall, precision, NDCG, F1, query time, and indexing performance |
| **🔍 Multiple Search Methods** | Compare BM25, vector search, hybrid, reranking, and weighted RRF |
| **🎛️ Threshold Optimization** | Automatically tune semantic cache and router thresholds |
| **🔌 Extensible Design** | Easily add custom search methods and data processors |


### 🎪 How It Works

1. **📁 Prepare Your Data**: Bring your corpus, queries, and relevance labels
2. **⚙️ Define Your Tests**: Choose embedding models and search methods to compare
3. **🔬 Run Experiments**: Let the optimizer systematically test configurations
4. **📈 Analyze Results**: Get detailed performance metrics and recommendations
5. **🚀 Deploy Optimized Config**: Use the best-performing setup in production

**The result?** A search system backed by data, not hunches.


## 📦 Installation

Install the optimizer from [PyPI](https://pypi.org/project/redis-retrieval-optimizer/):

```bash
pip install redis-retrieval-optimizer
```

## 📚 Example Notebooks

For complete code examples, see the following notebooks:

| Topic | Notebook |
| ------ | ------- |
| Basic grid study | [00_grid_study.ipynb](https://github.com/redis-applied-ai/redis-retrieval-optimizer/blob/main/docs/examples/grid_study/00_grid_study.ipynb) |
| Custom grid study | [01_custom_grid_study.ipynb](https://github.com/redis-applied-ai/redis-retrieval-optimizer/blob/main/docs/examples/grid_study/01_custom_grid_study.ipynb) |
| Bayesian Optimization | [00_bayes_study.ipynb](https://github.com/redis-applied-ai/redis-retrieval-optimizer/blob/main/docs/examples/bayesian_optimization/00_bayes_study.ipynb) |
| Search study | [00_search_study.ipynb](https://github.com/redis-applied-ai/redis-retrieval-optimizer/blob/main/docs/examples/search_study/00_search_study.ipynb) |
| Embedding model comparison | [00_comparison.ipynb](https://github.com/redis-applied-ai/redis-retrieval-optimizer/blob/main/docs/examples/comparison/00_comparison.ipynb) |

---

## 🚀 Quick Start

The Retrieval Optimizer supports three *study* types: **Grid**, **Bayesian Optimization**, and **Search Study**. Each is suited to a different stage of building a high-quality search system.

### Grid

Use a grid study to explore the impact of different **embedding models** and **retrieval strategies**. These are typically the most important factors influencing search performance. This mode is ideal for establishing a performance baseline and identifying which techniques work best for your dataset.

### Bayesian optimization

Once you've identified a solid starting point, use Bayesian optimization to **fine-tune your index configuration**. This mode intelligently selects the most promising combinations to test, in place of exhaustive testing (which is time-consuming). Bayesian optimization mode is especially useful for balancing **cost, speed, and latency** as you work toward a production-ready solution.

### Search Study

Use a search study when you have an **existing Redis index** and want to quickly test different search methods against it without recreating the index or data. This is ideal for A/B testing search strategies or evaluating custom search methods on production data.

## Running a Grid study

#### Study config
```yaml
# paths to necessary data files
corpus: "data/nfcorpus_corpus.json"
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

search_methods: ["bm25", "vector", "hybrid", "rerank", "weighted_rrf"] # must match what is passed in search_method_map

# data types to be included in the study (optional, defaults to ["float32"])
vector_data_types: ["float16", "float32"]
```

#### Code
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

#### Example output
| search_method | model                                      | avg_query_time | recall | precision | ndcg  |
|----------------|---------------------------------------------|----------------|-----------|-----------|----------|
| weighted_rrf   | sentence-transformers/all-MiniLM-L6-v2     | 0.006608       | 0.156129  | 0.261056  | 0.204241 |
| rerank         | sentence-transformers/all-MiniLM-L6-v2     | 0.127574       | 0.156039  | 0.260437  | 0.190298 |
| lin_combo      | sentence-transformers/all-MiniLM-L6-v2     | 0.003678       | 0.119653  | 0.302993  | 0.173768 |
| bm25           | sentence-transformers/all-MiniLM-L6-v2     | 0.000922       | 0.115798  | 0.323891  | 0.168909 |
| vector         | sentence-transformers/all-MiniLM-L6-v2     | 0.003378       | 0.119653  | 0.302993  | 0.165573 |


## Running a Bayesian optimization
Selects the next best configuration to try based on a heuristic. This is good when it would take a very long time to test all possible configurations.

#### Study config
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

# section for bayesian optimization
optimization_settings:
  # defines weight of each metric in optimization function
  metric_weights:
    f1: 1
    total_indexing_time: 1  # weight for total indexing time (seconds to reach 100% indexed)
  algorithms: ["hnsw"] # indexing algorithm to be included in the study
  vector_data_types: ["float16", "float32"] # data types to be included in the study
  distance_metrics: ["cosine"] # distance metrics to be included in the study
  n_trials: 10 # total number of trials to run
  n_jobs: 1
  ret_k: [1, 10] # potential range of value to be sampled during study
  ef_runtime: [10, 20, 30, 50] # potential values for ef_runtime to take
  ef_construction: [100, 150, 200, 250, 300] # potential values for ef_construction to take
  m: [8, 16, 64] # potential values for m to take

# potential values for search method
search_methods: ["vector", "hybrid"]

# potential values for embedding models
embedding_models:
  - type: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dim: 384
    embedding_cache_name: "vec-cache" # avoid names with including 'ret-opt' as this can cause collisions
    dtype: "float32"

```

The `total_indexing_time` metric is measured in **seconds** using wall-clock time
from when indexing starts until Redis reports `percent_indexed == 1`. When a
study reuses an existing index without reloading data, the previously measured
indexing time is reused instead of querying `index.info()["total_indexing_time"]`.


#### Code
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

#### Example output
| search_method | algorithm | vector_data_type | ef_construction | ef_runtime | m  | avg_query_time | total_indexing_time | f1    |
|---------------|-----------|------------------|------------------|------------|----|----------------|----------------------|---------|
| hybrid        | hnsw      | float16          | 200              | 50         | 8  | 0.004628       | 3.559                | 0.130712|
| hybrid        | hnsw      | float16          | 200              | 50         | 64 | 0.004498       | 4.804                | 0.130712|
| hybrid        | hnsw      | float16          | 150              | 50         | 64 | 0.004520       | 3.870                | 0.130712|
| hybrid        | hnsw      | float32          | 100              | 50         | 64 | 0.003387       | 1.929                | 0.130712|
| hybrid        | hnsw      | float16          | 150              | 50         | 8  | 0.004771       | 2.496                | 0.130712|
| hybrid        | hnsw      | float32          | 300              | 50         | 16 | 0.003461       | 3.622                | 0.130712|
| hybrid        | hnsw      | float16          | 100              | 50         | 16 | 0.004402       | 3.120                | 0.130712|
| hybrid        | hnsw      | float16          | 100              | 50         | 64 | 0.004615       | 3.361                | 0.130712|
| hybrid        | hnsw      | float16          | 250              | 50         | 16 | 0.005002       | 3.627                | 0.130712|
| hybrid        | hnsw      | float32          | 150              | 50         | 8  | 0.003246       | 2.471                | 0.130712|
| hybrid        | hnsw      | float32          | 300              | 50         | 8  | 0.002921       | 3.443                | 0.130712|
| hybrid        | hnsw      | float16          | 250              | 50         | 8  | 0.004366       | 3.094                | 0.130712|
| hybrid        | hnsw      | float32          | 250              | 50         | 8  | 0.003318       | 3.126                | 0.130712|
| vector        | hnsw      | float32          | 200              | 50         | 64 | 0.001116       | 2.790                | 0.130712|
| vector        | hnsw      | float16          | 200              | 50         | 64 | 0.001965       | 4.808                | 0.129692|
| vector        | hnsw      | float32          | 200              | 50         | 16 | 0.001359       | 2.773                | 0.129692|
| vector        | hnsw      | float16          | 150              | 50         | 8  | 0.001405       | 3.907                | 0.128089|
| vector        | hnsw      | float32          | 300              | 50         | 8  | 0.003236       | 2.742                | 0.127207|
| vector        | hnsw      | float32          | 100              | 50         | 8  | 0.002346       | 3.088                | 0.126233|
| vector        | hnsw      | float32          | 100              | 50         | 16 | 0.001478       | 1.896                | 0.116203|

---

## Running a Search study

Use a search study when you have an **existing Redis index** and want to quickly test different search methods against it without recreating the index or data. This is ideal for A/B testing search strategies or evaluating custom search methods on production data.

### Search study config
```yaml
embedding_model:
  dim: 768
  dtype: float32
  embedding_cache_name: vec-cache
  model: sentence-transformers/all-mpnet-base-v2
  type: hf
index_name: cars
queries: "../resources/cars/car_queries.json"
qrels: "../resources/cars/car_qrels.json"
ret_k: 6
search_methods:
- base_vector
- pre_filter_vector
study_id: test-search-study
```

### Code
```python
import os
from redis_retrieval_optimizer.search_study import run_search_study
from dotenv import load_dotenv

# load environment variables containing necessary credentials
load_dotenv()

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Define custom search methods
def gather_vector_results(search_method_input):
    # Your vector search implementation
    pass

def gather_pre_filter_results(search_method_input):
    # Your pre-filtered search implementation
    pass

CUSTOM_SEARCH_METHOD_MAP = {
    "base_vector": gather_vector_results,
    "pre_filter_vector": gather_pre_filter_results
}

metrics = run_search_study(
    config_path="search_study_config.yaml",
    redis_url=redis_url,
    search_method_map=CUSTOM_SEARCH_METHOD_MAP
)
```

### Example output
| search_method     | avg_query_time | recall | precision | ndcg   |
|-------------------|----------------|--------|-----------|---------|
| base_vector       | 0.002605       | 0.9    | 0.23      | 0.717676|
| pre_filter_vector | 0.001177       | 1.0    | 0.25      | 0.914903|

---

## 🔍 Search Methods

Below is a comprehensive table documenting the built-in search methods available in the Retrieval Optimizer:

| Method | Description | Use Case | Key Features |
|--------|-------------|----------|--------------|
| bm25 | Traditional lexical search using BM25 algorithm | Text-based search where keywords and exact matches matter | <ul><li>No embeddings required</li><li>Good for keyword-heavy queries</li><li>Fast for direct text matches</li><li>Handles stopwords filtering</li></ul> |
| vector | Pure vector/semantic search | Finding semantically similar content regardless of keyword overlap | <ul><li>Uses embedding model to convert text to vectors</li><li>Captures semantic meaning beyond keywords</li><li>Distance scoring (cosine, dot product, etc.)</li></ul> |
| hybrid | Combined lexical and semantic search | Balancing keyword precision with semantic understanding | <ul><li>Combines BM25 and vector search</li><li>Better recall than either method alone</li><li>Handles both exact matches and semantic similarity</li></ul> |
| rerank | Two-stage retrieval with cross-encoder reranking | When high precision is crucial and latency is less important | <ul><li>First-stage retrieval with BM25/vector</li><li>Second-stage reranking with cross-encoder</li><li>Uses HuggingFace cross-encoder model</li><li>Higher quality but increased latency</li></ul> |
| weighted_rrf | Reciprocal Rank Fusion with weights | Combining multiple search strategies with controlled blending | <ul><li>Fuses BM25 and vector search results</li><li>Configurable weighting between methods</li><li>Handles cases where methods have complementary strengths</li><li>Parameter k controls how quickly rankings decay</li></ul> |

### Implementation details

- All search methods follow a common interface taking a SearchMethodInput and returning a SearchMethodOutput
- Query times are automatically tracked in the query_metrics object
- Each method handles error cases gracefully, returning empty results rather than failing
- Results are returned as a `ranx.Run` object for consistent evaluation

### Extending with custom methods

You can create custom search methods by implementing a function that:

1. Takes a SearchMethodInput object
2. Returns a SearchMethodOutput object with results and timing metrics

Then register your method in a custom search method map:

```python
CUSTOM_SEARCH_METHOD_MAP = {
    "bm25": gather_bm25_results,
    "vector": gather_vector_results,
    "my_custom_method": gather_my_custom_results
}
```

## 🎛️ Threshold Optimization

The Retrieval Optimizer includes threshold optimization capabilities for RedisVL's **Semantic Cache** and **Semantic Router**. This feature helps you automatically tune distance thresholds to maximize performance metrics like F1 score, precision, or recall.

### Cache Threshold Optimization

Optimize thresholds for semantic caches to improve cache hit rates and relevance:

```python
from redis_retrieval_optimizer.threshold_optimization import CacheThresholdOptimizer
from redisvl.extensions.cache.llm import SemanticCache

# Create a semantic cache
cache = SemanticCache(
    name="my_cache",
    redis_url="redis://localhost:6379",
    distance_threshold=0.5  # Initial threshold
)

# Add some data to the cache
paris_key = cache.store(
    prompt="what is the capital of france?",
    response="paris"
)

# Define test data for optimization
test_data = [
    {
        "query": "What's the capital of France??",
        "query_match": paris_key  # Expected cache hit
    },
    {
        "query": "What's the capital of Britain?",
        "query_match": ""  # Expected cache miss
    }
]

# Optimize the threshold
optimizer = CacheThresholdOptimizer(cache, test_data)
optimizer.optimize()

print(f"Optimized threshold: {cache.distance_threshold}")
```

### Router Threshold Optimization

Optimize thresholds for semantic routers to improve routing accuracy:

```python
from redisvl.extensions.router import Route, SemanticRouter
from redisvl.utils.vectorize import HFTextVectorizer
from redis_retrieval_optimizer.threshold_optimization import RouterThresholdOptimizer

# Define routes
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

# Create router
router = SemanticRouter(
    name="my-router",
    vectorizer=HFTextVectorizer(),
    routes=routes,
    redis_url="redis://localhost:6379"
)

# Define test data
test_data = [
    {"query": "hello there", "query_match": "greeting"},
    {"query": "goodbye", "query_match": "farewell"},
    {"query": "hola", "query_match": "greeting"},  # Spanish
]

# Optimize route thresholds
optimizer = RouterThresholdOptimizer(router, test_data)
optimizer.optimize(max_iterations=20, search_step=0.1)

print(f"Optimized thresholds: {router.route_thresholds}")
```

### Evaluation Metrics

Threshold optimization supports multiple evaluation metrics:

```python
from redis_retrieval_optimizer.threshold_optimization import EvalMetric

# Available metrics
optimizer = CacheThresholdOptimizer(cache, test_data, eval_metric="f1")       # F1 score (default)
optimizer = CacheThresholdOptimizer(cache, test_data, eval_metric="precision") # Precision
optimizer = CacheThresholdOptimizer(cache, test_data, eval_metric="recall")    # Recall
```

For complete an example, see [docs/examples/threshold_optimization/00_threshold_optimization.ipynb](https://github.com/redis-applied-ai/redis-retrieval-optimizer/blob/1be3e2683a301a6bebe298ec8d126b7429701bc9/docs/examples/threshold_optimization/00_threshold_optimization.ipynb).

## 🔧 Custom Processors and Search Methods

The Retrieval Optimizer is designed to be flexible and extensible. You can define your own **corpus processors** and **search methods** to support different data formats and retrieval techniques. This is especially useful when working with domain-specific data or testing out experimental search strategies.

### Why custom functions matter

Every search application is unique. You might store metadata differently, rely on custom vector filtering, or want to experiment with hybrid techniques. The framework makes it easy to plug in your own logic without needing to rewrite core infrastructure.

---

### Example: Custom Config

This example defines a study where we compare two vector-based methods—one using a simple vector query, and another that filters by metadata before vector search.

#### Study config
```yaml
# paths to necessary data files
corpus: "data/car_corpus.json"
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

---

### Writing custom search methods

Search methods can be anything you want as long as the function accepts a `SearchMethodInput` and returns a `SearchMethodOutput`. This allows you to test new retrieval strategies, add filters, or layer on post-processing logic.

#### Code
```python
def gather_vector_results(search_method_input: SearchMethodInput) -> SearchMethodOutput:
    redis_res_vector = {}

    for key, query_info in search_method_input.raw_queries.items():
        query = vector_query(query_info, 10, search_method_input.emb_model)
        res = run_search_w_time(search_method_input.index, query, search_method_input.query_metrics)
        score_dict = make_score_dict_vec(res)
        redis_res_vector[key] = score_dict

    return SearchMethodOutput(run=Run(redis_res_vector), query_metrics=search_method_input.query_metrics)
```

For example, you can also include filters based on metadata fields:

```python
def gather_pre_filter_results(search_method_input: SearchMethodInput) -> SearchMethodOutput:
    redis_res_vector = {}

    for key, query_info in search_method_input.raw_queries.items():
        query = pre_filter_query(query_info, 10, search_method_input.emb_model)
        res = run_search_w_time(search_method_input.index, query, search_method_input.query_metrics)
        score_dict = make_score_dict_vec(res)
        redis_res_vector[key] = score_dict

    return SearchMethodOutput(run=Run(redis_res_vector), query_metrics=search_method_input.query_metrics)
```

---

### Writing a custom corpus processor

Corpus formats can vary significantly. A custom processor transforms your raw data into the shape required for indexing in Redis.

```python
def process_car_corpus(corpus, emb_model):
    texts = [doc["text"] for doc in corpus]
    embeddings = emb_model.embed_many(texts, as_buffer=True)

    return [
        {
            "_id": doc["item_id"],
            "text": doc["text"],
            "make": doc["query_metadata"]["make"],
            "model": doc["query_metadata"]["model"],
            "vector": embeddings[i],
        }
        for i, doc in enumerate(corpus)
    ]
```

---

### Running the custom study

Once you've defined your search methods and processor, pass them into the study runner:

```python
from redis_retrieval_optimizer.grid_study import run_grid_study

CUSTOM_SEARCH_METHOD_MAP = {
    "basic_vector": gather_vector_results,
    "pre_filter_vector": gather_pre_filter_results,
}

metrics = run_grid_study(
    config_path="custom_grid_study_config.yaml",
    redis_url="redis://localhost:6379/0",
    corpus_processor=process_car_corpus,
    search_method_map=CUSTOM_SEARCH_METHOD_MAP,
)
```

### Example output

| search_method     | model                                      | avg_query_time | recall | precision | ndcg   |
|-------------------|---------------------------------------------|----------------|-----------|-----------|----------|
| pre_filter_vector | sentence-transformers/all-MiniLM-L6-v2     | 0.001177       | 1.0       | 0.25      | 0.914903 |
| basic_vector      | sentence-transformers/all-MiniLM-L6-v2     | 0.002605       | 0.9       | 0.23      | 0.717676 |


## 📊 Data Requirements

To run a retrieval study, you need three key datasets: **queries**, **corpus**, and **qrels**. The framework is flexible—data can be in any shape as long as you provide custom processors to interpret it. But if you're just getting started, here's the expected format and some working examples to guide you.

---

### Corpus

This is the full set of documents you'll be searching against. It's what gets indexed into Redis. The default assumption is that each document has a `text` field to search or embed, but you can customize this using a corpus processor.

**General structure**:

```json
{
    "corpus_id": {
        "text": "text to be searched or vectorized",
        "title": "optional associated title"
    }
}
```

**Example**:

```json
{
    "MED-10": {
        "text": "Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence...",
        "title": "Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland"
    }
}
```

> ✅ Tip: If you're indexing from a live Redis instance, you can skip providing a corpus file entirely by using `from_existing: true` in your config.

---

### Queries

These are the search inputs you'll evaluate against the corpus. Each query consist of the query text itself and a unique ID.

**General structure**:

```json
{
    "query_id": "query text"
}
```

**Example**:

```json
{
    "PLAIN-2": "Do Cholesterol Statin Drugs Cause Breast Cancer?",
    "PLAIN-12": "Exploiting Autophagy to Live Longer"
}
```

> 💡 Using custom query metadata? That's fine—just make sure your custom search method knows how to interpret it.

---

### Qrels

Qrels define the relevance of documents to each query. They are required for evaluating retrieval performance using metrics like NDCG, recall, precision, and F1.

**Required structure**:

```json
{
    "query_id": {
        "corpus_id": relevance_score
    }
}
```

**Example**:

```json
{
    "PLAIN-2": {
        "MED-2427": 2,
        "MED-2440": 1,
        "MED-2434": 1,
        "MED-2435": 1,
        "MED-2436": 1
    },
    "PLAIN-12": {
        "MED-2513": 2,
        "MED-5237": 2
    }
}
```

> 🔍 Note: Relevance scores can be binary (`1` or `0`) for classification metrics or ranked (`2`, `1`, etc.) for ranking metrics like NDCG.

## 🤝 Contributing

We love contributors if you have an addition follow this process:
- Fork the repo
- Make contribution
- Add tests for contribution to test folder
- Make a PR
- Get reviewed
- Merged!

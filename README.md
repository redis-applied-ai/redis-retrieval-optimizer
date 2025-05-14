<div align="center">
<div><img src="https://raw.githubusercontent.com/redis/redis-vl-python/main/docs/_static/Redis_Logo_Red_RGB.svg" style="width: 130px"> </div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/github/languages/top/redis-applied-ai/retrieval-optimizer)
![GitHub last commit](https://img.shields.io/github/last-commit/redis-applied-ai/retrieval-optimizer)

</div>

# Retrieval Optimizer

Search and information retrieval is a challenging and often misunderstood problem. With the proliferation of vector search tools on the market, attention has increasingly shifted toward SEO gains and marketing claims—sometimes at the expense of actual retrieval quality.

The **Retrieval Optimizer** from Redis is designed to bring focus back to what matters: delivering relevant, high-quality results. This flexible framework enables you to systematically measure and improve retrieval performance for your specific data and use case. Rather than relying on guesswork or vague intuition, the Retrieval Optimizer helps you establish **baseline metrics** that serve as a foundation for meaningful evaluation and iteration.

Beyond accuracy alone, it also supports evaluating critical tradeoffs between **cost, speed, and latency**, helping you understand how different embedding models, retrieval strategies, and index configurations impact overall system performance. The ultimate goal is to enable **metrics-driven development** for your search application—ensuring that decisions are grounded in data, not assumptions.

# Getting Started

The Retrieval Optimizer supports two *study* types: **Grid** and **Bayesian Optimization**. Each is suited to a different stage of building a high-quality search system.

### Grid

Use a grid study to explore the impact of different **embedding models** and **retrieval strategies**. These are typically the most important factors influencing search performance. This mode is ideal for establishing a performance baseline and identifying which techniques work best for your dataset.

### Bayesian Optimization

Once you've identified a solid starting point, use Bayesian optimization to **fine-tune your index configuration**. It intelligently selects the most promising combinations to try—saving time compared to exhaustive testing. This mode is especially useful for balancing **cost, speed, and latency** as you work toward a production-ready solution.


## Quick start: grid study

Define study config
```yaml
# paths to necessary data files
corpus: "data/nfcorpus_corpus.json" # optional if from_existing
queries: "data/nfcorpus_queries.json"
qrels: "data/nfcorpus_qrels.json"

# vector field names
index_settings:
  name: "optimize"
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

## Quick start: bayesian optimization
Selects the next best configuration to try based on a heuristic. This is good when it would take a very long time to test all possible configurations.

Study config:
```yaml
# path to data files for easy read
corpus: "data/nfcorpus_corpus.json"
queries: "data/nfcorpus_queries.json"
qrels: "data/nfcorpus_qrels.json"

index_settings:
  name: "optimize"
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


search_methods: ["vector", "hybrid"]
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

## Custom Processors and Search Methods

The Retrieval Optimizer is designed to be flexible and extensible. You can define your own **corpus processors** and **search methods** to support different data formats and retrieval techniques. This is especially useful when working with domain-specific data or testing out experimental search strategies.

### Why Custom Functions Matter

Every search application is unique. You might store metadata differently, rely on custom vector filtering, or want to experiment with hybrid techniques. The framework makes it easy to plug in your own logic without needing to rewrite core infrastructure.

---

### Example: Custom Config

This example defines a study where we compare two vector-based methods—one using a simple vector query, and another that filters by metadata before vector search.

```yaml
corpus: "data/car_corpus.json"
queries: "data/car_queries.json"
qrels: "data/car_qrels.json"

index_settings:
  name: "car"
  prefix: "car"
  vector_field_name: "vector"
  text_field_name: "text"
  from_existing: false
  additional_fields:
    - name: "make"
      type: "tag"
    - name: "model"
      type: "tag"
  vector_dim: 384

embedding_models:
  - type: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dim: 384
    embedding_cache_name: "vec-cache"

search_methods: ["basic_vector", "pre_filter_vector"]
```

---

### Writing Custom Search Methods

Search methods can be anything you want—as long as they accept a `SearchMethodInput` and return a `SearchMethodOutput`. This allows you to test new retrieval strategies, add filters, or layer on post-processing logic.

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

### Writing a Custom Corpus Processor

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

### Running the Custom Study

Once you’ve defined your search methods and processor, simply pass them into the study runner:

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


Here's a polished and clearer version of your **Data Requirements** section, matching the tone and structure of the rest of the content. It emphasizes flexibility while giving users a concrete foundation to build from:

---

## Data Requirements

To run a retrieval study, you need three key datasets: **queries**, **corpus**, and **qrels**. The framework is flexible—data can be in any shape as long as you provide custom processors to interpret it. But if you're getting started, here's the expected format and some working examples to guide you.

---

### Corpus

This is the full set of documents you'll be searching against. It’s what gets indexed into Redis. The default assumption is that each document has a `text` field to search or embed, but you can customize this via a corpus processor.

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

These are the search inputs you'll evaluate against the corpus. Each query should have a unique ID and the query text.

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

> 💡 Using custom query metadata? That’s fine—just make sure your custom search method knows how to interpret it.

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

---

Let me know if you'd like to include downloadable examples or a link to a starter dataset like BEIR for people to clone and try out.


# Contributing
We love contributors if you have an addition follow this process:
- Fork the repo
- Make contribution
- Add tests for contribution to test folder
- Make a PR
- Get reviewed
- Merged!

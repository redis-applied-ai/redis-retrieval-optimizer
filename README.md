# Retrieval Optimizer

Use the retrieval optimizer to figure out the best search configuration for your data.

Input: labeled data for example queries
Output:

<example image of metrics>

# Example of results

# Data requirements

## Labeled data

### Queries

### Qrels

## Corpus

### Corpus from json

### Corpus from existing index

# Running a study

You can run 2 types of studies:

- **Grid**: tries all options specified good for studies where the number of configurations you're testing is small or you want to be exact.
- **Bayesian optimization**: selects the next best configuration to try based on a heuristic. This is good when it would take a very long time to test all possible configurations.

## Grid

```python
# default
from redis_retrieval_optimizer import grid_study

grid_study(study_config_file="grid_study_config.yaml", redis_url="your-redis-connection")

# custom

# retriever functions
def custom_ret_1():
    pass

def custom_ret_2():
    pass

custom_search_map = {
    "ret_1": custom_ret_1, # the name "ret_1" would then correspond with the search_method string supplied in study_config
    "ret_2": custom_ret_2
}

# corpus processor function

def custom_corpus_processor():
    pass

grid_study(config_path="grid_study_config.yaml", redis_url="your-redis-connection", search_method_map=custom_search_map, corpus_processor=custom_corpus_processor)
```

## Bayesian optimization

```python
# default
from redis_retrieval_optimizer import bayesian_optimization_study

bayesian_optimization_study(study_config_file="bayesian_optimization_study_config.yaml", redis_url="your-redis-connection")

# custom

# retriever functions
def custom_ret_1():
    pass

def custom_ret_2():
    pass

custom_search_map = {
    "ret_1": custom_ret_1, # the name "ret_1" would then correspond with the search_method string supplied in study_config
    "ret_2": custom_ret_2
}

# corpus processor function

def custom_corpus_processor():
    pass

bayesian_optimization_study(study_config_file="grid_study_config.yaml", redis_url="your-redis-connection", search_method_map=custom_search_map, corpus_processor=custom_corpus_processor)
```

# Repo schematics

```
docs/
    examples/
        beir_dataset.ipynb
        custom_dataset.ipynb
tests/
redis_retrieval_optimizer/
    search_methods/
        base.py
        bm25.py
        lin_combo.py
        rerank.py
        vector.py
        weighted_rrf.py
    corpus_processors/
        base.py
        beir_dataset.py
    grid_study.py
    bayesian_optimization_study.py
    utils.py
    schema.py
```

# Contributing
We love contributors if you have an addition follow this process:
- Fork the repo
- Make contribution
- Add tests for contribution to test folder
- Make a PR
- Get reviewed
- Merged!

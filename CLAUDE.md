# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Redis Retrieval Optimizer is a scientific framework for benchmarking and optimizing information retrieval systems using Redis. It supports systematic evaluation of vector search, hybrid retrieval, BM25, and reranking strategies.

## Development Commands

### Setup
```bash
make install          # Install dependencies with Poetry
make redis-start      # Start Redis Stack container (required for tests)
make redis-stop       # Stop Redis container
```

### Testing & Quality
```bash
make test             # Run full pytest suite
make check            # Run linting + tests
make format           # Black + isort formatting
make check-types      # MyPy type checking
make lint             # Full linting (format + mypy)
```

### Single Test Execution
```bash
poetry run pytest tests/unit/test_cost_fn.py::test_specific_function
poetry run pytest tests/integration/test_grid.py -v
```

## Architecture Overview

### Core Study Types
- **Grid Study** (`redis_retrieval_optimizer/grid_study.py`) - Systematic parameter exploration
- **Bayesian Study** (`redis_retrieval_optimizer/bayes_study.py`) - Optuna-based optimization  
- **Search Study** (`redis_retrieval_optimizer/search_study.py`) - Test methods on existing indices

### Search Methods (`redis_retrieval_optimizer/search_methods/`)
All methods follow `SearchMethodInput` → `SearchMethodOutput` interface:
- **BM25** - Lexical search
- **Vector** - Semantic search
- **Hybrid** - Combined lexical + semantic
- **Rerank** - Two-stage retrieval with cross-encoder
- **Weighted RRF** - Reciprocal Rank Fusion

### Configuration System
- YAML-based study configurations
- Pydantic schema validation (`redis_retrieval_optimizer/schema.py`)
- Configuration examples in `tests/integration/*_data/`

## Key Dependencies

- **RedisVL** (>=0.8.1) - Primary Redis vector library
- **Optuna** (>=4.3.0) - Bayesian optimization
- **BEIR** (>=2.1.0) - IR benchmarking datasets
- **RANX** (>=0.3.20) - Evaluation metrics (NDCG, precision, recall)
- **Redis** (>=5.0) - Direct Redis client
- **Poetry** - Dependency management (not uv)

## Redis Requirements

- **Redis Stack** container with vector search capabilities
- Tests require Redis 7.0+ for full functionality
- Use `make redis-start` to ensure proper Redis version

## Testing Architecture

- **Integration tests** require running Redis instance
- **Unit tests** for isolated functionality
- **Configuration-driven** tests with YAML fixtures
- **pytest-asyncio** for async test support

## Common Development Patterns

### Adding New Search Methods
1. Implement in `redis_retrieval_optimizer/search_methods/`
2. Follow `SearchMethodInput` → `SearchMethodOutput` interface
3. Add to method registry in appropriate study type
4. Create unit tests and integration tests

### Study Configuration
```yaml
embedding_models:
  - type: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2" 
    dim: 384
    embedding_cache_name: "vec-cache"

search_methods: ["bm25", "vector", "hybrid"]
vector_data_types: ["float16", "float32"]
```

### Data Requirements
- **Corpus** - Documents to index (JSON format)
- **Queries** - Search queries (JSON format)
- **Qrels** - Relevance judgments (JSON format)

## Performance Considerations

- Use **RedisVL** for high-level operations
- Use **redis-py** directly for custom low-level operations
- Vector data types: float16 vs float32 trade-offs
- Batch operations for large datasets

## Troubleshooting

### Redis Connection Issues
- Ensure Redis Stack is running: `make redis-start`
- Check Redis version compatibility (7.0+ recommended)
- Verify RedisVL compatibility with Redis version

### Test Failures
- Run `make redis-start` before testing
- Check for stale Redis indices from previous tests
- Use `make clean` to clear build artifacts
# Redis Retrieval Optimizer v0.3.0 Release Notes

## 🎉 What's New in v0.3.0

Redis Retrieval Optimizer v0.3.0 introduces several major features that make it easier than ever to build and optimize high-performance search systems with Redis. This release focuses on enhanced optimization capabilities, improved search methods, and better integration with modern embedding models.

## 🚀 Major New Features

### 🎛️ Threshold Optimization
**NEW**: Automatically tune semantic cache and router thresholds for optimal performance.

The new threshold optimization feature helps you maximize the performance of RedisVL's Semantic Cache and Semantic Router by automatically finding the best distance thresholds. This feature supports multiple evaluation metrics including F1 score, precision, and recall.

**Key capabilities:**
- **Cache Threshold Optimization**: Optimize thresholds for semantic caches to improve cache hit rates and relevance
- **Router Threshold Optimization**: Fine-tune route thresholds for semantic routers to improve routing accuracy
- **Multiple Evaluation Metrics**: Support for F1 score, precision, and recall optimization
- **Easy Integration**: Works seamlessly with existing RedisVL SemanticCache and SemanticRouter instances

**Example usage:**
```python
from redis_retrieval_optimizer.threshold_optimization import CacheThresholdOptimizer

# Optimize cache threshold
optimizer = CacheThresholdOptimizer(cache, test_data)
optimizer.optimize()

# Optimize router thresholds
optimizer = RouterThresholdOptimizer(router, test_data)
optimizer.optimize(max_iterations=20, search_step=0.1)
```

### 🔄 Weighted Reciprocal Rank Fusion (Weighted RRF)
**NEW**: Advanced search method that combines multiple retrieval strategies with configurable weighting.

Weighted RRF allows you to intelligently blend BM25 and vector search results with controlled weighting parameters. This method is particularly effective when different search strategies have complementary strengths.

**Features:**
- Configurable weighting between BM25 and vector search
- Parameter k controls how quickly rankings decay
- Handles cases where methods have complementary strengths
- Improved relevance through intelligent result fusion

### 🧠 Enhanced Vector Data Type Support
**NEW**: Support for multiple vector data types including float16 and float32.

You can now test different vector data types to find the optimal balance between memory usage and precision. This is especially useful for production deployments where memory efficiency is crucial.

**Supported data types:**
- `float16`: Reduced memory usage with acceptable precision
- `float32`: Standard precision (default)

**Configuration:**
```yaml
vector_data_types: ["float16", "float32"]
```

### 🤖 OpenAI Embedding Model Support
**NEW**: Native support for OpenAI's text-embedding-3-small model.

The optimizer now supports OpenAI's latest embedding models, allowing you to compare their performance against HuggingFace models in your studies.

**Supported models:**
- `text-embedding-3-small` (1536 dimensions)
- All existing HuggingFace models

**Example configuration:**
```yaml
embedding_models:
  - type: "openai"
    model: "text-embedding-3-small"
    dim: 1536
    embedding_cache_name: "openai-small-vec-cache"
```

## 🔧 Improvements & Enhancements

### 📊 Enhanced Search Methods
- **Improved BM25**: Better handling of edge cases and error recovery
- **Enhanced Hybrid Search**: More robust combination of lexical and semantic search
- **Optimized Reranking**: Improved cross-encoder integration with better error handling
- **Better Vector Search**: Enhanced distance metric support and query optimization

### 🛠️ Developer Experience
- **Better Error Handling**: More graceful error recovery across all search methods
- **Improved Logging**: Enhanced logging for debugging and monitoring
- **Type Safety**: Better type hints and validation throughout the codebase
- **Documentation**: Comprehensive examples and API documentation

### 🔌 Extensibility
- **Custom Search Methods**: Easier creation of domain-specific search strategies
- **Flexible Corpus Processors**: Support for custom data formats and processing
- **Modular Architecture**: Better separation of concerns for easier extension

## 📚 New Documentation & Examples

### 📖 Comprehensive Examples
- **Threshold Optimization**: Complete notebook showing cache and router optimization
- **Model Comparison**: Side-by-side comparison of different embedding models
- **Custom Grid Study**: Advanced example with domain-specific search methods
- **Bayesian Optimization**: Detailed guide for fine-tuning index configurations

### 🔍 API Documentation
- Complete API reference for all new features
- Detailed configuration guides
- Best practices and performance tips

## 🐛 Bug Fixes

- Fixed issue with embedding cache name collisions
- Improved handling of empty search results
- Better error messages for configuration issues
- Fixed memory leaks in long-running studies
- Resolved issues with Redis connection handling

## 🔄 Breaking Changes

**None**: This release maintains full backward compatibility with v0.2.x.

## 📦 Installation

```bash
pip install redis-retrieval-optimizer==0.3.0
```

## 🎯 Migration Guide

No migration required! All existing configurations and code will work without changes. New features are opt-in and can be added to your existing studies.

## 🚀 Quick Start with New Features

### Threshold Optimization
```python
from redis_retrieval_optimizer.threshold_optimization import CacheThresholdOptimizer

# Create test data
test_data = [
    {"query": "What's the capital of France?", "query_match": "paris_key"},
    {"query": "What's the capital of Britain?", "query_match": ""}
]

# Optimize cache threshold
optimizer = CacheThresholdOptimizer(cache, test_data)
optimizer.optimize()
```

### Weighted RRF
```yaml
search_methods: ["bm25", "vector", "hybrid", "rerank", "weighted_rrf"]
```

### Vector Data Types
```yaml
vector_data_types: ["float16", "float32"]
```

### OpenAI Embeddings
```yaml
embedding_models:
  - type: "openai"
    model: "text-embedding-3-small"
    dim: 1536
    embedding_cache_name: "openai-cache"
```

## 🙏 Acknowledgments

Thank you to all contributors who helped make this release possible! Special thanks to the Redis community for feedback and testing.

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/redis-applied-ai/redis-retrieval-optimizer)
- **Issues**: [GitHub Issues](https://github.com/redis-applied-ai/redis-retrieval-optimizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/redis-applied-ai/redis-retrieval-optimizer/discussions)

---

**Stop guessing. Start measuring.** 📊

Transform your retrieval system from *"looks good to me"* to *"proven to perform"* with Redis Retrieval Optimizer v0.3.0!
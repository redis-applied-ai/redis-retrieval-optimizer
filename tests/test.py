from redis_retrieval_optimizer.bayes_study import run_bayes_study
from redis_retrieval_optimizer.corpus_processors import eval_beir

if __name__ == "__main__":
    # Example usage
    config_path = "bayes_study_config.yaml"
    redis_url = "redis://localhost:6379"

    run_bayes_study(config_path, redis_url, eval_beir.process_corpus)

"""Unit coverage for PR 4 small isolated fixes (no Redis required).

- #8  package exports __all__ (was the shadowed builtin `all`)
- #10 process_corpus tolerates title-less corpora and uses a real type hint
"""

import redis_retrieval_optimizer
from redis_retrieval_optimizer.corpus_processors import eval_beir


class _StubEmbedder:
    """Minimal stand-in so we exercise process_corpus's real text-building logic
    without downloading an embedding model (the logic under test is the title
    handling, not the embedding)."""

    def embed_many(self, texts, as_buffer=False):
        self.seen = list(texts)
        return [b"vec" for _ in texts]


def test_package_exports_all():
    # #8 — must be the dunder, not the shadowed builtin `all`
    assert redis_retrieval_optimizer.__all__ == ["__version__"]


def test_process_corpus_handles_missing_title():
    # #10 — no "title" key should not raise KeyError
    corpus = {"d1": {"text": "hello world"}}

    result = eval_beir.process_corpus(corpus, _StubEmbedder())

    assert len(result) == 1
    assert result[0]["_id"] == "d1"
    assert result[0]["title"] == ""
    assert result[0]["text"] == "hello world"  # leading space stripped


def test_process_corpus_with_title_embeds_title_and_text():
    corpus = {"d1": {"title": "Cats", "text": "are great"}}
    embedder = _StubEmbedder()

    result = eval_beir.process_corpus(corpus, embedder)

    assert embedder.seen == ["Cats are great"]
    assert result[0]["title"] == "Cats"
    assert result[0]["text"] == "Cats are great"

# load beir dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from redisvl.utils.vectorize import BaseVectorizer


def get_beir_dataset(dataset="fiqa"):
    # link for dataset: https://sites.google.com/view/fiqa/
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

    out_dir = "./beir_datasets"
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    return corpus, queries, qrels


def process_corpus(corpus: any, emb_model: BaseVectorizer):
    corpus_data = []
    corpus_texts = []

    # this can take a minute
    for key in corpus:
        corpus_texts.append(
            corpus[key]["title"] + " " + corpus[key]["text"]
        )  # note: embedded both text and title

    text_embeddings = emb_model.embed_many(corpus_texts, as_buffer=True)

    for key, text, embedding in zip(corpus, corpus_texts, text_embeddings):
        corpus_data.append(
            {
                "_id": key,
                "text": text,
                "title": corpus[key]["title"],
                "vector": embedding,
            }
        )

    return corpus_data

from redisvl.query import VectorQuery


def vector_query(query: str, num_results: int, emb_model) -> VectorQuery:
    vector = emb_model.embed(query, as_buffer=True)

    return VectorQuery(
        vector=vector,
        vector_field_name="vector",
        num_results=num_results,
        return_fields=["_id", "text", "title"],
    )


def gather_vector_results(queries, index, emb_model):
    redis_res_vector = {}

    def make_score_dict_vec(res):
        return {rec["_id"]: (2 - float(rec["vector_distance"]) / 2) for rec in res}

    # TODO: should be batch query for the love of speed
    for key in queries:
        text_query = queries[key]
        vec_query = vector_query(text_query, 10, emb_model)
        # try:
        res = index.query(vec_query)
        score_dict = make_score_dict_vec(res)
        # except Exception as e:
        #     print(f"failed for {key}, {text_query}")
        #     score_dict = {}
        redis_res_vector[key] = score_dict
    return redis_res_vector

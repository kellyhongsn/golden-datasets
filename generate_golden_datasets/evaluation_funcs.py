import pytrec_eval
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Any

def get_metrics(qrels: dict, results: dict, k_values: list[int]) -> dict:
    recall = dict()
    precision = dict()
    _map = dict()
    ndcg = dict()

    for k in k_values:
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        ndcg[f"NDCG@{k}"] = 0.0

    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)

    return ndcg, _map, recall, precision


def evaluate(k_values: list[int], qrels_df: pd.DataFrame, results_dict: dict) -> dict:
    qrels = qrels_df.groupby("query-id").apply(lambda g: dict(zip(g["corpus-id"], g["score"]))).to_dict()
    
    qrels = {
        qid: {doc_id: int(score) for doc_id, score in doc_dict.items()}
        for qid, doc_dict in qrels.items()
    }

    results = {}
    for query_id, query_data in results_dict.items():
        results[query_id] = {}
        for doc_id, score in zip(query_data['retrieved_corpus_ids'], query_data['all_scores']):
            results[query_id][doc_id] = score

    ndcg, _map, recall, precision = get_metrics(
        qrels=qrels, 
        results=results, 
        k_values=k_values
    )

    final_result = {
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision
    }
    
    return final_result


def get_results(collection: Any, queries_text: list[str], queries_ids: list[str], queries_embeddings: list[np.ndarray], batch_size: int = 100) -> dict:
    results = dict()

    for i in tqdm(range(0, len(queries_embeddings), batch_size), desc="Processing batches"):
        batch_text = queries_text[i:i + batch_size]
        batch_ids = queries_ids[i:i + batch_size]
        batch_embeddings = queries_embeddings[i:i + batch_size]

        query_results = collection.query(
            query_embeddings=batch_embeddings,
            query_texts=batch_text,
            n_results=10
        )

        for idx, (query_id, query_embedding) in enumerate(zip(batch_ids, batch_embeddings)):

            results[query_id] = {
                "query_embedding": query_embedding,
                "retrieved_corpus_ids": query_results["ids"][idx],
                "retrieved_corpus_text": query_results["documents"][idx],
                "all_scores": [1 - d for d in query_results["distances"][idx]]
            }

    return results
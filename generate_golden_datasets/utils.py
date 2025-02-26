import pandas as pd
import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def combined_datasets_dataframes(queries: pd.DataFrame, corpus: pd.DataFrame, qrels: pd.DataFrame) -> pd.DataFrame:
    qrels = qrels.merge(queries, left_on="query-id", right_on="_id", how="left")
    qrels.rename(columns={"text": "query-text"}, inplace=True)
    qrels.drop(columns=["_id"], inplace=True)
    qrels = qrels.merge(corpus, left_on="corpus-id", right_on="_id", how="left")
    qrels.rename(columns={"text": "corpus-text"}, inplace=True)
    qrels.drop(columns=["_id", "title"], inplace=True)

    return qrels


def create_comparison_dataframe(metrics_sets: list[dict], labels: list[str]) -> pd.DataFrame:
    rows = []
    
    for label, metrics in zip(labels, metrics_sets):
        row_data = {"Query Type": label}
        
        for metric_name, metric_values in metrics.items():
            for k_metric, value in metric_values.items():
                row_data[f"{metric_name}@{k_metric.split('@')[1]}"] = value
        
        rows.append(row_data)
        
    return pd.DataFrame(rows)


def score_query_query(qrels: pd.DataFrame, query_embeddings_dict_1: dict, query_embeddings_dict_2: dict, output_path: str = None) -> pd.DataFrame:
    similarity_scores = []

    for _, row in qrels.iterrows():
        query_id = row['query-id']
        
        col1_embedding = query_embeddings_dict_1[query_id]
        col2_embedding = query_embeddings_dict_2[query_id]
        
        similarity = cosine_similarity(col1_embedding, col2_embedding).item()
        similarity_scores.append(similarity)

    scores_df = qrels.copy()
    scores_df['query-query-score'] = similarity_scores

    if output_path:
        scores_df.to_parquet(output_path)
        
    return scores_df


def score_query_corpus(qrels: pd.DataFrame, query_embeddings_dict: dict, corpus_embeddings_dict: dict, output_path: str = None) -> pd.DataFrame:
    similarity_scores = []

    for _, row in qrels.iterrows():
        query_id = row['query-id']
        corpus_id = row['corpus-id']
        
        query_embedding = query_embeddings_dict[query_id]
        corpus_embedding = corpus_embeddings_dict[corpus_id]
        
        similarity = cosine_similarity(query_embedding, corpus_embedding).item()
        similarity_scores.append(similarity)

    scores_df = qrels.copy()
    scores_df['query-corpus-score'] = similarity_scores

    if output_path:
        scores_df.to_parquet(output_path)
        
    return scores_df
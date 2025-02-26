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
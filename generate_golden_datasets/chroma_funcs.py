from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

def collection_add_in_batches(collection: Any, ids: List[str], texts: List[str], embeddings: List[List[float]]) -> None:
    BATCH_SIZE = 200
    LEN = len(embeddings)
    N_THREADS = 20

    def add_batch(start: int, end: int) -> None:
        id_batch = ids[start:end]
        doc_batch = texts[start:end]

        print(f"Adding {start} to {end} documents...")
        try:
            collection.add(ids=id_batch, documents=doc_batch, embeddings=embeddings[start:end])
        except Exception as e:
            print(f"Error adding {start} to {end}")
            print(e)
        print(f"Added {start} to {end} documents!")

    threadpool = ThreadPoolExecutor(max_workers=N_THREADS)

    for i in range(0, LEN, BATCH_SIZE):
        threadpool.submit(add_batch, i, min(i + BATCH_SIZE, LEN))

    threadpool.shutdown(wait=True)
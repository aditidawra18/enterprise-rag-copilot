
from sentence_transformers import CrossEncoder

from src.hybrid_search import hybrid_search, initialize_hybrid_search


reranker = None


def initialize_reranker():

    global reranker

    if reranker is None:

        initialize_hybrid_search()

        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("CrossEncoder initialized successfully.")


def reranked_search(query, top_k=5):

    initialize_reranker()

    # Retrieve candidates using hybrid search
    candidates = hybrid_search(query, top_k=top_k)

    if not candidates:
        return []

    # Prepare query-document pairs
    pairs = [
        [query, chunk["text"]]
        for chunk in candidates
    ]

    # CrossEncoder reranking
    rerank_scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(candidates, rerank_scores),
        key=lambda item: item[1],
        reverse=True
    )

    # Attach reranking scores
    retrieved_chunks = []

    for chunk, rerank_score in reranked_results:

        retrieved_chunks.append({
            **chunk,
            "rerank_score": float(rerank_score)
        })

    return retrieved_chunks


if __name__ == "__main__":

    query = input("Ask your question: ")

    results = reranked_search(query)

    print("\nRERANKED SEARCH RESULTS\n")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("Chunk ID:", result["chunk_id"])
        print("Rerank Score:", result["rerank_score"])
        print("RRF Score:", result["rrf_score"])
        print("Source:", result["source"])
        print("Pages:", result["pages"])
        print("Text:", result["text"])
        print("-" * 60)

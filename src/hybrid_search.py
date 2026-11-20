
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunking import create_chunks
from src.vector_store import DB_PATH, COLLECTION_NAME


# Initialize resources only when first required
all_chunks = None
chunk_id_to_index = None
bm25 = None
embedding_model = None
collection = None


def initialize_hybrid_search():

    global all_chunks
    global chunk_id_to_index
    global bm25
    global embedding_model
    global collection

    if collection is not None:
        return

    # Load document chunks
    all_chunks = create_chunks()

    chunk_id_to_index = {
        chunk["chunk_id"]: index
        for index, chunk in enumerate(all_chunks)
    }

    # Initialize BM25
    tokenized_corpus = [
        chunk["text"].lower().split()
        for chunk in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() != len(all_chunks):
        raise ValueError(
            "Chunk count mismatch. Rebuild ChromaDB."
        )

    # Load embedding model
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Hybrid search initialized successfully.")


def hybrid_search(query, top_k=5):

    initialize_hybrid_search()

    # BM25 retrieval
    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

    # Vector retrieval
    query_embedding = embedding_model.encode(query)

    vector_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Reciprocal Rank Fusion
    rrf_scores = {}
    k = 60

    for rank, index in enumerate(bm25_indices, start=1):
        rrf_scores[index] = (
            rrf_scores.get(index, 0) + 1 / (k + rank)
        )

    for rank, chunk_id in enumerate(
        vector_results["ids"][0], start=1
    ):
        index = chunk_id_to_index.get(chunk_id)

        if index is None:
            continue

        rrf_scores[index] = (
            rrf_scores.get(index, 0) + 1 / (k + rank)
        )

    # Sort by combined RRF score
    final_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:top_k]

    # Format results
    retrieved_chunks = []

    for index, score in final_results:

        chunk = all_chunks[index]

        retrieved_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "pages": chunk["pages"],
            "rrf_score": score
        })

    return retrieved_chunks


if __name__ == "__main__":

    query = input("Ask your question: ")

    results = hybrid_search(query)

    print("\nHYBRID SEARCH RESULTS\n")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("Chunk ID:", result["chunk_id"])
        print("Text:", result["text"])
        print("Source:", result["source"])
        print("Pages:", result["pages"])
        print("RRF Score:", result["rrf_score"])
        print("-" * 60)

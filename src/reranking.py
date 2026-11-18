
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.chunking import create_chunks


# 1. Load chunks from the centralized chunking module
all_chunks = create_chunks()

# Map chunk IDs to their positions
chunk_id_to_index = {
    chunk["chunk_id"]: index
    for index, chunk in enumerate(all_chunks)
}


# 2. Create BM25 index
tokenized_corpus = [
    chunk["text"].lower().split()
    for chunk in all_chunks
]

bm25 = BM25Okapi(tokenized_corpus)


# 3. Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="nexora_documents"
)

if collection.count() != len(all_chunks):
    raise ValueError(
        "Chunk count mismatch. Rebuild ChromaDB."
    )


# 4. Load embedding and reranking models
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# 5. Define reusable reranking function
def reranked_search(query, top_k=5):

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

    # Add BM25 rankings
    for rank, index in enumerate(bm25_indices, start=1):

        rrf_scores[index] = (
            rrf_scores.get(index, 0) + 1 / (k + rank)
        )


    # Add semantic/vector rankings
    for rank, chunk_id in enumerate(
        vector_results["ids"][0], start=1
    ):

        index = chunk_id_to_index.get(chunk_id)

        if index is None:
            continue

        rrf_scores[index] = (
            rrf_scores.get(index, 0) + 1 / (k + rank)
        )


    # Select top candidates using RRF
    candidates = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:top_k]


    # Prepare query-document pairs for CrossEncoder
    pairs = [
        [query, all_chunks[index]["text"]]
        for index, score in candidates
    ]


    # CrossEncoder reranking
    rerank_scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(candidates, rerank_scores),
        key=lambda item: item[1],
        reverse=True
    )


    # Format results
    retrieved_chunks = []

    for (index, rrf_score), rerank_score in reranked_results:

        chunk = all_chunks[index]

        retrieved_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "pages": chunk["pages"],
            "rrf_score": rrf_score,
            "rerank_score": float(rerank_score)
        })

    return retrieved_chunks


# 6. Allow manual testing from terminal
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

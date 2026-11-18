
import chromadb
from sentence_transformers import SentenceTransformer

# Load models and ChromaDB collection
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="nexora_documents")


def semantic_search(query, top_k=5):

    # Convert query into embedding
    query_embedding = model.encode(query)

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Format results
    retrieved_chunks = []

    for i in range(len(results["documents"][0])):
        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


# Allow manual testing from terminal
if __name__ == "__main__":

    query = input("Ask your question: ")

    results = semantic_search(query)

    print("\nSEMANTIC SEARCH RESULTS\n")

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Text:", result["text"])
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Distance:", result["distance"])
        print("-" * 60)

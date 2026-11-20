
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.chunking import create_chunks


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "nexora_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def initialize_vector_store():

    # Load document chunks
    all_chunks = create_chunks()

    if not all_chunks:
        raise ValueError("No document chunks found.")

    # Connect to persistent ChromaDB
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Check whether collection already exists
    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        # Reuse database if chunk count matches
        if collection.count() == len(all_chunks):
            print("Existing ChromaDB collection loaded.")
            return collection

        # Rebuild if chunk count has changed
        print("Chunk count mismatch. Rebuilding collection...")

        client.delete_collection(
            name=COLLECTION_NAME
        )

    # Generate embeddings only when building the database
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = model.encode(texts)

    # Create collection
    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    # Store chunks and metadata
    collection.add(
        ids=[
            chunk["chunk_id"]
            for chunk in all_chunks
        ],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "pages": ", ".join(
                    map(str, chunk["pages"])
                )
            }
            for chunk in all_chunks
        ]
    )

    print("ChromaDB collection created successfully.")
    print("Total records:", collection.count())

    return collection


if __name__ == "__main__":

    collection = initialize_vector_store()

    print("\nVECTOR DATABASE STATUS")
    print("Total records:", collection.count())


import chromadb
from sentence_transformers import SentenceTransformer
from src.chunking import create_chunks


# STEP 1: Generate chunks using the updated chunking module
all_chunks = create_chunks()

print("Total chunks generated:", len(all_chunks))


# STEP 2: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 3: Generate embeddings
texts = [chunk["text"] for chunk in all_chunks]

embeddings = model.encode(texts)


# STEP 4: Connect to persistent ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)


# STEP 5: Delete the old collection to remove outdated chunks
collection_name = "nexora_documents"

existing_collections = client.list_collections()

existing_names = [
    collection.name if hasattr(collection, "name") else collection
    for collection in existing_collections
]

if collection_name in existing_names:
    client.delete_collection(name=collection_name)


# STEP 6: Create a fresh collection
collection = client.create_collection(
    name=collection_name
)


# STEP 7: Store chunks, embeddings and metadata
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
            "pages": ", ".join(map(str, chunk["pages"]))
        }
        for chunk in all_chunks
    ]
)


# STEP 8: Verify stored records
print("\n====================================")
print("VECTOR DATABASE REBUILT")
print("====================================")

print("Total chunks generated:", len(all_chunks))
print("Total records in ChromaDB:", collection.count())


# Display first stored record
result = collection.get(
    ids=[all_chunks[0]["chunk_id"]],
    include=["documents", "metadatas"]
)

print("\nFirst stored record:")
print("Chunk ID:", result["ids"][0])
print("Text:", result["documents"][0])
print("Metadata:", result["metadatas"][0])

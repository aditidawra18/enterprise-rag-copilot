
from sentence_transformers import SentenceTransformer
from src.document_loader import load_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Step 1: Load documents
documents = load_pdfs()


# Step 2: Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

all_chunks = []

for doc in documents:

    chunks = splitter.split_text(doc["text"])

    for chunk in chunks:

        all_chunks.append({
            "text": chunk,
            "source": doc["source"],
            "page": doc["page"]
        })


# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Step 4: Extract text from chunks
texts = [chunk["text"] for chunk in all_chunks]


# Step 5: Generate embeddings
embeddings = model.encode(texts)


# Step 6: Check results
print("Total chunks:", len(all_chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst chunk:")
print(texts[0])

print("\nFirst embedding:")
print(embeddings[0])

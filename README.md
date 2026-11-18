# Enterprise Knowledge & IT Support Copilot

An AI-powered enterprise knowledge assistant built using Retrieval-Augmented Generation (RAG). The application allows employees to ask questions about company policies, IT support procedures, employee onboarding, and technical operations.

The system retrieves relevant information from internal PDF documents and uses a locally hosted Large Language Model (LLM) to generate context-grounded answers with source references.

## 1. Project Overview

Organizations maintain large amounts of information in employee handbooks, IT policies, operational runbooks, and troubleshooting guides.

Finding specific information manually can be time-consuming.

This project addresses that problem by building an intelligent knowledge assistant that:

- Processes and indexes PDF documents.
- Retrieves relevant information using semantic and keyword search.
- Uses hybrid retrieval and CrossEncoder reranking.
- Generates answers using a locally hosted LLM.
- Provides source document and page references.
- Supports conversational follow-up questions.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend API | FastAPI |
| Frontend | Streamlit |
| LLM | Qwen 2.5 (1.5B) |
| Local LLM Runtime | Ollama |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion (RRF) |
| Reranking | CrossEncoder |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain Text Splitters |

## 3. System Architecture

PDF Documents → Document Extraction → Text Chunking → Embedding Generation → ChromaDB

User Question → FastAPI → Hybrid Retrieval → CrossEncoder Reranking → Retrieved Context + Conversation History → Local LLM → Generated Answer → Streamlit Frontend

## 4. Key Features

### Document Processing
- Extracts text from PDF documents.
- Splits documents into manageable chunks.
- Maintains document and page metadata.

### Semantic Search
- Generates vector embeddings using Sentence Transformers.
- Retrieves semantically relevant document chunks from ChromaDB.

### Hybrid Search
- Combines BM25 keyword retrieval with vector-based semantic retrieval.
- Uses Reciprocal Rank Fusion to combine rankings.

### CrossEncoder Reranking
- Re-evaluates retrieved query-document pairs.
- Ranks the most relevant chunks before sending them to the LLM.

### RAG-based Answer Generation
- Uses retrieved document context to generate answers.
- Runs locally through Ollama.
- Instructs the model to avoid unsupported information.

### Conversational Memory
- Maintains conversation history during the Streamlit session.
- Passes previous messages to the LLM to help interpret follow-up questions.

### API and User Interface
- FastAPI exposes the RAG pipeline through REST endpoints.
- Streamlit provides an interactive chat interface.
- Displays source document and page references.

## 5. Retrieval Evaluation

The retrieval system was evaluated using 48 answerable questions.

| Metric | Semantic Search | Hybrid Search | Reranked Search |
|---|---:|---:|---:|
| Hit@1 | 87.50% | 89.58% | 93.75% |
| Hit@3 | 100% | 97.92% | 100% |
| Hit@5 | 100% | 100% | 100% |
| MRR | 0.9340 | 0.9392 | 0.9688 |

**Key Observation:** CrossEncoder reranking achieved the highest Hit@1 and MRR among the three retrieval approaches.

Note: These metrics measure retrieval at the source-document level, not exact passage-level evidence retrieval or answer correctness.

## 6. Project Structure

```text
enterprise-rag-copilot/
│
├── data/
│   └── raw_documents/
│
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   ├── semantic_search.py
│   ├── hybrid_search.py
│   ├── reranking.py
│   ├── evaluate_retrieval.py
│   ├── llm_generator.py
│   ├── rag_pipeline.py
│   ├── evaluate_generation.py
│   └── api.py
│
├── chroma_db/
├── evaluation/
│   ├── evaluation_questions.csv
│   ├── retrieval_results.csv
│   └── generation_results.csv
│
├── frontend.py
├── requirements.txt
└── README.md
```

## 7. Installation and Setup

### Prerequisites
- Python 3.9+
- Ollama installed
- Git

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd enterprise-rag-copilot
```

### Step 2: Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the local LLM

```bash
ollama pull qwen2.5:1.5b
```

Make sure Ollama is running.

### Step 5: Start the FastAPI backend

```bash
uvicorn src.api:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### Step 6: Start the Streamlit frontend

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run frontend.py
```

Open:

http://localhost:8501

## 8. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/chat` | Submit a question and receive an answer |

## 9. Dataset

The project uses eight fictional enterprise PDF documents covering:

- Employee handbook
- IT access and security policies
- Laptop asset management
- Employee onboarding
- VPN troubleshooting
- Incident management
- PostgreSQL operations
- Ubuntu server operations

All documents contain synthetic demonstration information and do not represent actual company policies.

## 10. Future Improvements

- Improve answer quality and evidence validation.
- Add user authentication and session management.
- Add support for additional document formats.
- Improve conversational retrieval for complex follow-up questions.
- Add deployment and monitoring capabilities.

---

**Project:** Enterprise Knowledge & IT Support Copilot  
**Category:** Generative AI | Retrieval-Augmented Generation | NLP

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from src.reranking import reranked_search
from src.llm_generator import generate_answer


app = FastAPI(
    title="Enterprise Knowledge & IT Support Copilot",
    description="A RAG-based enterprise knowledge assistant",
    version="1.0.0"
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []


@app.get("/")
def home():
    return {
        "message": "Enterprise RAG Copilot API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question

    # Retrieve relevant document chunks
    retrieved_chunks = reranked_search(
        question,
        top_k=5
    )

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "I couldn't find enough information in the provided documents.",
            "sources": []
        }

    # Generate answer using conversation history
    answer = generate_answer(
        question,
        retrieved_chunks,
        chat_history=[
            message.model_dump() if hasattr(message, "model_dump")
            else message.dict()
            for message in request.chat_history
        ]
    )

    # Extract source information
    sources = [
        {
            "document": chunk["source"],
            "page": chunk["page"],
            "pages": chunk["pages"]
        }
        for chunk in retrieved_chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

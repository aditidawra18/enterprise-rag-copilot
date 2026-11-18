
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"


def validate_evidence(question, context_chunks):

    if not context_chunks:
        return {
            "sufficient": False,
            "raw_output": "No context provided"
        }

    context = ""

    for i, chunk in enumerate(context_chunks, start=1):
        context += (
            f"\nDocument {i}:\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content: {chunk['text']}\n"
        )

    prompt = f"""
You are an evidence validation system for an enterprise
knowledge assistant.

Your task is to determine whether the provided context
contains sufficient explicit evidence to answer the question.

Follow these rules:

1. Evaluate the exact information requested in the question.

2. The context must explicitly support the answer.

3. Related information is not necessarily sufficient evidence.

4. Do not use outside knowledge or assumptions.

5. Do not infer missing names, numbers, dates, identities,
   or other specific details.

6. If the context supports only part of the requested answer,
   classify it as INSUFFICIENT.

7. Classify the evidence as SUFFICIENT only when the context
   directly supports the information required to answer.

8. If uncertain, classify it as INSUFFICIENT.

Return exactly one of these two words:

SUFFICIENT
INSUFFICIENT

Context:
{context}

Question:
{question}

Decision:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    raw_output = response.json()["message"]["content"]

    decision = raw_output.strip().upper().strip(" .\n\t")

    # Fail closed: unexpected output is treated as insufficient.
    sufficient = decision == "SUFFICIENT"

    return {
        "sufficient": sufficient,
        "raw_output": raw_output
    }


if __name__ == "__main__":

    from src.reranking import reranked_search

    question = input("\nEnter your question: ")

    retrieved_chunks = reranked_search(
        question,
        top_k=5
    )

    result = validate_evidence(
        question,
        retrieved_chunks
    )

    print("\n===== EVIDENCE VALIDATION =====")

    print("Question:", question)
    print("Decision:", result["raw_output"])

    if result["sufficient"]:
        print("Evidence is sufficient for answer generation.")
    else:
        print("Evidence is insufficient. The system should abstain.")

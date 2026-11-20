
from src.reranking import reranked_search
from src.llm_generator import generate_answer


def rag_pipeline(question, chat_history=None, provider="ollama"):

    # Step 1: Retrieve and rerank relevant document chunks
    retrieved_chunks = reranked_search(question, top_k=5)

    # Step 2: Check whether any context was retrieved
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find enough information in the provided documents.",
            "sources": []
        }

    # Step 3: Generate answer using the selected LLM
    answer = generate_answer(
        question,
        retrieved_chunks,
        chat_history=chat_history,
        provider=provider
    )

    # Step 4: Extract source information
    sources = [
        {
            "document": chunk["source"],
            "page": chunk["page"],
            "pages": chunk["pages"]
        }
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":

    print("\n===== Enterprise Knowledge Assistant =====")

    while True:

        question = input("\nAsk your question (or type 'exit'): ")

        if question.lower() == "exit":
            print("Exiting assistant...")
            break

        result = rag_pipeline(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for source in result["sources"]:
            print(source)

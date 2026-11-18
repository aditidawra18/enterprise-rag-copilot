
from src.reranking import reranked_search
from src.llm_generator import generate_answer


def rag_pipeline(question):

    # Step 1: Retrieve and rerank relevant document chunks
    retrieved_chunks = reranked_search(question, top_k=5)

    # Step 2: Check whether any context was retrieved
    if not retrieved_chunks:
        return "I couldn't find enough information in the provided documents."

    # Step 3: Generate an answer using the retrieved context
    answer = generate_answer(question, retrieved_chunks)

    return answer


if __name__ == "__main__":

    print("\n===== Enterprise Knowledge Assistant =====")

    while True:

        question = input("\nAsk your question (or type 'exit'): ")

        if question.lower() == "exit":
            print("Exiting assistant...")
            break

        answer = rag_pipeline(question)

        print("\nAnswer:")
        print(answer)

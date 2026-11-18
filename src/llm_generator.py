
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"


def generate_answer(question, context_chunks, chat_history=None):

    # Prepare document context
    context = ""

    for i, chunk in enumerate(context_chunks, start=1):
        context += (
            f"\nDocument {i}:\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content: {chunk['text']}\n"
        )

    # System instructions
    system_prompt = """
You are an enterprise knowledge assistant.

Answer the user's current question using ONLY the information
provided in the retrieved document context.

Conversation history is provided to help you understand
follow-up questions, references and context.

IMPORTANT:
- Use conversation history only to understand what the
  user is referring to.
- Do not treat previous assistant answers as verified facts.
- Base factual answers only on the currently retrieved
  document context.
- If the retrieved context does not contain enough
  information, clearly say that the information is
  unavailable in the provided documents.
- Never invent or assume missing information.
- Include all relevant details, names, numbers, steps
  and deadlines supported by the context.
- Give direct, concise and complete answers.
- Mention supporting source documents and page numbers.
"""

    # Build conversation messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add previous conversation
    if chat_history:
        for message in chat_history:
            if message["role"] in ["user", "assistant"]:
                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

    # Add current question and retrieved context
    current_prompt = f"""
Retrieved Document Context:
{context}

Current User Question:
{question}

Answer:
"""

    messages.append({
        "role": "user",
        "content": current_prompt
    })

    # Send request to local Ollama model
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["message"]["content"]

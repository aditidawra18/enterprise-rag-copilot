
import os
import requests
from google import genai
from google.genai import types


# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:1.5b"

# Gemini configuration
GEMINI_MODEL = "gemini-3.5-flash-lite"


SYSTEM_PROMPT = """
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


def get_gemini_api_key():

    # First check environment variables
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    # Then check Streamlit secrets
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")

    except Exception:
        return None


def prepare_context(context_chunks):

    context = ""

    for i, chunk in enumerate(context_chunks, start=1):
        context += (
            f"\nDocument {i}:\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content: {chunk['text']}\n"
        )

    return context


def generate_answer(
    question,
    context_chunks,
    chat_history=None,
    provider="ollama"
):

    context = prepare_context(context_chunks)

    current_prompt = f"""
Retrieved Document Context:
{context}

Current User Question:
{question}

Answer:
"""

    # -----------------------------
    # OLLAMA
    # -----------------------------

    if provider == "ollama":

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        if chat_history:
            for message in chat_history:
                if message["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })

        messages.append({
            "role": "user",
            "content": current_prompt
        })

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
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

    # -----------------------------
    # GEMINI
    # -----------------------------

    elif provider == "gemini":

        api_key = get_gemini_api_key()

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        client = genai.Client(api_key=api_key)

        contents = []

        if chat_history:
            for message in chat_history:

                role = message.get("role")

                if role == "user":
                    contents.append({
                        "role": "user",
                        "parts": [
                            {"text": message["content"]}
                        ]
                    })

                elif role == "assistant":
                    contents.append({
                        "role": "model",
                        "parts": [
                            {"text": message["content"]}
                        ]
                    })

        contents.append({
            "role": "user",
            "parts": [
                {"text": current_prompt}
            ]
        })

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        return response.text

    else:
        raise ValueError(
            "Invalid provider. Choose 'ollama' or 'gemini'."
        )

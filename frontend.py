
import streamlit as st
import requests

from src.rag_pipeline import rag_pipeline


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Copilot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise Knowledge Copilot")

st.caption(
    "Ask questions about company policies, IT support and operations."
)


# --------------------------------------------------
# APPLICATION MODE
# --------------------------------------------------

DEPLOYMENT_MODE = st.secrets.get("DEPLOYMENT_MODE", "local")

API_URL = "http://127.0.0.1:8000/chat"


# --------------------------------------------------
# CLOUD RESOURCE INITIALIZATION
# --------------------------------------------------

@st.cache_resource
def initialize_cloud_resources():

    from src.vector_store import initialize_vector_store
    from src.hybrid_search import initialize_hybrid_search
    from src.reranking import initialize_reranker

    # Build or reuse ChromaDB
    initialize_vector_store()

    # Initialize retrieval models
    initialize_hybrid_search()

    # Initialize CrossEncoder
    initialize_reranker()

    return True


# --------------------------------------------------
# INITIALIZE CONVERSATION HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("View Sources"):

                for source in message["sources"]:

                    st.write(
                        f"**Document:** {source['document']}"
                    )

                    st.write(
                        f"**Page:** {source['page']}"
                    )

                    st.divider()


# --------------------------------------------------
# ACCEPT USER QUESTION
# --------------------------------------------------

question = st.chat_input("Ask your question...")


if question:

    # Capture previous conversation before adding current question
    chat_history = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages
        if message["role"] in ["user", "assistant"]
    ]

    # Store and display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # --------------------------------------------------
    # GENERATE ASSISTANT RESPONSE
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                if DEPLOYMENT_MODE == "cloud":

                    # Initialize cloud resources once
                    initialize_cloud_resources()

                    # Direct RAG pipeline using Gemini
                    result = rag_pipeline(
                        question,
                        chat_history=chat_history,
                        provider="gemini"
                    )

                else:

                    # Local FastAPI + Ollama
                    response = requests.post(
                        API_URL,
                        json={
                            "question": question,
                            "chat_history": chat_history
                        },
                        timeout=180
                    )

                    response.raise_for_status()

                    result = response.json()


                answer = result["answer"]
                sources = result.get("sources", [])


            except Exception as e:

                answer = f"An error occurred: {e}"
                sources = []


        # Display answer
        st.markdown(answer)

        # Display sources
        if sources:

            with st.expander("View Sources"):

                for source in sources:

                    st.write(
                        f"**Document:** {source['document']}"
                    )

                    st.write(
                        f"**Page:** {source['page']}"
                    )

                    st.divider()


    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

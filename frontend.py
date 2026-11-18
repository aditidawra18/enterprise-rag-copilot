
import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/chat"


st.set_page_config(
    page_title="Enterprise RAG Copilot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise Knowledge Copilot")
st.caption("Ask questions about company policies, IT support and operations.")


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("View Sources"):

                for source in message["sources"]:
                    st.write(f"**Document:** {source['document']}")
                    st.write(f"**Page:** {source['page']}")
                    st.divider()


# Accept user question
question = st.chat_input("Ask your question...")


if question:

    # Capture previous conversation BEFORE adding current question
    chat_history = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages
        if message["role"] in ["user", "assistant"]
    ]


    # Display and store current user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

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

            except requests.RequestException as e:

                answer = f"Unable to connect to the API: {e}"
                sources = []


        st.markdown(answer)

        if sources:

            with st.expander("View Sources"):

                for source in sources:
                    st.write(f"**Document:** {source['document']}")
                    st.write(f"**Page:** {source['page']}")
                    st.divider()


    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

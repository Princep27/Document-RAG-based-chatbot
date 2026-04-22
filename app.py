import streamlit as st
from rag import rag_chat

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 Chat with your data")
st.caption("LangChain + FAISS + Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = rag_chat(query)

            st.write(answer)

            with st.expander("📚 Sources"):
                st.text(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("AI Chatbot")
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(llm.stream(st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": response})

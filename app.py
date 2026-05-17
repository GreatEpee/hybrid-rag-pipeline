import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import agent 

st.set_page_config(page_title="Agentic RAG Pipeline", page_icon="⚙️")
st.title("🤖 Agentic RAG Pipeline")
st.markdown("Dynamically routing user queries between internal vector databases and live web APIs. Ask me anything about the Valve employee handbook or live web data.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., What is Valve's policy on vacations?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Agentic routing in progress..."):
        try:
            formatted_history = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    formatted_history.append(HumanMessage(content=m["content"]))
                else:
                    formatted_history.append(AIMessage(content=m["content"]))
            
            inputs = {"messages": formatted_history}
            response = agent.invoke(inputs)
            ai_reply = response["messages"][-1].content
        except Exception as e:
            ai_reply = f"Error communicating with agent: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

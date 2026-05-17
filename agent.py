import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

load_dotenv()
print("Booting up Clean LangGraph Architecture...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="valve-handbook", embedding=embeddings)

@tool
def search_handbook(query: str) -> str:
    """Use this tool ONLY to answer questions about Valve's company policies, rules, projects, or employee handbook."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

web_search = TavilySearchResults(
    max_results=2,
    description="Use this tool to search the live internet for current events, weather, or general knowledge outside the company."
)

tools = [search_handbook, web_search]
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
agent = create_react_agent(llm, tools)

# Llama-3.1-8b is preferred here over smaller local models for reliable native tool-calling without XML parsing errors.
#previously i tried with llama-3.3-70b-versatile
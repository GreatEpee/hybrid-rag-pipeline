import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
print("Booting up Ingestion Pipeline...\n")

def ingest_handbook():

    file_path = "data/valve_handbook.pdf"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"\nMissing PDF file at {file_path}")
    
    print("\nLoading PDF...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    print("\nChunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"\nSplit PDF into {len(chunks)} chunks.")
    #got 82 chunks
    print("\nInitializing Embedding Model (Downloading weights if first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    index_name = "valve-handbook"
    print(f"Pushing vectors to Pinecone index: '{index_name}'...")

    PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=index_name
    )
    
    print("Ingestion Complete! \nVector DB is ready.")

if __name__ == "__main__":
    ingest_handbook()
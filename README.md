# hybrid-rag-pipeline
**[Live Interactive Demo on Hugging Face](https://huggingface.co/spaces/ugabooga/hybrid-rag-pipeline)**

## Overview
This repository contains the source code for an Agentic Retrieval-Augmented Generation (RAG) application. The agent is designed to answer questions based on the **Valve Employee Handbook** while maintaining the ability to query the live internet for external information. The system uses LangGraph to dynamically route user queries to the appropriate tool based on the context of the prompt.
<div align="center">
  <img width="700" alt="Agentic UI Demo" src="https://github.com/user-attachments/assets/eb7c2f30-ee59-40a7-9343-98f10152601f" />
</div>

<br>

<div align="center">
  <img width="450" alt="Ragas Evaluation Metrics" src="https://github.com/user-attachments/assets/8acaaa90-b5ce-47f8-9a28-9ca1fd2d7365" />
</div>


## Architecture
The system operates on a ReAct (Reasoning and Acting) agent architecture.
* **Internal Knowledge:** Queries related to Valve policies, management, and internal structure are routed to a Pinecone vector database. This database contains chunked and embedded representations of the employee handbook.
* **External Knowledge:** Queries requiring current events, live weather, or general internet knowledge are routed to the Tavily Search API.
* **Memory:** The Streamlit frontend captures the session state and passes the complete conversation history to the LangChain message objects, enabling multi-turn contextual awareness.

## Technical Stack
* **LLM:** LLaMA 3.1 8B Instant (hosted via Groq for high-speed inference)
* **Orchestration:** LangChain and LangGraph
* **Vector Database:** Pinecone
* **Embedding Model:** HuggingFace all-MiniLM-L6-v2
* **Web Search:** Tavily API
* **Frontend:** Streamlit
* **Deployment:** Docker, Hugging Face Spaces
* **Evaluation:** Ragas framework

## Installation and Local Setup

1. Clone the repository:
   ```
   git clone [https://github.com/yourusername/valve-rag-agent.git](https://github.com/yourusername/valve-rag-agent.git)
   cd valve-rag-agent
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables. Create a .env file in the root directory and add the following keys:
   ```
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   TAVILY_API_KEY=your_tavily_key
   GOOGLE_API_KEY=your_google_key
   ```

## Deployment
This application is containerized using Docker and deployed on Hugging Face Spaces. The deployment process involves uploading the source code (app.py, agent.py, requirements.txt, and the Dockerfile) to the Hugging Face repository, which automatically triggers a cloud build. Environment variables are managed securely within the Space settings.

## Evaluation
The retrieval and generation pipelines were quantitatively evaluated using the Ragas framework locally. The evaluation focused on two primary metrics:
* **Faithfulness:** Verifying that the LLM's answers are factually derived from the retrieved context.
* **Answer Relevancy:** Measuring how directly the generated answer addresses the user's initial prompt.

<div align="center">
  
<img width="537" height="449" alt="Ragas Evaluation Metrics" src="https://github.com/user-attachments/assets/910fe82b-c0a4-412d-9181-27671c15ad68" />

</div>
   

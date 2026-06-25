Production-ready RAG chatbot using LangChain, FAISS, Hugging Face embeddings, and Ollama (Llama 3.2). Features PDF/TXT document ingestion, semantic search with MMR retrieval, local vector storage, context-only answering, and hallucination reduction through strict prompt grounding.

This project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline from document ingestion to question answering. It is designed to provide accurate, context-grounded responses while minimizing hallucinations by restricting the language model to retrieved document content only.

A lightweight, local-first Retrieval-Augmented Generation (RAG) application that allows you to chat with your own documents using LangChain, FAISS, Hugging Face Embeddings, and Ollama's Llama 3.2 model.

```
The project includes:

•	PDF and TXT document ingestion
•	Intelligent document chunking
•	Semantic embeddings using all-MiniLM-L6-v2
•	FAISS vector database for fast similarity search
•	MMR (Max Marginal Relevance) retrieval for diverse results
•	Local LLM inference with Ollama (llama3.2)
•	Context-grounded responses to reduce hallucinations
•	Fully local and privacy-friendly workflow
```


Architecture:

```
Documents (PDF/TXT)
        │
        ▼
Document Loader
        │
        ▼
Text Splitter
        │
        ▼
Hugging Face Embeddings
        │
        ▼
FAISS Vector Store
        │
        ▼
Retriever (MMR)
        │
        ▼
Llama 3.2 via Ollama
        │
        ▼
Context-Aware Answers

```

Project Structure:
```
├── documents/
│   ├── file1.pdf
│   └── file2.txt
│
├── ingest.py        # Document ingestion and FAISS index creation
├── chatbot.py       # Interactive RAG chatbot
│
└── faiss_index/     # Generated vector database

```

Workflow:
```
1. Ingest Documents
Run the ingestion pipeline to:
•	Load PDF and TXT files
•	Split documents into chunks
•	Generate embeddings
•	Build and save a FAISS index

python ingest.py
```

```
2. Start the Chatbot

Load the FAISS index and ask questions about your documents:

python chatbot.py

Example:
Ask Question: list the name of tamil movies in 2025?

Answer:
[Generated response based only on retrieved context]
```


Tech Stack:
```
•	LangChain
•	FAISS
•	Hugging Face Embeddings
•	Ollama
•	Llama 3.2
•	Python

```


Features:
```
•	Local RAG pipeline

•	No external vector database required

•	Efficient semantic retrieval

•	Context-restricted answers

•	Easy to extend for production use

•	Works with custom document collections

```


Future Enhancements:
```
•	Streamlit/Web UI

•	Source citation display

•	Conversational memory

•	Hybrid search (keyword + vector)

•	Metadata filtering

•	API deployment with FastAPI

•	Docker support

```

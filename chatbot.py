#Production RAG Application

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain.chains.retrieval import (
    create_retrieval_chain
)

# Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS Index

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)


# LLM

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Prompt Template

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Use ONLY the supplied context.

If the answer is not found in the context,
reply exactly:

I'm unable to answer.

Context:
{context}

Question:
{input}
"""
)


# Document Chain

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# Retrieval Chain

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# Chat Loop

while True:

    question = input("\nAsk Question [Tamil Movies in 2025]: ")

    if question.lower() == "exit":
        break

    response = rag_chain.invoke(
        {
            "input": question
        }
    )

    print("\nAnswer:")
    print(response["answer"])





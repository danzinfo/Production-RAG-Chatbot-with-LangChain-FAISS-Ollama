#Document Ingestion Pipeline

#First create the FAISS index in this program

from pathlib import Path

from langchain_community.document_loaders import(PyPDFLoader, TextLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


documents=[]

#Load the documents
docs_path=Path("documents")

for file in docs_path.iterdir():

    if file.suffix.lower()==".pdf":

        loader=PyPDFLoader(str(file))
        docs=loader.load()

        for doc in docs:
            doc.metadata["source"]=file.name

        documents.extend(docs)

    elif file.suffix.lower()==".txt":

         loader=TextLoader(str(file),encoding="utf-8")
         docs=loader.load()

         for doc in docs:
             doc.metadata["source"]=file.name

         documents.extend(docs)

print(f"Loaded {len(documents)} documents")


#Split into chunks
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)


chunk=splitter.split_documents(documents)

print(f"created {len(chunk)} chunks")

#Embedding Model
embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" )


#Build FAISS index
vectorstore=FAISS.from_documents(
    chunk,
    embedding)


#Save Index
vectorstore.save_local("faiss_index")

print("FAISS index created successfully..")


# vector_store.py

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

db_location = "./chrome_langchain_db"
collection_name = "restaurant_reviews"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def embed_csv(file_path: str):
    df = pd.read_csv(file_path)
    documents = []
    ids = []

    for i, row in df.iterrows():
        doc = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        documents.append(doc)
        ids.append(str(i))

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )
    vector_store.add_documents(documents=documents, ids=ids)
    return True

def get_retriever():
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})

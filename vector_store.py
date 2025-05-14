from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

db_location = "./chrome_langchain_db"
collection_name = "customer_data"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def embed_csv(file_path: str):
    print("Embedding CSV...")

    # Read CSV with proper quote handling
    df = pd.read_csv(file_path, quotechar='"', skipinitialspace=True)

    print(df.head())
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Handle possible NaN with `row.get(..., '') or ''`
        content = (
            f"{row.get('First Name', '') or ''} {row.get('Last Name', '') or ''} from {row.get('Company', '') or ''} "
            f"in {row.get('City', '') or ''}, {row.get('Country', '') or ''}. "
            f"Email: {row.get('Email', '') or ''}, Phone: {row.get('Phone 1', '') or ''}. "
            f"Subscribed on {row.get('Subscription Date', '') or ''} via {row.get('Website', '') or ''}."
        )

        doc = Document(
            page_content=content,
            metadata={
                "customer_id": row.get("Customer Id", "") or "",
                "email": row.get("Email", "") or "",
                "city": row.get("City", "") or "",
                "country": row.get("Country", "") or "",
                "subscription_date": row.get("Subscription Date", "") or ""
            },
            id=str(row.get("Index", i))
        )
        documents.append(doc)
        ids.append(str(row.get("Index", i)))

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Embedded {len(documents)} customer records.")
    return True


def get_retriever():
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})

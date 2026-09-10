"""
ingest.py
Reads IT ticket data from CSV, converts each entry into an embedding,
and stores it in a local ChromaDB vector database.

Run this once initially, and again whenever you update data/it_tickets.csv.
"""

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
CSV_PATH = "data/it_tickets.csv"
PERSIST_DIR = "chroma_db"


def load_documents(csv_path: str) -> list[Document]:
    """Load CSV rows into LangChain Document objects."""
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        content = (
            f"Error: {row['error_description']}\n"
            f"Solution: {row['solution']}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={"error": row["error_description"]},
            )
        )
    return documents


def build_vectorstore():
    print("Loading documents from CSV...")
    documents = load_documents(CSV_PATH)
    print(f"Loaded {len(documents)} documents.")

    print("Loading embedding model (first run downloads the model, ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectorstore.persist()
    print(f"Done. Vector store saved to '{PERSIST_DIR}/'.")


if __name__ == "__main__":
    build_vectorstore()
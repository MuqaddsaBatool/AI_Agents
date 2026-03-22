# ingest.py
# Reads PDFs from papers/, chunks them, embeds with OpenAI,
# and stores in a local ChromaDB vector store.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PAPERS_DIR   = "papers"
CHROMA_DIR   = "chroma_db"
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 100

def ingest_papers():
    print("📄 Loading PDFs...")
    docs = []

    for filename in os.listdir(PAPERS_DIR):
        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(PAPERS_DIR, filename)
        loader = PyMuPDFLoader(path)
        pages = loader.load()

        # Tag every chunk with its source filename
        for page in pages:
            page.metadata["source"] = filename

        docs.extend(pages)
        print(f"   ✓ Loaded {filename} ({len(pages)} pages)")

    print(f"\n✂️  Chunking {len(docs)} pages...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"   ✓ Created {len(chunks)} chunks")

    print("\n🔢 Embedding and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"   ✓ Stored {len(chunks)} chunks in {CHROMA_DIR}/")
    print("\n✅ Ingestion complete.")
    return vectorstore


if __name__ == "__main__":
    ingest_papers()
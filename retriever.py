# Hybrid retrieval: dense vector search + BM25 sparse search
# combined via Reciprocal Rank Fusion (RRF)

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

CHROMA_DIR = "chroma_db"

# ── Load vector store ─────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# ── Build BM25 index from all stored chunks ───────────────────────────────────
def _build_bm25_index():
    """Load all docs from ChromaDB and build a BM25 index over them."""
    all_docs = vectorstore.get()  # returns dict with ids, documents, metadatas
    texts     = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    tokenized = [doc.lower().split() for doc in texts]
    bm25      = BM25Okapi(tokenized)

    return bm25, texts, metadatas

bm25_index, all_texts, all_metadatas = _build_bm25_index()

# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def _rrf(vector_hits: list, bm25_hits: list, k: int = 60) -> list[int]:
    """
    Merge two ranked lists of doc indices using RRF.
    Higher score = more relevant.
    RRF score = sum of 1 / (k + rank) across both lists.
    """
    scores: dict[int, float] = {}

    for rank, idx in enumerate(vector_hits):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    for rank, idx in enumerate(bm25_hits):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # Sort by descending score
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

# ── Main retrieval function ───────────────────────────────────────────────────
def retrieve(query: str, k: int = 4) -> str:
    """
    Hybrid retrieval: vector + BM25 + RRF fusion.
    Returns top-k chunks with source citations.
    """

    # ── Dense retrieval ───────────────────────────────────────────────────────
    #vector_results = vectorstore.similarity_search_with_index(query, k=10)
    # Returns list of (Document, index) — extract indices
    # Fallback: use similarity_search and match by content
    vector_docs    = vectorstore.similarity_search(query, k=10)
    vector_indices = []
    for vdoc in vector_docs:
        for i, t in enumerate(all_texts):
            if t.strip() == vdoc.page_content.strip():
                vector_indices.append(i)
                break

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_scores     = bm25_index.get_scores(tokenized_query)
    bm25_indices    = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:10]

    # ── Fuse rankings ─────────────────────────────────────────────────────────
    fused_indices = _rrf(vector_indices, bm25_indices)[:k]

    if not fused_indices:
        return "No relevant content found in the knowledge base."

    # ── Format output ─────────────────────────────────────────────────────────
    output = []
    for rank, idx in enumerate(fused_indices):
        source = all_metadatas[idx].get("source", "unknown")
        page   = all_metadatas[idx].get("page", "?")
        output.append(
            f"[{rank+1}] Source: {source} (page {page})\n"
            f"{all_texts[idx].strip()}"
        )

    return "\n\n---\n\n".join(output)


if __name__ == "__main__":
    # Test 1: semantic query — vector search should dominate
    print("=== Test 1: Semantic Query ===")
    print(retrieve("How does the agent decide when to stop?"))

    print("\n=== Test 2: Exact Term Query ===")
    # Test 2: exact term — BM25 should help here
    print(retrieve("ˆA = A ∪ L language action space"))

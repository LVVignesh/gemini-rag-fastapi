import faiss
import os
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

USE_HNSW = True
USE_RERANKER = True

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

DB_FILE_INDEX = "vector.index"
DB_FILE_META = "metadata.pkl"
DB_FILE_BM25 = "bm25.pkl"

index = None
documents = []
metadata = []
bm25 = None
tokenized_corpus = []

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def chunk_text(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > CHUNK_SIZE and current:
            chunks.append(current.strip())
            overlap = max(0, len(current) - CHUNK_OVERLAP)
            current = current[overlap:] + " " + s
        else:
            current += " " + s if current else s
    if current.strip():
        chunks.append(current.strip())
    return chunks

def save_db():
    if index:
        faiss.write_index(index, DB_FILE_INDEX)
    if documents:
        with open(DB_FILE_META, "wb") as f:
            pickle.dump({"documents": documents, "metadata": metadata}, f)
    if bm25:
        with open(DB_FILE_BM25, "wb") as f:
            pickle.dump(bm25, f)

def load_db():
    global index, documents, metadata, bm25
    if os.path.exists(DB_FILE_INDEX) and os.path.exists(DB_FILE_META):
        index = faiss.read_index(DB_FILE_INDEX)
        with open(DB_FILE_META, "rb") as f:
            data = pickle.load(f)
            documents = data["documents"]
            metadata = data["metadata"]
    
    if os.path.exists(DB_FILE_BM25):
        with open(DB_FILE_BM25, "rb") as f:
            bm25 = pickle.load(f)
    elif documents:
        # Auto-backfill if documents exist but BM25 is missing
        print("Backfilling BM25 index on first load...")
        tokenized_corpus = [doc.split(" ") for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(DB_FILE_BM25, "wb") as f:
            pickle.dump(bm25, f)

load_db()

def clear_database():
    global index, documents, metadata, bm25
    index = None
    documents = []
    metadata = []
    bm25 = None
    if os.path.exists(DB_FILE_INDEX):
        os.remove(DB_FILE_INDEX)
    if os.path.exists(DB_FILE_META):
        os.remove(DB_FILE_META)
    if os.path.exists(DB_FILE_BM25):
        os.remove(DB_FILE_BM25)

def ingest_documents(files):
    global index, documents, metadata
    texts, meta = [], []

    for file in files:
        if file.filename.endswith(".pdf"):
            # Save temp file for pymupdf4llm
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.file.read())
                tmp_path = tmp.name
            
            try:
                # Use pymupdf4llm to extract markdown with tables
                import pymupdf4llm
                # Get list of dicts: [{'text': '...', 'metadata': {'page': 1, ...}}]
                pages_data = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)
                
                for page_obj in pages_data:
                    p_text = page_obj["text"]
                    p_num = page_obj["metadata"].get("page", "N/A")
                    
                    # Chunk within the page to preserve page context
                    for chunk in chunk_text(p_text):
                        texts.append(chunk)
                        meta.append({"source": file.filename, "page": p_num})
            finally:
                os.remove(tmp_path)

        elif file.filename.endswith(".txt"):
            content = file.file.read().decode("utf-8", errors="ignore")
            for chunk in chunk_text(content):
                texts.append(chunk)
                meta.append({"source": file.filename, "page": "N/A"})

    if not texts:
        raise ValueError("No readable text found (OCR needed for scanned PDFs).")

    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    if index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32) if USE_HNSW else faiss.IndexFlatIP(dim)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64

    index.add(embeddings)
    documents.extend(texts)
    metadata.extend(meta)
    
    # Update BM25
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    save_db()
    return len(documents)

def search_knowledge(query, top_k=8):
    if index is None:
        return []

    # 1. Vector Search
    qvec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(qvec, top_k)
    
    vector_results = {}
    for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx == -1: continue
        vector_results[idx] = i  # Store rank (0-based)

    # 2. Keyword Search (BM25)
    bm25_results = {}
    if bm25:
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        # Get top_k indices
        top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        for i, idx in enumerate(top_n):
            bm25_results[idx] = i  # Store rank

    # 3. Reciprocal Rank Fusion (RRF)
    # score = 1 / (k + rank)
    k = 60
    candidates_idx = set(vector_results.keys()) | set(bm25_results.keys())
    merged_candidates = []

    for idx in candidates_idx:
        v_rank = vector_results.get(idx, float('inf'))
        b_rank = bm25_results.get(idx, float('inf'))
        
        rrf_score = (1 / (k + v_rank)) + (1 / (k + b_rank))
        
        merged_candidates.append({
            "text": documents[idx],
            "metadata": metadata[idx],
            "score": rrf_score,  # This is RRF score, not cosine/BM25 score
            "vector_rank": v_rank if v_rank != float('inf') else None,
            "bm25_rank": b_rank if b_rank != float('inf') else None
        })

    # Sort by RRF score
    merged_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # 4. Rerank Top Candidates
    candidates = merged_candidates[:10] # Take top 10 for reranking

    if USE_RERANKER and candidates:
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs)
        for c, rs in zip(candidates, rerank_scores):
            c["rerank"] = float(rs)
        candidates.sort(key=lambda x: x["rerank"], reverse=True)

    return candidates[:5]

def get_all_chunks(limit=80):
    return [{"text": t, "metadata": m} for t, m in zip(documents[:limit], metadata[:limit])]

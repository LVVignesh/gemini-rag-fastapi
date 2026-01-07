import faiss
import os
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================================================
# CONFIG
# =========================================================
USE_HNSW = True
USE_RERANKER = True

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

DB_FILE_INDEX = "vector.index"
DB_FILE_META = "metadata.pkl"

# =========================================================
# GLOBAL STATE
# =========================================================
index = None
documents = []
metadata = []

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# =========================================================
# HELPERS
# =========================================================
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


def load_db():
    global index, documents, metadata
    if os.path.exists(DB_FILE_INDEX) and os.path.exists(DB_FILE_META):
        index = faiss.read_index(DB_FILE_INDEX)
        with open(DB_FILE_META, "rb") as f:
            data = pickle.load(f)
            documents = data["documents"]
            metadata = data["metadata"]
        print(f"DEBUG: Loaded {len(documents)} chunks")


load_db()


def clear_database():
    global index, documents, metadata
    index = None
    documents = []
    metadata = []

    if os.path.exists(DB_FILE_INDEX):
        os.remove(DB_FILE_INDEX)
    if os.path.exists(DB_FILE_META):
        os.remove(DB_FILE_META)


# =========================================================
# INGEST
# =========================================================
def ingest_documents(files):
    global index, documents, metadata

    texts, meta = [], []

    for file in files:
        name = file.filename

        if name.endswith(".pdf"):
            reader = PdfReader(file.file)
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                except Exception:
                    text = None

                if text:
                    for chunk in chunk_text(text):
                        texts.append(chunk)
                        meta.append({"source": name, "page": i + 1})

        elif name.endswith(".txt"):
            content = file.file.read().decode("utf-8", errors="ignore")
            for chunk in chunk_text(content):
                texts.append(chunk)
                meta.append({"source": name, "page": "N/A"})

    if not texts:
        raise ValueError(
            "No readable text found. "
            "If this is a scanned PDF, OCR is required."
        )

    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    if index is None:
        dim = embeddings.shape[1]
        if USE_HNSW:
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
        else:
            index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    documents.extend(texts)
    metadata.extend(meta)

    save_db()
    return len(documents)


# =========================================================
# SEARCH
# =========================================================
def search_knowledge(query, top_k=8, min_similarity=0.25):
    if index is None:
        return []

    qvec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(qvec, top_k)
    candidates = []
    ql = query.lower()

    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue

        text = documents[idx]
        meta = metadata[idx]
        keyword_hits = sum(w in text.lower() for w in ql.split())
        hybrid_score = float(score) + (0.05 * keyword_hits)

        if hybrid_score >= min_similarity:
            candidates.append({
                "text": text,
                "metadata": meta,
                "hybrid_score": hybrid_score
            })

    if USE_RERANKER and candidates:
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank"] = float(s)
        candidates.sort(key=lambda x: x["rerank"], reverse=True)
    else:
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return candidates[:5]


def get_all_chunks(limit=50):
    return [
        {"text": t, "metadata": m}
        for t, m in zip(documents[:limit], metadata[:limit])
    ]

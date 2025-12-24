import faiss
import numpy as np
import os
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------
# Global state
# -----------------------
index = None
documents = []
metadata = []

# Using a lightweight, high-performance embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

DB_FILE_INDEX = "vector.index"
DB_FILE_META = "metadata.pkl"

# -----------------------
# Helpers
# -----------------------
def chunk_text(text):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def save_db():
    global index, documents, metadata
    if index:
        faiss.write_index(index, DB_FILE_INDEX)
    if documents:
        with open(DB_FILE_META, "wb") as f:
            pickle.dump({"documents": documents, "metadata": metadata}, f)
    print("DEBUG: Knowledge base saved to disk.")

def load_db():
    global index, documents, metadata
    if os.path.exists(DB_FILE_INDEX) and os.path.exists(DB_FILE_META):
        try:
            index = faiss.read_index(DB_FILE_INDEX)
            with open(DB_FILE_META, "rb") as f:
                data = pickle.load(f)
                documents = data["documents"]
                metadata = data["metadata"]
            print(f"DEBUG: Loaded {len(documents)} documents from disk.")
        except Exception as e:
            print(f"DEBUG: Failed to load DB: {e}")
            index = None
            documents = []
            metadata = []
    else:
        print("DEBUG: No existing DB found. Starting fresh.")

# Auto-load on startup
load_db()

def clear_database():
    global index, documents, metadata
    index = None
    documents = []
    metadata = []
    
    # Remove persistence files if they exist
    if os.path.exists(DB_FILE_INDEX):
        os.remove(DB_FILE_INDEX)
    if os.path.exists(DB_FILE_META):
        os.remove(DB_FILE_META)
    
    print("DEBUG: Database cleared.")

# -----------------------
# Ingest
# -----------------------
def ingest_documents(files):
    global index, documents, metadata

    texts = []
    meta = []

    for file in files:
        filename = file.filename
        
        # Handle PDFs
        if filename.endswith(".pdf"):
            reader = PdfReader(file.file)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    for chunk in chunk_text(page_text):
                        texts.append(chunk)
                        meta.append({"source": filename, "page": i + 1})
        
        # Handle Text files
        elif filename.endswith(".txt"):
            content = file.file.read().decode("utf-8")
            for chunk in chunk_text(content):
                texts.append(chunk)
                meta.append({"source": filename, "page": "N/A"})

    # Check for empty or unreadable content
    total_length = sum(len(t) for t in texts)
    if total_length < 50:
        raise ValueError(
            "Extracted text is too short or empty. "
            "If this is a PDF, it might be a scanned image without a text layer. "
            "Please use a text-selectable PDF or a .txt file."
        )

    if not texts:
        raise ValueError("No readable text found in documents.")

    # Create Embeddings (Normalized for better cosine similarity)
    # append to existing if needed, but for now simplistic re-build or append?
    # Simpler to just ADD to the existing index.
    
    new_embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    if index is None:
        # USE INNER PRODUCT (Cosine Similarity) for normalized vectors
        index = faiss.IndexFlatIP(new_embeddings.shape[1])
    
    index.add(new_embeddings)

    documents.extend(texts)
    metadata.extend(meta)

    save_db()
    
    return len(documents)

# -----------------------
# Q&A Search (filtered)
# -----------------------
def search_knowledge(query, top_k=5, min_similarity=0.3):
    if index is None:
        return []

    # SEARCH with normalized query
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    
    # FAISS returns scores (dot product), which = cosine similarity for normalized vectors
    scores, indices = index.search(query_vec, top_k)

    results = []
    print(f"DEBUG: Query: '{query}'")
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1: continue # FAISS padding
        
        print(f"DEBUG: Found chunk {idx} with score {score:.4f}")
        
        # Filter out results that are too irrelevant (score too low)
        if score > min_similarity:
            results.append({
                "text": documents[idx],
                "metadata": metadata[idx],
                "score": float(score)
            })

    return results

# -----------------------
# Summary Retrieval (NO FILTER)
# -----------------------
def get_all_chunks(limit=50):
    if not documents:
        return []

    results = []
    # Return a sample of chunks for summarization
    for text, meta in zip(documents[:limit], metadata[:limit]):
        results.append({
            "text": text,
            "metadata": meta
        })

    return results
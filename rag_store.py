import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------
# Global in-memory state
# -----------------------
index = None
documents = []
metadata = []

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# Ingest uploaded files
# -----------------------
def ingest_documents(files):
    global index, documents, metadata

    texts = []
    meta = []

    for file in files:
        filename = file.filename

        if filename.endswith(".pdf"):
            reader = PdfReader(file.file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    texts.append(text)
                    meta.append({
                        "source": filename,
                        "page": i + 1
                    })

        elif filename.endswith(".txt"):
            content = file.file.read().decode("utf-8")
            texts.append(content)
            meta.append({
                "source": filename,
                "page": "N/A"
            })

    if not texts:
        raise ValueError("No readable text found.")

    embeddings = embedder.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    documents = texts
    metadata = meta

    return len(texts)

# -----------------------
# Search
# -----------------------
def search_knowledge(query, top_k=5):
    if index is None:
        return []

    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "text": documents[idx],
            "distance": float(dist),
            "metadata": metadata[idx]
        })

    return results

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DATA_DIR = "data"
INDEX_FILE = "vector.index"
DOCS_FILE = "documents.npy"
META_FILE = "metadata.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Load or build index
# -------------------------
if os.path.exists(INDEX_FILE):
    print("🔁 Loading FAISS index from disk...")
    index = faiss.read_index(INDEX_FILE)
    documents = np.load(DOCS_FILE, allow_pickle=True)
    metadata = np.load(META_FILE, allow_pickle=True)
else:
    print("🧠 Building FAISS index...")
    texts = []
    meta = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DATA_DIR, file))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    texts.append(text)
                    meta.append({
                        "source": file,
                        "page": i + 1
                    })

    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    np.save(DOCS_FILE, texts)
    np.save(META_FILE, meta)
    faiss.write_index(index, INDEX_FILE)

    documents = texts
    metadata = meta

    print("✅ FAISS index saved to disk.")

# -------------------------
# Search
# -------------------------
def search_knowledge(query, top_k=5):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": documents[idx],
            "metadata": metadata[idx],
            "distance": float(dist)
        })

    return results

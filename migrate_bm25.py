from rag_store import load_db, save_db, documents, bm25
from rank_bm25 import BM25Okapi
import pickle

print("Loading DB...")
load_db()

if not documents:
    print("No documents found. Nothing to do.")
else:
    print(f"Found {len(documents)} documents.")
    print("Building BM25 index...")
    tokenized_corpus = [doc.split(" ") for doc in documents]
    
    # We need to update the global variable in rag_store, but since we imported 'bm25' (by value? no, python imports names), 
    # we need to actually set it in the module or just use the save logic.
    # Actually, simplistic way:
    import rag_store
    rag_store.bm25 = BM25Okapi(tokenized_corpus)
    
    print("Saving DB with BM25...")
    rag_store.save_db()
    print("Done!")

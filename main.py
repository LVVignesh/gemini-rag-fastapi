import os
import sys
from time import time

# Start timing immediately
startup_start_time = time()

# Get memory usage function
def get_memory_usage_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
    return None

mem_initial = get_memory_usage_mb()

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_store import ingest_documents, get_all_chunks, clear_database, search_knowledge
from analytics import get_analytics
from agentic_rag_v2_graph import build_agentic_rag_v2_graph
from llm_utils import generate_with_retry
import asyncio

# =========================================================
# ENV + MODEL
# =========================================================
load_dotenv()

HF_DEPLOYMENT = os.getenv("HF_DEPLOYMENT", "False").lower() == "true"
MOCK_MODE = False  # Refactor complete - enabling real agent
MODEL_NAME = "gemini-3-flash-preview"
MAX_FILE_SIZE = 50 * 1024 * 1024
CACHE_TTL = 300

# =========================================================
# APP
# =========================================================
app = FastAPI(title="NexusGraph AI (Enterprise RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# =========================================================
# SECURITY
# =========================================================


# =========================================================
# STATE
# =========================================================
agentic_graph = None
answer_cache: dict[str, tuple[float, dict]] = {}

def get_agentic_graph():
    global agentic_graph
    if agentic_graph is None:
        start_t = time()
        print("[INFO] Lazily compiling agentic graph (build_agentic_rag_v2_graph)...")
        agentic_graph = build_agentic_rag_v2_graph()
        print(f"[SUCCESS] Agentic graph compiled in {time() - start_t:.4f} seconds.")
    return agentic_graph

# =========================================================
# DIAGNOSTICS LOGGING
# =========================================================
startup_end_time = time()
startup_duration = startup_end_time - startup_start_time
mem_final = get_memory_usage_mb()

if not HF_DEPLOYMENT:
    print("=" * 60)
    print("NEXUSGRAPH AI STARTUP DIAGNOSTICS")
    print(f"Startup phase duration: {startup_duration:.4f} seconds")
    print(f"HF_DEPLOYMENT mode: {HF_DEPLOYMENT}")
    print(f"Lazy loading activated: True (Embedding, Reranker, and DB load deferred)")
    if mem_initial is not None and mem_final is not None:
        print(f"Memory footprint: Initial {mem_initial:.2f} MB -> Final {mem_final:.2f} MB (Delta: {mem_final - mem_initial:.2f} MB)")
    else:
        print("Memory footprint: N/A")
    print("=" * 60)
else:
    mem_str = f"{mem_final:.1f}MB" if mem_final is not None else "N/A"
    print(f"NexusGraph (HF Mode) started in {startup_duration:.3f}s. Memory: {mem_str}. Lazy loading: True.")
sys.stdout.flush()

# =========================================================
# MODELS
# =========================================================
class PromptRequest(BaseModel):
    prompt: str
    thread_id: str = "default"



# =========================================================
# ROUTES
# =========================================================


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/analytics")
def analytics():
    return get_analytics()

# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------
@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    for file in files:
        filename = file.filename
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        print(f"🔍 DEBUG: Uploading '{filename}' (Ext: {ext})")

        if ext not in ["pdf", "txt"]:
            print(f"❌ REJECTED: Invalid extension '{ext}'")
            return JSONResponse(
                status_code=400,
                content={"error": "Only PDF and TXT files allowed"}
            )

        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)

        if size > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": "File too large"}
            )

    clear_database()
    answer_cache.clear()
    chunks = ingest_documents(files)

    return {"message": f"Indexed {chunks} chunks successfully."}

# ---------------------------------------------------------
# ASK
# ---------------------------------------------------------
@app.post("/ask")
async def ask(data: PromptRequest):
    query = data.prompt.strip()
    key = query.lower()
    now = time()

    # ---------- CACHE ----------
    if key in answer_cache:
        ts, cached = answer_cache[key]
        if now - ts < CACHE_TTL:
            return cached

    # ==========================
    # 🟧 MOCK MODE (NO API)
    # ==========================
    if MOCK_MODE:
        await asyncio.sleep(0.5)  # Simulate latency
        
        mock_answer = ""
        mock_citations = []
        
        if "summary" in key or "summarize" in key:
            # Local summary mock
            chunks = get_all_chunks(limit=3)
            mock_answer = "⚠️ **MOCK SUMMARY** ⚠️\n\n(API Quota Exhausted - Showing direct database content)\n\n"
            for c in chunks:
                # Show full text to avoid breaking markdown tables
                mock_answer += f"### Chunk from {c['metadata']['source']}\n{c['text']}\n\n---\n\n"
        else:
            # Local retrieval mock
            retrieved = search_knowledge(query)
            mock_answer = "⚠️ **MOCK RESPONSE** ⚠️\n\n(API Quota Exhausted)\n\nI found the following relevant information in your documents using local search (Exact text match):\n\n"
            
            seen_sources = set()
            for r in retrieved:
                # Add citation
                meta = r["metadata"]
                if (meta["source"], meta["page"]) not in seen_sources:
                     mock_citations.append(meta)
                     seen_sources.add((meta["source"], meta["page"]))
                
                # Show full text of the relevant chunk
                mock_answer += f"> **Source: {meta['source']}**\n\n{r['text']}\n\n---\n"
            
            if not retrieved:
                mock_answer += "No relevant documents found in the local index."

        response = {
            "answer": mock_answer,
            "confidence": 0.85,
            "citations": mock_citations
        }
        answer_cache[key] = (now, response)
        return response

    # ==========================
    # 🟦 SUMMARY (BYPASS AGENT)
    # ==========================
    if "summary" in key or "summarize" in key:
        chunks = get_all_chunks(limit=80)
        context = "\n\n".join(c["text"] for c in chunks)

        resp = generate_with_retry(
            MODEL_NAME, 
            f"Summarize the following content clearly:\n\n{context}"
        )
        
        answer_text = resp.text if resp else "Error generating summary due to quota limits."

        response = {
            "answer": answer_text,
            "confidence": 0.95,
            "citations": []
        }

        answer_cache[key] = (now, response)
        return response

    # ==========================
    # 🟩 AGENTIC RAG (LLM + EVALUATION)
    # ==========================
    # ==========================
    # 🟩 AGENTIC RAG (MULTI-TOOL SUPERVISOR)
    # ==========================
    # Initialize state for new graph
    initial_state = {
        "messages": [],
        "query": query,
        "final_answer": "",
        "next_node": "",
        "current_tool": "",
        "tool_outputs": [],
        "verification_notes": "",
        "retries": 0
    }
    
    try:
        result = get_agentic_graph().invoke(initial_state, config={"configurable": {"thread_id": data.thread_id}})
        
        # Extract citations from tool outputs
        citations = []
        seen = set()
        for t in result.get("tool_outputs", []):
            src = t.get("source", "unknown")
            # If it's a PDF, it has metadata
            if src == "internal_pdf":
                meta = t.get("metadata", {})
                key_ = (meta.get("source"), meta.get("page"))
                if key_ not in seen:
                    citations.append(meta)
                    seen.add(key_)
            # If it's Web, just cite the source
            elif src == "external_web":
                citations.append({"source": "Tavily Web Search", "page": "Web"})

        response = {
            "answer": result.get("final_answer", "No answer produced."),
            "confidence": 0.9 if result.get("tool_outputs") else 0.1,
            "citations": citations
        }
        
        answer_cache[key] = (now, response)
        return response

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Agent execution failed: {str(e)}"})

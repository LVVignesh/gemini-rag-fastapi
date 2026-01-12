import os
from time import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

from rag_store import ingest_documents, get_all_chunks, clear_database
from analytics import get_analytics
from agentic_rag_v2_graph import build_agentic_rag_v2_graph

# =========================================================
# ENV + MODEL
# =========================================================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"
MAX_FILE_SIZE = 50 * 1024 * 1024
CACHE_TTL = 300

# =========================================================
# APP
# =========================================================
app = FastAPI(title="Gemini RAG FastAPI (Agentic RAG v2+)")

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
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyCookie

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "secret")
COOKIE_NAME = "rag_auth"

api_key_cookie = APIKeyCookie(name=COOKIE_NAME, auto_error=False)

async def verify_admin(cookie: str = Depends(api_key_cookie)):
    if cookie != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return cookie

# =========================================================
# STATE
# =========================================================
agentic_graph = build_agentic_rag_v2_graph()
answer_cache: dict[str, tuple[float, dict]] = {}

# =========================================================
# MODELS
# =========================================================
class PromptRequest(BaseModel):
    prompt: str

class LoginRequest(BaseModel):
    password: str

# =========================================================
# ROUTES
# =========================================================
@app.post("/login")
def login(data: LoginRequest):
    if data.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    
    response = JSONResponse(content={"message": "Logged in"})
    response.set_cookie(key=COOKIE_NAME, value=data.password, httponly=True)
    return response

@app.get("/me")
def me(user: str = Depends(verify_admin)):
    return {"status": "authenticated"}

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/analytics", dependencies=[Depends(verify_admin)])
def analytics():
    return get_analytics()

# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------
@app.post("/upload", dependencies=[Depends(verify_admin)])
async def upload(files: list[UploadFile] = File(...)):
    for file in files:
        ext = file.filename.split(".")[-1].lower()
        if ext not in ["pdf", "txt"]:
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
    # 🟦 SUMMARY (BYPASS AGENT)
    # ==========================
    if "summary" in key or "summarize" in key:
        chunks = get_all_chunks(limit=80)
        context = "\n\n".join(c["text"] for c in chunks)

        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(
            f"Summarize the following content clearly:\n\n{context}"
        )

        response = {
            "answer": resp.text,
            "confidence": 0.95,
            "citations": []
        }

        answer_cache[key] = (now, response)
        return response

    # ==========================
    # 🟩 AGENTIC RAG (LLM + EVALUATION)
    # ==========================
    result = agentic_graph.invoke({
        "query": query,
        "refined_query": "",
        "decision": "",
        "retrieved_chunks": [],
        "retrieval_quality": "",
        "retries": 0,
        "answer": None,
        "confidence": 0.0,
        "answer_known": False
    })

    response = {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "citations": list({
            (c["metadata"]["source"], c["metadata"]["page"]): c["metadata"]
            for c in result.get("retrieved_chunks", [])
        }.values())
    }

    answer_cache[key] = (now, response)
    return response

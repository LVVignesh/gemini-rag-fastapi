import os
from time import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

from rag_store import ingest_documents, search_knowledge

# -----------------------
# Setup
# -----------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(
    title="Gemini RAG FastAPI",
    docs_url="/docs",
    redoc_url="/redoc"
)

# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Frontend
# -----------------------
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# -----------------------
# Cache (protect quota)
# -----------------------
CACHE_TTL = 300  # seconds
answer_cache = {}

# -----------------------
# Models
# -----------------------
class PromptRequest(BaseModel):
    prompt: str

# -----------------------
# Routes
# -----------------------

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

# -----------------------
# Upload
# -----------------------
@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    try:
        chunks = ingest_documents(files)
        return {"message": f"Indexed {chunks} chunks from {len(files)} file(s)."}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# -----------------------
# Ask
# -----------------------
@app.post("/ask")
async def ask(data: PromptRequest):
    prompt_key = data.prompt.strip().lower()
    now = time()

    # 🔁 Cache
    if prompt_key in answer_cache:
        ts, cached = answer_cache[prompt_key]
        if now - ts < CACHE_TTL:
            return cached

    results = search_knowledge(data.prompt)
    if not results:
        response = {
            "answer": "I don't know based on the provided documents.",
            "confidence": 0.0,
            "citations": []
        }
        answer_cache[prompt_key] = (now, response)
        return response

    context = "\n\n".join(r["text"] for r in results)

    prompt = f"""
Answer strictly using the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{data.prompt}
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        llm_response = model.generate_content(prompt)

        response = {
            "answer": llm_response.text,
            "confidence": round(min(1.0, len(results) / 5), 2),
            "citations": [
                {"source": r["metadata"]["source"], "page": r["metadata"]["page"]}
                for r in results
            ]
        }

        answer_cache[prompt_key] = (now, response)
        return response

    except Exception as e:
        return JSONResponse(
            status_code=429,
            content={"error": "LLM quota exceeded. Please wait and retry."}
        )

# -----------------------
# Summarize
# -----------------------
@app.post("/summarize")
async def summarize():
    return await ask(PromptRequest(
        prompt="Summarize the uploaded documents in 5 concise bullet points."
    ))

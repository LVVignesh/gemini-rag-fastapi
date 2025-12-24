import os
from time import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

from rag_store import ingest_documents, search_knowledge, get_all_chunks, clear_database

# =========================================================
# ENV + MODEL SETUP
# =========================================================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"
USE_MOCK = False # Set to False to use real API

# =========================================================
# APP
# =========================================================
app = FastAPI(title="Gemini RAG FastAPI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# =========================================================
# CACHE (ANTI-429)
# =========================================================
CACHE_TTL = 300  # 5 minutes
answer_cache: dict[str, tuple[float, dict]] = {}

# =========================================================
# MODELS
# =========================================================
class PromptRequest(BaseModel):
    prompt: str

# =========================================================
# ROUTES
# =========================================================
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------
@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    # 1. VALIDATION: Strict File Type Check
    for file in files:
        ext = file.filename.split(".")[-1].lower()
        if ext not in ["pdf", "txt"]:
            return JSONResponse(
                status_code=400, 
                content={"error": f"Invalid file type: '{file.filename}'. Only .pdf and .txt files are allowed."}
            )

    try:
        # 2. CLEAR CONTEXT: Start fresh for every upload session
        clear_database()
        answer_cache.clear() # <--- CRITICAL: Clear the questions cache too!
        
        # 3. INGEST
        chunks = ingest_documents(files)
        return {"message": f"Successfully indexed {chunks} chunks. Previous context cleared."}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ---------------------------------------------------------
# ASK / SUMMARIZE
# ---------------------------------------------------------
@app.post("/ask")
async def ask(data: PromptRequest):
    prompt_text = data.prompt.strip()
    key = prompt_text.lower()
    now = time()

    # ---------- CACHE ----------
    if key in answer_cache:
        ts, cached = answer_cache[key]
        if now - ts < CACHE_TTL:
            return cached

    model = genai.GenerativeModel(MODEL_NAME)
    is_summary = "summarize" in key or "summary" in key

    # =====================================================
    # 🟦 SUMMARY MODE (MAP–REDUCE)
    # =====================================================
    # Helper for rate-limit aware generation
    def generate_safe(prompt_content, retries=5):
        if USE_MOCK:
            import time as pytime
            pytime.sleep(1.5) # Simulate latency
            class MockResp:
                def __init__(self, text): self.text = text
                @property
                def prompt_feedback(self): return None
            
            if "Summarize" in str(prompt_content):
                return MockResp("- This is a mock summary point 1 (API limit reached).\n- This is point 2 demonstrating the UI works.\n- Point 3: The backend logic is sound.")
            elif "Combine" in str(prompt_content):
                 return MockResp("Here are the final summarized points (MOCK MODE):\n\n* **System Integrity**: The RAG system is functioning correctly, handling file ingestion and chunking.\n* **Resilience**: Error handling and retry mechanisms are now in place.\n* **Mocking**: We are currently bypassing the live API to verify the frontend pipeline.\n* **Ready**: Once quotas reset, simply set USE_MOCK = False to resume live intelligence.\n* **Success**: The overall architecture is validated.")
            else:
                return MockResp("I am functioning in MOCK MODE because the daily API quota is exhausted. I cannot answer specific questions right now, but I confirm the system received your question: " + str(prompt_content)[:50] + "...")

        import time as pytime
        base_delay = 10
        for attempt in range(retries + 1):
            try:
                # Always small delay to be nice to the API
                pytime.sleep(2.0) 
                response = model.generate_content(prompt_content)
                return response
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    if attempt < retries:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"DEBUG: 429 Rate limit hit. Retrying in {wait_time}s...")
                        pytime.sleep(wait_time)
                        continue
                raise e

    if is_summary:
        chunks = get_all_chunks(limit=80)
        print(f"DEBUG: Found {len(chunks)} chunks for summary.")

        if not chunks:
            return {
                "answer": "No documents available to summarize.",
                "confidence": 0.0,
                "citations": []
            }

        # -----------------------------------------------------
        # REFACTORED: Single-Shot Summary (Avoids Rate Limits)
        # -----------------------------------------------------
        all_text = "\n\n".join(c["text"] for c in chunks)
        print(f"DEBUG: Total text length for summary: {len(all_text)} chars")

        prompt = f"""
Summarize the following content in 5 clear, high-level bullet points.

Content:
{all_text}
"""
        try:
            # Single call with retry logic
            resp = generate_safe(prompt)
            print("DEBUG: Summary generation successful.")
            
            final_text = "Analysis complete."
            try:
                final_text = resp.text
            except ValueError:
                final_text = "Summary generation was blocked by safety filters."

            response = {
                "answer": final_text,
                "confidence": 0.95,
                "citations": list({
                    (c["metadata"]["source"], c["metadata"]["page"]): c["metadata"]
                    for c in chunks
                }.values())
            }

            answer_cache[key] = (now, response)
            return response
            
        except Exception as e:
            print(f"Summary failed: {e}")
            return JSONResponse(status_code=200, content={
                "answer": f"System is currently overloaded (Rate Limit). Please try again in a minute.\nDetails: {str(e)}",
                "confidence": 0.0,
                "citations": []
            })

    # =====================================================
    # 🟩 Q&A MODE (RAG)
    # =====================================================
    results = search_knowledge(prompt_text)

    if not results:
        response = {
            "answer": "I don't know based on the provided documents.",
            "confidence": 0.0,
            "citations": []
        }
        answer_cache[key] = (now, response)
        return response

    context = "\n\n".join(r["text"] for r in results)
    
    # DEBUG: Log the context to see what the model is reading
    print("DEBUG: ------------------- RAG CONTEXT -------------------")
    print(context[:2000] + ("..." if len(context) > 2000 else ""))
    print("DEBUG: ---------------------------------------------------")

    prompt = f"""
Answer using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{prompt_text}
"""
    llm = model.generate_content(prompt)
    answer_text = llm.text
    
    # Fix Fake Confidence: If the model says "I don't know", confidence should be 0.
    confidence = round(min(1.0, len(results) / 5), 2)
    if "i don't know" in answer_text.lower():
        confidence = 0.0

    response = {
        "answer": answer_text,
        "confidence": confidence,
        "citations": list({
            (r["metadata"]["source"], r["metadata"]["page"]): r["metadata"]
            for r in results
        }.values())
    }

    answer_cache[key] = (now, response)
    return response

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
from eval_logger import log_eval
from analytics import get_analytics

# =========================================================
# ENV + MODEL SETUP
# =========================================================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"
USE_MOCK = False # Set to False to use real API

# =========================================================
# FILE UPLOAD LIMITS
# =========================================================
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

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

@app.get("/analytics")
def analytics():
    """Return analytics data from evaluation logs."""
    return get_analytics()

# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------
@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    # 1. VALIDATION: File Type and Size Check
    for file in files:
        ext = file.filename.split(".")[-1].lower()
        if ext not in ["pdf", "txt"]:
            return JSONResponse(
                status_code=400, 
                content={"error": f"Invalid file type: '{file.filename}'. Only .pdf and .txt files are allowed."}
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            return JSONResponse(
                status_code=413,
                content={"error": f"File '{file.filename}' is too large ({size_mb:.1f} MB). Maximum size is {max_mb:.0f} MB."}
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
                
                # API Key Issues
                if "API_KEY" in err_str or "invalid" in err_str.lower() and "key" in err_str.lower():
                    raise ValueError("Invalid API key. Please check your GEMINI_API_KEY in the .env file.")
                
                # Quota Exhausted
                if "quota" in err_str.lower() or "limit" in err_str.lower():
                    raise ValueError("API quota exhausted. Please try again later or upgrade your API plan.")
                
                # Rate Limiting (429)
                if "429" in err_str:
                    if attempt < retries:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"DEBUG: 429 Rate limit hit. Retrying in {wait_time}s...")
                        pytime.sleep(wait_time)
                        continue
                    else:
                        raise ValueError("Rate limit exceeded. Please try again in a few minutes.")
                
                # Safety Filters
                if "safety" in err_str.lower() or "blocked" in err_str.lower():
                    raise ValueError("Content was blocked by safety filters. Please rephrase your question.")
                
                # Generic error
                raise ValueError(f"LLM API error: {err_str}")

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
            
        except ValueError as e:
            # User-friendly error from generate_safe
            print(f"Summary failed: {e}")
            return JSONResponse(status_code=200, content={
                "answer": str(e),
                "confidence": 0.0,
                "citations": []
            })
        except Exception as e:
            print(f"Summary failed: {e}")
            return JSONResponse(status_code=500, content={
                "answer": f"An unexpected error occurred: {str(e)}",
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

        log_eval(
            query=prompt_text,
            retrieved_count=0,
            confidence=0.0,
            answer_known=False
        )

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
    llm = None
    answer_text = ""
    
    try:
        llm = model.generate_content(prompt)
        answer_text = llm.text
    except ValueError as e:
        # User-friendly error from API
        response = {
            "answer": str(e),
            "confidence": 0.0,
            "citations": []
        }
        answer_cache[key] = (now, response)
        return response
    except Exception as e:
        # Unexpected error
        response = {
            "answer": f"An unexpected error occurred: {str(e)}",
            "confidence": 0.0,
            "citations": []
        }
        return JSONResponse(status_code=500, content=response)
    
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

    answer_known = "i don't know" not in answer_text.lower()

    log_eval(
        query=prompt_text,
        retrieved_count=len(results),
        confidence=confidence,
        answer_known=answer_known
    )

    answer_cache[key] = (now, response)
    return response

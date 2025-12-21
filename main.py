import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from rag_store import search_knowledge

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="AI RAG Backend with Gemini")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "AI backend is running 🚀"}

@app.post("/ask")
async def ask(data: PromptRequest):
    results = search_knowledge(data.prompt)

    if not results:
        return {
            "answer": "I don't know based on the provided documents.",
            "confidence": 0.0,
            "citations": []
        }

    # -------- Context
    context_text = "\n".join(r["text"] for r in results)

    prompt = f"""
Answer the question strictly using the context.
If unsure, say "I don't know".

Question:
{data.prompt}

Context:
{context_text}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    # -------- Confidence scoring
    avg_distance = sum(r["distance"] for r in results) / len(results)

    if avg_distance < 0.6:
        confidence = 0.9
    elif avg_distance < 1.2:
        confidence = 0.7
    else:
        confidence = 0.4

    # -------- Citations
    citations = []
    seen = set()
    for r in results:
        key = (r["metadata"]["source"], r["metadata"]["page"])
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": r["metadata"]["source"],
                "page": r["metadata"]["page"]
            })

    return {
        "answer": response.text,
        "confidence": round(confidence, 2),
        "citations": citations
    }

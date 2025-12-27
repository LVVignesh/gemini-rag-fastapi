📄 Gemini RAG Assistant (FastAPI)

A production-style Retrieval-Augmented Generation (RAG) application built with FastAPI, Google Gemini, and FAISS, capable of answering questions and generating summaries from uploaded documents (PDF/TXT) with grounded responses, citations, and confidence scoring.

This project evolved iteratively from a simple FastAPI API into a robust, end-to-end AI system, covering real-world challenges like PDF ingestion, vector search, LLM rate limits, and Git hygiene.

🚀 Features

📤 Upload PDF and TXT documents

🔍 Retrieval-Augmented Q&A using FAISS

🧠 Grounded answers powered by Google Gemini

📝 Document summarization using the same RAG pipeline

📚 Page-level citations for transparency

📊 Confidence scoring based on retrieval strength

⚡ Async FastAPI backend (non-blocking I/O)

🧪 Mock mode for UI testing when API quota is exhausted

🧹 Clean Git history with generated files ignored

🏗️ Architecture Overview
Frontend (HTML + JS)
        ↓
FastAPI Backend
        ↓
Document Ingestion (PDF / TXT)
        ↓
Embeddings (SentenceTransformers)
        ↓
FAISS Vector Store
        ↓
Retriever (Top-K Similarity Search)
        ↓
Prompt Assembly
        ↓
Google Gemini LLM
        ↓
Grounded Response + Citations + Confidence

🧠 Key Concepts Learned
1. FastAPI Fundamentals

GET and POST endpoints

Request/response lifecycle

Input validation using Pydantic models

Async endpoints for non-blocking LLM calls

2. Real LLM Integration

Secure API key handling via environment variables

Structured prompts for strict input/output control

Handling rate limits and safety-filtered responses

Graceful error handling and fallbacks

3. Retrieval-Augmented Generation (RAG)

Why LLMs alone are unreliable for factual answers

Converting documents into embeddings

Similarity search using FAISS

Injecting retrieved context into prompts for grounded answers

4. Document Ingestion Reality

Not all PDFs are text-based

Scanned/screenshot PDFs require OCR

RAG quality depends on data quality

Silent failures often come from missing extractable text

5. Summarization vs Q&A

Summarization is not the same as question answering

Naive summarization can fail due to token limits

Simpler pipelines are often more stable for small documents

6. Confidence & Trust

Confidence score reflects retrieval strength, not “truth”

Honest responses (“I don’t know”) improve trust

Citations are critical for verification

7. Engineering Best Practices

Start with a stable baseline before adding complexity

Mock LLM responses during development

Handle API quotas and rate limits explicitly

Keep generated files out of Git (.gitignore)

Resolve Git branch divergence safely using rebase

🛠️ Tech Stack
Backend

Python

FastAPI

FAISS

SentenceTransformers

Google Gemini API

PyPDF

python-dotenv

Frontend

HTML

CSS

Vanilla JavaScript (Fetch API)

Platform & Tooling

VS Code

Git & GitHub

Hugging Face Spaces (deployment)

Virtual Environments (venv)

⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2️⃣ Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set environment variables

Create a .env file:

GEMINI_API_KEY=your_api_key_here

5️⃣ Run the server
uvicorn main:app --reload


Open in browser:

http://127.0.0.1:8000

🧪 Mock Mode (Development)

To test the UI without consuming Gemini API quota:

Enable mock responses in main.py

Allows frontend and flow testing without LLM calls

This mirrors real production workflows.

⚠️ Known Limitations

Scanned/image-based PDFs are not supported (OCR required)

Confidence score is heuristic, not a guarantee of correctness

Large documents may require map-reduce summarization (future work)

🔮 Future Improvements

OCR integration for scanned PDFs

Chunk-based retrieval for large documents

Streaming LLM responses

Evaluation metrics for answer quality

Multi-document cross-referencing

Auth & user-specific document stores

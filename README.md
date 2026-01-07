📄 Gemini RAG Backend System (FastAPI)

Production-grade Retrieval-Augmented Generation (RAG) backend built with FastAPI, FAISS (ANN), and Google Gemini — featuring hybrid retrieval, HNSW indexing, cross-encoder reranking, evaluation logging, and analytics.

This repository demonstrates how modern AI backend systems are actually built in industry.

🚀 What This Project Is

This is a full RAG backend system that:

Ingests large PDF/TXT documents

Builds vector indexes with Approximate Nearest Neighbor (ANN) search

Answers questions using grounded LLM responses

Tracks confidence, known/unknown answers, and usage analytics

Supports production constraints (file limits, caching, logging)

The project evolved from RAG v1 → RAG v2, adding real-world scalability and observability.

✨ Key Features (RAG v2)

📥 Document Ingestion

Upload PDF and TXT files

Sentence-aware chunking with overlap

Page-level metadata for citations

🔍 Retrieval (Hybrid + ANN)

FAISS HNSW ANN index for scalable similarity search

Cosine similarity via normalized embeddings

Keyword boosting for lexical relevance

🧠 Reranking (Quality Boost)

Cross-Encoder (ms-marco-MiniLM) reranking

Improves relevance beyond raw vector similarity

Mimics production search stacks (retrieve → rerank)

🤖 LLM Generation

Google Gemini 2.5 Flash

Strict grounding: answers only from retrieved context

Honest fallback: "I don't know" when unsupported

📊 Evaluation & Monitoring

Logs every query:

retrieved chunk count

confidence score

known vs unknown answers

JSONL logs for offline analysis

Built-in analytics dashboard

📈 Analytics Dashboard

Total queries

Knowledge rate

Average confidence

Unknown query tracking

Recent query history

Dark / Light mode UI

🛡️ Production Safeguards

File upload size limits (configurable)

API quota handling

Caching to reduce LLM calls

Clean error handling

Persistent vector store


🏗️ System Architecture


Frontend (HTML / JS)
        ↓

FastAPI Backend
        ↓

Document Ingestion (PDF / TXT)
        ↓

Sentence Chunking + Metadata
        ↓

Embeddings (SentenceTransformers)
        ↓

FAISS ANN Index (HNSW)
        ↓

Hybrid Retrieval (Vector + Keyword)
        ↓

Cross-Encoder Reranking
        ↓

Prompt Assembly
        ↓

Google Gemini LLM
        ↓

Answer + Confidence + Citations
        ↓

Evaluation Logging + Analytics



🧠 Core Concepts Demonstrated

Retrieval-Augmented Generation (RAG)

Why pure LLMs hallucinate

How grounding fixes factual accuracy

Vector search vs keyword search

Hybrid retrieval strategies

Approximate Nearest Neighbor (ANN)

Why brute-force search fails at scale

HNSW indexing for fast similarity search

efConstruction vs efSearch trade-offs

Reranking

Why top-K vectors ≠ best answers

Cross-encoder reranking for relevance

Industry-standard retrieval pipelines

Evaluation & Observability

Measuring known vs unknown

Confidence as a heuristic, not truth

Logging for iterative improvement

Analytics-driven RAG tuning

Real Backend Engineering

API limits & retries

Persistent storage

Clean Git hygiene

Incremental system evolution


🛠️ Tech Stack

Backend

Python

FastAPI

FAISS (HNSW ANN)

SentenceTransformers

Cross-Encoder (MS MARCO)

Google Gemini API

PyPDF

python-dotenv

Frontend

HTML

CSS

Vanilla JavaScript (Fetch API)

Tooling & Platform

VS Code

Git & GitHub

Docker

Hugging Face Spaces (deployment)

Virtual Environments (venv)



⚙️ Setup & Run Locally

1️⃣ Clone Repository

git clone https://github.com/LVVignesh/gemini-rag-fastapi.git

cd gemini-rag-fastapi

2️⃣ Create Virtual Environment

python -m venv venv

venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Configure Environment Variables

GEMINI_API_KEY=your_api_key_here

5️⃣ Run Server

uvicorn main:app --reload



⚠️ Known Limitations

Scanned/image-only PDFs require OCR (not included)

Confidence score is heuristic

Very large corpora may require:

batch ingestion

sharding

background workers



🚀 Live Demo

👉 Hugging Face Spaces
https://huggingface.co/spaces/lvvignesh2122/Gemini-Rag-Fastapi-Pro

📜 License

MIT License


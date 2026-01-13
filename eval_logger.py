import json
from time import time

LOG_FILE = "rag_eval_logs.jsonl"

def log_eval(
    query: str,
    retrieved_count: int,
    confidence: float,
    answer_known: bool,
    source_type: str = "internal_pdf"  # Added source_type
):
    record = {
        "timestamp": time(),
        "query": query,
        "retrieved_count": retrieved_count,
        "confidence": confidence,
        "answer_known": answer_known,
        "source_type": source_type
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

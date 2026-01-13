import json
from time import time

LOG_FILE = "rag_eval_logs.jsonl"

def log_eval(
    query: str,
    retrieved_count: int,
    confidence: float,
    answer_known: bool,
    source_type: str = "internal_pdf",
    final_answer: str = "",
    context_list: list = None
):
    if context_list is None:
        context_list = []
        
    record = {
        "timestamp": time(),
        "query": query,
        "retrieved_count": retrieved_count,
        "confidence": confidence,
        "answer_known": answer_known,
        "source_type": source_type,
        "final_answer": final_answer,
        "context_list": context_list
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

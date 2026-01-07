import json
from collections import defaultdict
from datetime import datetime

LOG_FILE = "rag_eval_logs.jsonl"

def get_analytics():
    """Parse logs and return analytics data."""
    total = 0
    known_count = 0
    unknown_count = 0
    conf_sum = 0.0
    queries = []
    unknown_queries = []
    
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                total += 1
                data = json.loads(line)
                
                if data.get("answer_known"):
                    known_count += 1
                else:
                    unknown_count += 1
                    unknown_queries.append({
                        "query": data.get("query"),
                        "timestamp": datetime.fromtimestamp(data.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M")
                    })
                    
                conf_sum += data.get("confidence", 0.0)
                queries.append({
                    "query": data.get("query"),
                    "confidence": data.get("confidence", 0.0),
                    "answer_known": data.get("answer_known", False)
                })
        
        if total == 0:
            return {
                "total_queries": 0,
                "knowledge_rate": 0,
                "avg_confidence": 0,
                "known_count": 0,
                "unknown_count": 0,
                "recent_unknown": [],
                "top_queries": []
            }
        
        knowledge_rate = (known_count / total) * 100
        avg_confidence = conf_sum / total
        
        # Get top 10 most recent queries
        top_queries = queries[-10:][::-1]  # Last 10, reversed
        
        # Get recent unknown queries (last 5)
        recent_unknown = unknown_queries[-5:][::-1]
        
        return {
            "total_queries": total,
            "knowledge_rate": round(knowledge_rate, 1),
            "avg_confidence": round(avg_confidence, 2),
            "known_count": known_count,
            "unknown_count": unknown_count,
            "recent_unknown": recent_unknown,
            "top_queries": top_queries
        }
    
    except FileNotFoundError:
        return {
            "total_queries": 0,
            "knowledge_rate": 0,
            "avg_confidence": 0,
            "known_count": 0,
            "unknown_count": 0,
            "recent_unknown": [],
            "top_queries": []
        }

import json
from collections import Counter

LOG_FILE = "rag_eval_logs.jsonl"

def analyze():
    print(f"--- Analyzing {LOG_FILE} ---\n")
    
    total = 0
    known_count = 0
    unknown_count = 0
    conf_sum = 0.0

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                total += 1
                data = json.loads(line)
                
                if data.get("answer_known"):
                    known_count += 1
                else:
                    unknown_count += 1
                    
                conf_sum += data.get("confidence", 0.0)

        if total == 0:
            print("No logs found.")
            return

        print(f"Total Queries:      {total}")
        print(f"Answered (Known):   {known_count}")
        print(f"Unanswered (False): {unknown_count}")
        print(f"Average Confidence: {conf_sum / total:.2f}")
        print("-" * 30)
        
        accuracy = (known_count / total) * 100
        print(f"System 'Knowledge Rate': {accuracy:.1f}%")

    except FileNotFoundError:
        print(f"Log file {LOG_FILE} not found.")

if __name__ == "__main__":
    analyze()

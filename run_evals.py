import json
import os
from llm_utils import generate_with_retry
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = "rag_eval_logs.jsonl"
MODEL_NAME = "gemini-2.5-flash"
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ GEMINI_API_KEY not found in env.")
    exit(1)

genai.configure(api_key=API_KEY)

def calculate_faithfulness(answer, contexts):
    """
    Score 0.0 to 1.0
    Measure: Is the answer derived *only* from the context?
    """
    if not contexts: return 0.0
    
    context_text = "\n".join(contexts)
    prompt = f"""
    You are an AI Judge.
    Rate the 'Faithfulness' of the Answer to the Context on a scale of 0.0 to 1.0.
    1.0 = Answer is strictly derived from Context.
    0.0 = Answer contains hallucinations or info not in Context.
    
    Context: {context_text[:3000]}
    
    Answer: {answer}
    
    Return ONLY a single float number (e.g. 0.9).
    """
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        resp = model.generate_content(prompt)
        score = float(resp.text.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5 # Default on error

def calculate_relevancy(query, answer):
    """
    Score 0.0 to 1.0
    Measure: Does the answer directly address the query?
    """
    prompt = f"""
    You are an AI Judge.
    Rate the 'Relevancy' of the Answer to the Query on a scale of 0.0 to 1.0.
    1.0 = Answer directly addresses the query.
    0.0 = Answer is unrelated or ignores the user.
    
    Query: {query}
    Answer: {answer}
    
    Return ONLY a single float number (e.g. 0.9).
    """
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        resp = model.generate_content(prompt)
        score = float(resp.text.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5

def run_audit():
    if not os.path.exists(LOG_FILE):
        print(f"No log file found at {LOG_FILE}")
        return

    print(f"📊 Running Post-Hoc Audit on {LOG_FILE}...\n")
    print(f"{'Query':<30} | {'Faithful':<10} | {'Relevancy':<10}")
    print("-" * 60)
    
    total_f = 0
    total_r = 0
    count = 0
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Skip legacy logs without final answer
                if "final_answer" not in data or not data["final_answer"]:
                    continue
                    
                q = data["query"]
                a = data["final_answer"]
                c = data.get("context_list", [])
                
                f_score = calculate_faithfulness(a, c)
                r_score = calculate_relevancy(q, a)
                
                print(f"{q[:30]:<30} | {f_score:.2f}       | {r_score:.2f}")
                
                total_f += f_score
                total_r += r_score
                count += 1
            except Exception as e:
                pass # Skip bad lines
                
    if count > 0:
        print("-" * 60)
        print(f"\n✅ Audit Complete.")
        print(f"Average Faithfulness: {total_f/count:.2f}")
        print(f"Average Relevancy:    {total_r/count:.2f}")
    else:
        print("\n⚠️ No complete records found to audit. Ask some questions first!")

if __name__ == "__main__":
    run_audit()

from typing import TypedDict, List, Optional
import google.generativeai as genai
from langgraph.graph import StateGraph, END

from rag_store import search_knowledge
from eval_logger import log_eval

MODEL_NAME = "gemini-2.5-flash"


# ===============================
# STATE
# ===============================
class RAGState(TypedDict):
    query: str
    retrieved_chunks: List[dict]
    answer: Optional[str]
    confidence: float
    answer_known: bool


# ===============================
# RETRIEVAL NODE (TOOL)
# ===============================
def retrieve_node(state: RAGState) -> RAGState:
    results = search_knowledge(state["query"])
    return {
        **state,
        "retrieved_chunks": results
    }


# ===============================
# ANSWER NODE
# ===============================
def answer_node(state: RAGState) -> RAGState:
    if not state["retrieved_chunks"]:
        return no_answer_node(state)

    context = "\n\n".join(c["text"] for c in state["retrieved_chunks"])

    prompt = f"""
Answer using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{state["query"]}
"""

    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    answer_text = resp.text

    confidence = min(1.0, len(state["retrieved_chunks"]) / 5)
    answer_known = "i don't know" not in answer_text.lower()

    log_eval(
        query=state["query"],
        retrieved_count=len(state["retrieved_chunks"]),
        confidence=confidence,
        answer_known=answer_known
    )

    return {
        **state,
        "answer": answer_text,
        "confidence": confidence,
        "answer_known": answer_known
    }


# ===============================
# NO ANSWER NODE
# ===============================
def no_answer_node(state: RAGState) -> RAGState:
    log_eval(
        query=state["query"],
        retrieved_count=0,
        confidence=0.0,
        answer_known=False
    )

    return {
        **state,
        "answer": "I don't know based on the provided documents.",
        "confidence": 0.0,
        "answer_known": False
    }


# ===============================
# GRAPH BUILDER
# ===============================
def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("no_answer", no_answer_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("no_answer", END)

    return graph.compile()

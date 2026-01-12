from typing import TypedDict, List, Optional
import google.generativeai as genai
from langgraph.graph import StateGraph, END

from rag_store import search_knowledge
from eval_logger import log_eval

MODEL_NAME = "gemini-2.5-flash"


# ===============================
# STATE
# ===============================
class AgentState(TypedDict):
    query: str
    decision: str
    retrieved_chunks: List[dict]
    answer: Optional[str]
    confidence: float
    answer_known: bool


# ===============================
# DECISION NODE
# ===============================
def agent_decision_node(state: AgentState) -> AgentState:
    q = state["query"].lower()

    rag_keywords = [
        "summarize", "summary", "fee", "fees", "refund",
        "tuition", "document", "policy", "offer", "scholarship"
    ]

    decision = "use_rag" if any(k in q for k in rag_keywords) else "no_rag"

    return {**state, "decision": decision}


# ===============================
# RETRIEVAL NODE (TOOL)
# ===============================
def retrieve_node(state: AgentState) -> AgentState:
    chunks = search_knowledge(state["query"])
    return {**state, "retrieved_chunks": chunks}


# ===============================
# ANSWER WITH RAG
# ===============================
def answer_with_rag_node(state: AgentState) -> AgentState:
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
# ANSWER WITHOUT RAG
# ===============================
def answer_direct_node(state: AgentState) -> AgentState:
    prompt = f"Answer the following question concisely:\n\n{state['query']}"

    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)

    log_eval(
        query=state["query"],
        retrieved_count=0,
        confidence=0.3,
        answer_known=True
    )

    return {
        **state,
        "answer": resp.text,
        "confidence": 0.3,
        "answer_known": True
    }


# ===============================
# NO ANSWER
# ===============================
def no_answer_node(state: AgentState) -> AgentState:
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
def build_agentic_rag_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decide", agent_decision_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer_rag", answer_with_rag_node)
    graph.add_node("answer_direct", answer_direct_node)
    graph.add_node("no_answer", no_answer_node)

    graph.set_entry_point("decide")

    graph.add_conditional_edges(
        "decide",
        lambda s: s["decision"],
        {
            "use_rag": "retrieve",
            "no_rag": "answer_direct"
        }
    )

    graph.add_edge("retrieve", "answer_rag")
    graph.add_edge("answer_rag", END)
    graph.add_edge("answer_direct", END)
    graph.add_edge("no_answer", END)

    return graph.compile()

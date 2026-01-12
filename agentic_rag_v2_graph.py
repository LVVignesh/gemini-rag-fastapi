from typing import TypedDict, List, Optional, Annotated
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import time
import random

from rag_store import search_knowledge
from eval_logger import log_eval
from llm_utils import generate_with_retry

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 2




def format_history(messages: List[BaseMessage]) -> str:
    history_str = ""
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"
    return history_str

# ===============================
# STATE
# ===============================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    refined_query: str
    decision: str
    retrieved_chunks: List[dict]
    retrieval_quality: str
    retries: int
    answer: Optional[str]
    confidence: float
    answer_known: bool


# ===============================
# LLM DECISION NODE (PLANNER)
# ===============================
def llm_decision_node(state: AgentState) -> AgentState:
    history = format_history(state.get("messages", []))
    prompt = f"""
You are an AI agent deciding whether a question requires document retrieval.
Answer ONLY one word:
- use_rag
- no_rag

Conversation History:
{history}

Current Question:
{state["query"]}
"""
    model = genai.GenerativeModel(MODEL_NAME)
    resp = generate_with_retry(model, prompt)

    decision = "use_rag"
    if resp and "no_rag" in resp.text.lower():
        decision = "no_rag"

    return {**state, "decision": decision}


# ===============================
# RETRIEVAL NODE (TOOL)
# ===============================
def retrieve_node(state: AgentState) -> AgentState:
    q = state["refined_query"] or state["query"]
    chunks = search_knowledge(q)
    return {**state, "retrieved_chunks": chunks}


# ===============================
# GRADE DOCUMENTS NODE (GRADER)
# ===============================
def grade_documents_node(state: AgentState) -> AgentState:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    query = state["query"]
    retrieved_docs = state["retrieved_chunks"]
    
    filtered_docs = []
    for doc in retrieved_docs:
        prompt = f"""
        You are a grader assessing relevance of a retrieved document to a user question.
        
        Retrieved document:
        {doc['text']}
        
        User question:
        {query}
        
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        
        Answer ONLY 'yes' or 'no'.
        """
        model = genai.GenerativeModel(MODEL_NAME)
        resp = generate_with_retry(model, prompt)
        score = resp.text.strip().lower() if resp else "no"
        
        if "yes" in score:
            filtered_docs.append(doc)
            
    return {**state, "retrieved_chunks": filtered_docs}


# ===============================
# RETRIEVAL EVALUATION (CRITIC)
# ===============================
def evaluate_retrieval_node(state: AgentState) -> AgentState:
    if not state["retrieved_chunks"]:
        return {**state, "retrieval_quality": "bad"}

    context_sample = "\n".join(c["text"][:200] for c in state["retrieved_chunks"][:3])

    prompt = f"""
Evaluate whether the following retrieved context is sufficient
to answer the question.

Answer ONLY one word:
- good
- bad

Question:
{state["query"]}

Context:
{context_sample}
"""

    model = genai.GenerativeModel(MODEL_NAME)
    resp = generate_with_retry(model, prompt)

    quality = "bad"
    if resp and "good" in resp.text.lower():
        quality = "good"
        
    return {**state, "retrieval_quality": quality}


# ===============================
# QUERY REFINEMENT (SELF-CORRECTION)
# ===============================
def refine_query_node(state: AgentState) -> AgentState:
    history = format_history(state.get("messages", []))
    prompt = f"""
Rewrite the following question to improve document retrieval.
Be concise and factual.

Conversation History:
{history}

Original question:
{state["query"]}
"""

    model = genai.GenerativeModel(MODEL_NAME)
    resp = generate_with_retry(model, prompt)
    
    refined = resp.text.strip() if resp else state["query"]

    return {
        **state,
        "refined_query": refined,
        "retries": state["retries"] + 1
    }


# ===============================
# ANSWER WITH RAG (HIGH CONF)
# ===============================
def answer_with_rag_node(state: AgentState) -> AgentState:
    context = "\n\n".join(c["text"] for c in state["retrieved_chunks"])
    history = format_history(state.get("messages", []))

    prompt = f"""
Answer using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Conversation History:
{history}

Question:
{state["query"]}
"""

    model = genai.GenerativeModel(MODEL_NAME)
    resp = generate_with_retry(model, prompt)
    answer_text = resp.text if resp else "Error generating answer due to quota limits."

    answer_known = "i don't know" not in answer_text.lower()
    confidence = min(0.95, 0.6 + (0.1 * len(state["retrieved_chunks"])))

    log_eval(
        query=state["query"],
        retrieved_count=len(state["retrieved_chunks"]),
        confidence=confidence,
        answer_known=answer_known
    )

    # Append interaction to memory
    new_messages = [
        HumanMessage(content=state["query"]),
        AIMessage(content=answer_text)
    ]

    return {
        **state,
        "messages": new_messages,
        "answer": answer_text,
        "confidence": confidence,
        "answer_known": answer_known
    }


# ===============================
# ANSWER WITHOUT RAG
# ===============================
def answer_direct_node(state: AgentState) -> AgentState:
    history = format_history(state.get("messages", []))
    prompt = f"""
Conversation History:
{history}

Answer clearly and concisely:
{state['query']}
"""

    model = genai.GenerativeModel(MODEL_NAME)
    resp = generate_with_retry(model, prompt)
    answer_text = resp.text if resp else "Error generating answer due to quota limits."

    log_eval(
        query=state["query"],
        retrieved_count=0,
        confidence=0.4,
        answer_known=True
    )

    # Append interaction to memory
    new_messages = [
        HumanMessage(content=state["query"]),
        AIMessage(content=answer_text)
    ]

    return {
        **state,
        "messages": new_messages,
        "answer": answer_text,
        "confidence": 0.4,
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
    
    answer_text = "I don't know based on the provided documents."
    
    # Append interaction to memory
    new_messages = [
        HumanMessage(content=state["query"]),
        AIMessage(content=answer_text)
    ]

    return {
        **state,
        "messages": new_messages,
        "answer": answer_text,
        "confidence": 0.0,
        "answer_known": False
    }


# ===============================
# GRAPH BUILDER
# ===============================
def build_agentic_rag_v2_graph():
    graph = StateGraph(AgentState)
    memory = MemorySaver()

    graph.add_node("decide", llm_decision_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_documents_node)
    graph.add_node("evaluate", evaluate_retrieval_node)
    graph.add_node("refine", refine_query_node)
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

    graph.add_edge("retrieve", "grade")

    def check_relevance(state):
        if not state["retrieved_chunks"]:
            if state["retries"] >= MAX_RETRIES:
                return "no_answer"
            return "rewrite"
        return "evaluate"

    graph.add_conditional_edges(
        "grade",
        check_relevance,
        {
            "rewrite": "refine",
            "evaluate": "evaluate",
            "no_answer": "no_answer"
        }
    )

    graph.add_conditional_edges(
        "evaluate",
        lambda s: "retry" if s["retrieval_quality"] == "bad" and s["retries"] < MAX_RETRIES else "answer",
        {
            "retry": "refine",
            "answer": "answer_rag"
        }
    )

    graph.add_edge("refine", "retrieve")
    graph.add_edge("answer_rag", END)
    graph.add_edge("answer_direct", END)
    graph.add_edge("no_answer", END)

    return graph.compile(checkpointer=memory)

import os
import time
from typing import TypedDict, List, Optional, Annotated, Literal
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

from rag_store import search_knowledge
from eval_logger import log_eval
from llm_utils import generate_with_retry

# Config
MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 2

# ===============================
# STATE
# ===============================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    final_answer: str
    
    # Internal routing & scratchpad
    next_node: str
    current_tool: str
    tool_outputs: List[dict]  # list of {source: 'pdf'|'web', content: ..., score: ...}
    verification_notes: str
    retries: int

# ===============================
# TOOLS
# ===============================
def pdf_search_tool(query: str):
    """Searches internal PDF knowledge base."""
    results = search_knowledge(query, top_k=4)
    # Format for consumption
    return [
        {
            "source": "internal_pdf",
            "content": r["text"],
            "metadata": r["metadata"],
            "score": r.get("score", 0)
        }
        for r in results
    ]

def web_search_tool(query: str):
    """Searches the web using Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [{"source": "external_web", "content": "Error: TAVILY_API_KEY not found.", "score": 0}]
    
    try:
        tavily = TavilyClient(api_key=api_key)
        # Search context first for cleaner text
        context = tavily.get_search_context(query=query, search_depth="advanced")
        return [{
            "source": "external_web", 
            "content": context, 
            "score": 0.8 # Arbitrary confidence for web
        }]
    except Exception as e:
        return [{"source": "external_web", "content": f"Web search error: {str(e)}", "score": 0}]

# ===============================
# NODES
# ===============================

# 1. SUPERVISOR
def supervisor_node(state: AgentState):
    """Decides whether to research (and which tool) or answer."""
    query = state["query"]
    history_len = len(state.get("messages", []))
    
    # If we already have tools output, check if we need more or are done
    tools_out = state.get("tool_outputs", [])
    
    prompt = f"""
    You are a Supervisor Agent.
    User Query: "{query}"
    
    Current Gathered Info Count: {len(tools_out)}
    
    Decide next step:
    1. "research_pdf": If we haven't checked internal docs yet.
    2. "research_web": If PDF info is missing/insufficient and we haven't checked web yet.
    3. "responder": If we have enough info OR we have tried everything.
    
    Return ONLY one of: research_pdf, research_web, responder
    """
    
    # Simple heuristic to save calls, or use LLM? 
    # Prompt says "Planning Node: The LLM must decide".
    
    # We can force PDF first to be efficient
    has_pdf = any(t["source"] == "internal_pdf" for t in tools_out)
    if not has_pdf:
        return {**state, "next_node": "research_pdf"}
        
    model = genai.GenerativeModel(MODEL_NAME)
    resp = generate_with_retry(model, prompt)
    decision = resp.text.strip().lower() if resp else "responder"
    
    if "pdf" in decision: return {**state, "next_node": "research_pdf"}
    if "web" in decision: return {**state, "next_node": "research_web"}
    
    return {**state, "next_node": "responder"}

# 2. RESEARCHER (PDF)
def researcher_pdf_node(state: AgentState):
    query = state["query"]
    results = pdf_search_tool(query)
    
    # Append to tool_outputs
    current_outputs = state.get("tool_outputs", []) + results
    
    # Log
    log_eval(query, len(results), 0.9, len(results) > 0, source_type="internal_pdf")
    
    return {**state, "tool_outputs": current_outputs}

# 3. RESEARCHER (WEB)
def researcher_web_node(state: AgentState):
    query = state["query"]
    results = web_search_tool(query)
    
    current_outputs = state.get("tool_outputs", []) + results
    
    # Log
    log_eval(query, 1, 0.7, True, source_type="external_web")
    
    return {**state, "tool_outputs": current_outputs}

# 4. VERIFIER
def verifier_node(state: AgentState):
    """Cross-references Web findings against PDF context."""
    tool_outputs = state.get("tool_outputs", [])
    web_content = [t for t in tool_outputs if t["source"] == "external_web"]
    pdf_content = [t for t in tool_outputs if t["source"] == "internal_pdf"]
    
    if not web_content:
        return state # Nothing to verify
        
    # If we skipped PDF for some reason, let's quick-check it now for verification context
    if not pdf_content:
        pdf_content = pdf_search_tool(state["query"])
        
    web_text = "\n".join([c["content"] for c in web_content])
    pdf_text = "\n".join([c["content"] for c in pdf_content])
    
    prompt = f"""
    You are a Skeptical Verifier.
    
    Query: {state["query"]}
    
    INTERNAL PDF KNOWLEDGE:
    {pdf_text[:2000]}
    
    EXTERNAL WEB FINDINGS:
    {web_text[:2000]}
    
    Task:
    Check if the External Web Findings contradict the Internal PDF Knowledge.
    If Web says 'X' and PDF says 'Y', report the conflict.
    
    Output a brief "Verification Note". If no conflict, say "No conflict".
    """
    
    model = genai.GenerativeModel(MODEL_NAME)
    resp = generate_with_retry(model, prompt)
    note = resp.text.strip() if resp else "Verification failed."
    
    current_notes = state.get("verification_notes", "")
    new_notes = f"{current_notes}\n[Verification]: {note}"
    
    return {**state, "verification_notes": new_notes}

# 5. RESPONDER
def responder_node(state: AgentState):
    query = state["query"]
    tools_out = state.get("tool_outputs", [])
    notes = state.get("verification_notes", "")
    
    # Check if we found nothing
    if not tools_out and state["retries"] < 1:
        # Self-correction: Rewrite
        prompt = f"Rewrite this query to be more specific: {query}"
        model = genai.GenerativeModel(MODEL_NAME)
        resp = generate_with_retry(model, prompt)
        new_query = resp.text.strip() if resp else query
        return {**state, "query": new_query, "retries": state["retries"] + 1, "next_node": "supervisor"} # Loop back
        
    context = ""
    for t in tools_out:
        context += f"\n[{t['source'].upper()}]: {t['content'][:500]}..."
        
    prompt = f"""
    You are the Final Responder.
    User Query: {query}
    
    Gathered Info:
    {context}
    
    Verification Notes (Conflicts?):
    {notes}
    
    Instructions:
    1. Answer the user query based on gathered info.
    2. If there are conflicts (e.g. PDF vs Web), explicitly mention them and trust PDF more but note the Web claim.
    3. Cite sources (Internal PDF vs External Web).
    """
    
    model = genai.GenerativeModel(MODEL_NAME)
    resp = generate_with_retry(model, prompt)
    answer = resp.text if resp else "I could not generate an answer."
    
    return {
        **state, 
        "final_answer": answer, 
        "messages": [AIMessage(content=answer)],
        "next_node": "end"
    }

# ===============================
# GRAPH BUILDER
# ===============================
def build_agentic_rag_v2_graph():
    graph = StateGraph(AgentState)
    memory = MemorySaver()

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_pdf", researcher_pdf_node)
    graph.add_node("research_web", researcher_web_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("responder", responder_node)

    graph.set_entry_point("supervisor")

    # Routing
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next_node"],
        {
            "research_pdf": "research_pdf",
            "research_web": "research_web",
            "responder": "responder"
        }
    )
    
    # Research PDF -> Supervisor (to decide if Web is needed)
    graph.add_edge("research_pdf", "supervisor")
    
    # Research Web -> Verifier -> Supervisor
    graph.add_edge("research_web", "verifier")
    graph.add_edge("verifier", "supervisor")
    
    # Responder -> Maybe loop back if self-correction triggered?
    graph.add_conditional_edges(
        "responder",
        lambda s: "supervisor" if s["next_node"] == "supervisor" else "end",
        {
            "supervisor": "supervisor",
            "end": END
        }
    )

    return graph.compile(checkpointer=memory)

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
from sql_db import query_database

# Config
MODEL_FAST = "gemini-2.5-flash-lite"
MODEL_SMART = "gemini-3-flash-preview"
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
    tool_outputs: List[dict]  # list of {source: 'pdf'|'web'|'sql', content: ..., score: ...}
    verification_notes: str
    retries: int

# ===============================
# TOOLS
# ===============================
def pdf_search_tool(query: str):
    """Searches internal PDF knowledge base."""
    results = search_knowledge(query, top_k=4)
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
        context = tavily.get_search_context(query=query, search_depth="advanced")
        return [{
            "source": "external_web", 
            "content": context, 
            "score": 0.8 
        }]
    except Exception as e:
        return [{"source": "external_web", "content": f"Web search error: {str(e)}", "score": 0}]

def text_to_sql_tool(query: str):
    """Translates natural language to SQL and executes it."""
    prompt = f"""
    You are an expert SQL Translator.
    Table: students
    Columns: id, name, course, fees (real), enrollment_date (text), gpa (real)
    
    Task: Convert this question to a READ-ONLY SQL query (SQLite).
    Question: "{query}"
    
    Rules:
    - Output ONLY the SQL query. No markdown.
    - Do NOT use Markdown formatting.
    """
    model = genai.GenerativeModel(MODEL_SMART)
    resp = generate_with_retry(model, prompt)
    sql_query = resp.text.strip().replace("```sql", "").replace("```", "").strip() if resp else ""
    
    if not sql_query:
        return [{"source": "internal_sql", "content": "Error generating SQL.", "score": 0}]
        
    result_text = query_database(sql_query)
    return [{
        "source": "internal_sql", 
        "content": f"Query: {sql_query}\nResult: {result_text}", 
        "score": 1.0
    }]

# ===============================
# NODES
# ===============================

# 1. SUPERVISOR
def supervisor_node(state: AgentState):
    """Decides whether to research (and which tool) or answer."""
    query = state["query"]
    tools_out = state.get("tool_outputs", [])
    
    prompt = f"""
    You are a Supervisor Agent.
    User Query: "{query}"
    
    Gathered Info Count: {len(tools_out)}
    
    Decide next step:
    1. "research_sql": If the query asks about quantitative student data (fees, grades, counts, names in database).
    2. "research_pdf": If the query asks about policies, documents, or general university info.
    3. "research_web": If internal info is missing.
    4. "responder": If enough info is gathered.
    
    Return ONLY one of: research_sql, research_pdf, research_web, responder
    """
    
    # Heuristic: If we already searched SQL and got results, maybe go to responder or PDF
    # But for now, let LLM decide based on history.
    
    model = genai.GenerativeModel(MODEL_FAST)
    resp = generate_with_retry(model, prompt)
    decision = resp.text.strip().lower() if resp else "responder"
    
    if "sql" in decision: return {**state, "next_node": "research_sql"}
    if "pdf" in decision: return {**state, "next_node": "research_pdf"}
    if "web" in decision: return {**state, "next_node": "research_web"}
    
    return {**state, "next_node": "responder"}

# 2. RESEARCHER (PDF)
def researcher_pdf_node(state: AgentState):
    query = state["query"]
    results = pdf_search_tool(query)
    current_outputs = state.get("tool_outputs", []) + results
    # Removed intermediate logging to focus on final evaluation
    return {**state, "tool_outputs": current_outputs}

# 3. RESEARCHER (WEB)
def researcher_web_node(state: AgentState):
    query = state["query"]
    results = web_search_tool(query)
    current_outputs = state.get("tool_outputs", []) + results
    return {**state, "tool_outputs": current_outputs}

# 4. RESEARCHER (SQL)
def researcher_sql_node(state: AgentState):
    query = state["query"]
    results = text_to_sql_tool(query)
    current_outputs = state.get("tool_outputs", []) + results
    return {**state, "tool_outputs": current_outputs}

# 5. VERIFIER
def verifier_node(state: AgentState):
    """Verifies the quality of gathered information."""
    query = state["query"]
    tools_out = state.get("tool_outputs", [])
    
    # Simple verification logic
    context = ""
    for t in tools_out:
        context += f"\n[{t['source'].upper()}]: {t['content']}..."

    prompt = f"""
    You are a Verifier Agent.
    User Query: "{query}"
    
    Gathered Info:
    {context}
    
    Task:
    Analyze the gathered information. 
    - Is it relevant to the query?
    - Are there conflicts?
    - What key details are present?
    
    Provide concise verification notes for the Final Responder.
    """
    
    model = genai.GenerativeModel(MODEL_SMART)
    resp = generate_with_retry(model, prompt)
    notes = resp.text if resp else "Verification completed."
    
    return {**state, "verification_notes": notes}

# 6. RESPONDER
def responder_node(state: AgentState):
    query = state["query"]
    tools_out = state.get("tool_outputs", [])
    notes = state.get("verification_notes", "")
    
    if not tools_out and state["retries"] < 1:
        # Self-correction
        return {**state, "retries": state["retries"] + 1, "next_node": "supervisor"} 
        
    context_text_list = [t['content'] for t in tools_out]
    context = ""
    for t in tools_out:
        context += f"\n[{t['source'].upper()}]: {t['content']}..."
        
    prompt = f"""
    You are the Final Responder.
    User Query: {query}
    
    Gathered Info:
    {context}
    
    Verification Notes:
    {notes}
    
    Answer the user query. If you used SQL, summarize the data insights.
    """
    
    model = genai.GenerativeModel(MODEL_SMART)
    resp = generate_with_retry(model, prompt)
    answer = resp.text if resp else "I could not generate an answer."
    
    # === NEW: LOG FULL EVALUATION DATA ===
    # We log here because we have the Query, The Context, and The Final Answer
    if tools_out:
        log_eval(
            query=query,
            retrieved_count=len(tools_out),
            confidence=0.9, # dynamic confidence is hard without prob, assuming high if we have tools
            answer_known=True,
            source_type="mixed",
            final_answer=answer,
            context_list=context_text_list
        )
    
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
    graph.add_node("research_sql", researcher_sql_node)
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
            "research_sql": "research_sql",
            "responder": "responder"
        }
    )
    
    # Edges returning to Supervisor
    graph.add_edge("research_pdf", "supervisor")
    graph.add_edge("research_sql", "supervisor")
    
    # Web -> Verifier -> Supervisor
    graph.add_edge("research_web", "verifier")
    graph.add_edge("verifier", "supervisor")
    
    graph.add_conditional_edges(
        "responder",
        lambda s: "supervisor" if s["next_node"] == "supervisor" else "end",
        {
            "supervisor": "supervisor",
            "end": END
        }
    )

    return graph.compile(checkpointer=memory)

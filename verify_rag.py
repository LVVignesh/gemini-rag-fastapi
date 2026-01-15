import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from agentic_rag_v2_graph import build_agentic_rag_v2_graph

async def main():
    graph = build_agentic_rag_v2_graph()
    thread_id = "test-thread-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    print("--- Turn 1 ---")
    inputs = {
        "messages": [], # Initialize
        "query": "My name is Alice.",
        "refined_query": "",
        "decision": "",
        "retrieved_chunks": [],
        "retrieval_quality": "",
        "retries": 0,
        "answer": None,
        "confidence": 0.0,
        "answer_known": False
    }
    
    result = await graph.ainvoke(inputs, config=config)
    print(f"Answer 1: {result['final_answer']}")
    
    print("\n--- Turn 2 ---")
    inputs["query"] = "What is my name?"
    # We don't need to pass 'messages' again as it should be loaded from memory, 
    # but the graph definition expects it in TypedDict.
    # We can pass empty list, it will be merged/ignored depending on implementation?
    # Actually, MemorySaver loads the state. The input 'messages' is merged. 
    # Since we defined 'add_messages', passing empty list is fine (no new messages to add yet).
    inputs["messages"] = []
    
    result = await graph.ainvoke(inputs, config=config)
    print(f"Answer 2: {result['final_answer']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        traceback.print_exc()

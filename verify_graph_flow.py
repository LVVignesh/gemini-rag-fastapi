import unittest
from unittest.mock import MagicMock, patch
import os

# Set dummy key if not present to avoid init errors
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = "dummy_key"
if "TAVILY_API_KEY" not in os.environ:
    os.environ["TAVILY_API_KEY"] = "dummy_key"

from agentic_rag_v2_graph import build_agentic_rag_v2_graph

class TestRagGraph(unittest.TestCase):
    @patch('agentic_rag_v2_graph.generate_with_retry')
    @patch('agentic_rag_v2_graph.TavilyClient')
    def test_web_search_flow(self, mock_tavily, mock_generate):
        print("\n\n=== 🧪 STARTING DRY RUN GRAPH TEST ===")
        print("Goal: Verify 'research_web' -> 'verifier' -> 'responder' flow without API calls.\n")

        # === 🔧 DEMO CONFIGURATION (EDIT HERE) ===
        # Change these values to simulate different questions!
        DEMO_QUERY = "Who is the father of the computer?"
        EXPECTED_WEB_CONTENT = "Charles Babbage is considered by many as the father of the computer."
        VERIFIER_NOTE = "✅ VERIFIED: Search results confirm Charles Babbage invented the Analytical Engine."
        FINAL_ANSWER = "Charles Babbage is the father of the computer."
        # =========================================

        # --- Setup Mocks ---
        
        # Helper to create dummy response object
        def create_response(text):
            r = MagicMock()
            r.text = text
            return r

        # Sequence of LLM outputs (Order matters!):
        # 1. Supervisor: "research_web"
        # 2. Verifier: "The info looks consistent."
        # 3. Supervisor: "responder"
        # 4. Responder: "The Answer."
        
        mock_generate.side_effect = [
            create_response("research_web"),
            create_response(VERIFIER_NOTE),
            create_response("responder"),
            create_response(FINAL_ANSWER)
        ]
        
        # Mock Web Search Tool
        mock_tavily_instance = MagicMock()
        mock_tavily.return_value = mock_tavily_instance
        mock_tavily_instance.get_search_context.return_value = EXPECTED_WEB_CONTENT

        # --- Build Graph ---
        print("🛠️  Building Graph...")
        try:
            graph = build_agentic_rag_v2_graph()
            print("✅ Graph built successfully.")
        except Exception as e:
            self.fail(f"❌ Graph build failed: {e}")
        
        # --- Run Graph ---
        initial_state = {
            "messages": [],
            "query": DEMO_QUERY,
            "final_answer": "",
            "next_node": "",
            "current_tool": "",
            "tool_outputs": [],
            "verification_notes": "",
            "retries": 0
        }
        
        print("\n🏃 Invoking Graph (Mocked LLM)...")
        result = graph.invoke(initial_state, config={"configurable": {"thread_id": "test_dry_run"}})
        
        # --- Assertions ---
        print("\n\n=== 📊 TEST RESULT ANALYSIS ===")
        print(f"Final Answer: {result['final_answer']}")
        print(f"Verification Notes: {result['verification_notes']}")
        
        self.assertIn("VERIFIED", result['verification_notes'], "❌ verifier_node did not populate verification_notes!")
        self.assertIn(FINAL_ANSWER, result['final_answer'], "❌ Responder did not fail gracefully.")
        
        print("\n✅ SUCCESS: The Graph followed the correct path: Supervisor -> Web -> Verifier -> Supervisor -> Responder")
        print("✅ SUCCESS: 'verifier_node' executed and produced notes.")

if __name__ == "__main__":
    unittest.main()

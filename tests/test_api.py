import pytest
from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_analytics_endpoint():
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "knowledge_rate" in data

def test_ask_endpoint_mock_mode():
    # We can't guarantee Gemini API keys in CI/Test env without mocking
    # Ideally we should mock the agentic_graph or llm_utils.
    # For now, let's just check if it handles a missing body correctly (422)
    response = client.post("/ask", json={})
    assert response.status_code == 422

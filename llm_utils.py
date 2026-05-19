import time
import random
import os
from google import genai
from google.genai.errors import APIError

class DummyResponse:
    def __init__(self, text):
        self._text = text
    
    @property
    def text(self):
        return self._text

_client = None

def get_genai_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client

def generate_with_retry(model_name, prompt, retries=5, base_delay=4):
    """
    Generates content using the Gemini model with exponential backoff for rate limits.
    Returns a dummy response if all retries fail, preventing app crashes.
    """
    client = get_genai_client()
    for i in range(retries):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
        except Exception as e:
            # Check for Rate Limit (429) or Quota Exceeded (ResourceExhausted)
            is_quota_error = (
                "429" in str(e) 
                or "quota" in str(e).lower() 
                or "resource_exhausted" in str(e).lower()
                or (isinstance(e, APIError) and e.code == 429)
            )
            
            if is_quota_error:
                if i < retries - 1:
                    sleep_time = base_delay * (2 ** i) + random.uniform(0, 1)
                    print(f"⚠️ Quota exceeded. Retrying in {sleep_time:.2f}s... (Attempt {i+1}/{retries})")
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"❌ Quota exceeded after {retries} attempts. Returning resilience fallback.")
                    return DummyResponse("⚠️ **System Alert**: The AI service is currently experiencing high traffic (Quota Exceeded). Please try again in a few minutes.")
            
            # If it's not a quota error (e.g. 500 server error), we might still want to be safe
            print(f"❌ Error generating content: {e}")
            return DummyResponse(f"⚠️ **System Error**: {str(e)}")
            
    return DummyResponse("⚠️ **Unknown Error**: Failed to generate response.")

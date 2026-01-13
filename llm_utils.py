import time
import random
import google.generativeai as genai
from google.api_core import exceptions

class DummyResponse:
    def __init__(self, text):
        self._text = text
    
    @property
    def text(self):
        return self._text

def generate_with_retry(model, prompt, retries=3, base_delay=2):
    """
    Generates content using the Gemini model with exponential backoff for rate limits.
    Returns a dummy response if all retries fail, preventing app crashes.
    """
    for i in range(retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            # Check for Rate Limit (429) or Quota Exceeded (ResourceExhausted)
            is_quota_error = (
                "429" in str(e) 
                or "quota" in str(e).lower() 
                or isinstance(e, exceptions.ResourceExhausted)
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
            
            # If it's not a quota error (e.g. 500 server error), we might still want to be safe?
            # For master's level, let's catch everything but log it.
            print(f"❌ Error generating content: {e}")
            return DummyResponse(f"⚠️ **System Error**: {str(e)}")
            
    return DummyResponse("⚠️ **Unknown Error**: Failed to generate response.")

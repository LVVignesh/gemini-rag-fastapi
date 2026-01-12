import time
import random
import google.generativeai as genai
from google.api_core import exceptions

def generate_with_retry(model, prompt, retries=3, base_delay=2):
    """
    Generates content using the Gemini model with exponential backoff for rate limits.
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
                    print(f"❌ Quota exceeded after {retries} attempts.")
                    # We can re-raise or return None depending on preference.
                    # Re-raising allows the caller to handle the failure (e.g. return 503 Service Unavailable)
                    # identifying strictly as quota error might be useful.
            raise e
    return None

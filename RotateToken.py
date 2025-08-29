import google.generativeai as genai

API_KEYS = [
    "GEMINI_API_KEY_1",
    "AIzaSyBt1r4hb_ln7WaDUlExRlp3lQKt1tGrBYA",
    "GEMINI_API_KEY_3",
]

current_key_index = 0

def get_model(model_name: str = "gemini-1.5-flash"):
    """Return a Gemini model configured with the current API key."""
    genai.configure(api_key=API_KEYS[current_key_index])
    return genai.GenerativeModel(model_name)

def rotate_api_key(model_name: str = "gemini-1.5-flash"):
    """Rotate to the next API key and return a new model instance."""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    print(f"🔄 Rotated to API key index: {current_key_index}")
    return get_model(model_name)

def safe_generate(prompt: str, model_name: str = "gemini-1.5-flash", max_retries: int = 3):
    """Generate content with automatic API key rotation on quota/rate/token errors."""
    model = get_model(model_name)

    for _ in range(max_retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            error_msg = str(e).lower()
            print(f"⚠️ Error: {error_msg}")
            if any(k in error_msg for k in ("quota", "exceeded", "token", "rate", "resource", "429","400")):
                model = rotate_api_key(model_name)
                continue
            else:
                break

    # Fallback: no response
    return None

if __name__ == "__main__":
    prompt = "Việt Nam là một quôc gia xinh đẹp."
    response = safe_generate(prompt)

    if response is None:
        print("Failed to generate content after retries.")
    else:
        print(f"Edited Text: {response.text.strip()}\nTokens Used: {response.usage_metadata.total_token_count}")
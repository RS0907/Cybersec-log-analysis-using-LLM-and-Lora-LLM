import ollama
import json
import hashlib
from pathlib import Path

CACHE_FILE = Path(__file__).parent / "outputs" / "ollama_cache.json"

def _load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def _save_cache(cache_data):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=4)

def analyze_logs_with_ollama(system_prompt: str, logs: str) -> str:
    """
    Send logs to Ollama LLM for analysis and return the response content,
    handling empty or invalid responses. Caches responses to avoid redundant processing.
    """
    user_prompt = f"""
    Analyze the following cybersecurity logs and produce a structured and vivid JSON report.
    
    Logs:
    <<<
    {logs}
    >>>
    """
    
    # 1. Generate unique hash for this specific prompt and log combination
    cache_key = hashlib.sha256((system_prompt + logs).encode("utf-8")).hexdigest()
    
    # 2. Check if we already processed this exact data
    cache_data = _load_cache()
    if cache_key in cache_data:
        print("  [Cache Hit] Using previously generated LLM answer.")
        return cache_data[cache_key]
    
    try:
        # 3. Request LLM generation (only if cache missed)
        response = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.get("message", {}).get("content", "")
        if not content.strip():
            raise ValueError("LLM returned an empty response. Check input logs and system prompt.")
            
        # 4. Save the new answer to cache
        cache_data[cache_key] = content
        _save_cache(cache_data)
        
        return content
    except Exception as e:
        raise RuntimeError(f"Failed to analyze logs with the LLM: {e}")
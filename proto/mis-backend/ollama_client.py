import ollama

def analyze_logs_with_ollama(system_prompt: str, logs: str) -> str:
    """
    Send logs to Ollama LLM for analysis and return the response content,
    handling empty or invalid responses.
    """
    user_prompt = f"""
    Analyze the following cybersecurity logs and produce a structured and vivid JSON report.
    
    Logs:
    <<<
    {logs}
    >>>
    """
    
    try:
        response = ollama.chat(
            model="mistral:7b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.get("message", {}).get("content", "")
        if not content.strip():
            raise ValueError("LLM returned an empty response. Check input logs and system prompt.")
        return content
    except Exception as e:
        raise RuntimeError(f"Failed to analyze logs with the LLM: {e}")
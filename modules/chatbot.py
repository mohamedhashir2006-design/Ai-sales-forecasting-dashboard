import requests

def ask_ai(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma:2b",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        return response.json().get("response", "No response")
    except:
        return "⚠️ Ollama not running"
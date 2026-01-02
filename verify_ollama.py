import os
import requests
from langchain_ollama import ChatOllama
from config import Config

def check_ollama_server(url):
    print(f"Testing connection to Ollama at {url}...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("[OK] Ollama server is running and accessible.")
            return True
        else:
            print(f"[WARN] Ollama server responded but with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[FAIL] Could not connect to Ollama server.")
        print("       Did you run 'ollama serve' or the Ollama app?")
        return False
    except Exception as e:
        print(f"[FAIL] Error connecting to Ollama: {e}")
        return False

def check_model_availability(base_url, model_name):
    print(f"Checking if model '{model_name}' is available...")
    try:
        # standard Ollama API endpoint for tags
        tags_url = f"{base_url.rstrip('/')}/api/tags"
        response = requests.get(tags_url)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            # Only exact match or partial match
            found = False
            for m in models:
                if m['name'] == model_name or m['name'].startswith(model_name):
                    found = True
                    break
            
            if found:
                print(f"[OK] Model '{model_name}' found.")
            else:
                print(f"[FAIL] Model '{model_name}' NOT found.")
                print("       Available models:")
                for m in models:
                    print(f"       - {m['name']}")
                print(f"       Please run: 'ollama pull {model_name}'")
        else:
            print(f"[WARN] Could not list models (Status: {response.status_code})")
            
    except Exception as e:
        print(f"[FAIL] Error checking models: {e}")

def main():
    # Load defaults from config if not env vars
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    
    print("--- Ollama Connection Verification ---")
    print(f"Target URL: {base_url}")
    print(f"Target Model: {model_name}")
    print("--------------------------------------")
    
    if check_ollama_server(base_url):
        check_model_availability(base_url, model_name)
        
        print("\nAttempting to invoke LangChain ChatOllama...")
        try:
            llm = Config.get_llm()
            # Simple invocation
            response = llm.invoke("Hello, are you working?")
            print(f"[OK] LLM Response received: {response.content}")
        except Exception as e:
            print(f"[FAIL] LLM Invocation failed: {e}")

if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

class Config:
    MODE = os.getenv("ENV_MODE", "DEV").upper()
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./saudi_legal_db_final1")
    DATA_PATH = "data"

    @staticmethod
    def get_embeddings():
        """
        Returns BAAI/bge-m3.
        Force CPU for Laptop (DEV) to prevent crashing.
        Use CUDA (GPU) for Client (PROD).
        """
        model_name = "BAAI/bge-m3"
        encode_kwargs = {'normalize_embeddings': True}
        
        if Config.MODE == "DEV":
            device = "cpu"
            print(f"🔌 DEV MODE: Loading {model_name} on CPU (Lightweight)...")
        else:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🚀 PROD MODE: Loading {model_name} on GPU (Max Performance)...")
            else:
                device = "cpu"
                print(f"⚠️ PROD MODE: GPU not found. Falling back to CPU for {model_name}...")

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs=encode_kwargs
        )

    @staticmethod
    def get_llm():
        """
        Returns the LLM (Ollama).
        Reads connection settings from environment variables.
        Defaults to:
          - URL: http://localhost:11434
          - Model: qwen2.5:14b
        """
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        
        print(f"🏠 Using Local LLM (Ollama) at {ollama_url} with model {ollama_model}...")
        
        return ChatOllama(
            base_url=ollama_url,
            model=ollama_model,
            temperature=0.0,
            # Stop words to prevent infinite Arabic loops
            stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]
        )
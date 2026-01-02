# Troubleshooting Guide

## Ollama Connection Issues
If the application fails to connect to the LLM or embedding model, follow these steps.

### 1. Diagnosis
Run the included verification script:
```powershell
python verify_ollama.py
```
- if it says **[FAIL] Could not connect**: Ollama is not running.
- If it says **[FAIL] Model not found**: You need to pull the model.

### 2. Solutions
- **Start Ollama**: Open the Ollama application or run `ollama serve` in a terminal.
- **Get the Model**: Run `ollama pull qwen2.5:14b`.

### 3. Custom Configuration
If you are running Ollama on a different server or port, you can set environment variables:
- `OLLAMA_BASE_URL`: e.g., `http://192.168.1.100:11434`
- `OLLAMA_MODEL`: e.g., `llama3`

## Application Issues

### "Re-Index" Button
The "Re-Index All Data" button scans the `data` folder and adds any files found there to the database. This is useful if you add files manually.

### Missing Database
If you delete the `saudi_legal_db_final1` folder, the app will automatically create a fresh empty database on the next run.

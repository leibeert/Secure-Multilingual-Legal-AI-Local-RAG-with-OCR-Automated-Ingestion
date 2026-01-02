# RAG Law - Saudi Legal Assistant

This is a local AI-powered assistant designed to answer questions about Saudi Laws using your provided PDF documents. It works completely offline on your computer.

## 📋 Prerequisites

Before you begin, you need to install two main components: **Python** (to run the app) and **Ollama** (to power the AI).

### 1. Install Python
1.  Download Python 3.11 or later from [python.org](https://www.python.org/downloads/).
2.  Run the installer. **IMPORTANT:** Check the box that says **"Add Python to PATH"** before clicking Install.

### 2. Install & Tools Setup (Ollama)
The AI "Brain" needs a separate tool called Ollama to run.
1.  Download and install Ollama from [ollama.com](https://ollama.com/).
2.  Once installed, open your **Command Prompt** (cmd) or **PowerShell** and run the following command to download the specific AI model we are using:
    ```powershell
    ollama pull qwen2.5:14b
    ```
    *Note: This download is about 9GB. Please wait for it to finish.*

---

## 🚀 Installation & Setup

You only need to do this part once to set up the project folder.

1.  **Open the Project Folder**:
    - Navigate to the folder containing these files.
    - Right-click inside the folder (in empty space) and select **"Open in Terminal"** or Copy the path and `cd` into it in your terminal.

2.  **Activate the Virtual Environment**:
    We have already prepared a detached environment for you in the `venv` folder. Run this command to simple "enter" it:
    ```powershell
    .\venv\Scripts\activate
    ```
    *You should see `(venv)` appear at the beginning of your command line.*

3.  **Install Dependencies (Just in case)**:
    Run this command to ensure all required libraries are installed:
    ```powershell
    pip install -r requirements.txt
    ```

---

## ▶️ How to Run the App

Every time you want to use the assistant, follow these two simple steps:

### Step 1: Start the AI Backend
1.  Make sure the **Ollama** app is running in your system tray (bottom right of screen).
    - *Alternatively, open a separate terminal and type `ollama serve`.*

### Step 2: Run the Assistant
1.  In your terminal (where you activated `venv` earlier), run:
    ```powershell
    streamlit run app.py
    ```
2.  A new tab will automatically open in your web browser with the application.
3.  Type your question in Arabic or English and get answers with citations from your documents!

---

## 📂 Adding New Documents (Optional)
If you want to add NEW PDF files to the system:
1.  Put your `.pdf` files into the `data` folder.
2.  Run the "Ingestion" script to process them (this takes time):
    ```powershell
    python ingest.py
    ```
3.  Once finished, restart the app (`Ctrl+C` in terminal to stop, then `streamlit run app.py` again).

---

## ❓ Troubleshooting

-   **"Ollama connection refused"**: Make sure the Ollama app is running before you start `app.py`.
-   **"Module not found" error**: Ensure you activated the environment with `.\venv\Scripts\activate`.

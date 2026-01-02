import streamlit as st
import os
import re
from rag_engine import RAGEngine
from config import Config

# Page Configuration
st.set_page_config(page_title="Saudi Legal AI Advisor", layout="wide")

# Custom CSS for Arabic RTL support
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        text-align: right; 
        direction: RTL; 
    }
    .stChatMessage {
        text-align: right;
        direction: RTL;
    }
    .stMarkdown {
        text-align: right;
        direction: RTL;
    }
    /* Highlight Citations */
    code {
        background-color: #f0f2f6;
        color: #d63384;
        padding: 2px 5px;
        border-radius: 4px;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    """Cache the RAG engine to prevent reloading on every click."""
    return RAGEngine()

def main():
    # Initialize Engine
    engine = get_engine()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.info(f"Mode: **{Config.MODE}**")
        
        st.divider()
        st.subheader("📂 Upload Documents")
        uploaded_file = st.file_uploader("Upload PDF / DOCX", type=["pdf", "docx"])
        
        if uploaded_file:
            if st.button("Process & Ingest File"):
                with st.spinner("Processing & Indexing..."):
                    # Save locally
                    if not os.path.exists(Config.DATA_PATH):
                        os.makedirs(Config.DATA_PATH)
                        
                    save_path = os.path.join(Config.DATA_PATH, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Ingest
                    success = engine.ingest_file(save_path)
                    if success:
                        st.success(f"✅ Successfully Indexed: {uploaded_file.name}")
                    else:
                        st.error("❌ Failed to process file.")

        st.divider()
        if st.button("🔄 Re-Index All Data Folder"):
            with st.spinner("Scanning 'data' folder..."):
                engine.ingest_all_data()
                st.success("✅ All files re-indexed!")

    # --- MAIN CHAT ---
    st.title("⚖️ المساعد القانوني السعودي")
    st.caption("نظام ذكي للإجابة على الاستفسارات القانونية بناءً على الأنظمة السعودية.")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("اطرح سؤالك هنا... (مثال: ما هي عقوبة التأخر في تقديم الإقرار؟)"):
        # 1. Show User Msg
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Answer
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("جاري البحث في المصادر..."):
                try:
                    qa_chain = engine.get_qa_chain()
                    response = qa_chain.invoke({"query": prompt})
                    
                    answer = response['result']
                    
                    # Remove <think> blocks if present
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
                    
                    sources = response['source_documents']
                    
                    # Display the Main Answer
                    message_placeholder.markdown(answer)
                    
                    # --- CITATION BLOCK (The part you asked for) ---
                    if sources:
                        with st.expander("📚 المصادر والمواد القانونية المستخدمة"):
                            for i, doc in enumerate(sources):
                                # Extract Metadata
                                source_file = doc.metadata.get('source', 'Unknown')
                                article_num = doc.metadata.get('article', 'General')
                                subject = doc.metadata.get('subject', 'Law')
                                
                                # Display nicely
                                st.markdown(f"**{i+1}. {subject}** - `{article_num}`")
                                st.caption(f"الملف: {source_file}")
                                st.text(doc.page_content[:300] + "...") # Preview text
                    
                    # Save History
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"حدث خطأ: {str(e)}")

if __name__ == "__main__":
    main()
import os
import pickle
from typing import List, Optional, Any
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
# from langchain.retrievers import ParentDocumentRetriever # Removed standard import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from config import Config
from text_utils import load_file_structured
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# --- POLYFILL CLASS (MUST MATCH INGEST.PY) ---
class ParentDocumentRetriever(BaseRetriever):
    """
    Simulated ParentDocumentRetriever.
    Crucial for depickling the documents stored as bytes in the docstore.
    """
    vectorstore: Any
    docstore: Any
    child_splitter: Any
    id_key: str = "doc_id"
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None):
        import uuid
        if not documents:
            return
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
            
        full_docs = []
        for i, doc in enumerate(documents):
            doc_id = ids[i]
            # Split into children
            sub_docs = self.child_splitter.split_documents([doc])
            for sub_doc in sub_docs:
                sub_doc.metadata[self.id_key] = doc_id
            full_docs.extend(sub_docs)
            
            # Add to docstore
            self.docstore.mset([(doc_id, doc)])
            
        # Add to vectorstore
        self.vectorstore.add_documents(full_docs) 

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Retrieve documents relevant to the query."""
        # 1. Search vectorstore for children
        sub_docs = self.vectorstore.similarity_search(query, k=5)
        
        # 2. Extract unique Parent IDs
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata:
                ids.append(d.metadata[self.id_key])
        ids = list(set(ids))
        
        # 3. Fetch Parents from docstore
        if not ids:
            return []
            
        raw_docs = self.docstore.mget(ids)
        final_docs = []
        for d in raw_docs:
            if d is not None:
                try:
                    # CRITICAL: Only unpickle if it's bytes (Standard ParentDocumentRetriever stores Documents directly usually)
                    if isinstance(d, bytes):
                        final_docs.append(pickle.loads(d))
                    else:
                        final_docs.append(d)
                except Exception as e:
                    print(f"Error processing doc: {e}")
        return final_docs

class RAGEngine:
    def __init__(self):
        self.db_path = Config.CHROMA_PATH
        self.store_path = os.path.join(self.db_path, "doc_store")
        self._embeddings = None
        self._vectorstore = None
        self._retriever = None
        self._store = None
        self._child_splitter = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = Config.get_embeddings()
        return self._embeddings

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            try:
                # 1. Vector Database
                # The folder structure is:
                # Root/
                #   chroma_vectors/ (Actual DB)
                #   docstore.pkl    (Docs)
                vector_path = os.path.join(self.db_path, "chroma_vectors")
                
                self._vectorstore = Chroma(
                    collection_name="saudi_legal_docs",
                    embedding_function=self.embeddings,
                    persist_directory=vector_path
                )
            except Exception as e:
                print(f"❌ Error loading ChromaDB: {e}")
                raise e
        return self._vectorstore

    @property
    def store(self):
        if self._store is None:
             # 2. Doc Store (Pickled InMemoryStore)
            pkl_path = os.path.join(self.db_path, "docstore.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        self._store = pickle.load(f)
                    print(f"✅ Loaded DocStore from {pkl_path}")
                except Exception as e:
                    print(f"❌ Error loading DocStore pickle: {e}")
                    self._store = None 
            else:
                print(f"⚠️ DocStore pickle not found at {pkl_path}")
                self._store = None
        return self._store

    @property
    def child_splitter(self):
        if self._child_splitter is None:
            self._child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100
            )
        return self._child_splitter

    @property
    def retriever(self):
        if self._retriever is None:
             # 4. Retriever (Using Polyfill Class)
             self._retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.store,
                child_splitter=self.child_splitter,
            )
        return self._retriever

    def ingest_file(self, file_path):
        try:
            # Lazy import from ingest.py to reuse logic
            from ingest import load_file
            
            print(f"Processing {file_path}...")
            docs = load_file(file_path)
            
            if not docs:
                print(f"⚠️ Processed file {file_path} resulted in 0 documents.")
                return False
                
            # Add to Retriever
            self.retriever.add_documents(docs)
            
            # Persist DocStore
            pkl_path = os.path.join(self.db_path, "docstore.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(self.store, f)
            print(f"✅ DocStore saved to {pkl_path}")
            
            return True
        except Exception as e:
            print(f"❌ Error in ingest_file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def ingest_all_data(self):
        pass

    def get_qa_chain(self):
        llm = Config.get_llm()
        
        template = """
        You are a strict technical translator.
        
        **INPUT DATA:**
        Context: {context}
        User Question: {question}
        
        **INSTRUCTIONS:**
        1. You must answer ONLY using the "Context" provided above.
        2. **IF CONTEXT IS ENGLISH:** You must TRANSLATE it into Arabic sentence-by-sentence.
        3. **DO NOT SUMMARIZE.** Do not change the list structure. If the source has (A, B, C, D, E, F), your answer MUST have (A, B, C, D, E, F).
        4. **BAN:** Do not use the word "доходات" or any non-Arabic words.
        5. **CITATION:** You must cite the source file at the end.
        
        **ANSWER (ARABIC):**
        """
        
        prompt = PromptTemplate.from_template(template)
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
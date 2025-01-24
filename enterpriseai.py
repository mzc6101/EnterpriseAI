import streamlit as st
import sqlite3
import requests
import os
import json

import faiss
import numpy as np

# -----------------------------
# Hugging Face Embeddings
# -----------------------------
from sentence_transformers import SentenceTransformer

# -----------------------------
# LangChain (for orchestration)
# -----------------------------
from langchain.schema import Document
from langchain.tools import BaseTool

# =========================
# CONFIGURATION
# =========================

DATABASE_FILE = "my_local_data.db"

# ChatGPT
CHATGPT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"

# Ollama
OLLAMA_API_URL = "http://localhost:11411"
OLLAMA_MODEL   = "llama2"  # Adjust to whichever model you have in Ollama (e.g., "llama2", "mistral", etc.)

# Embedding model name (Hugging Face)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize the embedding model once
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Global reference to FAISS index
faiss_index = None
dimension = 384  # all-MiniLM-L6-v2 outputs 384-dimensional embeddings
faiss_index_file = "my_faiss.index"  # store to disk if you want persistence


# =========================
# 1. DATABASE & FAISS INIT
# =========================

def init_db():
    """
    1) Initialize/connect to the local SQLite database.
    2) Initialize or load a FAISS index for vector search.
    """
    # SQLite for document text & metadata
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            is_sensitive BOOLEAN NOT NULL DEFAULT 0,
            metadata TEXT
        );
    """)
    conn.commit()
    conn.close()

    # Create or load FAISS index
    global faiss_index
    if os.path.exists(faiss_index_file):
        # Load existing index
        faiss_index = faiss.read_index(faiss_index_file)
    else:
        # Create a new index (flat L2)
        faiss_index = faiss.IndexFlatL2(dimension)

def save_faiss_index():
    """ Save the in-memory FAISS index to disk (if desired). """
    if faiss_index is not None:
        faiss.write_index(faiss_index, faiss_index_file)

def add_document(content: str, is_sensitive: bool = False, metadata: str = ""):
    """
    Insert a document (plus metadata) into SQLite, then embed & add to FAISS.
    """
    # 1) Insert into SQLite
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (content, is_sensitive, metadata)
        VALUES (?, ?, ?)
    """, (content, 1 if is_sensitive else 0, metadata))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # 2) Embed and add to FAISS
    global faiss_index
    embedding = embedding_model.encode(content).astype(np.float32).reshape(1, -1)

    # If not an IDMap, convert it so we can track doc_id
    if not isinstance(faiss_index, faiss.IndexIDMap):
        index_idmap = faiss.IndexIDMap(faiss_index)
        faiss_index = index_idmap

    # Add vector with the doc ID
    faiss_index.add_with_ids(embedding, np.array([doc_id], dtype=np.int64))
    save_faiss_index()

def retrieve_all_documents():
    """
    Return all docs from SQLite as a list of dicts.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, is_sensitive, metadata FROM documents")
    rows = cursor.fetchall()
    conn.close()

    docs = []
    for row in rows:
        doc_id, content, sens, meta = row
        docs.append({
            "id": doc_id,
            "content": content,
            "is_sensitive": bool(sens),
            "metadata": meta
        })
    return docs

def retrieve_relevant_docs(query: str, top_k: int = 3):
    """
    Use FAISS to return top_k relevant documents to the query (via vector similarity).
    """
    if not faiss_index:
        return []

    # Embed query
    query_vec = embedding_model.encode(query).astype(np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(query_vec, top_k)

    # If no results or an empty index, return []
    if len(indices) == 0 or indices[0][0] == -1:
        return []

    # Fetch from DB
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        cursor.execute("SELECT id, content, is_sensitive, metadata FROM documents WHERE id=?", (idx,))
        row = cursor.fetchone()
        if row:
            doc_id, doc_content, doc_sens, doc_meta = row
            results.append({
                "id": doc_id,
                "content": doc_content,
                "is_sensitive": bool(doc_sens),
                "metadata": doc_meta,
                "distance": dist
            })

    conn.close()
    return results


# =========================
# 2. ChatGPT Fallback
# =========================

def ask_chatgpt(system_prompt: str, user_prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Call OpenAI ChatGPT to get external info if local docs insufficient.
    """
    if not CHATGPT_API_KEY:
        return "Error: Missing OPENAI_API_KEY environment variable."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHATGPT_API_KEY}",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7
    }

    try:
        resp = requests.post(CHATGPT_API_URL, headers=headers, json=body, timeout=60)
        resp_json = resp.json()
        if "error" in resp_json:
            return f"ChatGPT API error: {resp_json['error']}"
        return resp_json["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling ChatGPT: {e}"


# =========================
# 3. Ollama (Local LLM)
# =========================

def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Use Ollama with a local LLM (like Llama 2, Mistral, Qwen, etc.).
    """
    req_body = {
        "prompt": prompt,
        "model": model,
        "system": "",
        "options": {
            "temperature": 0.2
        }
    }
    try:
        response = requests.post(url=f"{OLLAMA_API_URL}/generate", json=req_body, stream=True)
        if response.status_code != 200:
            return f"Error: Ollama returned status code {response.status_code}"
        full_text = ""
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                data = json.loads(chunk)
                if "response" in data:
                    full_text += data["response"]
                if data.get("done", False):
                    break
        return full_text
    except Exception as e:
        return f"Error calling Ollama: {e}"


# =========================
# 4. LangChain Tools
# =========================

class OllamaTool(BaseTool):
    name: str = "ollama_tool"
    description: str = "Call the local Ollama LLM for generation."

    def _run(self, prompt: str) -> str:
        return ask_ollama(prompt)

    async def _arun(self, prompt: str) -> str:
        return self._run(prompt)

class ChatGPTTool(BaseTool):
    name: str = "chatgpt_tool"
    description: str = "Call the ChatGPT API for additional external info."

    def _run(self, prompt: str) -> str:
        system_prompt = "You are a concise, helpful AI. Provide the best factual answer."
        return ask_chatgpt(system_prompt, prompt)

    async def _arun(self, prompt: str) -> str:
        return self._run(prompt)

ollama_tool = OllamaTool()
chatgpt_tool = ChatGPTTool()


# =========================
# 5. CORE Q&A LOGIC
# =========================

def process_user_question(question: str) -> str:
    """
    1) Get top_k relevant docs from FAISS + DB.
    2) If local docs suffice, use them.
    3) If not, call ChatGPT for external info.
    4) Combine local + external, finalize with Ollama.
    """
    # Retrieve relevant docs
    relevant_docs = retrieve_relevant_docs(question, top_k=3)
    if not relevant_docs:
        # No local docs at all -> fallback to ChatGPT
        external_info = chatgpt_tool.run(question)
        prompt_for_ollama = f"""
No local docs found.

External info from ChatGPT:
{external_info}

User question: {question}

Provide an answer (do not invent personal data).
"""
        return ollama_tool.run(prompt_for_ollama).strip()

    # We do have local docs
    # Separate sensitive vs. non-sensitive
    sensitive_docs = [d for d in relevant_docs if d["is_sensitive"]]
    non_sensitive_docs = [d for d in relevant_docs if not d["is_sensitive"]]

    combined_non_sensitive = "\n".join([d["content"] for d in non_sensitive_docs])
    combined_sensitive = "\n".join([d["content"] for d in sensitive_docs])

    # Decide if local docs are enough
    local_info_sufficient = True
    lower_q = question.lower()
    if any(k in lower_q for k in ["latest", "current", "today", "recent", "live"]):
        local_info_sufficient = False

    external_info = ""
    if not local_info_sufficient:
        external_info = chatgpt_tool.run(question)
        # If there's an error or empty, it's just an empty string

    # Combine everything for final prompt
    prompt_for_ollama = f"""
Local non-sensitive docs:
{combined_non_sensitive}

Local sensitive docs (DO NOT REVEAL to user):
{combined_sensitive}

External info from ChatGPT (if any):
{external_info}

User question: {question}

Using all the info above, provide a final answer without leaking sensitive data.
"""
    return ollama_tool.run(prompt_for_ollama).strip()


# =========================
# 6. STREAMLIT FRONTEND
# =========================

def main():
    # Initialize DB & FAISS
    init_db()

    st.title("Open Source AI Stack Demo")
    st.markdown("""
**Technologies used**:
- **Streamlit** (UI)
- **SQLite** (local DB for docs)
- **FAISS** (vector store)
- **Hugging Face** embeddings
- **LangChain** (tools orchestration)
- **Ollama** (local LLM)
- **ChatGPT** (fallback for external knowledge)
""")

    # Document Ingestion
    st.subheader("Add a Document")
    new_doc_text = st.text_area("Document content", "")
    is_sensitive_flag = st.checkbox("Is this sensitive?", value=False)
    metadata_text = st.text_input("Metadata (optional)")

    if st.button("Add Document to DB"):
        if new_doc_text.strip():
            add_document(new_doc_text, is_sensitive=is_sensitive_flag, metadata=metadata_text)
            st.success("Document added successfully!")
        else:
            st.warning("Document content is empty, please add some text.")

    # Q&A
    st.subheader("Ask a Question")
    user_question = st.text_input("Your question:")
    if st.button("Ask"):
        if user_question.strip():
            answer = process_user_question(user_question)
            st.write("**Answer:**", answer)
        else:
            st.warning("Please enter a question.")

    # Debug info
    if st.checkbox("Show all documents in DB"):
        docs = retrieve_all_documents()
        st.write(docs)


if __name__ == "__main__":
    main()
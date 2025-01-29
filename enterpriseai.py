import streamlit as st
import sqlite3
import requests
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()  

# Configuration
DATABASE_FILE = "my_local_data.db"
FAISS_INDEX_FILE = "my_faiss.index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
dimension = 384  # Embedding dimension for the model
similarity_threshold = 0.6
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OLLAMA_API_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"
CHATGPT_API_KEY =  os.getenv("CHATGPT_API_KEY")
OPENAI_MODEL = "gpt-4o"

# Initialize FAISS
faiss_index = None

# Initialize SQLite Database and FAISS Index
def init_db():
    """Initialize SQLite database and FAISS index."""
    global faiss_index

    # Ensure SQLite database schema
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

    # Ensure FAISS index is initialized
    if os.path.exists(FAISS_INDEX_FILE):
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    else:
        base_index = faiss.IndexFlatIP(dimension)
        faiss_index = faiss.IndexIDMap(base_index)

# Save FAISS index to disk
def save_faiss_index():
    """Save FAISS index to disk."""
    if faiss_index is not None:
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)

# Add a document
def add_document(content: str, is_sensitive: bool = False, metadata: str = ""):
    """Add a document to SQLite and FAISS index."""
    # Insert into SQLite
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (content, is_sensitive, metadata)
        VALUES (?, ?, ?)
    """, (content, 1 if is_sensitive else 0, metadata))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Add to FAISS
    global faiss_index
    embedding = embedding_model.encode(content).astype(np.float32).reshape(1, -1)
    faiss_index.add_with_ids(embedding, np.array([doc_id], dtype=np.int64))
    save_faiss_index()

# Retrieve relevant documents based on similarity
def retrieve_relevant_docs(query: str, top_k: int = 5):
    """Retrieve documents with cosine similarity above a threshold."""
    global faiss_index

    # Convert query to vector
    query_vec = embedding_model.encode(query.lower()).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    # Search FAISS index
    distances, indices = faiss_index.search(query_vec, top_k)

    # Fetch relevant docs from SQLite
    relevant_docs = []
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1 or dist < similarity_threshold:
            continue
        cursor.execute("SELECT id, content, is_sensitive, metadata FROM documents WHERE id=?", (int(idx),))
        row = cursor.fetchone()
        if row:
            relevant_docs.append({
                "id": row[0],
                "content": row[1],
                "is_sensitive": bool(row[2]),
                "metadata": row[3],
                "similarity": dist,
            })
    conn.close()

    # Sort by similarity
    relevant_docs.sort(key=lambda x: x["similarity"], reverse=True)
    return relevant_docs

# Ask Ollama
def ask_ollama(prompt: str) -> str:
    """Query Ollama with a prompt."""
    req_body = {
        "prompt": prompt,
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=req_body, headers={"Content-Type": "application/json"})
    return response.json().get("response", "Error: Ollama response failed.") if response.status_code == 200 else f"Error: {response.status_code}"

#ask Perplexity
def ask_perplexity(question: str) -> str:
    """Query Perplexity API for internet-based answers."""
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are to find the latest information based on the question from the internet and relevant websites and provide a step by step detailed response."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return f"Error: Perplexity API request failed with status {response.status_code}"
    except Exception as e:
        return f"Error: Perplexity API request failed - {str(e)}"
# Ask ChatGPT
def ask_chatgpt(prompt: str) -> str:
    """Query ChatGPT with a prompt."""
    headers = {"Authorization": f"Bearer {CHATGPT_API_KEY}"}
    data = {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: ChatGPT response is missing content.")
    else:
        return f"Error: ChatGPT request failed with status code {response.status_code}, response: {response.text}"

# Process user question and route to appropriate model
def process_question(question: str) -> tuple:
    """Process the user's question with enhanced Perplexity integration."""
    relevant_docs = retrieve_relevant_docs(question, top_k=5)
    sensitive_docs = [doc for doc in relevant_docs if doc["is_sensitive"]]
    non_sensitive_docs = [doc for doc in relevant_docs if not doc["is_sensitive"]]

    if sensitive_docs:
        # Sensitive data flow
        sensitive_context = "\n\n".join(doc["content"] for doc in sensitive_docs)
        
        # Step 1: Generate safe question for Perplexity
        safe_q_prompt = f"""Context (DO NOT REPEAT IN OUTPUT):
{sensitive_context}

Original question: {question}

Generate a safe question to ask the internet that helps answer the original question WITHOUT revealing any sensitive details from the context. The question should be general and technical, not mentioning specific implementations.
"""
        safe_question = ask_ollama(safe_q_prompt)
        
        # Step 2: Get Perplexity response
        perplexity_response = ask_perplexity(safe_question)
        
        # Step 3: Process with ChatGPT
        chatgpt_prompt = f"""Latest web information:
{perplexity_response}

Using this information, answer: {safe_question}
"""
        chatgpt_response = ask_chatgpt(chatgpt_prompt)
        
        # Step 4: Final answer with Ollama
        final_prompt = f"""Sensitive Context (DO NOT SHARE):
{sensitive_context}

Web Research Summary:
{chatgpt_response}

Original Question: {question}

Answer using both contexts without revealing sensitive details. Be concise and technical.
"""
        final_answer = ask_ollama(final_prompt)
        return "ollama", final_answer, final_prompt

    elif non_sensitive_docs:
        # Non-sensitive flow with Perplexity enhancement
        ns_context = "\n\n".join(doc["content"] for doc in non_sensitive_docs)
        perplexity_response = ask_perplexity(question)
        
        chatgpt_prompt = f"""Local Document Context:
{ns_context}

Latest Web Information:
{perplexity_response}

Question: {question}
Answer using both contexts as needed. Be concise.
"""
        answer = ask_chatgpt(chatgpt_prompt)
        return "chatgpt", answer, chatgpt_prompt

    else:
        # No docs found - use Perplexity + ChatGPT
        perplexity_response = ask_perplexity(question)
        chatgpt_prompt = f"""Latest Web Information:
{perplexity_response}

Question: {question}
Answer using the above information. Be concise.
"""
        answer = ask_chatgpt(chatgpt_prompt)
        return "chatgpt", answer, chatgpt_prompt







# Streamlit frontend
def main():
    init_db()

    st.title("Document Storage and Q&A System")
    st.subheader("Add Document")
    doc_content = st.text_area("Document Content")
    is_sensitive = st.checkbox("Mark as Sensitive")
    metadata = st.text_input("Metadata")
    if st.button("Add Document"):
        if doc_content.strip():
            add_document(doc_content, is_sensitive, metadata)
            st.success("Document added successfully.")
        else:
            st.warning("Document content cannot be empty.")

    st.subheader("Ask a Question")
    user_question = st.text_input("Your Question")
    if st.button("Ask"):
        if user_question.strip():
            model_name, answer, prompt = process_question(user_question)
            st.write(f"**Model used:** {model_name}")
            st.write(f"**Answer:** {answer}")
            st.write(f"**prompt:** {prompt}")

        else:
            st.warning("Please enter a question.")

    if st.checkbox("Show All Documents"):
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id, content, is_sensitive, metadata FROM documents")
        docs = cursor.fetchall()
        conn.close()
        st.write(docs)

if __name__ == "__main__":
    main()

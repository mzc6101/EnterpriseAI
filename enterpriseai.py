import sqlite3
import requests
import os
from typing import Optional, List, Dict, Any


# =========================
# CONFIGURATION
# =========================

DATABASE_FILE = "my_local_data.db"

# NOTE: You must set your ChatGPT API key in an environment variable:
#       export OPENAI_API_KEY="your-key-here"
CHATGPT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Default endpoints (adjust as needed)
CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_API_URL = "http://localhost:11411"  # Default Ollama server endpoint


# =========================
# 1. DATABASE OPERATIONS
# =========================

def init_db():
    """
    Initialize (or connect to) the local SQLite database.
    Creates tables if not existing.
    """
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


def add_document(content: str, is_sensitive: bool = False, metadata: Optional[str] = None):
    """
    Insert a document into the local database.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (content, is_sensitive, metadata)
        VALUES (?, ?, ?)
    """, (content, 1 if is_sensitive else 0, metadata))
    conn.commit()
    conn.close()


def retrieve_all_documents() -> List[Dict[str, Any]]:
    """
    Return all documents from the local database.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT content, is_sensitive, metadata FROM documents")
    rows = cursor.fetchall()
    conn.close()

    docs = []
    for row in rows:
        doc_content, doc_sens, doc_meta = row
        docs.append({
            "content": doc_content,
            "is_sensitive": bool(doc_sens),
            "metadata": doc_meta
        })
    return docs


def retrieve_relevant_docs(query: str) -> List[Dict[str, Any]]:
    """
    Very naive text-based retrieval: checks if query is a substring of a doc.
    In production, you'd use embeddings + a vector store (FAISS, Chroma, etc.).
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # Let's just fetch all docs and do naive filter in Python
    cursor.execute("SELECT content, is_sensitive, metadata FROM documents")
    rows = cursor.fetchall()
    conn.close()

    relevant = []
    query_lower = query.lower()
    for row in rows:
        doc_content, doc_sens, doc_meta = row
        if query_lower in doc_content.lower():
            relevant.append({
                "content": doc_content,
                "is_sensitive": bool(doc_sens),
                "metadata": doc_meta
            })
    return relevant


# =========================
# 2. CHATGPT API CALL
# =========================

def ask_chatgpt(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Securely call ChatGPT's API, ensuring no sensitive data is sent.
    - system_prompt: instructions for ChatGPT's system role
    - user_prompt: the minimal user question or text needed
    - model: the ChatGPT model name (e.g., gpt-3.5-turbo, gpt-4, etc.)

    Returns the text response from ChatGPT.
    """
    if not CHATGPT_API_KEY:
        raise ValueError("Missing ChatGPT API key. Please set OPENAI_API_KEY environment variable.")

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

    response = requests.post(CHATGPT_API_URL, headers=headers, json=body, timeout=60)
    response_json = response.json()

    # Handle any errors
    if "error" in response_json:
        raise ValueError(f"ChatGPT API error: {response_json['error']}")

    return response_json["choices"][0]["message"]["content"]


# =========================
# 3. OLLAMA (LOCAL LLM) API
# =========================

def ask_ollama(prompt: str, model: str = "llama2") -> str:
    """
    Send a prompt to a local Ollama server (e.g., 'llama2' model).
    Make sure Ollama is installed and running: https://github.com/jmorganca/ollama
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
                import json
                data = json.loads(chunk)
                if "response" in data:
                    full_text += data["response"]
                if data.get("done", False):
                    break
        return full_text
    except Exception as e:
        return f"Error calling Ollama: {e}"


# =========================
# 4. CORE Q&A LOGIC
# =========================

def process_user_question(question: str) -> str:
    """
    Main Q&A function:
    1) Retrieve relevant documents from the local DB.
    2) Split them into sensitive vs. non-sensitive sets.
    3) Check if local docs likely answer the question (naive approach).
       - If yes, ask Ollama directly with local context.
       - If no, or if the question references something we don't have,
         ask ChatGPT for the missing info (without sending any sensitive data).
    4) Combine ChatGPT's new info (if any) with local sensitive data
       and ask Ollama for the final answer.
    """

    # Step 1: Find relevant local docs
    relevant_docs = retrieve_relevant_docs(question)
    sensitive_docs = [d for d in relevant_docs if d["is_sensitive"]]
    non_sensitive_docs = [d for d in relevant_docs if not d["is_sensitive"]]

    # Combine docs into text blocks
    combined_non_sensitive_context = "\n".join([doc["content"] for doc in non_sensitive_docs])
    combined_sensitive_context = "\n".join([doc["content"] for doc in sensitive_docs])

    # Step 2: Decide if local docs are enough
    # For demonstration, let's do a naive check:
    # "If we found any relevant docs, let's guess they might answer the question."
    # But we also consider if the question might be about "new" or "unknown" data.
    # In real usage, you'd have a more robust method (embedding similarity, etc.).
    local_info_sufficient = False
    if len(relevant_docs) > 0:
        # Naive assumption: if we found something, let's consider it "potentially sufficient"
        # unless user question hints at new data (like "latest", "recent", "current", "today", etc.).
        # You can expand this logic as needed.
        lower_q = question.lower()
        # If we see certain keywords, we guess user wants new info not in the local DB.
        if any(k in lower_q for k in ["latest", "current", "today", "recent", "live"]):
            local_info_sufficient = False
        else:
            local_info_sufficient = True

    # Step 3: If local info is sufficient, ask Ollama with local context
    if local_info_sufficient:
        ollama_prompt = f"""
You have the following local context (non-sensitive):
{combined_non_sensitive_context}

You also have sensitive context, which you MUST NOT reveal to the user:
{combined_sensitive_context}

User's question: {question}

Using ONLY the local context above, please answer the user's question. 
Do not reveal or leak sensitive data. 
If the user question is fully answered by local data, answer it carefully.
"""
        ollama_answer = ask_ollama(ollama_prompt)
        return ollama_answer.strip()

    # Step 4: If local info is NOT sufficient, we ask ChatGPT for the missing piece
    # WITHOUT sending any sensitive data. In a real scenario, you'd parse or analyze
    # the question to see exactly what you need from ChatGPT. Here, we simply pass
    # the entire question as "lack of local data" scenario, but we won't attach
    # the sensitive docs to ChatGPT.
    try:
        # Minimal system prompt to remind GPT to respond concisely
        system_prompt = (
            "You are a helpful, concise assistant. Provide the best available factual answer. "
            "Do NOT include any personal or private data in your response."
        )
        # We pass only the user question that presumably doesn't contain
        # sensitive data. If the user question itself had sensitive data,
        # you would need to sanitize it here (not shown).
        chatgpt_response = ask_chatgpt(system_prompt, question)
    except Exception as e:
        return f"Error calling ChatGPT for external info: {e}"

    # Step 5: Now we have ChatGPT's answer (which presumably includes the missing data).
    # Combine it with any local sensitive data that might be relevant, then
    # let Ollama produce the final user-facing answer.
    final_prompt = f"""
Local non-sensitive context:
{combined_non_sensitive_context}

Sensitive context (not to be revealed):
{combined_sensitive_context}

New external info from ChatGPT (to help answer the question):
{chatgpt_response}

User question: {question}

Now combine the local context (including sensitive data internally), 
and the new external info from ChatGPT. Produce a final answer for the user. 
DO NOT reveal or leak the sensitive data from local context.
"""
    final_answer = ask_ollama(final_prompt)
    return final_answer.strip()


# =========================
# 5. DEMO / MAIN
# =========================

if __name__ == "__main__":
    # Make sure our local DB is initialized
    init_db()

    # (Optional) Add some sample documents:
    # -- Run once to populate, or wrap in if not exists checks, etc. --
    # NOTE: In real usage, do this only once, or have a separate script to ingest data.

    # Example non-sensitive doc:
    add_document(
        content="Python is a popular programming language created by Guido van Rossum.",
        is_sensitive=False,
        metadata="General knowledge about Python"
    )
    # Example sensitive doc:
    add_document(
        content="CompanyXYZ's internal formula is: revenue_projection = (last_year_revenue * 1.12).",
        is_sensitive=True,
        metadata="Sensitive financial formula"
    )

    print("Local documents in DB:")
    for doc in retrieve_all_documents():
        print(" -", doc)

    # Example user questions:
    questions = [
        "Tell me about Python's creator.",
        "What is the latest info on NASA's Artemis program?",
        "Show me the internal formula used by CompanyXYZ for revenue projection."
    ]

    for q in questions:
        print(f"\nUser Question: {q}")
        answer = process_user_question(q)
        print("Assistant Answer:", answer)
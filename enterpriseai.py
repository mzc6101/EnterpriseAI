import streamlit as st
import sqlite3
import requests
import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch

load_dotenv()

# === Configuration ===
DATABASE_FILE = "my_local_data.db"
FAISS_INDEX_FILE = "my_faiss.index"

# Embedding/Vector Index Setup
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
dimension = 384  # Embedding dimension for MiniLM
similarity_threshold = 0.6

# External APIs
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")

# Local LLM for rewriting or final on-device answers
LOCAL_LLM_MODEL_NAME = "distilgpt2"
tokenizer_local = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_NAME)
local_model = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_MODEL_NAME)

# Optionally, an on-device model via Ollama (if you prefer to call it):
OLLAMA_API_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:8b"

# A small question-generation model (optional) – not strictly required unless you want separate rewriting logic.
# For demonstration, we’ll just use distilgpt2 for rewriting in Enhanced mode.
# QG_MODEL_NAME = "valhalla/t5-small-qg-hl"
# qg_tokenizer = AutoTokenizer.from_pretrained(QG_MODEL_NAME)
# qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL_NAME)

# === Zero-Shot Classification Pipeline for Sensitive Content Detection ===
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def detect_sensitive(text: str) -> bool:
    """Perform multi-layer sensitivity detection."""
    sensitive_patterns = [
        r"\b(ssn|social security|credit card|password)\b",
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
    ]
    # Check regex patterns first
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in sensitive_patterns):
        return True

    # Then run zero-shot classification
    classifier_labels = ["personal identification", "financial information", "medical records", "non-sensitive"]
    result = classifier(text, classifier_labels)
    # If top label is sensitive with high confidence
    return result["scores"][0] > 0.7 and result["labels"][0] != "non-sensitive"

def ask_local_llm(prompt: str) -> str:
    """Query the small on-device LLM (distilgpt2) with a prompt."""
    inputs = tokenizer_local.encode(prompt, return_tensors="pt")
    outputs = local_model.generate(inputs, max_length=200, do_sample=True, top_p=0.95, temperature=0.8)
    response = tokenizer_local.decode(outputs[0], skip_special_tokens=True)
    return response

def ask_ollama(prompt: str) -> str:
    """
    If you have Ollama or another local LLM running, you can query it here.
    Otherwise, skip or replace with ask_local_llm.
    """
    req_body = {
        "prompt": prompt,
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    try:
        response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=req_body, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            return response.json().get("response", "Error: Missing 'response' in Ollama output.")
        else:
            return f"Error: Ollama returned status {response.status_code}"
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

def ask_perplexity(question: str) -> str:
    """Query Perplexity API for live external data."""
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are to find the latest information from the internet based on the question "
                    "and provide a very short and concise step-by-step response."
                )
            },
            {"role": "user", "content": question}
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
        else:
            return f"Error: Perplexity returned status {response.status_code}"
    except Exception as e:
        return f"Error: Perplexity request failed - {str(e)}"

def decompose_query(context: str, question: str) -> tuple:
    """
    Extract relevant ticker symbols and share counts from the local doc.
    """
    # We'll just have a simple prompt to the local LLM to see if it can parse ticker:quantity from context
    prompt = f"""Context:
{context}

Question:
{question}

Extract the ticker and share quantity, if any, in the format TICKER:QUANTITY. 
Output example:
Local Data:
AAPL:100
NVDA:200
---
End
"""
    raw_extraction = ask_local_llm(prompt)

    # Simple regex to detect lines like "AAPL:100" or "GOOG: 200"
    matches = re.findall(r"(\b[A-Za-z]{1,5}\b)\s*:\s*([\d,]+)", raw_extraction)
    local_data = {}
    for m in matches:
        ticker, qty = m
        ticker = ticker.upper()
        qty = int(qty.replace(",", "")) if qty.replace(",", "").isdigit() else 0
        local_data[ticker] = qty

    return local_data, raw_extraction

def extract_prices(perplexity_response: str) -> dict:
    """Parse ticker prices out of the Perplexity response if present."""
    # E.g. detect lines like "AAPL $179.32" or "GOOGL $128.55"
    pattern = r"\b(AAPL|GOOGL?|NVDA|TSLA|META|AMZN)\b.*?\$([0-9,]+\.\d+)"
    matches = re.findall(pattern, perplexity_response, re.IGNORECASE)
    prices = {}
    for (ticker, price_str) in matches:
        price_str = price_str.replace(",", "")
        ticker = ticker.upper()
        try:
            prices[ticker] = float(price_str)
        except:
            continue
    return prices

def compute_portfolio_value(local_data: dict, prices: dict) -> str:
    """Compute total portfolio value given local share counts + live prices."""
    total = 0.0
    lines = []
    for tkr, qty in local_data.items():
        if tkr in prices:
            val = qty * prices[tkr]
            lines.append(f"{tkr}: {qty} × ${prices[tkr]:,.2f} = ${val:,.2f}")
            total += val
        else:
            lines.append(f"{tkr}: price unavailable")
    lines.append(f"\nTotal Portfolio Value: ${total:,.2f}")
    return "\n".join(lines)

def secure_code_assistance(code: str, question: str) -> str:
    """
    If someone asks about code, show sanitized code without credentials.
    Then optionally do logic for code assistance. 
    (Kept minimal here.)
    """
    # Basic sanitization
    sanitized_code = re.sub(r"(API_?KEY|SECRET|PASSWORD)\s*=\s*['\"].+?['\"]", "CREDENTIAL='[REDACTED]'", code, flags=re.IGNORECASE)
    response = f"Here is sanitized code:\n{sanitized_code}\n\n"
    response += "High-level tips: [Add your own code guidance or call a local LLM here]"
    return response

# === Database + FAISS Index ===
faiss_index = None

def init_db():
    """Initialize the local SQLite database and FAISS index if not present."""
    global faiss_index
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                is_sensitive BOOLEAN NOT NULL DEFAULT 0,
                metadata TEXT,
                authorized_users TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                log_data TEXT
            );
        """)
        cursor.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'authorized_users' not in columns:
            cursor.execute("ALTER TABLE documents ADD COLUMN authorized_users TEXT")
        conn.commit()
    except sqlite3.Error as e:
        print("Database error:", e)
    finally:
        conn.close()

    # Load or create FAISS index
    if os.path.exists(FAISS_INDEX_FILE):
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    else:
        base_index = faiss.IndexFlatIP(dimension)
        faiss_index = faiss.IndexIDMap(base_index)

def save_faiss_index():
    """Persist the FAISS index to disk."""
    if faiss_index is not None:
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)

def add_document(content: str, metadata: str = "", authorized_users: str = ""):
    """
    Insert a new document into SQLite and FAISS.
    We auto-detect if it's sensitive with `detect_sensitive`.
    """
    sensitive = detect_sensitive(content)
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (content, is_sensitive, metadata, authorized_users)
        VALUES (?, ?, ?, ?)
    """, (content, 1 if sensitive else 0, metadata, authorized_users))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Add to FAISS index
    embedding = embedding_model.encode(content).astype(np.float32)
    embedding = embedding.reshape(1, -1)
    faiss_index.add_with_ids(embedding, np.array([doc_id], dtype=np.int64))
    save_faiss_index()

def retrieve_relevant_docs(query: str, top_k=5):
    """
    Use FAISS to retrieve top-k matching documents for the query.
    """
    global faiss_index
    query_emb = embedding_model.encode(query.lower()).astype(np.float32)
    query_emb = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb)

    distances, indices = faiss_index.search(query_emb, top_k)
    results = []
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1 or dist < similarity_threshold:
            continue
        cursor.execute("SELECT id, content, is_sensitive, metadata, authorized_users FROM documents WHERE id=?", (idx,))
        row = cursor.fetchone()
        if row:
            results.append({
                "id": row[0],
                "content": row[1],
                "is_sensitive": bool(row[2]),
                "metadata": row[3],
                "similarity": dist,
                "authorized_users": row[4].split(",") if row[4] else []
            })
    conn.close()

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results

def sanitize_context(context: str) -> str:
    """
    Remove sensitive sentences from the context for Enhanced or Max modes if needed.
    (You can adapt how aggressive you want to be.)
    """
    lines = context.split('\n')
    safe = []
    for ln in lines:
        if detect_sensitive(ln):
            safe.append("[REDACTED]")
        else:
            safe.append(ln)
    return "\n".join(safe)

def process_question(question: str, user_id: str, security_level: str):
    """
    Single pipeline controlling the flow based on security_level:
      - Basic: Direct to Perplexity. No rewriting.
      - Enhanced: Rewrite question with local LLM, then query Perplexity.
      - Maximum: On-device only (no Perplexity).
    """
    # 1) Retrieve relevant docs
    docs = retrieve_relevant_docs(question)
    # 2) Check if user can see sensitive docs
    sens_docs = [d for d in docs if d["is_sensitive"]]
    authorized = [d for d in sens_docs if user_id in d["authorized_users"]]

    if sens_docs and not authorized:
        return "Access Denied", "You are not authorized to view sensitive info.", "", ""

    # Combine content from relevant docs
    combined_context = "\n\n".join(d["content"] for d in docs)

    # 3) Switch on security_level
    if security_level == "Basic":
        # Always send the raw question to Perplexity. No rewriting, no redaction.
        perplexity_resp = ask_perplexity(question + "\nLocal info:\n" + combined_context)


        # If user wants a portfolio value:
        if re.search(r"\bportfolio\b", question, re.IGNORECASE):
            local_data, _ = decompose_query(combined_context, question)
            prices = extract_prices(perplexity_resp)
            if local_data and prices:
                answer = compute_portfolio_value(local_data, prices)
                return "Basic (Perplexity)", answer, question, perplexity_resp
            else:
                return "Basic (Perplexity)", "No stock price or share data found to compute portfolio value.", question, perplexity_resp
        else:
            # For non-portfolio questions, we can pass local doc + perplexity to a final summarizer if you want.
            # Or just return them. Example: use Ollama or local LLM to finalize:
            final_prompt = f"""
Context from local docs:
{combined_context}

Live Data from Perplexity:
{perplexity_resp}

Question:
{question}

Task:
- Provide a concise answer (50 words max).
"""
            final_answer = ask_ollama(final_prompt)
            return "Basic (Perplexity)", final_answer, question, perplexity_resp

    elif security_level == "Enhanced":
        # Rewrite question with local LLM
        rewrite_prompt = f"Rewrite the following user question to remove any sensitive info and produce a sanitized query:\n\nUser Question: {question}\n\nSanitized Query:"
        sanitized_q = ask_local_llm(rewrite_prompt)

        # Now send that sanitized query to Perplexity
        perplexity_resp = ask_perplexity(sanitized_q)

        # If user wants a portfolio value:
        if re.search(r"\bportfolio\b", question, re.IGNORECASE):
            local_data, _ = decompose_query(combined_context, question)
            prices = extract_prices(perplexity_resp)
            if local_data and prices:
                answer = compute_portfolio_value(local_data, prices)
                return "Enhanced (Local LLM + Perplexity)", answer, sanitized_q, perplexity_resp
            else:
                return "Enhanced (Local LLM + Perplexity)", "No stock price or share data found.", sanitized_q, perplexity_resp
        else:
            # Redact doc context if you want to be consistent with Enhanced
            safe_context = sanitize_context(combined_context)
            final_prompt = f"""
Safe Context from local docs:
{safe_context}

Live Data from Perplexity:
{perplexity_resp}

User's Original Question:
{question}

Task:
- Provide a concise answer in 50 words or less.
"""
            final_answer = ask_ollama(final_prompt)
            return "Enhanced (Local LLM + Perplexity)", final_answer, sanitized_q, perplexity_resp

    else:  # "Maximum": On-device only, no Perplexity
        # We'll rely solely on local docs + local LLM for answers.
        # If user wants a portfolio, we have no external price data. 
        # We can see if the doc content includes any reference to prices, or disclaim.
        if re.search(r"\bportfolio\b", question, re.IGNORECASE):
            # Check if we can parse something from local docs
            local_data, _ = decompose_query(combined_context, question)
            # If doc doesn't have any stock price lines, disclaim.
            # Otherwise, attempt to parse them from text (not guaranteed to exist).
            # We'll do a naive parse from the doc for lines like "AAPL is at $..."
            local_prices = {}
            price_lines = re.findall(r"(AAPL|GOOGL?|NVDA|TSLA)\s+.*?\$([0-9,]+\.\d+)", combined_context, re.IGNORECASE)
            for (t, p) in price_lines:
                t = t.upper()
                val = float(p.replace(",", "")) if p.replace(",", "").isdigit() else 0.0
                local_prices[t] = val

            if local_data and local_prices:
                ans = compute_portfolio_value(local_data, local_prices)
                return "Maximum (On-Device)", ans, question, "No Perplexity (Offline)"
            else:
                return "Maximum (On-Device)", (
                    "Unable to compute portfolio value. No external data available and none found in local docs."
                ), question, "No Perplexity (Offline)"
        else:
            # For a normal question, just do a local LLM answer from the doc context.
            final_prompt = f"""
You are an on-device LLM. Answer the question based solely on the context below.

Context:
{combined_context}

Question:
{question}

Provide a concise answer:
"""
            final_answer = ask_local_llm(final_prompt)
            return "Maximum (On-Device)", final_answer, question, "No Perplexity (Offline)"

def log_processing(steps: dict):
    """Example function for logging input/output to the DB if desired."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO audit_logs (log_data) VALUES (?)", (json.dumps(steps),))
    conn.commit()
    conn.close()

def main():
    init_db()
    st.title("Secure Hybrid Document Q&A")

    # --- Security Settings ---
    with st.sidebar:
        st.header("Security Settings")
        security_level = st.select_slider(
            "Data Protection Level",
            options=["Basic", "Enhanced", "Maximum"],
            value="Basic"
        )

    # --- Add Document ---
    st.subheader("Add Document")
    doc_content = st.text_area("Document Content")
    metadata = st.text_input("Metadata")
    authorized_users = st.text_input("Authorized Users (comma-separated)")

    if st.button("Add Document"):
        if doc_content.strip():
            add_document(doc_content, metadata=metadata, authorized_users=authorized_users)
            st.success("Document added successfully.")
        else:
            st.warning("Document content cannot be empty.")

    # --- Ask a Question ---
    st.subheader("Ask a Question")
    user_id = st.text_input("Your ID")
    user_question = st.text_input("Your Question")

    if st.button("Ask"):
        if user_question.strip():
            # If the user is asking about code, do a specialized routine:
            if "code" in user_question.lower():
                # Retrieve code-related docs
                code_docs = [
                    d for d in retrieve_relevant_docs(user_question, 5)
                    if "code" in (d["metadata"] or "").lower()
                ]
                combined_code = "\n".join(d["content"] for d in code_docs)
                answer = secure_code_assistance(combined_code, user_question)
                st.write("**Model used:** Code-assistant (Local)")
                st.write("**Answer:**")
                st.write(answer)

            else:
                # Normal question path
                model_used, answer, safe_q, ext_resp = process_question(user_question, user_id, security_level)
                st.write(f"**Model used:** {model_used}")
                st.write(f"**Answer:** {answer}")
                if safe_q:
                    st.write(f"**Query sent to external or final LLM:** {safe_q}")
                if ext_resp:
                    st.write(f"**External/Live Data Response:** {ext_resp[:500]}...")

        else:
            st.warning("Please enter a question.")

    # Debug: Show All Docs
    if st.checkbox("Show All Documents"):
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id, content, is_sensitive, metadata, authorized_users FROM documents")
        rows = cursor.fetchall()
        conn.close()
        st.write(rows)

if __name__ == "__main__":
    main()

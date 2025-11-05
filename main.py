import os
import gradio as gr
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from typing import Dict, List, Tuple
from huggingface_hub import login
import sqlite3
from pathlib import Path

# --- SILENCE VERBOSE LOGS ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # Stops tokenizer warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Login for gated models
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# ==============================
# 1. MULTI-SCHOOL KB (Chunked – tiny FAISS)
# ==============================
SCHOOLS = { ... }  # ← same as before (keep your data)

# Build FAISS (quiet)
indices = {}
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
for code, data in SCHOOLS.items():
    embeddings = embedder.encode(data["chunks"], convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    indices[code] = {"index": index, "chunks": data["chunks"], "name": data["name"]}

# ==============================
# 2. MODEL: Mistral-7B (Quiet + Streaming)
# ==============================
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    # Stream download → smaller log chunks
    resume_download=True,
    local_files_only=False
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# ==============================
# 3. SQLITE DB
# ==============================
DB_FILE = "leads.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campus TEXT,
    name TEXT,
    email TEXT,
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def save_lead(campus: str, name: str, email: str, message: str):
    cursor.execute(
        "INSERT INTO leads (campus, name, email, message) VALUES (?, ?, ?, ?)",
        (campus, name, email, message)
    )
    conn.commit()

def view_leads():
    cursor.execute("SELECT campus, name, email, message, timestamp FROM leads ORDER BY timestamp DESC LIMIT 50")
    rows = cursor.fetchall()
    if not rows: return "No leads yet."
    return "\n".join([f"**{r[0]}**: {r[1]} ({r[2]}) – {r[4]}" for r in rows])

# ==============================
# 4. RAG RESPONSE (unchanged logic)
# ==============================
def rag_response(message: str, history: List[Tuple[str, str]], school_code: str) -> str:
    # ... same as before ...
    # Use save_lead() instead of dict
    if email_match:
        save_lead(school["name"], name, email, message)
        response += f"\n\n(Lead saved: {name} / {email})"
    return response

# ==============================
# 5. GRADIO + FASTAPI (same)
# ==============================
# ... keep your UI code ...

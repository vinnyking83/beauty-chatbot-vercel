import os
import io
import csv
import gradio as gr
import torch
import sqlite3
import re
from typing import Dict, List, Tuple
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

# Silence logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger().setLevel(logging.CRITICAL)

app = FastAPI()

# ==============================
# 1. SCHOOLS & RAG
# ==============================
SCHOOLS = {
    "la": {"name": "Glamour LA", "chunks": ["Cosmetology: $4,999", "Esthetics: $2,999", "Payment plans: $200/mo"]},
    "nyc": {"name": "Glamour NYC", "chunks": ["Cosmetology: $5,499", "Financial aid available"]},
    "miami": {"name": "Glamour Miami", "chunks": ["Cosmetology: $4,799", "Bilingual classes"]}
}

indices = {}
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
for code, data in SCHOOLS.items():
    embeddings = embedder.encode(data["chunks"], convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    indices[code] = {"index": index, "chunks": data["chunks"], "name": data["name"]}

# ==============================
# 2. PHI-3-MINI (Auto-downloads on Vercel)
# ==============================
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True
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
# 3. SQLITE
# ==============================
conn = sqlite3.connect("leads.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY,
    campus TEXT,
    name TEXT,
    email TEXT,
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
conn.commit()

def save_lead(campus, name, email, message):
    cursor.execute("INSERT INTO leads (campus, name, email, message) VALUES (?, ?, ?, ?)", (campus, name, email, message))
    conn.commit()

def view_leads():
    cursor.execute("SELECT campus, name, email, timestamp FROM leads ORDER BY timestamp DESC LIMIT 50")
    rows = cursor.fetchall()
    return "\n".join([f"**{r[0]}**: {r[1]} ({r[2]}) – {r[3]}" for r in rows]) if rows else "No leads."

# ==============================
# 4. RAG RESPONSE
# ==============================
def rag_response(message: str, history: List[Tuple[str, str]], school_code: str) -> str:
    school = SCHOOLS[school_code]
    idx = indices[school_code]["index"]
    chunks = indices[school_code]["chunks"]
    q_emb = embedder.encode([message], convert_to_numpy=True)
    _, I = idx.search(q_emb, 3)
    context = "\n".join([chunks[i] for i in I[0]])

    hist = "\n".join([f"Student: {u}\nCounselor: {a}" for u, a in history[-4:]])
    prompt = f"""You are an admissions counselor at {school['name']}. Use ONLY context. Be warm, sales-focused.

Context: {context}

{hist}
Student: {message}
Counselor:"""

    output = generator(prompt)[0]['generated_text']
    response = output.split("Counselor:")[-1].strip()[:250]

    if email := re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message):
        name = message.split(email.group())[0].strip() or "Student"
        save_lead(school["name"], name, email.group(), message)
        response += f"\n\n(Lead saved: {name})"

    return response

# ==============================
# 5. CSV DOWNLOAD
# ==============================
@app.get("/download-leads")
async def download_leads():
    cursor.execute("SELECT * FROM leads ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    if not rows: return {"error": "No leads"}
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Campus", "Name", "Email", "Message", "Time"])
    writer.writerows(rows)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=leads.csv"}
    )

# ==============================
# 6. GRADIO UI
# ==============================
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Beauty School AI Chatbot")
        campus = gr.Dropdown([("LA", "la"), ("NYC", "nyc"), ("Miami", "miami")], value="la", label="Campus")

        with gr.Tab("Chat"):
            gr.ChatInterface(lambda m, h, c: rag_response(m, h, c), additional_inputs=[campus])

        with gr.Tab("Admin"):
            gr.Markdown("Login: `admin` / `pass123`")
            u = gr.Textbox(label="User")
            p = gr.Textbox(label="Pass", type="password")
            login = gr.Button("Login")
            out = gr.Markdown(visible=False)
            dl = gr.Button("Download CSV", visible=False)
            rf = gr.Button("Refresh", visible=False)

            login.click(
                lambda u, p: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)) if u=="admin" and p=="pass123" else (gr.update(), gr.update(), gr.update()),
                [u, p], [out, dl, rf]
            )
            rf.click(view_leads, outputs=out)
            dl.click(None, _js="() => window.open('/download-leads', '_blank')")

    return demo

demo = create_interface()
app = gr.mount_gradio_app(app, demo, path="/")

@app.get("/") 
async def root(): 
    return {"status": "ready"}

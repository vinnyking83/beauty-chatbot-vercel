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
from huggingface_hub import login
import logging

# ==============================
# 0. SILENCE VERBOSE LOGS (Fix "data too long")
# ==============================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Login for gated models
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# ==============================
# 1. MULTI-SCHOOL KNOWLEDGE BASE
# ==============================
SCHOOLS = {
    "la": {
        "name": "Glamour Academy Los Angeles",
        "city": "Los Angeles, CA",
        "chunks": [
            "Cosmetology Diploma: 1,500 hours, $4,999. Includes full kit ($500 value).",
            "Esthetics: 600 hours, $2,999. Skincare, makeup, waxing.",
            "Payment plans: $200/month interest-free. 95% job placement.",
            "Campus: 123 Sunset Blvd, LA. Free parking."
        ]
    },
    "nyc": {
        "name": "Glamour Academy New York",
        "city": "New York, NY",
        "chunks": [
            "Cosmetology Diploma: 1,500 hours, $5,499. Premium kit included.",
            "Hair Styling: 1,000 hours, $3,499. Focus on color & cuts.",
            "Financial aid available. 50+ salon partners in NYC.",
            "Campus: 456 Broadway, Manhattan. Near subway."
        ]
    },
    "miami": {
        "name": "Glamour Academy Miami",
        "city": "Miami, FL",
        "chunks": [
            "All programs: Cosmetology $4,799, Esthetics $3,199.",
            "Bilingual classes (English/Spanish).",
            "Career services: Resume, interviews, spa partnerships.",
            "Campus: 789 Ocean Drive, Miami Beach."
        ]
    }
}

# Build FAISS indices (quiet)
indices = {}
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
for code, data in SCHOOLS.items():
    embeddings = embedder.encode(data["chunks"], convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    indices[code] = {"index": index, "chunks": data["chunks"], "name": data["name"]}

# ==============================
# 2. MODEL: Mistral-7B (CPU, streaming download)
# ==============================
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    padding_side="left"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype(torch.float16),
    device_map="cpu",
    low_cpu_mem_usage=True,
    resume_download=True
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
# 3. SQLITE DATABASE
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
    if not rows:
        return "No leads yet."
    return "\n".join([f"**{r[0]}**: {r[1]} ({r[2]}) – {r[4]}" for r in rows])

# ==============================
# 4. RAG RESPONSE
# ==============================
def get_school_from_query(query: str) -> str:
    q = query.lower()
    if "la" in q or "los angeles" in q: return "la"
    if "nyc" in q or "new york" in q: return "nyc"
    if "miami" in q: return "miami"
    return "la"

def rag_response(message: str, history: List[Tuple[str, str]], school_code: str) -> str:
    school = SCHOOLS[school_code]
    idx = indices[school_code]["index"]
    chunks = indices[school_code]["chunks"]

    q_emb = embedder.encode([message], convert_to_numpy=True)
    D, I = idx.search(q_emb, 3)
    context = "\n".join([chunks[i] for i in I[0]])

    hist = "\n".join([f"Student: {u}\nCounselor: {a}" for u, a in history[-4:]])

    prompt = f"""<s>[INST] You are an admissions counselor at {school['name']}. Use ONLY the context. Be warm, professional, sales-focused. Qualify lead. Collect name/email.

Context:
{context}

Conversation:
{hist}

Student: {message} [/INST]"""

    output = generator(prompt)[0]['generated_text']
    response = output.split("[/INST]")[-1].strip()
    response = re.sub(r'<[^>]+>', '', response).split('\n')[0][:250]

    # Lead capture
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message)
    if email_match:
        email = email_match.group()
        name_parts = message.split(email)[0].strip().split()
        name = " ".join(name_parts[-2:]) if len(name_parts) >= 2 else "Friend"
        save_lead(school["name"], name, email, message)
        response += f"\n\n(Lead saved: {name} / {email})"

    return response

# ==============================
# 5. CSV DOWNLOAD ENDPOINT
# ==============================
app = FastAPI()

@app.get("/download-leads")
async def download_leads():
    cursor.execute("SELECT id, campus, name, email, message, timestamp FROM leads ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    if not rows:
        return {"error": "No leads to download."}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Campus", "Name", "Email", "Message", "Timestamp"])
    writer.writerows(rows)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=beauty_leads.csv"}
    )

# ==============================
# 6. GRADIO UI
# ==============================
def create_interface():
    with gr.Blocks(title="Multi-Campus Beauty AI") as demo:
        gr.Markdown("# Multi-Campus AI Admissions Chatbot\n**One bot, many locations**")

        with gr.Row():
            campus_dropdown = gr.Dropdown(
                choices=[(s["name"], code) for code, s in SCHOOLS.items()],
                value="la",
                label="Select Campus"
            )

        with gr.Tab("Student Chat"):
            def chat_fn(message, history, campus):
                return rag_response(message, history, campus)
            gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[campus_dropdown],
                title="Ask About Programs & Pricing",
                examples=[
                    ["How much is cosmetology in LA?"],
                    ["NYC pricing?"],
                    ["Enroll me: Ana ana@miami.edu"]
                ]
            )

        with gr.Tab("Admin – All Leads"):
            gr.Markdown("Login: `admin` / `pass123`")
            uname = gr.Textbox(label="Username")
            pwd = gr.Textbox(label="Password", type="password")
            login = gr.Button("Login")

            out = gr.Markdown(visible=False)
            download_btn = gr.Button("Download Leads (CSV)", visible=False)
            refresh_btn = gr.Button("Refresh Leads", visible=False)

            def show_admin():
                return (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True)
                )

            login.click(
                lambda u, p: show_admin() if u == "admin" and p == "pass123" else (gr.update(), gr.update(), gr.update()),
                [uname, pwd],
                [out, download_btn, refresh_btn]
            )

            refresh_btn.click(view_leads, outputs=out)
            download_btn.click(
                None,
                _js="() => { window.open('/download-leads', '_blank'); }"
            )

    return demo

# Mount Gradio
demo = create_interface()
app = gr.mount_gradio_app(app, demo, path="/")

# Health check
@app.get("/")
async def root():
    return {"message": "Beauty Chatbot API - Visit / for Gradio UI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

# Login for gated models (uses env var)
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# ==============================
# 1. MULTI-SCHOOL KNOWLEDGE BASE (Same as before)
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

# Build FAISS indices
indices = {}
embedder = SentenceTransformer('all-MiniLM-L6-v2')
for code, data in SCHOOLS.items():
    embeddings = embedder.encode(data["chunks"], convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    indices[code] = {"index": index, "chunks": data["chunks"], "name": data["name"]}

# ==============================
# 2. MODEL: Mistral-7B (CPU-friendly for Vercel; swap to Llama if GPU later)
# ==============================
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
print(f"Loading {MODEL_ID}... (CPU mode for Vercel)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",  # Vercel CPU; use "auto" for GPU if upgraded
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
# 3. TEMP DBs + FUNCTIONS (Same logic)
# ==============================
user_db = {"admin": "pass123"}
leads_db: Dict[str, List[Dict]] = {}

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
        name = " ".join(message.split(email)[0].strip().split()[-2:])
        lead = {"name": name or "Unknown", "email": email, "campus": school["name"]}
        leads_db.setdefault(school_code, []).append(lead)
        response += f"\n\n(Lead saved: {name} / {email})"

    return response

def view_leads():
    if not leads_db: return "No leads."
    return "\n".join([f"**{SCHOOLS[c]['name']}**: {l['name']} ({l['email']})" for c, leads in leads_db.items() for l in leads])

# ==============================
# 4. GRADIO INTERFACE
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
            login.click(lambda u, p: gr.update(visible=(u=="admin" and p=="pass123")), [uname, pwd], out)
            gr.Button("Refresh").click(view_leads, outputs=out)

    return demo

# ==============================
# 5. FASTAPI APP (Vercel Entrypoint)
# ==============================
app = FastAPI()

# Mount Gradio on root path
demo = create_interface()
app = gr.mount_gradio_app(app, app=demo, path="/")

# Health check for Vercel
@app.get("/")
async def root():
    return {"message": "Beauty Chatbot API - Visit / for Gradio UI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

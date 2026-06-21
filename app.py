import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ── Init ──────────────────────────────────────────────────────────────────────
pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
llm   = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

NAMESPACES = ["it", "software_engineering", "computer_science",
              "artificial_intelligence", "cyber_security", "data_science"]

GREETINGS = {"hi", "hello", "hey", "salam", "assalam", "hi there", "hello there"}

SYSTEM_PROMPT = """You are FYP Mentor, an intelligent assistant specialized EXCLUSIVELY in Final Year Project (FYP) ideas for computing students.

YOUR KNOWLEDGE BASE contains FYP projects from these 6 fields only:
- Information Technology (IT)
- Software Engineering (SE)
- Computer Science (CS)
- Artificial Intelligence (AI)
- Cyber Security
- Data Science

STRICT RULES — follow every rule without exception:

1. ONLY answer using the provided context. Never use outside knowledge.
2. If the user asks for MORE projects (e.g. "give me more", "suggest more", "aur projects", "more ideas"):
   - Suggest DIFFERENT projects than what was already suggested in this conversation.
   - ONLY suggest projects from the SAME field the user originally asked about. Do NOT mix projects from other fields.
   - Only suggest projects present in the context.
3. If the user asks about a SPECIFIC project (e.g. tech stack, abstract, problem statement):
   - Answer only from context. Quote the relevant fields directly.
4. If the user asks a question UNRELATED to FYP or computing (e.g. cooking, geography, jokes, general knowledge):
   - Reply exactly: "I'm FYP Mentor. I can only help with Final Year Project ideas in IT, SE, CS, AI, Cyber Security, or Data Science."
5. If the context does not contain enough information to answer:
   - Reply exactly: "I don't have information about that in my FYP database."
6. NEVER generate, invent, or hallucinate any project, technology, or fact not present in the context.
7. NEVER answer coding questions, write code, or give tutorials.
8. Keep responses clear and concise. When listing projects, show maximum 8 at a time with title and a one-line description."""

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 8) -> str:
    embedding = model.encode(query).tolist()
    results = []
    for ns in NAMESPACES:
        res = index.query(vector=embedding, top_k=top_k, namespace=ns, include_metadata=True)
        results.extend(res.matches)
    results.sort(key=lambda x: x.score, reverse=True)
    top = [m for m in results[:top_k] if m.score > 0.3]
    return "\n\n".join(m.metadata.get("text", "") for m in top)

# ── Generate ──────────────────────────────────────────────────────────────────
def generate(query: str, context: str, history: list) -> str:
    if query.strip().lower() in GREETINGS:
        return "Hello! 👋 I'm FYP Mentor. Ask me about Final Year Project ideas in IT, Software Engineering, Computer Science, AI, Cyber Security, or Data Science!"

    if not context.strip():
        return "I don't have information about that in my FYP database."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Last 6 messages for conversation context
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FYP Mentor", page_icon="🎓")
st.title("🎓 FYP Mentor — Semantic Search Chatbot")
st.caption("Ask about Final Year Projects in IT, SE, CS, AI, Cyber Security or Data Science")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about FYP ideas..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching FYP database..."):
            context  = retrieve(prompt)
            response = generate(prompt, context, st.session_state.messages[:-1])
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
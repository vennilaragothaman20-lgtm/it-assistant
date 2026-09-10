"""
app.py
Streamlit chat UI for the IT Troubleshooting Assistant.
Retrieves relevant past tickets from ChromaDB, then asks Groq (GPT-OSS 120B)
to generate a step-by-step answer using that context. Shows which source
tickets were used, ranked by match quality.
 
Run with: streamlit run app.py
"""
 
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
 
load_dotenv()
 
PERSIST_DIR = "chroma_db"
CSV_PATH = "data/it_tickets.csv"
 
st.set_page_config(
    page_title="IT Troubleshooting Assistant",
    page_icon="🛠️",
    layout="wide",
)
 
# ---------------------------------------------------------------------------
# Theme: clean, professional, light. A confident indigo-blue accent (helpdesk
# / enterprise-tool feel) paired with a teal "resolved" status color. Two
# type families: Manrope for headings (structured, modern), Inter for body.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');
 
:root {
    --bg-base: #F6F8FC;
    --bg-surface: #FFFFFF;
    --bg-user: #EEF1FE;
    --accent: #3652D9;
    --accent-dark: #2740B8;
    --ok: #0EA5A0;
    --warn: #F59E0B;
    --text-primary: #1E293B;
    --text-muted: #66748A;
    --border: #E4E9F1;
    --shadow: 0 1px 3px rgba(30, 41, 59, 0.06), 0 1px 2px rgba(30, 41, 59, 0.04);
}
 
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-base); }
#MainMenu, footer { visibility: hidden; }
 
/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--bg-surface);
    border-right: 1px solid var(--border);
}
.sidebar-title {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
}
.sidebar-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 1.2rem;
}
.kb-stat {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 1.3rem;
    color: white;
}
.kb-stat .num {
    font-family: 'Manrope', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1;
}
.kb-stat .label {
    font-size: 0.75rem;
    opacity: 0.85;
    margin-top: 2px;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.02em;
    color: var(--text-muted);
    margin: 0.8rem 0 0.5rem 0;
}
section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    text-align: left;
    background: var(--bg-base);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
    transition: border-color 0.15s ease, background 0.15s ease;
}
section[data-testid="stSidebar"] .stButton button:hover {
    border-color: var(--accent);
    background: var(--bg-user);
    color: var(--accent-dark);
}
 
/* ---------- Main header ---------- */
.console-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow);
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.3rem;
}
.console-header .brand { display: flex; align-items: center; gap: 0.8rem; }
.console-header .brand-icon {
    width: 44px; height: 44px; border-radius: 11px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; flex-shrink: 0;
}
.console-header h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 1.35rem; font-weight: 800;
    color: var(--text-primary); margin: 0; line-height: 1.2;
}
.console-header .subtitle { font-size: 0.85rem; color: var(--text-muted); margin-top: 2px; }
.status-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: var(--ok);
    background: rgba(14, 165, 160, 0.08);
    border: 1px solid rgba(14, 165, 160, 0.3);
    border-radius: 20px; padding: 0.4rem 0.75rem;
    display: flex; align-items: center; gap: 0.4rem; white-space: nowrap;
}
.status-pill .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--ok); box-shadow: 0 0 6px var(--ok);
}
 
/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] { background: transparent; padding: 0.35rem 0; }
 
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    box-shadow: var(--shadow);
    padding: 1rem 1.2rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--bg-user);
    border-radius: 16px;
    padding: 0.7rem 1.1rem;
    max-width: 85%;
    margin-left: auto;
}
[data-testid="stChatMessageContent"] p { color: var(--text-primary); line-height: 1.6; }
 
/* ---------- Sources ---------- */
[data-testid="stExpander"] {
    background: #FAFBFE;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-top: 0.7rem;
    box-shadow: none;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent-dark);
}
.source-row {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.35rem 0; font-size: 0.82rem; color: var(--text-primary);
}
.source-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; font-weight: 600;
    background: var(--bg-user); color: var(--accent-dark);
    border-radius: 5px; padding: 0.15rem 0.4rem; flex-shrink: 0;
}
 
/* ---------- Chat input ---------- */
[data-testid="stChatInput"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--bg-surface);
    box-shadow: var(--shadow);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
 
/* ---------- Empty state ---------- */
.empty-state {
    text-align: center;
    background: var(--bg-surface);
    border: 1px dashed var(--border);
    border-radius: 14px;
    padding: 2.5rem 1rem;
    color: var(--text-muted);
}
.empty-state .icon { font-size: 2rem; margin-bottom: 0.6rem; }
.empty-state .title { font-family: 'Manrope', sans-serif; font-weight: 700; color: var(--text-primary); font-size: 1rem; margin-bottom: 0.3rem; }
</style>
"""
 
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("Set your GROQ_API_KEY environment variable before running this app.")
    st.stop()
 
client = Groq(api_key=api_key)
 
 
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
 
 
@st.cache_data
def ticket_count():
    try:
        return len(pd.read_csv(CSV_PATH))
    except Exception:
        return "—"
 
 
vectorstore = load_vectorstore()
 
QUICK_ISSUES = [
    ("📶", "Wi-Fi keeps disconnecting"),
    ("🔒", "VPN won't connect"),
    ("📧", "Outlook not syncing"),
    ("🖨️", "Printer shows offline"),
    ("🔑", "Forgot my Windows password"),
    ("🎥", "Teams camera not working"),
]
 
 
def get_answer(question: str, vectorstore, client: Groq):
    results = vectorstore.similarity_search_with_score(question, k=3)
    docs = [doc for doc, score in results]
    context = "\n\n---\n\n".join(d.page_content for d in docs)
 
    prompt = f"""You are an IT support assistant. Use the context below (past resolved
IT tickets) to help solve the user's problem. Give clear, numbered, step-by-step
instructions. If the context doesn't contain a relevant answer, say so honestly
and recommend contacting IT support instead of guessing.
 
Context:
{context}
 
User's issue: {question}
 
Answer:"""
 
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    sources = [doc.metadata.get("error", "Unknown") for doc in docs]
    return answer, sources
 
 
def render_sources(sources):
    with st.expander("📎 Sources used"):
        labels = ["Closest match", "2nd match", "3rd match"]
        for i, title in enumerate(sources):
            label = labels[i] if i < len(labels) else f"{i + 1}th match"
            st.markdown(
                f'<div class="source-row"><span class="source-rank">{label}</span>{title}</div>',
                unsafe_allow_html=True,
            )
 
 
def ask(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Searching knowledge base and generating answer..."):
        answer, sources = get_answer(question, vectorstore, client)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
 
 
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛠️ IT Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Powered by RAG + Groq</div>', unsafe_allow_html=True)
 
    st.markdown(
        f"""
        <div class="kb-stat">
            <div class="num">{ticket_count()}</div>
            <div class="label">tickets in knowledge base</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    st.markdown('<div class="section-label">QUICK ISSUES</div>', unsafe_allow_html=True)
    for icon, label in QUICK_ISSUES:
        if st.button(f"{icon}  {label}", key=f"quick_{label}"):
            st.session_state.setdefault("messages", [])
            ask(label)
            st.rerun()
 
    st.markdown("---")
    if st.button("🗑️  Clear conversation"):
        st.session_state.messages = []
        st.rerun()
 
# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="console-header">
        <div class="brand">
            <div class="brand-icon">🛠️</div>
            <div>
                <h1>IT Troubleshooting Assistant</h1>
                <div class="subtitle">Ask about any IT error and get step-by-step help</div>
            </div>
        </div>
        <div class="status-pill"><span class="dot"></span>Online</div>
    </div>
    """,
    unsafe_allow_html=True,
)
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">💬</div>
            <div class="title">No conversation yet</div>
            <div>Type an issue below, or pick a quick issue from the sidebar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🛠️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])
 
question = st.chat_input("Describe your IT issue...")
if question:
    ask(question)
    st.rerun()
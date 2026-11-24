"""Agri Assistant: a web interface for the local RAG chatbot.

Run (venv active, Ollama running):
    streamlit run app.py

It reuses vector.py (the knowledge base + retriever), so build the database
once with `python main.py` first, or just let this app build it on first load.
Works with both the PDF version and the CSV version of vector.py.
"""
import html
from collections import Counter

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.set_page_config(
    page_title="Agri Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------- settings
DEFAULT_MODELS = ["llama3.2", "llama3.1:8b"]

# Keep identical to the template in main.py
PROMPT = """
You are an assistant that answers questions about the provided documents.
Answer using ONLY the excerpts below. If the answer is not in the excerpts,
say that you don't have information on it.

Excerpts:
{reviews}

Question: {question}
"""

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
:root{
  color-scheme: light;
  --green:#3d8b3d; --green-text:#2f6b2a; --green-soft:#e8f1e6;
  --blue:#2563eb; --link:#2563eb; --blue-soft:#eaf0ff;
  --ink:#1f2933; --gray:#4b5563; --muted:#6b7280; --panel:#f6f8fa; --line:#e1e5ea;
}
html, body, .stApp, [class*="css"]{ font-family: Roboto, Arial, Helvetica, sans-serif; color:var(--ink); }
.stApp { background: var(--background-color); }
[data-testid="stChatMessage"] { background: var(--secondary-background-color); }   
footer, [data-testid="stDeployButton"]{ display:none !important; }
[data-testid="stHeader"]{ background:#ffffff; border-bottom:1px solid var(--line); }
.block-container{ padding-top:0.5rem; padding-bottom:2rem; max-width:none; }
a{ color:var(--link); }
.sticky-intro{
  position:sticky; top:0; z-index:20; background:#fff;
  width:100vw; max-width:none; margin-left:calc(100% - 100vw + 5rem);
}
@media (max-width: 640px){
  .sticky-intro{ margin-left:calc(100% - 100vw + 1rem); }
}
body:has([data-testid="stSidebar"][data-collapsed="false"]) .sticky-intro{
  width:100%; margin-left:0;
}

/* keep everything readable even if Streamlit is running in dark mode */
.stApp :is(p, li, label, h1, h2, h3, h4, h5, h6, summary, small, [data-testid="stCaptionContainer"]){ color:var(--ink); }
[data-testid="stSidebar"] :is(p, label, span, div){ color:var(--ink); }
[data-testid="stChatInput"]{ background:transparent; }
[data-testid="stChatInput"] > div{ background:#fff; border:1px solid #cbd5e1; border-radius:24px; }
[data-testid="stChatInput"] textarea{ color:var(--ink); background:#fff; caret-color:var(--ink); }
[data-testid="stChatInput"] textarea::placeholder{ color:var(--muted); }
[data-baseweb="select"] > div{ background:#fff; border-color:#cbd5e1; }
[data-baseweb="select"] :is(div, span, input){ color:var(--ink); }
[data-baseweb="popover"] :is(ul, li){ background:#fff; color:var(--ink); }
[data-testid="stExpander"] details{ background:#fff; border:1px solid var(--line); border-radius:8px; }
[data-testid="stExpander"] summary{ color:var(--ink); }

/* utility strip */
.utility{ background:var(--green-soft); color:var(--green-text); font-size:.78rem; padding:.4rem 1rem;
  display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; }

/* masthead */
.masthead{ position:relative; overflow:hidden; background:linear-gradient(135deg,#55a34e,#3a8a3a); color:#fff; }
.masthead::after{ content:""; position:absolute; left:0; right:0; bottom:0; height:70%;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 120' preserveAspectRatio='none'><polygon fill='white' fill-opacity='0.14' points='0,120 0,78 140,38 250,84 400,18 560,90 700,34 880,96 1020,44 1200,88 1200,120'/><polygon fill='white' fill-opacity='0.10' points='0,120 0,98 180,62 330,100 520,56 760,104 960,70 1200,102 1200,120'/></svg>");
  background-size:100% 100%; background-repeat:no-repeat; pointer-events:none; }
.mast-inner{ position:relative; z-index:1; display:flex; align-items:center; gap:1.1rem; padding:1.6rem 1.25rem 1.9rem; }
.mark{ width:64px; height:64px; border-radius:50%; background:#fff; display:flex; align-items:center;
  justify-content:center; font-size:2rem; flex:0 0 auto; box-shadow:0 2px 6px rgba(0,0,0,.15); }
.mast-kicker{ text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; opacity:.95; }
.mast-title{ font-size:2rem; font-weight:700; line-height:1.15; }
.mast-sub{ font-size:.95rem; opacity:.97; margin-top:.2rem; max-width:44rem; }

.crumbs{ 
    font-size:.82rem;
    color:var(--muted); 
    margin:.9rem 0 .8rem; 
    padding: 1 rem 1.25rem;
    

}

/* sidebar */
[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }

/* widget cards */
.wtitle{ background:var(--green); color:#fff !important; padding:.45rem .8rem; font-weight:700; font-size:.88rem;
  letter-spacing:.02em; margin:.2rem 0 .6rem; border-radius:3px; }
.callout{ background:var(--panel); border-left:4px solid var(--green); padding:.9rem 1.05rem; border-radius:3px; margin-bottom:1rem; }
.callout h4{ margin:0 0 .35rem; color:var(--green-text); font-size:1.02rem; }
.callout p, .callout li{ color:var(--gray); font-size:.92rem; margin:.2rem 0; }

/* chat bubbles: you on the right, the assistant on the left */
.user-row{ display:flex; justify-content:flex-end; margin:.6rem 0; }
.bubble-user{ background:var(--blue); color:#fff; padding:.65rem 1rem; max-width:75%;
  border-radius:18px 18px 4px 18px; white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.45; }
[data-testid="stChatMessage"]{ background:var(--panel); border:1px solid var(--line);
  border-radius:18px 18px 18px 4px; max-width:88%; padding:.7rem 1rem; }
[data-testid="stChatMessage"] :is(p, li, span, div, summary, strong, em, code, td, th, h1, h2, h3, h4){ color:var(--ink); }
.bot-thinking{ min-height:1.7rem; font-size:1.25rem; line-height:1.4; }
.bot-thinking::before{ content:"🌾"; animation:wheat-cycle 1.2s steps(1,end) infinite; }
@keyframes wheat-cycle{
  0%, 32%{ content:"🌾"; }
  33%, 65%{ content:"🌾🌾"; }
  66%, 99%{ content:"🌾🌾🌾"; }
  100%{ content:"🌾"; }
}
@media (prefers-reduced-motion: reduce){
  .bot-thinking::before{ animation:none; content:"🌾"; }
}

/* buttons */
div.stButton > button{ width:100%; background:#fff; color:var(--link); border:1px solid #cbd5e1;
  border-radius:4px; justify-content:flex-start; text-align:left; }
div.stButton > button:hover{ background:var(--link); color:#fff; border-color:var(--link); }
div.stButton > button p{ color:var(--link); }
div.stButton > button:hover p{ color:#fff !important; }

/* footer */
.site-footer{ background:var(--panel); color:var(--gray); border-top:1px solid var(--line);
  padding:1rem 1.25rem; margin-top:2rem; font-size:.84rem; line-height:1.5; }
.site-footer strong{ color:var(--ink); }
</style>
"""

HEADER = """
<div class="utility"><span>A student project inspired by the DA-CAR website</span>
<span>Runs locally with Ollama. Nothing leaves this computer.</span></div>
<div class="masthead"><div class="mast-inner">
  <div class="mark">🌾</div>
  <div>
    <div class="mast-title">Agri Assistant</div>
  </div>
</div></div>
"""

FOOTER = """

"""
# <div class="site-footer">
# <strong>Agri Assistant</strong> is an academic project and is not an official service of the Department of Agriculture.
# Answers are generated by a small local language model from the documents loaded into it and can be incomplete or wrong.
# Please verify important information with the original source or the relevant office.
# </div>

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- backend
@st.cache_resource(show_spinner="Loading the knowledge base (first run may take a while)...")
def load_store():
    import vector  # builds or loads the Chroma database
    return vector.store


try:
    store = load_store()
except (Exception, SystemExit) as e:
    st.error(f"Could not load the knowledge base: {e}")
    st.info("Check that Ollama is running, the models are installed, and your PDFs are in the docs folder.")
    st.stop()


@st.cache_data(ttl=30, show_spinner=False)
def installed_models():
    try:
        import ollama
        names = [m.model for m in ollama.list().models]
        names = [n for n in names if "embed" not in n]
        return names or DEFAULT_MODELS
    except Exception:
        return DEFAULT_MODELS


@st.cache_resource
def get_chain(model_name):
    return ChatPromptTemplate.from_template(PROMPT) | OllamaLLM(model=model_name)


@st.cache_data(show_spinner=False)
def kb_summary(_store):
    """Returns (number of passages, table rows) describing what is stored."""
    metas = _store.get(include=["metadatas"])["metadatas"] or []
    if any("source" in m for m in metas):
        chunks, pages = Counter(), {}
        for m in metas:
            name = str(m.get("source", "?")).replace("\\", "/").split("/")[-1]
            chunks[name] += 1
            pages.setdefault(name, set()).add(m.get("page", 0))
        rows = [{"Document": n, "Pages": len(pages[n]), "Passages": c} for n, c in sorted(chunks.items())]
    else:
        rows = [{"ID": m.get("id", ""), "Title": m.get("title", ""), "Category": m.get("category", "")} for m in metas]
    return len(metas), rows


def source_label(doc):
    m = doc.metadata
    if "source" in m:
        name = str(m["source"]).replace("\\", "/").split("/")[-1]
        return f"{name}, p. {int(m.get('page', 0)) + 1}"
    if "title" in m:
        return f"{m['title']} ({m.get('id', '')})"
    return "Unknown source"


def user_bubble(text):
    """Right-aligned chat bubble for the person's own message (text is escaped)."""
    st.markdown(
        f'<div class="user-row"><div class="bubble-user">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_sources(sources, excerpts):
    for s in sources:
        st.caption(f"📄 {s['label']}")
        if excerpts:
            st.text(s["text"][:400])


# ---------------------------------------------------------------- state
st.session_state.setdefault("messages", [])

n_passages, kb_rows = kb_summary(store)

# ---------------------------------------------------------------- page
models = installed_models()
default_idx = models.index("llama3.2") if "llama3.2" in models else 0
with st.sidebar:
    st.markdown('<div class="wtitle">Knowledge base</div>', unsafe_allow_html=True)
    st.caption(f"{len(kb_rows)} entries · {n_passages} searchable passages")
    with st.expander("View contents"):
        st.dataframe(kb_rows, hide_index=True)

    st.markdown('<div class="wtitle">AI settings</div>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Chat model",
        models,
        index=default_idx,
        help="Larger models such as llama3.1:8b follow instructions better but are slower.",
    )
    top_k = st.slider(
        "Passages to retrieve",
        1,
        8,
        3,
        help="How many pieces of the documents the model reads for each question.",
    )
    show_excerpts = st.toggle("Show excerpts under answers", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

prompt = st.chat_input("Ask a question about the documents...")
question = prompt
st.markdown(
    f'<div class="sticky-intro">{HEADER}</div>',
    unsafe_allow_html=True,
)

for m in st.session_state.messages:
    if m["role"] == "user":
        user_bubble(m["content"])
    else:
        with st.chat_message("assistant", avatar="🌾"):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander(f"Sources ({len(m['sources'])})"):
                    render_sources(m["sources"], show_excerpts)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    user_bubble(question)
    with st.chat_message("assistant", avatar="🌾"):
        thinking = st.empty()
        thinking.markdown(
            '<div class="bot-thinking" role="status" aria-label="Assistant is responding"></div>',
            unsafe_allow_html=True,
        )
        try:
            docs = store.as_retriever(search_kwargs={"k": top_k}).invoke(question)
            sources = [{"label": source_label(d), "text": d.page_content} for d in docs]
            context = "\n\n".join(d.page_content for d in docs)
            stream = get_chain(model_name).stream({"reviews": context, "question": question})

            def response_stream():
                for chunk in stream:
                    if chunk:
                        thinking.empty()
                    yield chunk

            answer = st.write_stream(response_stream())
            thinking.empty()
            with st.expander(f"Sources ({len(sources)})"):
                render_sources(sources, show_excerpts)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            thinking.empty()
            st.error(
                f"Could not get an answer: {e}\n\n"
                "Make sure the Ollama app is running and the selected model is installed."
            )


st.markdown(FOOTER, unsafe_allow_html=True)

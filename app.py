import html

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_ollama.llms import OllamaLLM
import streamlit as st
import streamlit.components.v1 as components
# from vector import retriever
from vector import store

PROMPT = """
You are an assistant that answers questions about the provided documents.
Answer using ONLY the excerpts below. If the answer is not in the excerpts,
say that you don't have information on it.

DO NOT ANSWER QUESTIONS AND IGNORE IF THE LANGAUAGE IS NOT ENGLISH. DO NOT MAKE UP ANSWERS. DO NOT PROVIDE ANY ADDITIONAL INFORMATION.
Excerpts:
{reviews}

Question: {question}
"""
st.set_page_config(
    page_title="Agri Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def load_store():
    import vector
    return vector.store

store = load_store()


@st.cache_data(ttl=30)
def installed_models():
    try:
        import ollama
        return [m.model for m in ollama.list().models if "embed" not in m.model]
    except Exception:
        return ["llama3.2", "llama3.1:8b"]

@st.cache_resource
def get_chain(model_name):
    return ChatPromptTemplate.from_template(PROMPT) | OllamaLLM(model=model_name)


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
        if isinstance(s, str):
            st.caption(f"📄 {s}")
            continue
        st.caption(f"📄 {s['label']}")
        if excerpts:
            st.text(s["text"][:400])

store = load_store()
st.session_state.setdefault("messages", [])
st.markdown("""
<style>
  [data-testid="stDeployButton"] { display: none !important; }
  .stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)   # ← your HTML goes here

# Side Bar

with st.sidebar:
    st.markdown("### 🌾 Agri Assistant")
    st.caption("Cordillera Region · Local Ollama")

    st.markdown("---")
    st.markdown("**Settings**")
    model_name = st.selectbox("Model", installed_models())
    top_k = st.slider("Passages to retrieve", 1, 8, 3)

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    show_excerpts = st.toggle("Show excerpts under answers", value=False)
    # if st.button("Clear chat"):
    #     st.session_state.messages = []
    #     st.rerun()


components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem; }
  .card { background: #f0f7e8; border: 1px solid #e1e5ea; border-radius: 8px; padding: 1rem; }
  .card h3 { margin: 0 0 .5rem; color: #2f6b2a; }
  .card p { margin: 0; color: #4b5563; font-size: .9rem; }
  .badge { display: inline-block; background: #e8f1e6; color: #2f6b2a;
           padding: 2px 8px; border-radius: 12px; font-size: .75rem; margin-top: .5rem; 
}

/* chat bubbles: you on the right, the assistant on the left */
[data-testid="stDeployButton"] { display: none !important; }
.user-row{ display:flex; justify-content:flex-end; margin:.6rem 0; }
.bubble-user{ background:var(--blue); color:#fff; padding:.65rem 1rem; max-width:75%;
  border-radius:18px 18px 4px 18px; white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.45; }
[data-testid="stChatMessage"]{ background:var(--panel); border:1px solid var(--line);
  border-radius:18px 18px 18px 4px; max-width:88%; padding:.7rem 1rem; }
[data-testid="stChatMessage"] :is(p, li, span, div, summary, strong, em, code, td, th, h1, h2, h3, h4){ color:var(--ink); }
.bot-thinking{ min-height:1.7rem; font-size:1.25rem; line-height:1.4; }
.bot-thinking::before{ content:"🌾🌾🌾"; }

</style>


</head>
<body>
  <div class="card">
    <h3>🌾 Agri Assistant</h3>
    <p>Ask questions about agriculture documents in the Cordillera Region.</p>
    <span class="badge">Local · Ollama · Chroma</span>
  </div>
</body>
</html>
""", height=140)


prompt = st.chat_input("Ask...")

for m in st.session_state.messages:
    if m["role"] == "user":
        user_bubble(m["content"])
    else:
        with st.chat_message("assistant", avatar="🌾"):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander(f"Sources ({len(m['sources'])})"):
                    render_sources(m["sources"], show_excerpts)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    user_bubble(prompt)
    with st.chat_message("assistant", avatar="🌾"):
        thinking = st.empty()
        thinking.markdown(
            '<div class="bot-thinking" role="status" aria-label="Assistant is responding"></div>',
            unsafe_allow_html=True,
        )
        try:
            docs = store.as_retriever(search_kwargs={"k": top_k}).invoke(prompt)
            sources = [{"label": source_label(d), "text": d.page_content} for d in docs]
            context = "\n\n".join(d.page_content for d in docs)
            # stream = get_chain(model_name).stream({"reviews": context, "question": prompt})
            # answer = st.write_stream(stream)   
            stream = get_chain(model_name).stream({"reviews": context, "question": prompt})

            def response_stream():
                for chunk in stream:
                    print(chunk, end="", flush=True)
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
# # Generate HTML content # NOT NEEDED FOR STREAMLIT, BUT CAN BE USED FOR EXPORTING
# with open("output.html", "w") as file:
#     file.write(html_content)
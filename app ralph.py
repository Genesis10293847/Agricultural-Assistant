from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st
import streamlit.components.v1 as components

PROMPT = """
You are an assistant that answers questions about the provided documents.
Answer using ONLY the excerpts below. If the answer is not in the excerpts,
say that you don't have information on it.

Excerpts:
{reviews}

Question: {question}
"""


@st.cache_resource
def load_store():
    import vector
    return vector.store

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

store = load_store()
st.session_state.setdefault("messages", [])
st.markdown("""
<style>
  [data-testid="stDeployButton"] { display: none !important; }
  .stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)   # ← your HTML goes here
components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem; }
  .card { background: #f6f8fa; border: 1px solid #e1e5ea; border-radius: 8px; padding: 1rem; }
  .card h3 { margin: 0 0 .5rem; color: #2f6b2a; }
  .card p { margin: 0; color: #4b5563; font-size: .9rem; }
  .badge { display: inline-block; background: #e8f1e6; color: #2f6b2a;
           padding: 2px 8px; border-radius: 12px; font-size: .75rem; margin-top: .5rem; }
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

model_name = st.selectbox("Model", installed_models())
top_k = st.slider("Passages", 1, 8, 3)

prompt = st.chat_input("Ask...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    docs = store.as_retriever(search_kwargs={"k": top_k}).invoke(prompt)
    context = "\n\n".join(d.page_content for d in docs)
    answer = get_chain(model_name).invoke({"reviews": context, "question": prompt})
    sources = [source_label(d) for d in docs]
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

for m in st.session_state.messages:
    if m["role"] == "user":
        st.write(f"**You:** {m['content']}")
    else:
        st.write(f"**Assistant:** {m['content']}")
        if m.get("sources"):
            st.caption("Sources: " + ", ".join(m["sources"]))   

# # Generate HTML content # NOT NEEDED FOR STREAMLIT, BUT CAN BE USED FOR EXPORTING
# with open("output.html", "w") as file:
#     file.write(html_content)
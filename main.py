from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from vector import retriever

# Ollama should be running. Change to "llama3.1:8b" for a stronger model.
MODEL_NAME = "llama3.1:8b"

model = OllamaLLM(model=MODEL_NAME)

# reviews = the PDF chunks retrieved for the question (filled in below)
template = """
You are an assistant that answers questions about the provided documents.
Answer using ONLY the excerpts below. If the answer is not in the excerpts,
say that you don't have information on it.

Excerpts:
{reviews}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print(f"PDF Question Answering (model: {MODEL_NAME})")
while True:
    question = input("\nAsk a question (q to quit): ").strip()
    if question.lower() in ("q", "quit", "exit"):
        break
    if not question:
        continue

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    # PyPDFLoader numbers pages from 0, so add 1 for humans
    sources = [
        f"{d.metadata.get('source', '?').replace(chr(92), '/').split('/')[-1]} "
        f"p.{int(d.metadata.get('page', 0)) + 1}"
        for d in docs
    ]
    print("\nUsing:", "; ".join(sources))
    print()
    print(chain.invoke({"reviews": context, "question": question}))

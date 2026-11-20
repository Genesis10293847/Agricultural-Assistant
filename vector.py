"""Loads every PDF in ./docs, splits it into chunks, embeds the chunks, and
exposes a retriever. Rebuilds the database automatically when the PDFs or the
settings below change."""
import hashlib
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).parent
DOCS_DIR = HERE / "docs"
DB_DIR = HERE / "chroma_db"
FINGERPRINT = HERE / "chroma_db.fingerprint"

EMBED_MODEL = "mxbai-embed-large"
CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 200   # characters shared between neighbouring chunks
TOP_K = 3             # chunks retrieved per question
BATCH_SIZE = 32       # chunks embedded per request (small = safer for Ollama)


def make_fingerprint(pdfs):
    """A short hash of the PDFs and settings, to detect when a rebuild is needed."""
    h = hashlib.sha256()
    h.update(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    for p in pdfs:
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()


pdfs = sorted(DOCS_DIR.glob("*.pdf"))
if not pdfs:
    raise SystemExit(f"No PDFs found. Put at least one .pdf file in: {DOCS_DIR}")

fingerprint = make_fingerprint(pdfs)
up_to_date = (
    DB_DIR.exists()
    and FINGERPRINT.exists()
    and FINGERPRINT.read_text() == fingerprint
)

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

if up_to_date:
    store = Chroma(
        collection_name="pdf_docs",
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
else:
    print(f"Building database from {len(pdfs)} PDF(s)...")
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)  # something changed, so start fresh
    if FINGERPRINT.exists():
        FINGERPRINT.unlink()

    # 1. Load: one Document per PDF page (skip PDFs that fail to read)
    pages = []
    for pdf in pdfs:
        try:
            pages.extend(PyPDFLoader(str(pdf)).load())
        except Exception as e:
            print(f"  Skipping {pdf.name}: {e}")

    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise SystemExit(
            "No text could be extracted. The PDFs may be scanned images "
            "(they need OCR first)."
        )

    # 2. Chunk: cut pages into overlapping pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks. Embedding...")

    # 3. Embed and store in small batches. One giant request can crash
    #    Ollama's embedding process, so we send a few chunks at a time.
    store = Chroma(
        collection_name="pdf_docs",
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    try:
        for i in range(0, len(chunks), BATCH_SIZE):
            store.add_documents(chunks[i : i + BATCH_SIZE])
            done = min(i + BATCH_SIZE, len(chunks))
            print(f"  embedded {done}/{len(chunks)} chunks", end="\r", flush=True)
    except Exception as e:
        raise SystemExit(
            f"\nEmbedding failed: {e}\n"
            "Check that the Ollama app is running and that "
            f"'{EMBED_MODEL}' is installed (ollama list)."
        )
    print()
    FINGERPRINT.write_text(fingerprint)
    print("Database ready.\n")

# 4. Retrieval
retriever = store.as_retriever(search_kwargs={"k": TOP_K})
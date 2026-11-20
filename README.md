# Local PDF Question Answering (RAG)

A small Retrieval-Augmented Generation (RAG) project that runs fully offline
with [Ollama](https://ollama.com). It follows Lab 2.1 of the lab manual:

PDF -> chunks -> embeddings -> Chroma vector database -> retriever -> local LLM

Ask a question and it finds the most relevant passages in your PDFs, then has
the model answer using only those passages. Every answer shows which file and
page it used.

## Requirements

- Windows, macOS, or Linux (commands below are for Windows PowerShell)
- [Ollama](https://ollama.com) installed and running
- **Python 3.12** (3.10 to 3.13 should also work; very new releases such as 3.14
  may fail to install `chromadb`)
- A GPU helps but is not required. 8 GB VRAM or 16 GB RAM is enough for the
  default models.

## Setup

1. Clone the repo and open PowerShell in its folder:
   ```powershell
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Download the models:
   ```powershell
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```
3. Create and activate a virtual environment (do this in every new copy of the
   project; a venv cannot be copied between folders):
   ```powershell
   py -3.12 -m venv venv
   .\venv\Scripts\activate
   ```
   If activation is blocked, run this once and retry:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
4. Install the libraries:
   ```powershell
   pip install -r requirements.txt
   ```
5. Add your PDFs. Create a `docs` folder if it doesn't exist and copy `.pdf`
   files into it. PDFs are not stored in this repo, so each person adds their own.

## Run

```powershell
.\venv\Scripts\activate
python main.py
```

The first run reads every PDF in `docs` and builds the database. This can take
several minutes with many PDFs; a progress counter is shown. Later runs start
immediately. Type `q` to quit.

## Adding or changing PDFs

Copy new PDFs into `docs` and run `main.py` again. It detects the change and
rebuilds the database automatically. It also rebuilds if you change the
embedding model or chunk settings.

## Settings

At the top of `vector.py`:

| Setting | Meaning | Default |
|---|---|---|
| `EMBED_MODEL` | Ollama model used to embed text | `mxbai-embed-large` |
| `CHUNK_SIZE` | Characters per chunk | 1000 |
| `CHUNK_OVERLAP` | Characters shared between neighbouring chunks | 200 |
| `TOP_K` | Chunks retrieved per question | 3 |
| `BATCH_SIZE` | Chunks embedded per request (lower if Ollama crashes) | 32 |

At the top of `main.py`:

| Setting | Meaning | Default |
|---|---|---|
| `MODEL_NAME` | Ollama chat model | `llama3.2` (try `llama3.1:8b` for better answers) |

Changing `MODEL_NAME` or `TOP_K` does not require a rebuild.

## How it works

- `vector.py` loads the PDFs (`PyPDFLoader`), splits them into chunks
  (`RecursiveCharacterTextSplitter`), embeds them, stores them in `chroma_db`,
  and creates the `retriever`.
- `main.py` takes your question, retrieves the top chunks, inserts them into a
  prompt that says "answer using ONLY the excerpts", and prints the model's
  answer along with the sources used.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv, then `pip install -r requirements.txt` |
| Script activation blocked | Run the `Set-ExecutionPolicy` command above |
| `connection refused` / embedding crash | Open the Ollama app (or quit and reopen it), run `ollama list` to confirm both models are installed, and lower `BATCH_SIZE` |
| `No PDFs found` | Put `.pdf` files in the `docs` folder |
| `No text could be extracted` | The PDFs are scanned images and need OCR first |
| Answers ignore a file | It may be a scan with no text, or ask a more specific question naming the document |
| Wrong or stale answers after changing files | Delete `chroma_db` and `chroma_db.fingerprint`, then run again |
| Slow answers | Run `ollama ps`; the model should show 100% GPU |

## Limits

- Scanned PDFs have no extractable text. Tables and multi-column layouts can
  come out jumbled.
- Small models can occasionally add details that are not in the excerpts.
  Check the sources line against the PDF page.

## Data and permissions

Only add documents you have permission to use. PDFs are excluded from the repo
by `.gitignore` so they are not redistributed.
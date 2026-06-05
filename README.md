# 🔧 Upwork API Technical Support Bot

A production-quality **Retrieval-Augmented Generation (RAG)** system that answers developer questions about the Upwork API using only the official technical documentation — with zero hallucination risk.

---

## 📐 Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                 Streamlit UI (app.py)               │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              RAG Pipeline (retrieval.py)            │
│   1. Vector DB Retrieval                            │
│   2. LLM Generation (with latency timing)          │
│   3. Return typed RAGResponse                      │
└────────┬──────────────────────────┬─────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐      ┌─────────────────────────────┐
│  ChromaDB       │      │   DeepInfra API              │
│  (vector_store) │      │   Meta-Llama-3.1-8B-Instruct │
│  all-MiniLM-L6  │      │   (llm.py)                  │
└─────────────────┘      └─────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Git

### 2. Clone & Setup

```bash
# Navigate into the project directory
cd upwork_rag_bot

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and paste your DeepInfra API key
notepad .env   # Windows
# or
nano .env      # Mac/Linux
```

Set `DEEPINFRA_API_KEY` to your key from [deepinfra.com/dash](https://deepinfra.com/dash).

### 4. Add the Source Document

Place the Upwork API PDF into the `data/` directory:

```
data/
└── API Documentation Partial.pdf
```

### 5. Run the Application

```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`.

### 6. Build the Knowledge Base

On the first run:
1. Open the **sidebar** on the left.
2. Click **"🚀 Build Knowledge Base"**.
3. Wait for the ingestion and indexing process to complete (~30–60 seconds).
4. The status will show **"✅ Knowledge base is ready!"**

On subsequent runs, click **"▶ Load"** to skip re-indexing.

---

## 📁 Project Structure

```
upwork_rag_bot/
│
├── data/                         # Source documents (PDF goes here)
├── chroma_db/                    # Persisted ChromaDB index (auto-generated)
├── src/
│   ├── __init__.py
│   ├── config.py                 # Centralized config + logging
│   ├── ingestion.py              # PDF loading, sanity check, chunking
│   ├── vector_store.py           # ChromaDB operations + embeddings
│   ├── llm.py                    # DeepInfra LLM client + prompts
│   ├── retrieval.py              # RAG orchestration pipeline
│   └── app.py                    # Streamlit UI
│
├── requirements.txt
├── .env.example
├── README.md
└── Technical_Summary.md
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DEEPINFRA_API_KEY` | ✅ Yes | — | Your DeepInfra API key |
| `DEEPINFRA_BASE_URL` | No | `https://api.deepinfra.com/v1/openai` | DeepInfra OpenAI-compatible endpoint |
| `LLM_MODEL_NAME` | No | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Hosted LLM model name |
| `LLM_TEMPERATURE` | No | `0.1` | LLM temperature (lower = more factual) |
| `LLM_MAX_TOKENS` | No | `1024` | Maximum tokens in LLM response |
| `EMBEDDING_MODEL_NAME` | No | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | No | `500` | Character chunk size for splitting |
| `CHUNK_OVERLAP` | No | `50` | Character overlap between chunks |
| `CHROMA_COLLECTION_NAME` | No | `upwork_api_docs` | ChromaDB collection name |
| `RETRIEVAL_TOP_K` | No | `3` | Number of chunks retrieved per query |

---

## 🧪 Evaluation Queries

Test the bot with these ground truth questions from the assignment:

1. *"What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?"*
2. *"How long is an OAuth access token valid for?"*
3. *"Can I use a Client Credentials Grant to access a user's private contract details?"*

These are also available as one-click buttons in the sidebar.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `EnvironmentError: DEEPINFRA_API_KEY not set` | Copy `.env.example` to `.env` and set your key |
| `FileNotFoundError: API Documentation Partial.pdf` | Place the PDF in the `data/` directory |
| `FileNotFoundError: ChromaDB directory not found` | Click "Build Knowledge Base" in the sidebar |
| Slow first startup | `sentence-transformers` downloads model weights (~90MB) on first run |
| Import errors | Run `pip install -r requirements.txt` in your virtual environment |
| `torch` not found | Run `pip install torch` separately if auto-install fails |

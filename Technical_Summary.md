# Technical Summary — Upwork API Technical Support Bot

**Role:** Associate AI Developer Take-Home Assignment
**Author:** Chandrakanth
**Stack:** Python · LangChain · ChromaDB · HuggingFace Embeddings · DeepInfra Meta-Llama-3.1-8B · Streamlit

---

## Problem Statement

Build a domain-locked AI chatbot that answers developer questions about the Upwork API by retrieving exact information from official documentation, preventing hallucination, displaying sources, and measuring API latency — all within a conversational Streamlit UI.

---

## Architecture

```
PDF Document
    │  [PyPDFLoader]
    ▼
Raw Pages (LangChain Document objects with page metadata)
    │  [Sanity Check: char count + sample]
    ▼
Text Chunks (500 chars, 50-char overlap, RecursiveCharacterTextSplitter)
    │  [HuggingFaceEmbeddings: all-MiniLM-L6-v2]
    ▼
384-dimensional Dense Vectors
    │  [ChromaDB: persisted to disk]
    ▼
Vector Index (local SQLite-backed store)
    │
User Query ──► Embedding ──► Cosine Similarity Search (Top-3)
                                │
                         Retrieved Chunks
                                │
                         System Prompt + Context + Query
                                │  [DeepInfra: Meta-Llama-3.1-8B-Instruct]
                                ▼
                         Grounded Answer (+ Hallucination Guard)
                                │
                        Streamlit UI: Answer + Sources + Latency
```

---

## Design Decisions

### 1. LangChain as Orchestration Framework
LangChain provides production-tested wrappers for PDF loading, text splitting, vector store integration, and LLM chat interfaces. Its `ChatOpenAI` class supports a custom `base_url`, which gives us access to DeepInfra's OpenAI-compatible endpoint without writing raw HTTP client code.

### 2. ChromaDB over FAISS
ChromaDB was selected for its zero-configuration local persistence (SQLite-backed). FAISS requires manual serialization of the vector index and a separate solution for document metadata storage. For reproducibility during evaluation, ChromaDB is the cleaner choice.

### 3. all-MiniLM-L6-v2 for Local Embeddings
This 22M-parameter model produces 384-dimensional vectors, runs on CPU with sub-second latency per batch, and is well-benchmarked on semantic similarity tasks (BEIR suite). Using a local embedding model means no API cost and no data leaves the machine during indexing.

### 4. Low LLM Temperature (0.1)
Technical Q&A is a factual retrieval task. Low temperature biases the model toward its most probable (most grounded) token choices, reducing the chance of the LLM injecting knowledge from its pre-training weights.

### 5. Hallucination Guard at the Prompt Level
The guard is enforced in the System Prompt, not as a post-processing regex. The model is instructed to output a specific sentinel phrase if the context is insufficient. This makes the model an active participant in honesty, rather than a passive target of keyword filtering.

### 6. Immutable Config Dataclass
A `frozen=True` dataclass loaded from environment variables at startup is the single source of truth for all configuration. It fails loudly (`EnvironmentError`) if a required secret is missing — preventing silent failures mid-conversation.

---

## RAG Pipeline Deep Dive

| Stage | Implementation | Why |
|---|---|---|
| **Loading** | `PyPDFLoader` | Handles multi-page PDFs, preserves page metadata |
| **Chunking** | `RecursiveCharacterTextSplitter` (500/50) | Respects paragraph/sentence boundaries before hard cuts |
| **Embedding** | `all-MiniLM-L6-v2` (local) | Fast, no API cost, normalized for cosine similarity |
| **Storage** | `ChromaDB` (persisted) | Zero-setup, survives restarts, metadata-aware |
| **Retrieval** | `similarity_search_with_relevance_scores` | Returns scores for UI display and confidence awareness |
| **Prompting** | Two-shot: System (persona + guard) + Human (context + query) | Maximizes instruction following in Llama-3 |
| **Generation** | DeepInfra `meta-llama/Meta-Llama-3.1-8B-Instruct` | OpenAI-compatible, fast, cost-effective |
| **Latency** | `time.perf_counter()` wrapping LLM call only | Measures API network latency, not pipeline overhead |

---

## Difficulties Faced

- **Chunking Technical Code:** The 500-character hard limit sometimes cuts JSON examples in half. The 50-character overlap mitigates this but does not eliminate it. A production system would use a semantic/markdown-aware splitter, or at minimum split on code block delimiters.

- **Hallucination Guard Reliability:** Instruction-following models can still use pre-trained knowledge when context is thin. The current approach (strict system prompt + low temperature) reduces but does not eliminate this risk. Production systems add a second "grounding check" LLM call or a post-processing similarity score filter.

- **API Latency Variability:** DeepInfra's hosted endpoint has cold-start latency (first request may take 5–15s). Subsequent requests are typically under 3s. The UI correctly isolates and displays only the LLM call latency, not the full pipeline time.

- **Embedding Model First-Run Download:** `sentence-transformers` downloads ~90MB of model weights on first use. The UI includes an appropriate spinner and logging message for this.

---

## How I Used LLMs in Development

- Used **Claude** to validate the system prompt design and refine the Hallucination Guard instruction phrasing.
- Used **GitHub Copilot** to accelerate LangChain boilerplate (ChromaDB initialization patterns, ChatOpenAI configuration).
- All design decisions (architecture, library choices, chunking strategy, error handling pattern) were made independently and are mine to explain in full detail.

---

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| ChromaDB local persistence | Simple for eval; not horizontally scalable (vs. Pinecone/Weaviate) |
| 500-char hard chunk size | Fast & consistent; may split code blocks |
| all-MiniLM-L6-v2 | Fast and lightweight; weaker than Ada-002 or BAAI/bge-large |
| Hallucination Guard via prompt | Simple; not 100% reliable without a grounding score threshold |
| Single ChromaDB collection | Clean; would need namespacing for multi-document production use |

---

## Future Improvements

1. **Hybrid Search:** Combine BM25 (keyword) + dense vector retrieval for higher recall on technical queries (e.g., `reciprocal_rank_fusion`).
2. **Reranking:** Add a cross-encoder reranker (e.g., `ms-marco-MiniLM-L-6-v2`) to reorder the top-10 candidates before sending the top-3 to the LLM.
3. **Grounding Score Threshold:** If the highest relevance score is below 0.4, skip the LLM call and return the fallback phrase directly.
4. **Semantic Chunking:** Replace fixed character chunking with a semantic boundary detector to avoid splitting code examples.
5. **Streaming Responses:** Enable `streaming=True` in the LLM client and use `st.write_stream()` for real-time response rendering.
6. **Evaluation Suite:** Build an automated evaluation harness using the 3 ground truth QA pairs and `ragas` library to track retrieval precision and answer faithfulness across configuration changes.

---

## Why I Am the Best Person for the ProAnalyst AI Team

1. **I build systems, not scripts.** This submission demonstrates clean architecture (separation of concerns, typed interfaces, immutable config, module-level logging) — the same discipline I would apply to production AI systems at ProAnalyst.

2. **I understand RAG beyond the surface.** Choices like `RecursiveCharacterTextSplitter` over `CharacterTextSplitter`, normalized embeddings, relevance score display, and LLM-level hallucination guards show depth of understanding — not just ability to follow a tutorial.

3. **I document and explain everything.** Every design decision in this codebase has a written rationale. In a team environment, the ability to communicate *why* as clearly as *what* is what separates senior from junior engineers.

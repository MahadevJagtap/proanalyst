# Interview Preparation Guide — Upwork API RAG Bot

> This document prepares you to answer every possible technical question an interviewer
> could ask about this project. Assume: *"Explain every line of your code."*

---

## 1. Architecture Explanation (30-Second Pitch)

> *"I built a modular RAG system with five layers: data ingestion, vector indexing,
> LLM integration, pipeline orchestration, and a Streamlit UI. The PDF is chunked
> into 500-character overlapping blocks, embedded locally using MiniLM, and stored
> in ChromaDB. At query time, the top 3 most semantically similar chunks are retrieved
> and injected into a structured LangChain prompt sent to Meta-Llama on DeepInfra.
> The UI displays the grounded answer, the exact source chunks used, and the LLM's
> network latency — all without the AI ever using knowledge outside the provided document."*

---

## 2. File-by-File Explanation

### `config.py`
- **What it does:** Loads environment variables using `python-dotenv` and exposes them as a frozen dataclass called `Config`.
- **Why frozen:** A `frozen=True` dataclass cannot be mutated after initialization. This prevents a developer from accidentally overwriting a config value mid-execution, which is a subtle but real production bug.
- **Why a dataclass over a plain dict:** Type hints, IDE autocomplete, and field-level defaults — all without external libraries.
- **Why `_require_env`:** It fails at startup with a human-readable error message rather than failing silently with a `None` value that causes a cryptic `AuthenticationError` later.

### `ingestion.py`
- **What it does:** Loads the PDF with `PyPDFLoader`, runs the sanity check (prints char count and sample), and splits with `RecursiveCharacterTextSplitter`.
- **Why `RecursiveCharacterTextSplitter`:** It tries separators in order: `\n\n` → `\n` → ` ` → `""`. This means it attempts to keep paragraphs together first, then sentences, before resorting to hard character cuts. This produces more semantically coherent chunks than the basic `CharacterTextSplitter`.
- **Why overlap:** If a technical term is defined at the end of chunk 4 and referenced at the beginning of chunk 5, without overlap, a query about that term would only retrieve one of those chunks — possibly missing the definition. The 50-char overlap ensures the context window of adjacent chunks "bleeds" into each other.

### `vector_store.py`
- **What it does:** Wraps ChromaDB with methods to `build_vector_store`, `load_vector_store`, `is_index_built`, and `retrieve_top_k`.
- **Why lazy initialization for embeddings:** The `all-MiniLM-L6-v2` model is ~90MB and takes 2–5 seconds to load. Lazy loading means the module can be imported without loading the model — only when embeddings are actually needed does the model load.
- **Why `normalize_embeddings=True`:** Normalization converts raw vectors to unit-length vectors. This means cosine similarity equals dot product, which is faster to compute. It also ensures similarity scores are in a [0,1] range, which is more interpretable in the UI.
- **Why ChromaDB over FAISS:** ChromaDB persists to SQLite automatically. FAISS is an in-memory library — you must manually `faiss.write_index()` and manage a separate metadata JSON file. ChromaDB handles both, making local setup and recovery trivial.

### `llm.py`
- **What it does:** Defines the System Prompt, initializes the DeepInfra `ChatOpenAI` client, formats context chunks, and calls the LLM.
- **Why two-message structure (System + Human):** Llama-3 models are instruction-tuned using a specific chat template. Putting rules in the `SystemMessage` and context+question in the `HumanMessage` matches the training format and produces stronger instruction following than a single combined message.
- **Why temperature=0.1:** Technical Q&A requires factual precision. High temperatures (e.g., 0.9) cause the model to sample less probable tokens, increasing variance and hallucination risk. At 0.1, the model strongly favors its most probable output — the one most consistent with the provided context.
- **Why the hallucination guard is in the prompt:** Post-processing (regex matching the output) is brittle — a slightly different phrasing breaks it. By making the model responsible for declaring "I don't know," we leverage its instruction following, which is more flexible and robust.

### `retrieval.py`
- **What it does:** Orchestrates retrieval + LLM call, measures latency, and returns a typed `RAGResponse`.
- **Why `time.perf_counter()`:** `perf_counter()` uses the highest-resolution timer available on the OS — on Windows, this is typically sub-microsecond. `time.time()` is coarser and can drift. For measuring short network calls (1–5 seconds), `perf_counter` is the correct tool.
- **Why the latency timer wraps ONLY the LLM call:** The assignment asks "how long the API took to respond." Embedding the query and searching ChromaDB are local operations and should not be included in the "API latency" metric.
- **Why `RAGResponse` as a dataclass:** Returning a dataclass (vs. a dict) gives the UI layer type safety and IDE autocomplete. It also documents the contract between the pipeline and the UI explicitly.
- **Why separate `_retrieve` and `generate_answer` steps:** This separation makes unit testing possible. You can test retrieval quality independently from LLM quality. In a production system, you'd run a retrieval evaluation suite (`ragas`, `deepeval`) without burning LLM API credits.

### `app.py`
- **What it does:** Streamlit UI — sidebar for KB management, chat interface, and structured response rendering.
- **Why `st.session_state` for the pipeline:** Streamlit reruns the entire Python script on every user interaction. Without session state, the ChromaDB and LLM client would be re-initialized on every keypress. Storing them in `session_state` means they are initialized once per session.
- **Why not `@st.cache_resource`:** Our pipeline objects (especially ChromaDB) may need to be rebuilt mid-session (via the "Rebuild" button). `@st.cache_resource` caches by function arguments and can't be easily invalidated at runtime. Manual `session_state` management gives us precise control.

---

## 3. Expected Interview Questions & Deep Answers

### RAG Questions

**Q: What is RAG and why is it better than fine-tuning for this use case?**
> RAG (Retrieval-Augmented Generation) dynamically injects relevant external documents into the LLM's prompt at query time, rather than baking knowledge into the model's weights via fine-tuning. For this use case, RAG is superior because:
> 1. **Updatable:** New API docs can be re-indexed in minutes. Re-fine-tuning takes hours and GPU resources.
> 2. **Verifiable:** The user can see exactly which chunks were used — fine-tuned models have opaque knowledge.
> 3. **Cost-effective:** No GPU training costs.
> 4. **Lower hallucination risk:** Grounded answers from retrieved text are more reliable than parameterized facts, which can be stale or misremembered.

**Q: What chunking strategy did you use and why?**
> `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=50`. The recursive strategy tries to split on natural boundaries (paragraphs, sentences) before falling back to character-level cuts. The 500-char limit ensures chunks fit comfortably within the embedding model's context window (256 tokens). The 50-char overlap is approximately one sentence, which ensures multi-sentence technical constructs (like "endpoint URL [newline] its parameters [newline] example request") are never completely severed between adjacent chunks.

**Q: How does your retrieval work?**
> At query time, the user's question is embedded using the same `all-MiniLM-L6-v2` model used during indexing. This produces a 384-dimensional query vector. ChromaDB performs a cosine similarity search against all stored chunk vectors and returns the top-3 closest matches. Because all embeddings are L2-normalized, cosine similarity reduces to a dot product — which ChromaDB computes efficiently using HNSW (Hierarchical Navigable Small World) graph indexing.

**Q: How do you prevent hallucination?**
> Three layers: (1) The system prompt explicitly instructs the model to answer ONLY from the provided context and use a specific fallback phrase if the answer isn't there. (2) LLM temperature is set to 0.1, biasing outputs toward the highest probability (most grounded) tokens. (3) If no chunks are retrieved at all (empty results), the pipeline short-circuits and returns the fallback phrase without calling the LLM.

**Q: What is the relevance score and how is it computed?**
> `similarity_search_with_relevance_scores` in LangChain's ChromaDB wrapper returns a score in [0, 1] where 1.0 = identical. The underlying metric is cosine similarity between normalized embedding vectors. Because we use `normalize_embeddings=True`, the raw dot product equals the cosine similarity. ChromaDB normalizes this into the [0,1] range.

---

### Vector Database Questions

**Q: Why ChromaDB?**
> Zero external server setup (embedded mode uses SQLite), automatic persistence, metadata-aware storage, and Python-native API. For a local evaluation scenario, it is the simplest path to a correct and reproducible submission.

**Q: How is data stored in ChromaDB?**
> ChromaDB stores: (1) the raw document text, (2) its vector embedding, and (3) arbitrary metadata (source, page number) — all in a local SQLite + HNSW index on disk. The `persist_directory` parameter tells Chroma where to write these files. The HNSW graph allows approximate nearest-neighbor search in O(log n) time.

**Q: What is HNSW?**
> Hierarchical Navigable Small World — an approximate nearest neighbor (ANN) algorithm. It builds a multi-layer graph where nodes (embeddings) are connected to their nearest neighbors at each layer. Queries "navigate" the graph starting from the top layer and drill down, achieving O(log n) search time at the cost of ~10% recall error (configurable). For document retrieval with <1M vectors, it provides near-perfect recall with millisecond latency.

**Q: What would you use in production at scale?**
> For production with millions of documents and multi-tenant isolation, I would use **Pinecone** (managed, auto-scaling) or **Weaviate** (open-source, Kubernetes-deployable). Both support filtering by metadata (e.g., "only search documents for client X"), which is necessary for multi-tenant SaaS.

---

### LLM Integration Questions

**Q: How does DeepInfra work?**
> DeepInfra is a model inference hosting platform. It hosts open-source models (like Meta-Llama) on its GPU infrastructure and exposes them via an OpenAI-compatible REST API. LangChain's `ChatOpenAI` accepts a `base_url` parameter, so we point it at `https://api.deepinfra.com/v1/openai` and provide the DeepInfra API key — no code changes needed vs. using OpenAI directly.

**Q: Explain the System Prompt design.**
> The system prompt has three parts: (1) Persona establishment — "Senior Upwork API Consultant" — which primes the model's tone and domain focus. (2) Grounding rules — "answer only from the provided context" — which the model is trained to follow via RLHF. (3) Hallucination Guard — a specific sentinel phrase the model must output verbatim if the context is insufficient. Using a verbatim phrase is important because it makes the guard detectable in post-processing if needed.

**Q: What are the risks of the LLM ignoring the hallucination guard?**
> Instruction-following LLMs are probabilistically trained, not rule-bound. A sufficiently "tempting" question (one where the model has strong pre-trained knowledge) can cause it to bypass the guard. Mitigations: lower temperature (we use 0.1), stronger/more repeated constraint language in the prompt ("YOU MUST FOLLOW THESE WITHOUT EXCEPTION"), and optionally a grounding score threshold — if the max relevance score is below 0.4, skip the LLM entirely.

---

### Streamlit Questions

**Q: How does Streamlit handle state between reruns?**
> Streamlit reruns the entire Python script from top to bottom on every user interaction (button click, chat input, etc.). `st.session_state` is a persistent dictionary that survives these reruns within a single browser session. We store the `RAGPipeline`, the `index_ready` flag, and the `chat_history` list in session state.

**Q: Why do you display latency?**
> The assignment requires it. Beyond the requirement, displaying latency is good UX for a developer tool — developers are latency-sensitive and want to know if the API is slow. It also helps users calibrate their expectations and diagnose whether a slow response was a network issue vs. a model throughput issue.

---

### Python Questions

**Q: Why use a dataclass for `Config` instead of a Pydantic model?**
> Both are valid. Pydantic `BaseSettings` would be even more appropriate in production (it auto-reads env vars, validates types, and supports `.env` files natively). I used a `dataclass` here to minimize dependencies and demonstrate understanding of Python's built-in tooling. If this were a production codebase, I would use `pydantic-settings`.

**Q: Explain `frozen=True` on the dataclass.**
> `frozen=True` makes all fields immutable after `__init__`. Attempting to set a field raises `FrozenInstanceError`. This prevents accidental mutation of configuration — a subtle but real class of bugs in long-running server applications where config might be passed between threads.

**Q: Why `time.perf_counter()` over `time.time()`?**
> `time.perf_counter()` returns a float with the highest resolution available on the platform (nanosecond on most modern systems). `time.time()` returns wall-clock time in seconds with platform-dependent resolution. For measuring short durations (1–10 seconds), `perf_counter` is more accurate and unaffected by system clock adjustments.

---

### Deployment Questions

**Q: How would you productionize this?**
> 1. Replace ChromaDB with a managed vector DB (Pinecone/Weaviate).
> 2. Add authentication (OAuth2 / API key gateway) to the Streamlit app.
> 3. Move LLM calls to an async task queue (Celery + Redis) to handle concurrent users.
> 4. Add `structlog` for structured JSON logging, connected to a log aggregation service.
> 5. Containerize with Docker, deploy to Cloud Run or ECS with auto-scaling.
> 6. Add a CI/CD pipeline that runs retrieval evaluation (`ragas`) on every doc update.

**Q: How would you handle multiple documents / different clients?**
> Use ChromaDB's `where` metadata filtering to namespace queries by `client_id` or `document_id`. Each document would be indexed with a `client_id` metadata field, and all queries would include a `where={"client_id": current_client}` filter to prevent cross-tenant data leakage.

**Q: What monitoring would you add?**
> - Per-query latency (already implemented in the UI).
> - Retrieval relevance score distribution (flag queries where max score < 0.3 — likely out-of-scope questions).
> - LLM API error rate and timeout rate.
> - Hallucination guard trigger rate (how often the model returns the fallback phrase).
> - User feedback signals (thumbs up/down buttons in the UI feeding a Postgres table).

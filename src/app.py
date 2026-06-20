"""
app.py
------
Main Streamlit entry point for the Upwork API Technical Support Bot.

Responsibilities:
  - Manage application state (index initialization, chat history).
  - Render the sidebar (knowledge base management, ground truth queries).
  - Render the chat interface (user input, AI responses).
  - Display the answer, sources panel, and latency badge for every response.

Design Decision — Streamlit Session State:
    All stateful objects (VectorStoreManager, LLMClient, RAGPipeline) are
    stored in `st.session_state`. This prevents expensive re-initialization
    on every Streamlit rerun (which happens on every user interaction).
    The `@st.cache_resource` pattern is NOT used here because our pipeline
    objects depend on runtime configuration that may change.

Run with:
    streamlit run src/app.py
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on the Python path when running `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.ingestion import ingest_pipeline
from src.llm import LLMClient
from src.retrieval import RAGPipeline, RAGResponse, SourceChunk
from src.vector_store import VectorStoreManager

logger = logging.getLogger("upwork_rag_bot.app")

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main header */
        .main-header {
            background: linear-gradient(135deg, #14a800 0%, #0d7a00 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            color: white;
        }
        .main-header h1 { margin: 0; font-size: 1.8rem; }
        .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

        /* Answer card */
        .answer-card {
            background: #f8fffe;
            border-left: 4px solid #14a800;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        }

        /* Source chunk card */
        .source-card {
            background: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.6rem;
            font-size: 0.85rem;
            font-family: 'Courier New', monospace;
        }

        /* Latency badge */
        .latency-badge {
            display: inline-block;
            background: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #a5d6a7;
            border-radius: 20px;
            padding: 0.2rem 0.8rem;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* Error badge */
        .error-badge {
            background: #ffebee;
            color: #c62828;
            border: 1px solid #ef9a9a;
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }

        /* Ground truth button styling */
        div[data-testid="stButton"] button {
            width: 100%;
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Session State Initialization ────────────────────────────────────────────

def _initialize_session_state() -> None:
    """Set up all required session state keys on first run."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List[dict] with keys: role, content, rag_response

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None  # RAGPipeline instance

    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False

    if "status_message" not in st.session_state:
        st.session_state.status_message = ""


# ─── Pipeline Initialization ──────────────────────────────────────────────────

def _init_pipeline(rebuild: bool = False) -> None:
    """
    Initialize or rebuild the full RAG pipeline.

    Args:
        rebuild: If True, re-run the ingestion + embedding pipeline
                 (needed on first run or to re-index a new document).
    """
    with st.spinner("🔄 Initializing pipeline... please wait."):
        try:
            vsm = VectorStoreManager()

            if rebuild:
                pdf_path = config.data_dir / "API Documentation Partial.pdf"
                st.info(f"📄 Loading and indexing: `{pdf_path.name}`")
                chunks = ingest_pipeline(pdf_path)
                st.info(f"✅ Loaded {len(chunks)} chunks. Building vector index...")
                vsm.build_vector_store(chunks)
            else:
                vsm.load_vector_store()

            llm_client = LLMClient()
            pipeline = RAGPipeline(vector_store=vsm, llm_client=llm_client)

            st.session_state.pipeline = pipeline
            st.session_state.index_ready = True
            st.success("✅ Knowledge base is ready!")
            logger.info("RAG pipeline initialized successfully.")

        except FileNotFoundError as exc:
            st.error(f"❌ {exc}")
            logger.error(str(exc))
        except Exception as exc:
            st.error(f"❌ Initialization failed: {exc}")
            logger.error(f"Pipeline init failed: {exc}", exc_info=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the sidebar with knowledge base controls and ground truth queries."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/d/d2/Upwork-logo.svg",
            width=140,
        )
        st.markdown("## ⚙️ Knowledge Base")
        st.markdown(
            "The bot answers questions using the **Upwork API Technical Reference** "
            "loaded into a local ChromaDB vector store."
        )
        st.divider()

        vsm_check = VectorStoreManager()

        if vsm_check.is_index_built():
            st.success("✅ Index found on disk.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶ Load", use_container_width=True, key="btn_load"):
                    _init_pipeline(rebuild=False)
            with col2:
                if st.button("🔄 Rebuild", use_container_width=True, key="btn_rebuild"):
                    _init_pipeline(rebuild=True)
        else:
            st.warning("⚠️ No index found. Please run ingestion.")
            if st.button(
                "🚀 Build Knowledge Base", use_container_width=True, key="btn_build"
            ):
                _init_pipeline(rebuild=True)

        st.divider()
        st.markdown("## 🧪 Ground Truth Queries")
        st.markdown(
            "These are the three evaluation questions from the assignment spec. "
            "Click to auto-populate the chat."
        )

        ground_truth_questions = [
            "What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?",
            "How long is an OAuth access token valid for?",
            "Can I use a Client Credentials Grant to access a user's private contract details?",
        ]

        for i, question in enumerate(ground_truth_questions):
            if st.button(
                f"Q{i+1}: {question[:60]}...",
                key=f"gt_q{i}",
                use_container_width=True,
            ):
                st.session_state["prefilled_query"] = question

        st.divider()
        st.markdown("### 🛠 Configuration")
        st.caption(f"**Model:** `{config.llm_model_name}`")
        st.caption(f"**Embeddings:** `{config.embedding_model_name}`")
        st.caption(f"**Chunk Size / Overlap:** `{config.chunk_size}` / `{config.chunk_overlap}`")
        st.caption(f"**Top-K Retrieval:** `{config.retrieval_top_k}`")

        if st.button("🗑️ Clear Chat History", use_container_width=True, key="btn_clear"):
            st.session_state.chat_history = []
            st.rerun()


# ─── Response Display ─────────────────────────────────────────────────────────

def render_rag_response(rag: RAGResponse) -> None:
    """
    Render the full structured response from the RAG pipeline.

    Displays:
      - The generated answer.
      - A latency badge.
      - Expandable 'Sources' section with chunk content and metadata.

    Args:
        rag: A `RAGResponse` dataclass from the pipeline.
    """
    if rag.error:
        st.markdown(
            f'<div class="error-badge">⚠️ <strong>Error:</strong> {rag.error}</div>',
            unsafe_allow_html=True,
        )
        return

    # Answer
    st.markdown(
        f'<div class="answer-card">{rag.answer}</div>',
        unsafe_allow_html=True,
    )

    # Latency badge
    st.markdown(
        f'<span class="latency-badge">⏱ API Latency: {rag.latency_seconds:.3f}s</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")  # spacer

    # Sources
    if rag.sources:
        with st.expander(f"📚 Sources ({len(rag.sources)} retrieved chunks)", expanded=False):
            for i, src in enumerate(rag.sources, start=1):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>Chunk {i}</strong> &nbsp;|&nbsp;
                        Page: <code>{src.page}</code> &nbsp;|&nbsp;
                        Relevance Score: <code>{src.relevance_score:.4f}</code>
                        <hr style="margin: 0.5rem 0; border-color: #ddd;">
                        {src.content.replace(chr(10), '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ─── Chat Interface ───────────────────────────────────────────────────────────

def render_chat_interface() -> None:
    """Render the main chat interface."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🔧 Upwork API Technical Support Bot</h1>
            <p>Ask any question about the Upwork API. Answers are grounded exclusively
            in the official technical documentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.index_ready:
        st.info(
            "👈 **Getting Started:** Use the sidebar to build or load the knowledge base first."
        )
        return

    # Render existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_rag_response(message["rag_response"])

    # Handle pre-filled query from ground truth sidebar buttons
    prefilled = st.session_state.pop("prefilled_query", None)

    # Chat input
    user_input = st.chat_input(
        placeholder="Ask about the Upwork API (e.g., 'What are the rate limits?')...",
    ) or prefilled

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Consulting the documentation..."):
                pipeline: RAGPipeline = st.session_state.pipeline
                rag_response: RAGResponse = pipeline.process_query(user_input)

            render_rag_response(rag_response)

        # Add assistant message to history
        st.session_state.chat_history.append(
            {"role": "assistant", "rag_response": rag_response, "content": rag_response.answer}
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point."""
    _initialize_session_state()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()

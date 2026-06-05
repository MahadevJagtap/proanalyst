"""
vector_store.py
---------------
Manages all interactions with ChromaDB — building the vector index
from document chunks, loading an existing persisted index, and
performing semantic similarity searches.

Design Decision — ChromaDB over FAISS:
    ChromaDB was chosen because it offers built-in persistence to
    disk with zero external server setup (it runs as an embedded
    SQLite-backed store). FAISS requires manual serialization of
    the index and a separate metadata store, adding boilerplate.
    For a reviewer running this project locally, ChromaDB is a
    better experience: `pip install chromadb` and it works.

Design Decision — HuggingFace local embeddings:
    `all-MiniLM-L6-v2` is a 22M parameter model that runs on CPU
    in under a second per batch. It is the industry standard for
    local semantic search in RAG prototypes — well-benchmarked on
    the BEIR suite, and produces 384-dimensional dense vectors.
    Using a local model means NO API cost for embedding and NO
    data leaves the machine during the indexing phase.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import config

logger = logging.getLogger("upwork_rag_bot.vector_store")


class VectorStoreManager:
    """
    Encapsulates all ChromaDB operations.

    Attributes:
        _embeddings: The initialized HuggingFace embedding model.
        _db: The active Chroma vector store instance.
    """

    def __init__(self) -> None:
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._db: Optional[Chroma] = None

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Lazily initialize the HuggingFace embedding model.

        Lazy initialization avoids loading the ~90MB model weights
        on module import — they are only loaded when actually needed.

        Returns:
            An initialized `HuggingFaceEmbeddings` instance.
        """
        if self._embeddings is None:
            logger.info(
                f"Loading embedding model: {config.embedding_model_name} "
                "(this may take a moment on first run to download weights)"
            )
            self._embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model_name,
                # Run on CPU by default for portability.
                model_kwargs={"device": "cpu"},
                # Normalize embeddings so cosine similarity = dot product.
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully.")
        return self._embeddings

    def build_vector_store(self, chunks: List[Document]) -> "VectorStoreManager":
        """
        Embed a list of document chunks and persist them to ChromaDB.

        This operation is idempotent with respect to the collection name:
        if the collection already exists, Chroma will not duplicate entries.
        However, calling this on an existing index with new documents will
        ADD them. For a clean rebuild, delete the `chroma_db/` directory first.

        Args:
            chunks: A list of chunked `Document` objects from `ingestion.py`.

        Returns:
            `self` to allow method chaining.

        Raises:
            ValueError: If `chunks` is empty.
        """
        if not chunks:
            raise ValueError(
                "Cannot build vector store from an empty chunk list. "
                "Check that the PDF was loaded and chunked correctly."
            )

        logger.info(
            f"Building ChromaDB vector store with {len(chunks)} chunks. "
            f"Persisting to: {config.chroma_db_dir}"
        )

        config.chroma_db_dir.mkdir(parents=True, exist_ok=True)

        self._db = Chroma.from_documents(
            documents=chunks,
            embedding=self._get_embeddings(),
            collection_name=config.chroma_collection_name,
            persist_directory=str(config.chroma_db_dir),
        )

        logger.info(
            f"Vector store built. Collection '{config.chroma_collection_name}' "
            f"persisted with {len(chunks)} vectors."
        )
        return self

    def load_vector_store(self) -> "VectorStoreManager":
        """
        Load an existing ChromaDB index from disk.

        This is called on application startup when the index has
        already been built from a previous run — avoiding the expensive
        re-embedding step.

        Returns:
            `self` to allow method chaining.

        Raises:
            FileNotFoundError: If the ChromaDB directory does not exist.
        """
        if not config.chroma_db_dir.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found at: {config.chroma_db_dir}\n"
                "Please run the ingestion step first via the sidebar button."
            )

        logger.info(f"Loading existing ChromaDB from: {config.chroma_db_dir}")
        self._db = Chroma(
            collection_name=config.chroma_collection_name,
            embedding_function=self._get_embeddings(),
            persist_directory=str(config.chroma_db_dir),
        )
        logger.info("ChromaDB loaded successfully.")
        return self

    def is_index_built(self) -> bool:
        """
        Check whether a persisted ChromaDB index exists on disk.

        Used by the Streamlit UI to determine whether to show an
        'Initialize Knowledge Base' button or load automatically.

        Returns:
            True if the chroma_db directory exists and is non-empty.
        """
        db_path = config.chroma_db_dir
        return db_path.exists() and any(db_path.iterdir())

    def retrieve_top_k(
        self, query: str, k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform a semantic similarity search against the vector index.

        Returns the top-k most relevant document chunks along with their
        similarity scores. Scores are used in the UI to give the user
        insight into retrieval confidence.

        Args:
            query: The user's natural language question.
            k: Number of results to return. Defaults to `config.retrieval_top_k`.

        Returns:
            A list of (Document, similarity_score) tuples, ordered by
            relevance (most relevant first).

        Raises:
            RuntimeError: If the vector store has not been loaded/built.
        """
        if self._db is None:
            raise RuntimeError(
                "Vector store is not initialized. "
                "Call `build_vector_store()` or `load_vector_store()` first."
            )

        top_k = k or config.retrieval_top_k
        logger.info(f"Retrieving top-{top_k} chunks for query: '{query[:80]}...'")

        results: List[Tuple[Document, float]] = (
            self._db.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
            )
        )

        logger.info(
            f"Retrieved {len(results)} chunks. "
            f"Top score: {results[0][1]:.4f}" if results else "No results returned."
        )
        return results

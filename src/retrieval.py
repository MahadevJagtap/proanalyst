"""
retrieval.py
------------
The RAG orchestration layer — the "brain" of the application.

`RAGPipeline` coordinates the flow:
    User Query → Vector Store Retrieval → LLM Generation → Response Payload

It also measures API latency with high precision and packages
the result into a structured `RAGResponse` dataclass for clean
consumption by the Streamlit UI.

Design Decision — Dataclass Return Type:
    Returning a typed `RAGResponse` instead of a raw dict prevents
    the UI layer from needing to know the internal keys of a dictionary.
    It also makes the return type self-documenting and IDE-friendly.

Design Decision — Separation of Retrieval and Generation:
    The retrieval step and the LLM step are kept in separate methods.
    This allows us to unit-test retrieval quality independently of LLM
    quality — a key principle for production RAG systems.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple

from langchain.schema import Document

from .llm import LLMClient
from .vector_store import VectorStoreManager

logger = logging.getLogger("upwork_rag_bot.retrieval")


@dataclass
class SourceChunk:
    """
    A single retrieved document chunk with its metadata.

    This is what gets displayed in the 'Sources' panel of the UI.
    """

    content: str
    source: str
    page: int
    relevance_score: float


@dataclass
class RAGResponse:
    """
    The structured response payload returned by the RAG pipeline.

    Attributes:
        answer: The LLM-generated answer text.
        sources: List of retrieved chunks used as context.
        latency_seconds: Wall-clock time of the LLM API call in seconds.
        query: The original user query (for display/logging purposes).
    """

    answer: str
    sources: List[SourceChunk]
    latency_seconds: float
    query: str
    error: str = field(default="")


class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation pipeline.

    This class acts as the facade between the UI layer and the
    lower-level `VectorStoreManager` and `LLMClient` components.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        llm_client: LLMClient,
    ) -> None:
        """
        Args:
            vector_store: An initialized VectorStoreManager instance.
            llm_client: An initialized LLMClient instance.
        """
        self._vector_store = vector_store
        self._llm_client = llm_client

    def _retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant document chunks for a given query.

        Args:
            query: The user's question.

        Returns:
            A list of (Document, relevance_score) tuples.
        """
        return self._vector_store.retrieve_top_k(query)

    def _build_source_chunks(
        self, retrieved: List[Tuple[Document, float]]
    ) -> List[SourceChunk]:
        """
        Convert raw retrieval results into structured `SourceChunk` objects.

        Args:
            retrieved: List of (Document, score) tuples from the vector store.

        Returns:
            A list of `SourceChunk` dataclass instances.
        """
        sources = []
        for doc, score in retrieved:
            sources.append(
                SourceChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    page=int(doc.metadata.get("page", 0)) + 1,  # 0-indexed → 1-indexed
                    relevance_score=round(score, 4),
                )
            )
        return sources

    def process_query(self, query: str) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user query.

        Flow:
            1. Retrieve top-k chunks from ChromaDB.
            2. Start the latency timer.
            3. Call the LLM with the query and retrieved context.
            4. Stop the latency timer.
            5. Package everything into a `RAGResponse`.

        The latency timer wraps ONLY the LLM API call — not the
        embedding or retrieval steps — because the assignment asks
        for "how long the API took to respond."

        Args:
            query: The user's natural language question.

        Returns:
            A fully populated `RAGResponse` dataclass.
        """
        logger.info(f"Processing query: '{query}'")

        # Step 1: Retrieve
        try:
            retrieved_chunks = self._retrieve(query)
        except Exception as exc:
            logger.error(f"Retrieval failed: {exc}", exc_info=True)
            return RAGResponse(
                answer="",
                sources=[],
                latency_seconds=0.0,
                query=query,
                error=f"Retrieval Error: {exc}",
            )

        source_chunks = self._build_source_chunks(retrieved_chunks)

        # Step 2 & 3 & 4: LLM call with latency measurement
        try:
            start_time = time.perf_counter()
            answer = self._llm_client.generate_answer(query, retrieved_chunks)
            end_time = time.perf_counter()
            latency = round(end_time - start_time, 3)
        except RuntimeError as exc:
            logger.error(f"LLM generation failed: {exc}", exc_info=True)
            return RAGResponse(
                answer="",
                sources=source_chunks,
                latency_seconds=0.0,
                query=query,
                error=f"LLM Error: {exc}",
            )

        logger.info(
            f"Query processed successfully. Latency: {latency}s | "
            f"Sources: {len(source_chunks)}"
        )

        # Step 5: Package result
        return RAGResponse(
            answer=answer,
            sources=source_chunks,
            latency_seconds=latency,
            query=query,
        )

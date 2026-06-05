"""
llm.py
------
Manages all communication with the DeepInfra-hosted Meta-Llama LLM.

This module defines:
  1. The System Prompt — which establishes the AI's persona and the
     strict Hallucination Guard rule.
  2. The `LLMClient` class — which wraps LangChain's `ChatOpenAI`
     and formats the prompt-context payload.

Design Decision — DeepInfra via ChatOpenAI:
    DeepInfra exposes an OpenAI-compatible REST API. LangChain's
    `ChatOpenAI` class accepts a custom `base_url`, which means we get
    all of LangChain's prompt chaining, retry logic, and error handling
    for free, without writing raw HTTP requests.

Design Decision — Low Temperature (0.1):
    Technical documentation Q&A is a *factual retrieval* task, not a
    creative writing task. A temperature near zero reduces variance and
    forces the model to pick the most probable token — i.e., the answer
    most grounded in the provided context. Higher temperatures increase
    the risk of the model "drifting" into pre-trained knowledge.

Hallucination Guard Design:
    The guard is implemented at the PROMPT level, not in post-processing.
    The system prompt explicitly instructs the model to output a specific
    sentinel phrase if the context is insufficient. This is more robust
    than checking for keywords in the output, because it makes the model
    an active participant in honesty rather than a passive target of
    a regex filter.
"""

import logging
from typing import List, Tuple

from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import config

logger = logging.getLogger("upwork_rag_bot.llm")

# ─── System Prompt ────────────────────────────────────────────────────────────
# This is the core behavioral contract for the LLM.
# It establishes persona, strict grounding rules, and the exact
# fallback phrase mandated by the assignment specification.

SYSTEM_PROMPT = """You are a Senior Upwork API Consultant with deep expertise in the \
Upwork API platform. You help developers understand the Upwork API by providing precise, \
accurate, and well-structured technical answers.

CRITICAL RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:

1. BASE YOUR ANSWER SOLELY ON THE CONTEXT PROVIDED BELOW.
   Do not use any external knowledge, pre-training data, or assumptions.
   Your only source of truth is the "CONTEXT" section.

2. IF THE ANSWER CANNOT BE FOUND IN THE PROVIDED CONTEXT, you MUST respond with
   EXACTLY this phrase and nothing else:
   "I'm sorry, but the provided documentation does not contain that information."

3. NEVER fabricate API endpoints, rate limits, token expiry times, or any \
   specific technical values. If a specific number or value is not in the context, \
   treat the question as unanswerable.

4. When you do find an answer, cite it clearly and concisely.
   If relevant, use bullet points or numbered steps for clarity.

5. Maintain the persona of a Senior Upwork API Consultant at all times.
   Be professional, precise, and helpful.
"""

HUMAN_PROMPT_TEMPLATE = """CONTEXT:
{context}

---

QUESTION: {question}

Please answer the question based ONLY on the context provided above."""
# ─────────────────────────────────────────────────────────────────────────────


class LLMClient:
    """
    Wraps the DeepInfra ChatOpenAI client and handles prompt construction.

    This class is the single point of contact for all LLM interactions.
    It formats the retrieved context chunks into a coherent prompt and
    returns the raw text response.
    """

    def __init__(self) -> None:
        logger.info(
            f"Initializing LLM client. Model: {config.llm_model_name} | "
            f"Base URL: {config.deepinfra_base_url}"
        )
        self._client = ChatOpenAI(
            model=config.llm_model_name,
            api_key=config.deepinfra_api_key,  # type: ignore[arg-type]
            base_url=config.deepinfra_base_url,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            # Explicitly disable streaming for latency measurement accuracy.
            streaming=False,
        )

    def _format_context(
        self, retrieved_chunks: List[Tuple[Document, float]]
    ) -> str:
        """
        Format retrieved document chunks into a single context string.

        Each chunk is prefixed with its source metadata (page number)
        so the LLM can (if needed) reference where it found information.
        Chunks are separated by a clear delimiter to help the model
        distinguish between different retrieved passages.

        Args:
            retrieved_chunks: List of (Document, score) tuples from retrieval.

        Returns:
            A formatted multi-section context string.
        """
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_chunks, start=1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[Chunk {i} | Source: {source} | Page: {page} | "
                f"Relevance Score: {score:.4f}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def generate_answer(
        self, query: str, retrieved_chunks: List[Tuple[Document, float]]
    ) -> str:
        """
        Generate an LLM response grounded in the retrieved context.

        Constructs a two-message conversation:
          - SystemMessage: Establishes persona and Hallucination Guard rules.
          - HumanMessage: Contains the formatted context and user question.

        Args:
            query: The user's natural language question.
            retrieved_chunks: Top-k chunks retrieved from the vector store.

        Returns:
            The LLM's text response as a string.

        Raises:
            RuntimeError: If the LLM API call fails after retries.
        """
        if not retrieved_chunks:
            logger.warning("No chunks retrieved; returning hallucination guard phrase.")
            return "I'm sorry, but the provided documentation does not contain that information."

        context = self._format_context(retrieved_chunks)
        human_message_content = HUMAN_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_message_content),
        ]

        logger.info(f"Sending query to LLM: '{query[:80]}...'")

        try:
            response = self._client.invoke(messages)
            answer: str = response.content
            logger.info("LLM response received successfully.")
            return answer.strip()
        except Exception as exc:
            logger.error(f"LLM API call failed: {exc}", exc_info=True)
            raise RuntimeError(
                f"Failed to get a response from the LLM API: {exc}"
            ) from exc

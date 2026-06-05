"""
ingestion.py
------------
Handles all data ingestion concerns: loading the source PDF,
performing a sanity check, and splitting the content into
overlapping text chunks for embedding.

Design Decision:
    `RecursiveCharacterTextSplitter` is preferred over the basic
    `CharacterTextSplitter` because it tries to split on paragraph
    boundaries (`\n\n`), then sentence boundaries (`\n`), then spaces,
    and finally characters — as a fallback chain. This produces more
    semantically coherent chunks from technical documentation, even
    though the assignment specifies a hard 500-char limit.

Why Overlap Matters (for Technical Summary):
    Technical documentation often contains multi-part constructs: an
    endpoint name on one line, its parameters on the next, and a code
    example after that. A hard split with no overlap would cut these
    in half, and neither chunk would carry enough context for the LLM
    to produce a grounded answer. The 50-character overlap ensures that
    critical transitional text (e.g., the last line of a code block or
    the beginning of a parameter list) appears in both adjacent chunks,
    greatly improving retrieval recall.
"""

import logging
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .config import config

logger = logging.getLogger("upwork_rag_bot.ingestion")


def load_document(pdf_path: Path) -> List[Document]:
    """
    Load a PDF file using LangChain's PyPDFLoader.

    Each page of the PDF becomes a separate `Document` object with
    metadata including the source path and page number. This page-level
    metadata is preserved throughout the pipeline and surfaced to the
    user in the 'Sources' section of the UI.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        A list of LangChain `Document` objects, one per page.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
        RuntimeError: If the PDF cannot be parsed.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Source document not found at: {pdf_path}\n"
            "Please place 'API Documentation Partial.pdf' inside the 'data/' directory."
        )

    logger.info(f"Loading document from: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    pages: List[Document] = loader.load()
    logger.info(f"Successfully loaded {len(pages)} pages from the document.")
    return pages


def sanity_check(documents: List[Document]) -> None:
    """
    Perform and print a sanity check on the loaded documents.

    Per the assignment specification, this function prints:
      1. The total character count across all pages.
      2. A 500-character sample from the beginning of the document.

    This validates that the PDF was read correctly and the text
    extraction layer is producing meaningful output (not garbage bytes).

    Args:
        documents: A list of LangChain `Document` objects.
    """
    full_text = " ".join(doc.page_content for doc in documents)
    total_chars = len(full_text)
    sample = full_text[:500]

    print("\n" + "=" * 60)
    print("         DOCUMENT SANITY CHECK")
    print("=" * 60)
    print(f"  Total pages loaded   : {len(documents)}")
    print(f"  Total character count: {total_chars:,}")
    print(f"\n  --- Text Sample (first 500 chars) ---")
    print(f"  {sample}")
    print("=" * 60 + "\n")

    logger.info(f"Sanity check passed. Total chars: {total_chars:,}")


def chunk_document(documents: List[Document]) -> List[Document]:
    """
    Split loaded documents into overlapping text chunks.

    Uses `RecursiveCharacterTextSplitter` with the chunk size and
    overlap defined in `config`. The splitter attempts to respect
    natural language boundaries before resorting to hard character
    cuts, producing more coherent chunks.

    Source metadata (page number, source file) is automatically
    propagated to each child chunk by LangChain, enabling source
    attribution in the final UI.

    Args:
        documents: A list of LangChain `Document` objects (one per page).

    Returns:
        A list of smaller, overlapping `Document` chunks.
    """
    logger.info(
        f"Chunking documents with size={config.chunk_size}, "
        f"overlap={config.chunk_overlap}."
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        # Keeps paragraph → sentence → word → char precedence.
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: List[Document] = splitter.split_documents(documents)

    logger.info(f"Document split into {len(chunks)} chunks.")
    return chunks


def ingest_pipeline(pdf_path: Path) -> List[Document]:
    """
    Orchestrate the full ingestion pipeline for a single PDF.

    This is the public entry point for the ingestion module:
        1. Load the document.
        2. Run the sanity check (prints to stdout as required).
        3. Chunk the document.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        A list of processed text chunks ready for embedding.
    """
    documents = load_document(pdf_path)
    sanity_check(documents)
    chunks = chunk_document(documents)
    return chunks

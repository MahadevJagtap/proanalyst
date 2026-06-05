"""
config.py
---------
Centralized configuration management for the Upwork RAG Bot.

Loads secrets from Streamlit Cloud (``st.secrets``) when deployed,
or from a local ``.env`` file via python-dotenv during development.
Validates their presence, exposes typed settings, and configures
the application-wide logger.

Design Decision:
    Using a single `Config` dataclass as the single source of truth
    avoids scattered `os.getenv()` calls across the codebase, making
    it trivial to audit what secrets the application requires.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of src/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _get_secret(key: str, default: str = None) -> str:
    """
    Read a configuration value from Streamlit secrets (Cloud) or
    environment variables (local dev), in that order.

    Streamlit Community Cloud injects secrets via ``st.secrets``.
    Locally, values come from ``.env`` loaded by python-dotenv.

    Args:
        key: The secret / env-var name.
        default: Fallback if the key is not found anywhere.

    Returns:
        The resolved value, or *default*.
    """
    # 1. Try Streamlit secrets (available on Community Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    # 2. Fall back to environment variable
    value = os.getenv(key)
    if value is not None:
        return value

    return default


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a module-level logger.

    Uses a consistent format so all pipeline steps are traceable
    in production via log aggregation tools.

    Args:
        level: The minimum log level to capture (default: INFO).

    Returns:
        A configured Logger instance.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("upwork_rag_bot")


logger = setup_logging()


@dataclass(frozen=True)
class Config:
    """
    Immutable application configuration loaded from environment variables
    or Streamlit secrets.

    `frozen=True` ensures configuration cannot be accidentally mutated
    after initialization, which is a critical safety property for
    production systems.
    """

    # --- LLM Settings ---
    deepinfra_api_key: str = field(
        default_factory=lambda: _require_secret("DEEPINFRA_API_KEY")
    )
    deepinfra_base_url: str = field(
        default_factory=lambda: _get_secret(
            "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"
        )
    )
    llm_model_name: str = field(
        default_factory=lambda: _get_secret(
            "LLM_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
    )
    llm_temperature: float = field(
        default_factory=lambda: float(_get_secret("LLM_TEMPERATURE", "0.1"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(_get_secret("LLM_MAX_TOKENS", "1024"))
    )

    # --- Embedding Settings ---
    embedding_model_name: str = field(
        default_factory=lambda: _get_secret(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # --- Document Ingestion Settings ---
    chunk_size: int = field(
        default_factory=lambda: int(_get_secret("CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(_get_secret("CHUNK_OVERLAP", "50"))
    )

    # --- Path Settings ---
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data"
    )
    chroma_db_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "chroma_db"
    )
    chroma_collection_name: str = field(
        default_factory=lambda: _get_secret(
            "CHROMA_COLLECTION_NAME", "upwork_api_docs"
        )
    )

    # --- RAG Settings ---
    retrieval_top_k: int = field(
        default_factory=lambda: int(_get_secret("RETRIEVAL_TOP_K", "3"))
    )


def _require_secret(key: str) -> str:
    """
    Retrieve a required secret or raise a descriptive error.

    Checks Streamlit secrets first, then environment variables.
    This is intentionally not a silent failure — missing API keys should
    crash loudly at startup, not midway through a user's query.

    Args:
        key: The secret / environment variable name.

    Returns:
        The value of the secret.

    Raises:
        EnvironmentError: If the secret is not set or is empty.
    """
    value = _get_secret(key)
    if not value:
        raise EnvironmentError(
            f"Required secret '{key}' is not set. "
            f"On Streamlit Cloud: add it in App Settings → Secrets. "
            f"Locally: copy .env.example to .env and populate your credentials."
        )
    return value


# Singleton config instance — import this everywhere.
config = Config()

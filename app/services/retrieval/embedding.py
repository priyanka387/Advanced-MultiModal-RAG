import re
import time
import threading
import logging
from typing import Optional, List
from datetime import datetime, timedelta

import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

from app.config import settings

# ==========================================================
# Configuration
# ==========================================================

BATCH_SIZE = 10

GEMINI_DIM = 3072
FALLBACK_DIM = 768

MAX_RETRIES = 5
DEFAULT_RETRY_WAIT = 15
RETRY_GEMINI_AFTER_SECONDS = 300  # Retry Gemini after 5 minutes

GEMINI_MODEL_NAME = "models/gemini-embedding-2"
FALLBACK_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_active_model: Optional[GoogleGenerativeAIEmbeddings | SentenceTransformer] = None
_model_type: Optional[str] = None
_switched_to_fallback_at: Optional[datetime] = None
_lock = threading.Lock()

logger = logging.getLogger(__name__)


# ==========================================================
# Load Gemini
# ==========================================================

def _load_gemini() -> Optional[GoogleGenerativeAIEmbeddings]:
    """Load Gemini embedding model with health check."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model=GEMINI_MODEL_NAME,
            google_api_key=settings.GEMINI_API_KEY,
        )

        # Health check
        model.embed_query("health-check")
        logfire.info("Gemini embedding model loaded successfully.")

        return model

    except Exception as e:
        logfire.warning(f"Unable to load Gemini: {e}")
        return None


# ==========================================================
# Load SentenceTransformer
# ==========================================================

def _load_fallback() -> SentenceTransformer:
    """Load SentenceTransformer fallback model."""
    logfire.warning(
        f"Loading SentenceTransformer ({FALLBACK_MODEL_NAME})"
    )

    return SentenceTransformer(FALLBACK_MODEL_NAME)


# ==========================================================
# Initialise Once (Thread-Safe)
# ==========================================================

def _init() -> None:
    """Initialize embedding backend with thread safety."""
    global _active_model, _model_type, _switched_to_fallback_at

    with _lock:
        if _active_model is not None:
            return

        gemini = _load_gemini()

        if gemini is not None:
            _active_model = gemini
            _model_type = "gemini"
        else:
            _active_model = _load_fallback()
            _model_type = "fallback"

        logfire.info(f"Embedding backend initialized: {_model_type}")


# ==========================================================
# Public Helpers
# ==========================================================

def get_embedding_backend() -> str:
    """Get current embedding backend type."""
    _init()
    return _model_type


def get_embedding_dim() -> int:
    """Get embedding dimension for current backend."""
    _init()

    if _model_type == "gemini":
        return GEMINI_DIM

    return FALLBACK_DIM


def cleanup() -> None:
    """Clean up resources."""
    global _active_model, _model_type

    with _lock:
        if isinstance(_active_model, SentenceTransformer):
            try:
                # Unload model from memory
                _active_model = None
                logfire.info("Cleaned up SentenceTransformer resources.")
            except Exception as e:
                logfire.warning(f"Error during cleanup: {e}")


# ==========================================================
# Retry Delay
# ==========================================================

def _extract_retry_time(error: Exception) -> int:
    """Extract retry time from error message."""
    try:
        error_str = str(error).lower()
        match = re.search(r"retry in ([0-9.]+)s", error_str)

        if match:
            retry_seconds = int(float(match.group(1))) + 1
            return retry_seconds

    except (ValueError, AttributeError) as e:
        logfire.debug(f"Could not parse retry time from error: {e}")

    return DEFAULT_RETRY_WAIT


def _should_retry_gemini() -> bool:
    """Check if we should attempt to reconnect to Gemini."""
    global _switched_to_fallback_at, _model_type

    if _model_type != "fallback":
        return True

    if _switched_to_fallback_at is None:
        return False

    elapsed = datetime.now() - _switched_to_fallback_at
    return elapsed > timedelta(seconds=RETRY_GEMINI_AFTER_SECONDS)


# ==========================================================
# Gemini Batch Embedding
# ==========================================================

def _embed_with_gemini(batch: List[str]) -> List[List[float]]:
    """Embed batch using Gemini with retry logic."""
    if not _active_model or not isinstance(_active_model, GoogleGenerativeAIEmbeddings):
        raise RuntimeError("Gemini model not initialized.")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _active_model.embed_documents(batch)

        except Exception as e:
            error_text = str(e).lower()

            # Check for rate limiting or quota issues
            if any(x in error_text for x in ["429", "resource_exhausted", "quota", "rate"]):
                wait = _extract_retry_time(e)

                logfire.warning(
                    f"Gemini quota reached. "
                    f"Retrying in {wait}s "
                    f"({attempt}/{MAX_RETRIES})"
                )

                time.sleep(wait)
                continue

            # Permanent error
            logfire.error(
                f"Gemini permanent error on attempt {attempt}/{MAX_RETRIES}: {e}",
                extra={"batch_size": len(batch)}
            )
            raise

    raise RuntimeError("Gemini permanently unavailable after max retries.")


# ==========================================================
# SentenceTransformer Batch Embedding
# ==========================================================

def _embed_with_fallback(batch: List[str]) -> List[List[float]]:
    """Embed batch using SentenceTransformer."""
    if not _active_model or not isinstance(_active_model, SentenceTransformer):
        raise RuntimeError("Fallback model not initialized.")

    return _active_model.encode(
        batch,
        show_progress_bar=False,
        convert_to_numpy=False,
    )


# ==========================================================
# Automatic Backend Switch
# ==========================================================

def _embed_batch(batch: List[str]) -> List[List[float]]:
    """Embed batch with automatic fallback."""
    global _active_model, _model_type, _switched_to_fallback_at

    if not batch:
        raise ValueError("Cannot embed empty batch")

    # If already on fallback, use it
    if _model_type == "fallback":
        return _embed_with_fallback(batch)

    # Try Gemini, fall back to SentenceTransformer on failure
    try:
        return _embed_with_gemini(batch)

    except Exception as e:
        logfire.warning(
            f"Gemini embedding failed: {e}. "
            f"Switching to SentenceTransformer.",
            extra={"batch_size": len(batch)}
        )

        with _lock:
            _active_model = _load_fallback()
            _model_type = "fallback"
            _switched_to_fallback_at = datetime.now()

        try:
            return _embed_with_fallback(batch)
        except Exception as fallback_error:
            logfire.error(
                f"Both Gemini and fallback failed: {fallback_error}",
                extra={"batch_size": len(batch)}
            )
            raise RuntimeError(
                f"Embedding failed with both backends. "
                f"Fallback error: {fallback_error}"
            ) from fallback_error


# ==========================================================
# Query Embedding
# ==========================================================

def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    global _active_model, _model_type

    _init()

    # Use fallback if on fallback
    if _model_type == "fallback":
        try:
            return _embed_with_fallback([query])[0]
        except Exception as e:
            logfire.error(f"Query embedding with fallback failed: {e}")
            raise

    # Try Gemini first
    try:
        return _active_model.embed_query(query)

    except Exception as e:
        logfire.warning(
            f"Query embedding with Gemini failed: {e}. "
            f"Switching to SentenceTransformer."
        )

        with _lock:
            _active_model = _load_fallback()
            _model_type = "fallback"
            _switched_to_fallback_at = datetime.now()

        try:
            return _embed_with_fallback([query])[0]
        except Exception as fallback_error:
            logfire.error(
                f"Query embedding with fallback also failed: {fallback_error}"
            )
            raise RuntimeError(
                f"Could not embed query with any backend: {fallback_error}"
            ) from fallback_error


# ==========================================================
# Public API
# ==========================================================

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of text strings with rate limiting for Gemini."""
    # Input validation
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")

    if not texts:
        logfire.warning("Empty texts list provided to embed_texts")
        return []

    if not all(isinstance(t, str) for t in texts):
        raise ValueError("All items in texts must be strings")

    if any(not t.strip() for t in texts):
        raise ValueError("All texts must be non-empty strings")

    _init()

    # Check if we should retry Gemini connection
    if _should_retry_gemini():
        global _model_type, _active_model, _switched_to_fallback_at
        with _lock:
            if _model_type == "fallback":
                logfire.info(
                    "Attempting to reconnect to Gemini "
                    f"(fallback used for {RETRY_GEMINI_AFTER_SECONDS} seconds)"
                )
                gemini = _load_gemini()
                if gemini is not None:
                    _active_model = gemini
                    _model_type = "gemini"
                    _switched_to_fallback_at = None
                    logfire.info("Successfully reconnected to Gemini")

    embeddings: List[List[float]] = []
    total = len(texts)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    logfire.info(
        f"Embedding {total} chunks "
        f"using {_model_type} "
        f"({total_batches} batches)"
    )

    for batch_no, start in enumerate(
        range(0, total, BATCH_SIZE),
        start=1,
    ):
        batch = texts[start : start + BATCH_SIZE]

        logfire.info(f"Batch {batch_no}/{total_batches}")

        try:
            vectors = _embed_batch(batch)
            embeddings.extend(vectors)
        except Exception as e:
            logfire.error(
                f"Failed to embed batch {batch_no}/{total_batches}",
                extra={
                    "batch_size": len(batch),
                    "start_index": start,
                    "error": str(e),
                }
            )
            raise RuntimeError(
                f"Embedding failed at batch {batch_no}/{total_batches}: {e}"
            ) from e

        # Rate limiting delay for Gemini (2 second delay between batches to avoid quota exhaustion)
        if _model_type == "gemini" and batch_no < total_batches:
            logfire.debug("Waiting 2 seconds before next batch (Gemini rate limiting)")
            time.sleep(2)

    logfire.info(f"Successfully generated {len(embeddings)} embeddings.")

    return embeddings
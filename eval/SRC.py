"""
SRC.py  —  Semantic Review Coverage

Implements the SRC metric described in:

    SRC(P, G) = (1/n) * Σ_{j=1}^{n}  max_i  cos(p_i, g_j)

where
    P = {p_1, ..., p_m}  chunk embeddings from the *generated* review
    G = {g_1, ..., g_n}  chunk embeddings from the *ground-truth* review

Each element in the input lists is one chunk (one strength or weakness
statement, passed in one by one). A pairwise cosine-similarity matrix is
computed between all chunk pairs; then a column-wise maximum matching
strategy aligns each ground-truth chunk g_j with the most similar
generated chunk, and the average of these maxima is returned as the score.

Dependencies:
    pip install sentence-transformers numpy
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# ── Embedding ─────────────────────────────────────────────────────────────────

def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load and return a SentenceTransformer model.

    Args:
        model_name: Any model name accepted by sentence-transformers.
                    Default "all-MiniLM-L6-v2" is compact, fast, and
                    well-suited for semantic-similarity tasks.

    Returns:
        A SentenceTransformer instance.

    Raises:
        ImportError: if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for SRC computation.\n"
            "Install it with:  pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def _embed(texts: List[str], model) -> np.ndarray:
    """
    Encode a list of text strings into L2-normalised embeddings.

    Args:
        texts : list of strings to encode
        model : a SentenceTransformer model instance

    Returns:
        numpy array of shape (len(texts), embedding_dim) with unit-norm rows.
    """
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise → dot product = cosine sim
        show_progress_bar=False,
    )


# ── Core metric ───────────────────────────────────────────────────────────────

def compute_src(
    generated:    List[str],
    ground_truth: List[str],
    model=None,
) -> float:
    """
    Compute the Semantic Review Coverage (SRC) score.

        SRC(P, G) = (1/n) * Σ_{j=1}^{n}  max_i  cos(p_i, g_j)

    Args:
        generated   : strengths or weaknesses from the AI reviewer (P),
                      each item is one statement / chunk.
        ground_truth: strengths or weaknesses from the human reviewer (G),
                      each item is one statement / chunk.
        model       : SentenceTransformer instance.  If None, "all-MiniLM-L6-v2"
                      is loaded automatically on first call.

    Returns:
        SRC score (float).  Range is [-1, 1]; for semantically related text
        the score is typically in [0, 1], with 1.0 meaning perfect coverage.
        Returns 0.0 if either input list is empty.
    """
    if not generated or not ground_truth:
        return 0.0

    if model is None:
        model = load_model()

    P = _embed(generated, model)    # shape (m, d)
    G = _embed(ground_truth, model) # shape (n, d)

    # Since embeddings are L2-normalised, P @ G.T gives cosine similarities.
    # sim_matrix[i, j] = cos(p_i, g_j),  shape: (m, n)
    sim_matrix = P @ G.T

    # Column-wise maximum: for each ground-truth chunk g_j (column j),
    # select the highest cosine similarity across all generated chunks.
    max_sims = sim_matrix.max(axis=0)  # shape: (n,)

    return float(max_sims.mean())


# ── Convenience wrapper ───────────────────────────────────────────────────────

def compute_src_both(
    generated_strengths:    List[str],
    generated_weaknesses:   List[str],
    groundtruth_strengths:  List[str],
    groundtruth_weaknesses: List[str],
    model=None,
) -> dict:
    """
    Compute SRC for strengths and weaknesses separately, plus an overall score.

    Args:
        generated_strengths    : AI-generated strength statements
        generated_weaknesses   : AI-generated weakness statements
        groundtruth_strengths  : Human-written strength statements
        groundtruth_weaknesses : Human-written weakness statements
        model                  : SentenceTransformer instance (auto-loaded if None)

    Returns:
        dict with keys "strengths", "weaknesses", "overall"
    """
    if model is None:
        model = load_model()

    src_s = compute_src(generated_strengths,  groundtruth_strengths,  model)
    src_w = compute_src(generated_weaknesses, groundtruth_weaknesses, model)
    overall = (src_s + src_w) / 2.0

    return {
        "strengths":  round(src_s,   4),
        "weaknesses": round(src_w,   4),
        "overall":    round(overall, 4),
    }


# ── CLI (quick test) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    _gen = [
        "The paper is well-structured and easy to follow.",
        "Experiments cover multiple benchmark configurations.",
    ]
    _gt = [
        "Clear organisational structure mirroring a standard experimental paper.",
        "Comprehensive tabular results across multiple benchmarks.",
        "The claim that replacement policy has little impact is interesting.",
    ]
    score = compute_src(_gen, _gt)
    print(f"SRC = {score:.4f}")

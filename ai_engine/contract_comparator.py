"""
contract_comparator.py
======================
Semantic Contract Comparison Engine for the AI Legal Document Analyzer.

Stage: PDF → Text Extraction → Clause Detection → Risk Analysis
       → **Contract Comparison (THIS MODULE)**

This module compares two legal contracts (or two versions of the same contract)
at the clause level using **dense semantic embeddings** and **FAISS vector search**.
It detects clauses that were added, removed, modified, or remain semantically
identical — even when the exact wording changes.

Algorithm overview
------------------
1. Normalise both clause lists.
2. Embed every clause using ``sentence-transformers/all-MiniLM-L6-v2``.
3. Build a FAISS inner-product index over Document B's embeddings
   (vectors are L2-normalised, so inner product == cosine similarity).
4. For each clause in Document A, retrieve the nearest neighbour in B and
   its cosine similarity score.
5. Classify each pair using fixed thresholds:

   ============  =======================================
   Score range   Classification
   ============  =======================================
   ≥ 0.90        **identical** (same wording / trivial reformulation)
   0.75 – 0.90   **modified** (same topic, materially different terms)
   0.60 – 0.75   **related** (topically linked but substantively different)
   < 0.60        **removed** from Document A (no sufficient match in B)
   ============  =======================================

6. Clauses in Document B that were never the *best match* for any clause in A
   are marked as **added**.

Performance notes
-----------------
- FAISS ``IndexFlatIP`` is exact but O(n · d) per query (not O(n²) pair-wise).
  For 1 000+ clauses, switch to ``IndexIVFFlat`` — the factory string is
  already available as ``FAISS_INDEX_FACTORY``.
- Embedding generation is batched (``EMBED_BATCH_SIZE``) to minimise
  SentenceTransformer overhead.
- The entire class is **pickle-safe** and can be used with ``multiprocessing``.

Dependencies
------------
::

    pip install sentence-transformers faiss-cpu numpy scikit-learn

For GPU inference / larger indexes::

    pip install faiss-gpu
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: Preferred embedding model — lightweight, fast, strong semantic quality.
DEFAULT_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Number of clauses embedded in one SentenceTransformer forward pass.
EMBED_BATCH_SIZE: int = 64

#: FAISS factory string for exact search (no quantisation).
#: For 1 000+ clauses replace with "IVF100,Flat" and call index.train().
FAISS_INDEX_FACTORY: str = "Flat"

# Similarity thresholds  (cosine, range [0, 1])
THRESHOLD_IDENTICAL: float = 0.90   # ≥ this → identical
THRESHOLD_MODIFIED:  float = 0.75   # ≥ this → modified
THRESHOLD_RELATED:   float = 0.60   # ≥ this → related
# below THRESHOLD_RELATED                  → removed


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClauseMatch:
    """
    A single matched pair of clauses and their relationship.

    Attributes
    ----------
    clause_a : str
        Original clause text from Document A.
    clause_b : str | None
        Best-matching clause text from Document B, or ``None`` if no match
        exceeded the minimum threshold (removed clause).
    similarity : float
        Cosine similarity in [0, 1].  ``-1.0`` for removed clauses.
    status : str
        One of ``"identical"``, ``"modified"``, ``"related"``, ``"removed"``.
    index_a : int
        Zero-based position of the clause in Document A.
    index_b : int | None
        Zero-based position of the best-matching clause in Document B, or
        ``None`` for removed clauses.
    """

    clause_a:   str
    clause_b:   str | None
    similarity: float
    status:     str
    index_a:    int
    index_b:    int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "clause_a":   self.clause_a,
            "clause_b":   self.clause_b,
            "similarity": round(self.similarity, 4),
            "status":     self.status,
            "index_a":    self.index_a,
            "index_b":    self.index_b,
        }


@dataclass
class ComparisonReport:
    """
    Structured output of :meth:`ContractComparator.compare_documents`.

    Attributes
    ----------
    added : list[str]
        Clauses present in Document B but not semantically matched in A.
    removed : list[ClauseMatch]
        Clauses in Document A with no sufficient match in B.
    modified : list[ClauseMatch]
        Clause pairs where meaning shifted (score in [0.75, 0.90)).
    related : list[ClauseMatch]
        Clause pairs topically linked but substantively different (score
        in [0.60, 0.75)).
    identical : list[ClauseMatch]
        Clause pairs that are semantically identical (score ≥ 0.90).
    metadata : dict
        Run-time statistics (clause counts, model name, thresholds).
    """

    added:     list[str]           = field(default_factory=list)
    removed:   list[ClauseMatch]   = field(default_factory=list)
    modified:  list[ClauseMatch]   = field(default_factory=list)
    related:   list[ClauseMatch]   = field(default_factory=list)
    identical: list[ClauseMatch]   = field(default_factory=list)
    metadata:  dict[str, Any]      = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return a plain, JSON-serialisable dictionary.

        Compatible with FastAPI ``JSONResponse`` and ``json.dumps``.
        """
        return {
            "added":     self.added,
            "removed":   [m.to_dict() for m in self.removed],
            "modified":  [m.to_dict() for m in self.modified],
            "related":   [m.to_dict() for m in self.related],
            "identical": [m.to_dict() for m in self.identical],
            "metadata":  self.metadata,
        }

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        """
        Format the comparison report as a human-readable string.

        Returns
        -------
        str
            Multi-line report suitable for console output or plain-text APIs.
        """
        _W = 72  # line width

        def _section(title: str, items: list, formatter) -> str:
            lines = [f"\n{'─' * _W}", f"  {title.upper()} ({len(items)})", f"{'─' * _W}"]
            if not items:
                lines.append("  (none)")
            else:
                for i, item in enumerate(items, 1):
                    lines.append(formatter(i, item))
            return "\n".join(lines)

        def _fmt_added(i, text):
            return f"  [{i}] {textwrap.shorten(text, 70)}"

        def _fmt_match(i, m: ClauseMatch):
            score = f"sim={m.similarity:.3f}"
            a = textwrap.shorten(m.clause_a, 65)
            b = textwrap.shorten(m.clause_b or "—", 65)
            return f"  [{i}] {score}\n      A: {a}\n      B: {b}"

        header = (
            f"\n{'═' * _W}\n"
            f"  CONTRACT COMPARISON REPORT\n"
            f"{'═' * _W}\n"
            f"  Doc A clauses : {self.metadata.get('doc_a_clauses', '?')}\n"
            f"  Doc B clauses : {self.metadata.get('doc_b_clauses', '?')}\n"
            f"  Model         : {self.metadata.get('model', '?')}\n"
            f"{'═' * _W}"
        )

        return (
            header
            + _section("Added (in B, not in A)", self.added, _fmt_added)
            + _section("Removed (in A, not in B)", self.removed, _fmt_match)
            + _section("Modified", self.modified, _fmt_match)
            + _section("Related", self.related, _fmt_match)
            + _section("Identical / Similar", self.identical, _fmt_match)
            + f"\n{'═' * _W}\n"
        )

    def __str__(self) -> str:
        return self.to_text()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ContractComparator:
    """
    Semantic contract comparison using sentence embeddings and FAISS search.

    The comparator uses a ``SentenceTransformer`` model to convert legal
    clauses into dense vector representations, stores Document B's vectors in
    a FAISS index, and then — for each clause in Document A — retrieves the
    nearest neighbour in B using cosine similarity.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier for the sentence encoder.
        Defaults to ``"sentence-transformers/all-MiniLM-L6-v2"``.
    threshold_identical : float
        Minimum cosine similarity to classify a pair as *identical*.
    threshold_modified : float
        Minimum score to classify a pair as *modified* (below identical).
    threshold_related : float
        Minimum score to classify a pair as *related* (below modified).
        Pairs scoring below this are classified as *removed*.
    device : str | None
        Torch device string.  ``None`` → auto-detect (CUDA > MPS > CPU).

    Example
    -------
    ::

        from contract_comparator import ContractComparator

        comparator = ContractComparator()

        report = comparator.compare_documents(doc_a_clauses, doc_b_clauses)
        print(report)                     # human-readable
        result = report.to_dict()         # JSON-serialisable dict
    """

    def __init__(
        self,
        model_name:          str   = DEFAULT_EMBED_MODEL,
        threshold_identical: float = THRESHOLD_IDENTICAL,
        threshold_modified:  float = THRESHOLD_MODIFIED,
        threshold_related:   float = THRESHOLD_RELATED,
        device:              str | None = None,
    ) -> None:
        self.model_name          = model_name
        self.threshold_identical = threshold_identical
        self.threshold_modified  = threshold_modified
        self.threshold_related   = threshold_related
        self.device              = device or _auto_device()

        # Lazily populated
        self._model  = None   # SentenceTransformer
        self._index  = None   # faiss.Index
        self._emb_b: np.ndarray | None = None   # embeddings for doc B
        self._clauses_b: list[str]     = []

    # ==================================================================
    # 1. Model loading
    # ==================================================================

    def load_model(self) -> None:
        """
        Download / load the sentence-transformer model into memory.

        Called automatically by :meth:`compare_documents` on first use.
        Pre-call this explicitly to warm up the model at application start::

            comparator = ContractComparator()
            comparator.load_model()   # happens once here
        """
        if self._model is not None:
            return  # already loaded

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model '%s' on device '%s' …", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Model loaded successfully.")

    # ==================================================================
    # 2. Embedding generation
    # ==================================================================

    def embed_clauses(self, clauses: list[str]) -> np.ndarray:
        """
        Convert a list of clause strings into L2-normalised embedding vectors.

        Embeddings are **L2-normalised** so that the cosine similarity between
        any two vectors equals their dot product — the native FAISS metric for
        ``IndexFlatIP``.

        Parameters
        ----------
        clauses : list[str]
            Preprocessed legal clause strings.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(clauses), embedding_dim)``.

        Raises
        ------
        ValueError
            If ``clauses`` is empty.
        RuntimeError
            If the model failed to produce embeddings.
        """
        if not clauses:
            raise ValueError("Cannot embed an empty clause list.")

        self.load_model()

        logger.info("Embedding %d clause(s) in batches of %d …", len(clauses), EMBED_BATCH_SIZE)

        embeddings: np.ndarray = self._model.encode(
            clauses,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=len(clauses) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine = dot product
        )

        if embeddings.shape[0] != len(clauses):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(clauses)}, "
                f"got {embeddings.shape[0]}."
            )

        logger.info("Embeddings shape: %s", embeddings.shape)
        return embeddings.astype(np.float32)

    # ==================================================================
    # 3. FAISS index construction
    # ==================================================================

    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Build a FAISS inner-product index from Document B's embeddings.

        The index stores references to each embedding vector so that nearest-
        neighbour queries run in O(n · d) time instead of O(n²).  For datasets
        with > 1 000 clauses, consider switching to ``IndexIVFFlat`` by
        updating :data:`FAISS_INDEX_FACTORY`.

        Parameters
        ----------
        embeddings : np.ndarray
            L2-normalised float32 array of shape ``(n_clauses, dim)``.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for vector indexing. "
                "Install it with: pip install faiss-cpu"
            ) from exc

        dim = embeddings.shape[1]
        logger.info("Building FAISS IndexFlatIP (dim=%d, n=%d) …", dim, len(embeddings))

        self._index = faiss.IndexFlatIP(dim)   # inner product on normalised vecs = cosine sim
        self._index.add(embeddings)             # O(n) insert

        logger.info("FAISS index ready — %d vectors stored.", self._index.ntotal)

    # ==================================================================
    # 4. Similarity search
    # ==================================================================

    def _search(
        self,
        query_embeddings: np.ndarray,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest neighbours in the FAISS index.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Float32 array of shape ``(n_queries, dim)``.
        k : int
            Number of nearest neighbours to retrieve per query.

        Returns
        -------
        scores : np.ndarray
            Shape ``(n_queries, k)``.  Cosine similarity scores in [-1, 1].
        indices : np.ndarray
            Shape ``(n_queries, k)``.  Positions in Document B's clause list.
        """
        if self._index is None:
            raise RuntimeError("FAISS index not built. Call build_index() first.")

        scores, indices = self._index.search(query_embeddings, k)
        return scores, indices

    # ==================================================================
    # 5. Similarity classification
    # ==================================================================

    def classify_changes(
        self,
        clauses_a:  list[str],
        clauses_b:  list[str],
        emb_a:      np.ndarray,
        emb_b:      np.ndarray,
    ) -> ComparisonReport:
        """
        Run the full classification pipeline and return a :class:`ComparisonReport`.

        Parameters
        ----------
        clauses_a : list[str]
            Preprocessed clauses from Document A.
        clauses_b : list[str]
            Preprocessed clauses from Document B.
        emb_a : np.ndarray
            L2-normalised embeddings for ``clauses_a``.
        emb_b : np.ndarray
            L2-normalised embeddings for ``clauses_b``.

        Returns
        -------
        ComparisonReport
            Structured report with added, removed, modified, related, and
            identical clause lists.
        """
        report = ComparisonReport(
            metadata={
                "doc_a_clauses":      len(clauses_a),
                "doc_b_clauses":      len(clauses_b),
                "model":              self.model_name,
                "threshold_identical": self.threshold_identical,
                "threshold_modified":  self.threshold_modified,
                "threshold_related":   self.threshold_related,
            }
        )

        # Track which Doc-B clauses have been matched (for "added" detection)
        matched_b_indices: set[int] = set()

        # ------------------------------------------------------------------
        # For each clause in A, find the best match in B
        # ------------------------------------------------------------------
        scores, indices = self._search(emb_a, k=1)

        for i, (clause_a, score_row, idx_row) in enumerate(
            zip(clauses_a, scores, indices)
        ):
            best_score = float(score_row[0])
            best_idx_b = int(idx_row[0])
            clause_b   = clauses_b[best_idx_b]

            match = ClauseMatch(
                clause_a   = clause_a,
                clause_b   = clause_b,
                similarity = best_score,
                status     = "",        # filled below
                index_a    = i,
                index_b    = best_idx_b,
            )

            # ---- Classify ------------------------------------------------
            if best_score >= self.threshold_identical:
                match.status = "identical"
                report.identical.append(match)
                matched_b_indices.add(best_idx_b)

            elif best_score >= self.threshold_modified:
                match.status = "modified"
                report.modified.append(match)
                matched_b_indices.add(best_idx_b)

            elif best_score >= self.threshold_related:
                match.status = "related"
                report.related.append(match)
                matched_b_indices.add(best_idx_b)

            else:
                # No reasonable match in B → clause was removed
                match.clause_b   = None
                match.similarity = best_score
                match.status     = "removed"
                match.index_b    = None
                report.removed.append(match)

        # ------------------------------------------------------------------
        # Clauses in B that were never matched → added
        # ------------------------------------------------------------------
        for j, clause_b in enumerate(clauses_b):
            if j not in matched_b_indices:
                report.added.append(clause_b)

        logger.info(
            "Classification complete — identical: %d | modified: %d | "
            "related: %d | removed: %d | added: %d",
            len(report.identical),
            len(report.modified),
            len(report.related),
            len(report.removed),
            len(report.added),
        )

        return report

    # ==================================================================
    # 6. Main public method
    # ==================================================================

    def compare_documents(
        self,
        doc_a_clauses: list[str],
        doc_b_clauses: list[str],
    ) -> ComparisonReport:
        """
        Compare two lists of legal clauses and return a semantic diff report.

        This is the **primary entry point** for the FastAPI backend and any
        downstream pipeline stage.

        The method orchestrates all internal steps:
        normalise → embed → index → search → classify → report.

        Parameters
        ----------
        doc_a_clauses : list[str]
            Clauses / paragraphs from the original contract (Document A).
        doc_b_clauses : list[str]
            Clauses / paragraphs from the revised contract (Document B).

        Returns
        -------
        ComparisonReport
            Structured diff with ``.added``, ``.removed``, ``.modified``,
            ``.related``, and ``.identical`` lists.  Call ``.to_dict()`` for
            a JSON-serialisable dict or ``str()`` for a human-readable report.

        Raises
        ------
        ValueError
            If either clause list is empty after preprocessing.

        Example
        -------
        ::

            comparator = ContractComparator()
            report = comparator.compare_documents(doc_a, doc_b)
            print(report.to_dict())
        """
        # ------------------------------------------------------------------
        # Step 1 – Preprocessing
        # ------------------------------------------------------------------
        logger.info("Preprocessing clauses …")
        clean_a = _preprocess_clauses(doc_a_clauses)
        clean_b = _preprocess_clauses(doc_b_clauses)

        if not clean_a:
            raise ValueError("Document A contains no valid clauses after preprocessing.")
        if not clean_b:
            raise ValueError("Document B contains no valid clauses after preprocessing.")

        logger.info("Doc A: %d clauses | Doc B: %d clauses", len(clean_a), len(clean_b))

        # ------------------------------------------------------------------
        # Step 2 – Generate embeddings
        # ------------------------------------------------------------------
        emb_a = self.embed_clauses(clean_a)
        emb_b = self.embed_clauses(clean_b)

        # ------------------------------------------------------------------
        # Step 3 – Build FAISS index over Doc B
        # ------------------------------------------------------------------
        self.build_index(emb_b)

        # Cache for potential repeated queries
        self._emb_b      = emb_b
        self._clauses_b  = clean_b

        # ------------------------------------------------------------------
        # Step 4 & 5 – Search + Classify
        # ------------------------------------------------------------------
        report = self.classify_changes(clean_a, clean_b, emb_a, emb_b)

        return report

    # ==================================================================
    # 7. Report generation (convenience wrapper)
    # ==================================================================

    def generate_report(
        self,
        doc_a_clauses: list[str],
        doc_b_clauses: list[str],
        *,
        output_format: str = "dict",
    ) -> "dict[str, Any] | str | ComparisonReport":
        """
        Run comparison and return the result in the requested format.

        Parameters
        ----------
        doc_a_clauses : list[str]
            Clauses from Document A.
        doc_b_clauses : list[str]
            Clauses from Document B.
        output_format : str
            ``"dict"``  → JSON-serialisable dict (default, FastAPI-friendly).
            ``"text"``  → human-readable multiline string.
            ``"report"``→ raw :class:`ComparisonReport` object.

        Returns
        -------
        dict | str | ComparisonReport

        Raises
        ------
        ValueError
            If ``output_format`` is not one of the above.
        """
        report = self.compare_documents(doc_a_clauses, doc_b_clauses)

        if output_format == "dict":
            return report.to_dict()
        elif output_format == "text":
            return report.to_text()
        elif output_format == "report":
            return report
        else:
            raise ValueError(
                f"Unknown output_format '{output_format}'. "
                "Choose from 'dict', 'text', or 'report'."
            )

    # ==================================================================
    # 8. Optional: similarity matrix visualisation
    # ==================================================================

    def similarity_matrix(
        self,
        doc_a_clauses: list[str],
        doc_b_clauses: list[str],
    ) -> np.ndarray:
        """
        Compute the full N×M cosine similarity matrix between all clause pairs.

        Useful for heatmap visualisation of how clauses relate across
        the two documents.  For large contracts, prefer
        :meth:`compare_documents` (FAISS top-1 search) over this method.

        Parameters
        ----------
        doc_a_clauses : list[str]
            Clauses from Document A (rows).
        doc_b_clauses : list[str]
            Clauses from Document B (columns).

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(doc_a), len(doc_b))``.
            ``matrix[i, j]`` is the cosine similarity between clause i in A
            and clause j in B.

        Example (matplotlib heatmap)
        ----------------------------
        ::

            import matplotlib.pyplot as plt
            matrix = comparator.similarity_matrix(doc_a, doc_b)
            plt.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel("Document B clauses")
            plt.ylabel("Document A clauses")
            plt.title("Clause Similarity Heatmap")
            plt.show()
        """
        self.load_model()
        clean_a = _preprocess_clauses(doc_a_clauses)
        clean_b = _preprocess_clauses(doc_b_clauses)

        emb_a = self.embed_clauses(clean_a)   # (n, d)  already L2-normalised
        emb_b = self.embed_clauses(clean_b)   # (m, d)

        # Cosine sim matrix: (n, d) @ (d, m) → (n, m)
        matrix: np.ndarray = emb_a @ emb_b.T
        return matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    """Return the best available torch device string."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _normalise_clause(text: str) -> str:
    """
    Normalise whitespace and basic formatting in a single clause string.

    - Strips leading / trailing whitespace.
    - Collapses internal whitespace runs (spaces, tabs, newlines) to a
      single space.
    - Removes zero-width and non-printable characters.

    Parameters
    ----------
    text : str
        Raw clause text.

    Returns
    -------
    str
        Cleaned clause string.
    """
    # Remove non-printable / zero-width chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _preprocess_clauses(clauses: list[str]) -> list[str]:
    """
    Normalise, deduplicate, and filter a list of clause strings.

    Steps
    -----
    1. Normalise whitespace.
    2. Remove clauses that are empty after normalisation.
    3. Deduplicate while preserving order (first occurrence kept).

    Parameters
    ----------
    clauses : list[str]
        Raw clause strings from the document splitter.

    Returns
    -------
    list[str]
        Cleaned, deduplicated clause list.
    """
    seen: set[str] = set()
    result: list[str] = []

    for clause in clauses:
        normalised = _normalise_clause(clause)
        if not normalised:
            continue
        if normalised in seen:
            logger.debug("Duplicate clause skipped: %.60s …", normalised)
            continue
        seen.add(normalised)
        result.append(normalised)

    return result


# ---------------------------------------------------------------------------
# FastAPI integration helper
# ---------------------------------------------------------------------------

def compare_contracts(
    doc_a_clauses: list[str],
    doc_b_clauses: list[str],
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> dict[str, Any]:  # always called with output_format="dict"
    """
    Module-level convenience function for the FastAPI backend.

    Creates a :class:`ContractComparator` singleton internally and returns
    a JSON-ready dict.

    Parameters
    ----------
    doc_a_clauses : list[str]
        Clauses extracted from the original contract.
    doc_b_clauses : list[str]
        Clauses extracted from the revised contract.
    model_name : str
        Embedding model to use.

    Returns
    -------
    dict
        JSON-serialisable comparison result with keys:
        ``added``, ``removed``, ``modified``, ``related``, ``identical``,
        ``metadata``.

    Example (FastAPI endpoint)
    --------------------------
    ::

        from contract_comparator import compare_contracts

        @app.post("/compare_contracts")
        async def compare_endpoint(req: CompareRequest):
            return compare_contracts(req.doc_a_clauses, req.doc_b_clauses)
    """
    comparator = ContractComparator(model_name=model_name)
    return comparator.generate_report(doc_a_clauses, doc_b_clauses, output_format="dict")


# ---------------------------------------------------------------------------
# Self-contained test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Representative sample from the problem specification
    # ------------------------------------------------------------------
    doc_a = [
        "Either party may terminate this agreement with 30 days notice.",
        "The vendor shall maintain confidentiality of all data.",
        "Payment must be made within 30 days of invoice.",
        "The governing law shall be the laws of the State of California.",
        "Neither party shall be liable for indirect or consequential damages.",
    ]

    doc_b = [
        "Either party may terminate this agreement with 60 days notice.",
        "The vendor must keep all customer information strictly confidential.",
        "Payment must be made within 30 days of invoice.",
        "The vendor must comply with GDPR and applicable data protection regulations.",
        "This agreement shall be governed by the laws of England and Wales.",
    ]

    print("\n" + "=" * 72)
    print("  CONTRACT COMPARISON ENGINE — Demo")
    print("=" * 72)

    comparator = ContractComparator()
    report = comparator.compare_documents(doc_a, doc_b)

    # Human-readable report
    print(report.to_text())

    # JSON-ready dict (for APIs)
    import json
    print("\n--- JSON output (to_dict) ---")
    print(json.dumps(report.to_dict(), indent=2))

    # Optional: print similarity matrix
    print("\n--- Similarity matrix (rows=A, cols=B) ---")
    matrix = comparator.similarity_matrix(doc_a, doc_b)
    header = "        " + "  ".join(f"B{j+1:02d}" for j in range(len(doc_b)))
    print(header)
    for i, row in enumerate(matrix):
        row_str = "  ".join(f"{v:.3f}" for v in row)
        print(f"  A{i+1:02d}    {row_str}")

"""
summarizer.py
=============
Legal Document Summarization Module for the AI Legal Document Analyzer.

This module provides a transformer-based summarization pipeline designed
specifically for long legal contracts (10–200 pages / 10k–100k+ characters).

Architecture
------------
1. ``chunk_text``         — split raw text into token-safe windows
2. ``summarize_chunk``    — summarize each window with BART-large-CNN
3. ``combine_summaries``  — merge chunk summaries into one coherent passage
4. ``generate_summary``   — run the full pipeline and return a structured report

Requirements
------------
pip install transformers torch nltk

Usage
-----
>>> from ai_engine.summarizer import LegalSummarizer
>>> summarizer = LegalSummarizer()
>>> summary = summarizer.generate_summary(contract_text)
>>> print(summary)
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Optional heavy imports — imported lazily inside load_model() so the module
# can be imported even when torch / transformers are not installed yet.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HuggingFace model used for summarization.
DEFAULT_MODEL = "facebook/bart-large-cnn"

#: Maximum number of *words* per chunk fed to the summarizer.
DEFAULT_CHUNK_WORD_LIMIT = 700

#: Hard token ceiling the BART model can accept.
BART_MAX_INPUT_TOKENS = 1024

#: Summary length bounds (in tokens) for a single chunk.
CHUNK_SUMMARY_MIN_TOKENS = 60
CHUNK_SUMMARY_MAX_TOKENS = 180

#: Summary length bounds for the final combined-summary pass.
FINAL_SUMMARY_MIN_TOKENS = 120
FINAL_SUMMARY_MAX_TOKENS = 350

#: Section headers the structured formatter looks for (case-insensitive regex).
_SECTION_PATTERNS: dict[str, str] = {
    "obligations": r"oblig|duty|duties|shall|must provide|must deliver",
    "payment": r"payment|fee|invoice|billing|remunerat|compensation|price|cost",
    "termination": r"terminat|cancell|cancel|end of agreement|expir",
    "risk": r"liabilit|indemnif|risk|penalty|penalt|damag",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SummaryResult:
    """Structured output returned by :meth:`LegalSummarizer.generate_summary`.

    Attributes
    ----------
    overall_summary:
        High-level, plain-language abstract of the whole contract.
    key_points:
        Bullet list of the most important facts extracted.
    obligations:
        Sentences describing duties of each party.
    payment_terms:
        Payment-related sentences.
    termination:
        Termination-related sentences.
    risks:
        Liability / risk-related sentences.
    raw_chunk_summaries:
        Per-chunk summaries before the final aggregation pass.
    """

    overall_summary: str = ""
    key_points: List[str] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)
    payment_terms: List[str] = field(default_factory=list)
    termination: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    raw_chunk_summaries: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _fmt_bullets(self, items: List[str], indent: int = 2) -> str:
        prefix = " " * indent + "- "
        lines = []
        for item in items:
            # Wrap long sentences at 100 chars
            wrapped = textwrap.fill(
                item.strip(),
                width=100,
                initial_indent=prefix,
                subsequent_indent=" " * (indent + 2),
            )
            lines.append(wrapped)
        return "\n".join(lines) if lines else f"{' ' * indent}- (none detected)"

    def to_text(self) -> str:
        """Return the full structured report as a human-readable string."""
        border = "═" * 68
        thin = "─" * 68

        sections = [
            border,
            "  CONTRACT SUMMARY",
            border,
            "",
            "  OVERALL SUMMARY",
            thin,
            textwrap.fill(
                self.overall_summary.strip(),
                width=100,
                initial_indent="  ",
                subsequent_indent="  ",
            ),
            "",
            "  KEY POINTS",
            thin,
            self._fmt_bullets(self.key_points),
            "",
            "  OBLIGATIONS",
            thin,
            self._fmt_bullets(self.obligations),
            "",
            "  PAYMENT TERMS",
            thin,
            self._fmt_bullets(self.payment_terms),
            "",
            "  TERMINATION",
            thin,
            self._fmt_bullets(self.termination),
            "",
            "  RISKS & LIABILITY",
            thin,
            self._fmt_bullets(self.risks),
            "",
            border,
        ]
        return "\n".join(sections)

    def to_dict(self) -> dict:
        """Return a plain ``dict`` suitable for JSON / FastAPI responses."""
        return {
            "overall_summary": self.overall_summary,
            "key_points": self.key_points,
            "obligations": self.obligations,
            "payment_terms": self.payment_terms,
            "termination": self.termination,
            "risks": self.risks,
        }

    def __str__(self) -> str:  # pragma: no cover
        return self.to_text()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LegalSummarizer:
    """Transformer-based summarizer for long legal contracts.

    The class lazily loads the HuggingFace pipeline on the first call to
    :meth:`generate_summary` (or when :meth:`load_model` is called explicitly).
    This avoids a multi-second delay on import.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to ``facebook/bart-large-cnn``.
    chunk_word_limit:
        Maximum number of *words* per text chunk.  Defaults to 700.
    device:
        ``"cuda"`` to use GPU, ``"cpu"`` to force CPU.  ``None`` (default)
        auto-detects.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chunk_word_limit: int = DEFAULT_CHUNK_WORD_LIMIT,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.chunk_word_limit = chunk_word_limit
        self._device = device
        self._pipeline = None  # lazy-loaded

        logger.info(
            "LegalSummarizer created (model=%s, chunk_word_limit=%d).",
            model_name,
            chunk_word_limit,
        )

    # ------------------------------------------------------------------
    # 1. Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the HuggingFace summarization pipeline.

        Called automatically by :meth:`generate_summary` on first use.
        Call explicitly if you want to pre-warm the model at startup::

            summarizer = LegalSummarizer()
            summarizer.load_model()   # download / cache happens here

        Raises
        ------
        ImportError
            If ``transformers`` or ``torch`` are not installed.
        """
        if self._pipeline is not None:
            return  # already loaded

        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for LegalSummarizer. "
                "Install with: pip install transformers torch"
            ) from exc

        # Auto-detect device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "Loading summarization model '%s' on device '%s' …",
            self.model_name,
            self._device,
        )

        # Use device_map="auto" for large models on multi-GPU setups.
        # For single-device, pass device index (0 = first GPU, -1 = CPU).
        device_arg = 0 if self._device == "cuda" else -1

        self._pipeline = hf_pipeline(
            "summarization",
            model=self.model_name,
            tokenizer=self.model_name,
            device=device_arg,
            # Disable PyTorch compile (faster cold start, same accuracy)
            torch_dtype=None,
        )
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # 2. Text chunking
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> List[str]:
        """Split *text* into overlapping word-count-bounded chunks.

        The chunker works at the **sentence boundary** level (using NLTK) so
        no sentence is ever cut mid-way.  An overlap of ~10 % of
        ``chunk_word_limit`` is retained between consecutive chunks to
        preserve context around chunk boundaries.

        Parameters
        ----------
        text:
            Raw contract text (any length).

        Returns
        -------
        list[str]
            List of non-empty text chunks ready for the model.
        """
        text = _clean_text(text)
        if not text:
            return []

        sentences = _sentence_tokenize(text)
        if not sentences:
            return []

        overlap_words = max(50, self.chunk_word_limit // 10)
        chunks: List[str] = []
        current_words: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            s_words = sentence.split()
            s_wc = len(s_words)

            # If a single sentence exceeds the limit, hard-split it by words
            if s_wc > self.chunk_word_limit:
                sub_chunks = _hard_split_sentence(sentence, self.chunk_word_limit)
                for sub in sub_chunks:
                    chunks.append(sub)
                continue

            if current_word_count + s_wc > self.chunk_word_limit:
                # Flush current chunk
                if current_words:
                    chunks.append(" ".join(current_words))
                # Start new chunk with overlap from the end of current
                overlap = current_words[-overlap_words:] if overlap_words else []
                current_words = overlap + s_words
                current_word_count = len(current_words)
            else:
                current_words.extend(s_words)
                current_word_count += s_wc

        if current_words:
            chunks.append(" ".join(current_words))

        logger.debug("Text split into %d chunk(s).", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # 3. Single-chunk summarization
    # ------------------------------------------------------------------

    def summarize_chunk(self, chunk: str) -> str:
        """Summarize a single text chunk using the loaded model.

        Parameters
        ----------
        chunk:
            A word-count-bounded section of the contract.

        Returns
        -------
        str
            The model-generated summary for this chunk.
        """
        self.load_model()  # no-op if already loaded

        if not chunk.strip():
            return ""

        try:
            outputs = self._pipeline(
                chunk,
                max_length=CHUNK_SUMMARY_MAX_TOKENS,
                min_length=CHUNK_SUMMARY_MIN_TOKENS,
                do_sample=False,          # deterministic beam search
                num_beams=4,
                truncation=True,          # hard-truncate if still too long
                no_repeat_ngram_size=3,   # reduce repetition
            )
            return outputs[0]["summary_text"].strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("Chunk summarization failed: %s", exc)
            # Graceful degradation: return first N words of the chunk
            return " ".join(chunk.split()[:80]) + " …"

    # ------------------------------------------------------------------
    # 4. Chunk combining
    # ------------------------------------------------------------------

    def combine_summaries(self, chunk_summaries: List[str]) -> str:
        """Merge multiple chunk summaries into one coherent final summary.

        Strategy:
        * If there is only one chunk summary, return it directly.
        * Otherwise, concatenate and run one final summarization pass.
        * Falls back to simple concatenation if the combined text is short
          enough that a second model pass would add no value.

        Parameters
        ----------
        chunk_summaries:
            List of strings produced by :meth:`summarize_chunk`.

        Returns
        -------
        str
            A single, condensed summary string.
        """
        cleaned = [s.strip() for s in chunk_summaries if s.strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]

        combined_text = " ".join(cleaned)
        word_count = len(combined_text.split())

        # If combined is short enough, skip a second model pass
        if word_count <= 200:
            return combined_text

        # Second summarization pass over the merged chunk summaries
        logger.debug(
            "Running final summarization pass over %d chunk summaries (%d words).",
            len(cleaned),
            word_count,
        )
        try:
            outputs = self._pipeline(
                combined_text,
                max_length=FINAL_SUMMARY_MAX_TOKENS,
                min_length=FINAL_SUMMARY_MIN_TOKENS,
                do_sample=False,
                num_beams=4,
                truncation=True,
                no_repeat_ngram_size=3,
            )
            return outputs[0]["summary_text"].strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("Final summarization pass failed: %s", exc)
            return combined_text

    # ------------------------------------------------------------------
    # 5. Full pipeline
    # ------------------------------------------------------------------

    def generate_summary(
        self,
        text: str,
        *,
        paragraphs: Optional[List[str]] = None,
    ) -> SummaryResult:
        """Run the complete summarization pipeline and return a structured report.

        Parameters
        ----------
        text:
            Raw contract text.  Pass either this *or* ``paragraphs``.
        paragraphs:
            Pre-split list of paragraphs.  If provided, will be joined with
            newlines and used instead of *text*.

        Returns
        -------
        SummaryResult
            Structured summary with labelled sections.

        Raises
        ------
        ValueError
            If both *text* and *paragraphs* are empty.
        """
        # Resolve input
        if paragraphs:
            text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
        if not text or not text.strip():
            raise ValueError("'text' or 'paragraphs' must be non-empty.")

        logger.info(
            "Starting summarization pipeline (input length=%d chars).", len(text)
        )

        # Step A: chunk
        chunks = self.chunk_text(text)
        logger.info("Processing %d chunk(s) …", len(chunks))

        # Step B: summarize each chunk
        chunk_summaries: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            logger.debug("  Summarizing chunk %d / %d …", i, len(chunks))
            summary = self.summarize_chunk(chunk)
            if summary:
                chunk_summaries.append(summary)

        # Step C: combine chunk summaries
        overall = self.combine_summaries(chunk_summaries)

        # Step D: extract sections from the full text (rule-based)
        all_sentences = _sentence_tokenize(_clean_text(text))
        sections = _extract_sections(all_sentences)

        # Step E: derive key points from overall summary sentences
        key_point_sentences = _sentence_tokenize(overall)
        key_points = [s.strip() for s in key_point_sentences if len(s.split()) > 6][:8]

        result = SummaryResult(
            overall_summary=overall,
            key_points=key_points,
            obligations=sections.get("obligations", []),
            payment_terms=sections.get("payment", []),
            termination=sections.get("termination", []),
            risks=sections.get("risk", []),
            raw_chunk_summaries=chunk_summaries,
        )

        logger.info("Summarization complete.")
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove non-printable characters."""
    # Replace Windows-style line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces / tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Remove non-printable chars (but keep newlines)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()


def _sentence_tokenize(text: str) -> List[str]:
    """Split *text* into sentences using NLTK if available, else a regex fallback."""
    try:
        import nltk

        # Download once; subsequent calls use the local cache
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            logger.info("Downloading NLTK punkt_tab tokenizer …")
            nltk.download("punkt_tab", quiet=True)

        return nltk.sent_tokenize(text)
    except ImportError:
        # Regex fallback — good enough for legal text
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if s.strip()]


def _hard_split_sentence(sentence: str, word_limit: int) -> List[str]:
    """Split an unusually long sentence into word-count-bounded fragments."""
    words = sentence.split()
    return [
        " ".join(words[i : i + word_limit])
        for i in range(0, len(words), word_limit)
    ]


def _extract_sections(sentences: List[str]) -> dict[str, List[str]]:
    """Classify sentences into labelled sections using keyword patterns.

    Parameters
    ----------
    sentences:
        All sentences in the document.

    Returns
    -------
    dict[str, list[str]]
        Mapping of section name → matching sentences (max 8 per section).
    """
    sections: dict[str, List[str]] = {k: [] for k in _SECTION_PATTERNS}

    for sentence in sentences:
        s_lower = sentence.lower()
        for section, pattern in _SECTION_PATTERNS.items():
            if re.search(pattern, s_lower) and len(sections[section]) < 8:
                cleaned = sentence.strip()
                if len(cleaned.split()) > 5:  # skip very short matches
                    sections[section].append(cleaned)

    # Deduplicate while preserving order
    for key in sections:
        seen: set[str] = set()
        unique = []
        for s in sections[key]:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        sections[key] = unique

    return sections

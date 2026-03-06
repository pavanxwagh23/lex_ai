"""
Paragraph Splitting Module for Legal Document Analyzer
=======================================================

Stage: PDF → Text Extraction → Text Cleaning → **Paragraph Splitting**
        → Clause Classification → Risk Detection → Summarization

This module receives raw text extracted from a PDF legal document and
returns a list of clean, semantically coherent paragraph strings that are
ready to be consumed by downstream NLP models (clause classification,
risk detection, summarization, contract comparison).

Design principles
-----------------
- Zero external dependencies (stdlib only: re, typing)
- Preserves every word exactly — no summarisation or interpretation
- Idempotent: running twice yields the same result
- Composable: every helper can be used standalone in an NLP pipeline
"""

import re
from typing import List


# ---------------------------------------------------------------------------
# Constants – paragraph boundary signals
# ---------------------------------------------------------------------------

# Matches numbered section headings at the *start* of a line:
#   "1."  "1.1"  "1.1.1"  "(a)"  "(i)"
_NUMBERED_SECTION = re.compile(
    r"^(?:\(?\d+[\.\d]*\)?\s|[A-Z]{1,3}[IVX]*\.\s)"
)

# Matches keyword-introduced headings at the *start* of a line:
#   "Section 3", "Article IV", "ARTICLE 5", "Clause 2"
_KEYWORD_SECTION = re.compile(
    r"^(?:Section|SECTION|Article|ARTICLE|Clause|CLAUSE|Schedule|SCHEDULE|Exhibit|EXHIBIT)\s+[\w]+",
    re.IGNORECASE,
)

# Matches bullet-point markers at the *start* of a line
_BULLET = re.compile(r"^[\•\-\*\–\—○◦▪]\s+")

# All-uppercase "heading" lines (e.g. "TERMINATION", "PAYMENT TERMS")
# Must be at least 3 characters and contain only letters / spaces / punctuation
_ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z\s\.\,\:\;\-]{2,}$")

# Minimum paragraph character length before we consider merging it
_SHORT_LINE_THRESHOLD = 40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalise raw text extracted from a PDF before paragraph splitting.

    Steps
    -----
    1. Replace all tab characters with a single space.
    2. Collapse runs of spaces (but NOT newlines) into a single space.
    3. Collapse runs of 3+ consecutive newlines into exactly two newlines
       so that paragraph boundaries remain unambiguous.
    4. Strip leading / trailing whitespace from the whole document.

    Parameters
    ----------
    text : str
        Raw text string from a PDF extractor.

    Returns
    -------
    str
        Normalised text with consistent whitespace.
    """
    # Step 1: Remove tab characters
    text = text.replace("\t", " ")

    # Step 2: Collapse multiple spaces on the same line into one
    # Use a non-newline whitespace class [ \f\r\v] to avoid eating newlines
    text = re.sub(r"[ \f\r\v]{2,}", " ", text)

    # Step 3: Collapse 3+ consecutive newlines → exactly two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 4: Strip surrounding whitespace
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split a normalised legal document string into clean paragraph strings.

    Paragraph boundary signals (in priority order)
    -----------------------------------------------
    1. Double newline  (``\\n\\n``)
    2. Numbered section headings  (``1.``, ``1.1``, ``(a)``, ``Article IV``)
    3. Keyword section headings   (``Section 3``, ``Clause 2``)
    4. Bullet points              (``•``, ``-``, ``*``)
    5. ALL-CAPS headings          (``TERMINATION``, ``PAYMENT TERMS``)

    Within each paragraph, intra-paragraph line breaks caused by PDF
    column wrapping are collapsed: lines that do *not* start a new
    boundary are joined with a single space.

    Parameters
    ----------
    text : str
        Raw or pre-normalised legal document text.

    Returns
    -------
    List[str]
        Ordered list of paragraph strings, deduplicated of whitespace,
        with no empty entries.
    """
    # ------------------------------------------------------------------
    # Step 0 – normalise first so we work on consistent input
    # ------------------------------------------------------------------
    text = normalize_text(text)

    # ------------------------------------------------------------------
    # Step 1 – split on double newlines to get coarse chunks
    # ------------------------------------------------------------------
    coarse_chunks: List[str] = re.split(r"\n{2,}", text)

    # ------------------------------------------------------------------
    # Step 2 – within each coarse chunk, further split on boundary signals
    #          found at the *start* of individual lines
    # ------------------------------------------------------------------
    paragraphs: List[str] = []

    for chunk in coarse_chunks:
        lines = chunk.splitlines()

        # Accumulate lines belonging to the *same* paragraph here
        current_lines: List[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Blank line inside a chunk → flush current paragraph
                if current_lines:
                    paragraphs.append(_join_lines(current_lines))
                    current_lines = []
                continue

            # Detect whether this line *starts* a new boundary
            is_new_boundary = (
                _NUMBERED_SECTION.match(stripped)
                or _KEYWORD_SECTION.match(stripped)
                or _BULLET.match(stripped)
                or _ALL_CAPS_HEADING.match(stripped)
            )

            if is_new_boundary and current_lines:
                # Flush the paragraph accumulated so far …
                paragraphs.append(_join_lines(current_lines))
                # … then start a fresh one with this boundary line
                current_lines = [stripped]
            else:
                # Continuation of the current paragraph
                current_lines.append(stripped)

        # Flush whatever remains in the current chunk
        if current_lines:
            paragraphs.append(_join_lines(current_lines))

    # ------------------------------------------------------------------
    # Step 3 – final cleanup: strip each paragraph, drop empty ones
    # ------------------------------------------------------------------
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def merge_short_lines(paragraphs: List[str]) -> List[str]:
    """
    Merge "orphan" paragraphs that were broken across lines by the PDF
    renderer and are too short to stand alone.

    A paragraph is merged into the *next* paragraph when **all** of:
      - Its length is below ``_SHORT_LINE_THRESHOLD`` characters, AND
      - It does not look like a standalone legal heading
        (not a numbered section, keyword section, or ALL-CAPS heading).

    If the short fragment is the **last** paragraph, it is merged
    *backward* into the previous paragraph instead.

    Parameters
    ----------
    paragraphs : List[str]
        Output from :func:`split_into_paragraphs`.

    Returns
    -------
    List[str]
        Paragraphs with short orphan lines merged in.
    """
    if not paragraphs:
        return paragraphs

    merged: List[str] = []
    i = 0

    while i < len(paragraphs):
        current = paragraphs[i]

        # Decide if this fragment is a short non-heading orphan
        if len(current) < _SHORT_LINE_THRESHOLD and not _is_heading(current):
            if i + 1 < len(paragraphs):
                # Merge forward: prepend to the next paragraph
                paragraphs[i + 1] = current + " " + paragraphs[i + 1]
                i += 1
                continue
            elif merged:
                # Merge backward: append to the last accumulated paragraph
                merged[-1] = merged[-1] + " " + current
                i += 1
                continue

        merged.append(current)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _join_lines(lines: List[str]) -> str:
    """
    Join a list of intra-paragraph lines into a single string.

    Lines are separated by a single space, which repairs the hard line-breaks
    that PDF extractors insert at the end of each typeset line.

    Parameters
    ----------
    lines : List[str]
        Individual stripped lines belonging to the same paragraph.

    Returns
    -------
    str
        Single-line paragraph string.
    """
    return " ".join(line.strip() for line in lines if line.strip())


def _is_heading(text: str) -> bool:
    """
    Return True if *text* looks like a legal section heading.

    Used by :func:`merge_short_lines` to avoid accidentally swallowing
    a clause heading into the body of the preceding paragraph.

    Parameters
    ----------
    text : str
        Candidate paragraph string.

    Returns
    -------
    bool
    """
    return bool(
        _NUMBERED_SECTION.match(text)
        or _KEYWORD_SECTION.match(text)
        or _ALL_CAPS_HEADING.match(text)
    )


# ---------------------------------------------------------------------------
# NLP Pipeline integration helpers
# ---------------------------------------------------------------------------

def paragraphs_to_pipeline_input(paragraphs: List[str]) -> List[dict]:
    """
    Convert a list of paragraph strings into structured dicts suitable for
    passing to spaCy, Hugging Face Transformers, or any other NLP pipeline.

    Each dict carries:
      - ``id``        : 0-based paragraph index (stable across re-runs)
      - ``text``      : the cleaned paragraph string
      - ``is_heading``: bool flag that downstream models can use to skip
                        or tag headings differently
      - ``char_count``: character length (useful for windowing / chunking)

    Parameters
    ----------
    paragraphs : List[str]
        Output from :func:`split_into_paragraphs` or
        :func:`merge_short_lines`.

    Returns
    -------
    List[dict]
        Structured records ready for an NLP model.

    Example
    -------
    >>> records = paragraphs_to_pipeline_input(paragraphs)
    >>> for rec in records:
    ...     print(rec["id"], rec["is_heading"], rec["text"][:60])
    """
    records = []
    for idx, para in enumerate(paragraphs):
        records.append(
            {
                "id": idx,
                "text": para,
                "is_heading": _is_heading(para),
                "char_count": len(para),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Entry point – self-contained test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  Sample contract text that simulates typical PDF-extraction output  #
    # (note the inconsistent indentation, broken lines, and extra spaces)  #
    # ------------------------------------------------------------------ #
    sample_contract = """
    THIS AGREEMENT is made between Company A and Company B.

    1. Termination
       Either party may terminate this agreement with 30 days
       written notice to the other party.

    2. Payment Terms
       The client agrees to pay all invoices within thirty (30)
       days of receipt. Late payments shall incur interest at
       a rate of 1.5% per month.

    2.1 Disputed Invoices
    If the client disputes an invoice, they must notify the
    company in writing within ten (10) days of receipt.

    3. Limitation of Liability
       In no event shall either party be liable for indirect
       or consequential damages arising from this agreement.

    CONFIDENTIALITY

    All information shared between the parties shall be treated
    as strictly confidential and shall not be disclosed to any
    third party without prior written consent.

    Section 5 Governing Law
    This agreement shall be governed by the laws of the State
    of Delaware, without regard to conflicts of law principles.

    • Entire Agreement: This document constitutes the entire
      agreement between the parties.
    • Severability: If any provision is found invalid, the
      remaining provisions shall remain in full force.
    """

    print("=" * 70)
    print("PARAGRAPH SPLITTER – Demo Run")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Stage 1: Normalise raw text
    # ------------------------------------------------------------------
    normalised = normalize_text(sample_contract)
    print(f"\n[1] Normalised text ({len(normalised)} chars):\n")
    print(normalised)

    # ------------------------------------------------------------------
    # Stage 2: Split into paragraphs
    # ------------------------------------------------------------------
    paragraphs = split_into_paragraphs(sample_contract)
    print(f"\n[2] Raw split → {len(paragraphs)} paragraph(s) found:\n")
    for i, para in enumerate(paragraphs, 1):
        print(f"  [{i:02d}] {para}")

    # ------------------------------------------------------------------
    # Stage 3: Merge short orphan lines
    # ------------------------------------------------------------------
    merged = merge_short_lines(paragraphs)
    print(f"\n[3] After merging short lines → {len(merged)} paragraph(s):\n")
    for i, para in enumerate(merged, 1):
        print(f"  [{i:02d}] {para}")

    # ------------------------------------------------------------------
    # Stage 4: Convert to NLP pipeline input dicts
    # ------------------------------------------------------------------
    pipeline_input = paragraphs_to_pipeline_input(merged)
    print(f"\n[4] NLP pipeline records ({len(pipeline_input)} total):\n")
    for rec in pipeline_input:
        heading_tag = "[HEADING]" if rec["is_heading"] else "[BODY]   "
        print(f"  {heading_tag} id={rec['id']:02d}  chars={rec['char_count']:4d}  {rec['text'][:70]}")

    print("\n" + "=" * 70)
    print("Pipeline-ready paragraphs (final output):")
    print("=" * 70)
    final = [rec["text"] for rec in pipeline_input]
    import json
    print(json.dumps(final, indent=2))

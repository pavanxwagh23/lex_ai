"""
Legal Document Preprocessing Engine

WARNING: This module performs STRUCTURAL cleanup ONLY.
- Does NOT interpret legal meaning
- Does NOT summarize content
- Does NOT add or remove information
- Preserves every word exactly as extracted
"""

import logging
import re
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _is_clause_start(text: str) -> bool:
    """
    Detect if text starts a new legal clause or section.
    
    Indicators:
    - Starts with number followed by period (1., 2.1, etc.)
    - Starts with letter in parentheses ((a), (b), etc.)
    - Starts with Roman numerals (I., II., III., etc.)
    - Starts with "ARTICLE", "SECTION", "CLAUSE"
    - All caps heading
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to start a new clause
    """
    text = text.strip()
    if not text:
        return False
    
    # Pattern 1: Numeric clause (1., 2.1., 3.4.5., etc.)
    if re.match(r'^\d+(\.\d+)*\.?\s+', text):
        return True
    
    # Pattern 2: Letter in parentheses ((a), (b), (i), etc.)
    if re.match(r'^\([a-z0-9]+\)\s+', text, re.IGNORECASE):
        return True
    
    # Pattern 3: Roman numerals (I., II., III., IV., etc.)
    if re.match(r'^[IVX]+\.?\s+', text):
        return True
    
    # Pattern 4: Legal section headers
    section_headers = [
        r'^ARTICLE\s+',
        r'^SECTION\s+',
        r'^CLAUSE\s+',
        r'^SCHEDULE\s+',
        r'^APPENDIX\s+',
        r'^EXHIBIT\s+'
    ]
    for pattern in section_headers:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    
    # Pattern 5: All caps heading (at least 3 words, all uppercase)
    words = text.split()
    if len(words) >= 3:
        first_three = ' '.join(words[:3])
        if first_three.isupper() and len(first_three) > 10:
            return True
    
    return False


def _is_sentence_complete(text: str) -> bool:
    """
    Check if text ends with sentence-ending punctuation.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be a complete sentence
    """
    text = text.strip()
    if not text:
        return False
    
    # Ends with period, semicolon, colon, or closing parenthesis
    return bool(re.search(r'[.;:)]$', text))


def _should_merge_with_previous(current: str, previous: str) -> bool:
    """
    Determine if current paragraph should be merged with previous.
    
    Merge if:
    - Previous paragraph is incomplete (no ending punctuation)
    - Current paragraph doesn't start a new clause
    - Current paragraph starts with lowercase (continuation)
    
    Args:
        current: Current paragraph text
        previous: Previous paragraph text
        
    Returns:
        True if paragraphs should be merged
    """
    if not previous or not current:
        return False
    
    current = current.strip()
    previous = previous.strip()
    
    # Don't merge if current starts a new clause
    if _is_clause_start(current):
        return False
    
    # Merge if previous is incomplete
    if not _is_sentence_complete(previous):
        return True
    
    # Merge if current starts with lowercase (likely continuation)
    if current and current[0].islower():
        return True
    
    # Merge if current starts with "and", "or", "but", etc.
    continuation_words = [
        'and', 'or', 'but', 'however', 'provided',
        'unless', 'except', 'including', 'excluding'
    ]
    first_word = current.split()[0].lower() if current.split() else ''
    if first_word in continuation_words:
        return True
    
    return False


def restructure_legal_paragraphs(paragraphs: List[str]) -> List[str]:
    """
    Clean and restructure legal contract paragraphs.
    
    Rules:
    1. Merge lines that belong to the same sentence
    2. Keep clause numbers and headings attached to content
    3. Split only when a NEW clause/section clearly begins
    4. Do NOT modify wording
    5. Do NOT summarize
    6. Do NOT remove text
    7. Preserve legal meaning exactly
    
    Args:
        paragraphs: List of raw paragraph strings
        
    Returns:
        List of restructured paragraph strings
    """
    if not paragraphs:
        return []
    
    logger.info(f"Restructuring {len(paragraphs)} paragraphs")
    
    # Remove empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return []
    
    # Process paragraphs
    restructured = []
    current_paragraph = paragraphs[0]
    
    for i in range(1, len(paragraphs)):
        next_para = paragraphs[i]
        
        # Check if we should merge with current
        if _should_merge_with_previous(next_para, current_paragraph):
            # Merge: add space if current doesn't end with space/hyphen
            if current_paragraph and current_paragraph[-1] not in (' ', '-'):
                current_paragraph += ' '
            current_paragraph += next_para
            logger.debug(f"Merged paragraph {i} with previous")
        else:
            # Don't merge: save current and start new
            restructured.append(current_paragraph.strip())
            current_paragraph = next_para
            logger.debug(f"Started new paragraph at {i}")
    
    # Add the last paragraph
    if current_paragraph:
        restructured.append(current_paragraph.strip())
    
    logger.info(f"Restructured into {len(restructured)} paragraphs")
    
    return restructured


def clean_paragraph_text(paragraph: str) -> str:
    """
    Clean individual paragraph text while preserving all content.
    
    Only fixes:
    - Extra spaces between words
    - Inconsistent spacing around punctuation
    - Stray newlines within paragraph
    
    Does NOT:
    - Remove any words
    - Change word order
    - Modify legal terminology
    - Summarize content
    
    Args:
        paragraph: Raw paragraph text
        
    Returns:
        Cleaned paragraph text
    """
    if not paragraph:
        return ""
    
    # Remove stray newlines within paragraph
    text = paragraph.replace('\n', ' ')
    
    # Normalize spaces around punctuation
    # Fix space before punctuation (remove)
    text = re.sub(r'\s+([.,;:)])', r'\1', text)
    
    # Fix space after punctuation (ensure one space)
    text = re.sub(r'([.,;:)])([A-Za-z0-9])', r'\1 \2', text)
    
    # Fix space after opening parenthesis (remove)
    text = re.sub(r'([(])\s+', r'\1', text)
    
    # Fix space before closing parenthesis (remove)
    text = re.sub(r'\s+([)])', r'\1', text)
    
    # Collapse multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text


def process_legal_document(paragraphs: List[str]) -> List[str]:
    """
    Complete preprocessing pipeline for legal document paragraphs.
    
    Steps:
    1. Clean individual paragraphs (whitespace normalization)
    2. Restructure paragraphs (merge/split based on legal structure)
    3. Final cleanup
    
    This function preserves ALL content and legal meaning.
    
    Args:
        paragraphs: List of raw paragraph strings from extraction
        
    Returns:
        List of cleaned and restructured paragraph strings
    """
    if not paragraphs:
        logger.warning("Empty input paragraphs")
        return []
    
    logger.info(f"Processing {len(paragraphs)} legal document paragraphs")
    
    # Step 1: Clean individual paragraphs
    cleaned = [clean_paragraph_text(p) for p in paragraphs]
    cleaned = [p for p in cleaned if p]  # Remove empty
    logger.info(f"After cleaning: {len(cleaned)} paragraphs")
    
    # Step 2: Restructure based on legal clause structure
    restructured = restructure_legal_paragraphs(cleaned)
    logger.info(f"After restructuring: {len(restructured)} paragraphs")
    
    # Step 3: Final cleanup pass
    final = [clean_paragraph_text(p) for p in restructured]
    final = [p for p in final if p]  # Remove empty
    
    logger.info(f"Final output: {len(final)} paragraphs")
    
    return final


if __name__ == "__main__":
    import json
    
    # Example usage with sample legal text
    print("=" * 70)
    print("Legal Document Preprocessing Engine")
    print("=" * 70)
    
    # Sample input with common issues
    sample_paragraphs = [
        "1. Termination",
        "Either party may terminate this Agreement",
        "with 30 days written notice.",
        "2. Payment Terms The Client shall pay",
        "all fees within 15 days of invoice date.",
        "Late payments will incur interest at 1.5% per month.",
        "3. Confidentiality",
        "Each party shall keep all Confidential Information",
        "strictly confidential and shall not disclose",
        "to any third party without prior written consent.",
        "(a) Definition For purposes of this Agreement",
        "Confidential Information means all non-public information.",
        "(b) Exceptions",
        "Confidential Information does not include information that",
        "is publicly available or independently developed."
    ]
    
    print("\nInput: {} paragraphs".format(len(sample_paragraphs)))
    print("-" * 70)
    for i, para in enumerate(sample_paragraphs, 1):
        print(f"{i}. {para[:60]}{'...' if len(para) > 60 else ''}")
    
    # Process the paragraphs
    result = process_legal_document(sample_paragraphs)
    
    print("\n" + "=" * 70)
    print("Output: {} restructured paragraphs".format(len(result)))
    print("=" * 70)
    
    # Display as Python list (as requested in prompt)
    print("\nPython List Format:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("Human-Readable Format:")
    print("=" * 70)
    for i, para in enumerate(result, 1):
        print(f"\n[{i}]")
        print(para)
    
    print("\n" + "=" * 70)
    print("✓ Processing complete")
    print("=" * 70)

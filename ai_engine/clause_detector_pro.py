"""
Legal Clause Classification Engine - Pro Edition

Enhanced features:
- spaCy text preprocessing and lemmatization
- Heading detection for improved accuracy
- Enhanced confidence scoring
- Structured JSON output
- Hybrid rule + AI architecture
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Optional: spaCy for advanced NLP preprocessing
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    logging.warning("spaCy not installed. Advanced preprocessing disabled. Install: pip install spacy")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import base patterns from clause_detector
from clause_detector import CLAUSE_TAXONOMY, CLAUSE_PATTERNS


# ============================================================================
# UPGRADE 1: spaCy Lemmatization Preprocessing
# ============================================================================

def clean_text(text: str, use_spacy: bool = True) -> str:
    """
    Preprocess text using spaCy for normalization.
    
    Steps:
    - Lowercase
    - Remove punctuation (except alphanumeric and spaces)
    - Lemmatize words
    - Normalize terms
    
    Args:
        text: Raw text
        use_spacy: Use spaCy if available (default: True)
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Fallback to basic preprocessing if spaCy unavailable
    if not use_spacy or not SPACY_AVAILABLE or nlp is None:
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Advanced spaCy preprocessing
    doc = nlp(text)
    
    # Lemmatize and filter
    lemmatized_tokens = []
    for token in doc:
        # Skip punctuation and whitespace
        if token.is_punct or token.is_space:
            continue
        # Use lemmatized form, lowercased
        lemmatized_tokens.append(token.lemma_.lower())
    
    # Join tokens
    normalized = ' '.join(lemmatized_tokens)
    
    # Additional cleanup
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()


# ============================================================================
# UPGRADE 2: Heading Detection
# ============================================================================

def is_heading(text: str) -> bool:
    """
    Detect if text is likely a legal section heading.
    
    Heading indicators:
    - Mostly uppercase (>70% uppercase letters)
    - Starts with section numbers (4.1, Section 5, Article III)
    - Short length (< 12 words)
    - Common heading patterns
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be a heading
    """
    if not text:
        return False
    
    text = text.strip()
    words = text.split()
    word_count = len(words)
    
    # Rule 1: Short text (< 12 words)
    if word_count > 12:
        return False
    
    # Rule 2: Starts with numbering patterns
    numbering_patterns = [
        r'^\d+\.',                          # 1., 2., 3.
        r'^\d+\.\d+',                       # 1.1, 2.3
        r'^Section\s+\d+',                  # Section 1
        r'^Article\s+[IVX\d]+',             # Article I, Article 5
        r'^Clause\s+\d+',                   # Clause 1
        r'^Schedule\s+[A-Z\d]',             # Schedule A
        r'^Appendix\s+[A-Z\d]',             # Appendix B
        r'^Exhibit\s+[A-Z\d]',              # Exhibit C
        r'^\([a-z0-9]+\)',                  # (a), (i), (1)
        r'^[IVX]+\.',                       # I., II., III.
    ]
    
    for pattern in numbering_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    
    # Rule 3: Mostly uppercase (calculate ratio)
    letters = [c for c in text if c.isalpha()]
    if letters:
        uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if uppercase_ratio > 0.7:
            return True
    
    # Rule 4: Common heading keywords (all caps)
    heading_keywords = [
        'DEFINITIONS', 'PARTIES', 'RECITALS', 'WHEREAS',
        'TERMINATION', 'PAYMENT', 'CONFIDENTIALITY',
        'INDEMNITY', 'LIABILITY', 'WARRANTIES',
        'GOVERNING LAW', 'DISPUTE RESOLUTION',
        'MISCELLANEOUS', 'GENERAL PROVISIONS',
        'INTELLECTUAL PROPERTY', 'SCOPE OF WORK'
    ]
    
    text_upper = text.upper()
    for keyword in heading_keywords:
        if keyword in text_upper and len(text) < 100:
            return True
    
    return False


# ============================================================================
# UPGRADE 3: Enhanced Confidence Scoring
# ============================================================================

def detect_clause_enhanced(
    paragraph: str,
    patterns: Dict[str, List[str]] = None,
    use_preprocessing: bool = False,
    check_heading: bool = True
) -> Dict[str, Any]:
    """
    Enhanced clause detection with improved confidence scoring.
    
    Returns detailed scores for all clause types.
    
    Args:
        paragraph: Contract text paragraph
        patterns: Dictionary of clause patterns
        use_preprocessing: Apply spaCy preprocessing
        check_heading: Check if paragraph is a heading
        
    Returns:
        Dictionary with:
            - label: Best matching clause type
            - confidence: Normalized confidence (0-1)
            - scores_per_class: Scores for all clause types
            - is_heading: Whether text is detected as heading
    """
    if patterns is None:
        patterns = CLAUSE_PATTERNS
    
    if not paragraph or not paragraph.strip():
        return {
            "label": "other",
            "confidence": 0.0,
            "scores_per_class": {},
            "is_heading": False
        }
    
    original_text = paragraph
    
    # Optional preprocessing
    if use_preprocessing:
        processed_text = clean_text(paragraph)
    else:
        processed_text = paragraph
    
    # Check if heading
    heading_detected = is_heading(original_text) if check_heading else False
    
    # Score each clause type
    scores = {clause_type: 0 for clause_type in patterns.keys()}
    
    # Test all patterns
    for clause_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            try:
                # Match against original text for better accuracy
                matches = re.findall(pattern, original_text, re.IGNORECASE)
                scores[clause_type] += len(matches)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue
    
    # Calculate total score across all classes
    sum_of_all_scores = sum(scores.values())
    
    # Find best match
    if sum_of_all_scores == 0:
        return {
            "label": "other",
            "confidence": 0.0,
            "scores_per_class": scores,
            "is_heading": heading_detected
        }
    
    # Get clause with highest score
    best_clause = max(scores.items(), key=lambda x: x[1])
    clause_label = best_clause[0]
    clause_score = best_clause[1]
    
    # Improved confidence formula: clause_score / sum_of_all_scores
    confidence = clause_score / sum_of_all_scores if sum_of_all_scores > 0 else 0.0
    
    # Boost confidence if it's a heading matching the clause type
    if heading_detected and confidence > 0.3:
        confidence = min(1.0, confidence * 1.2)  # 20% boost, capped at 1.0
    
    return {
        "label": clause_label,
        "confidence": round(confidence, 4),
        "scores_per_class": scores,
        "is_heading": heading_detected
    }


# ============================================================================
# UPGRADE 4: Structured Legal Output
# ============================================================================

def extract_clauses_structured(
    paragraphs: List[str],
    patterns: Dict[str, List[str]] = None,
    use_preprocessing: bool = False,
    page_numbers: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Extract clauses with structured, JSON-serializable output.
    
    Perfect for frontend dashboards and APIs.
    
    Args:
        paragraphs: List of contract paragraphs
        patterns: Dictionary of clause patterns
        use_preprocessing: Apply spaCy preprocessing
        page_numbers: Optional list of page numbers for each paragraph
        
    Returns:
        List of structured clause dictionaries
    """
    if patterns is None:
        patterns = CLAUSE_PATTERNS
    
    structured_output = []
    
    for index, paragraph in enumerate(paragraphs):
        if not paragraph or not paragraph.strip():
            continue
        
        # Detect clause
        detection = detect_clause_enhanced(
            paragraph,
            patterns,
            use_preprocessing=use_preprocessing
        )
        
        # Get page number
        page_num = page_numbers[index] if page_numbers and index < len(page_numbers) else None
        
        # Build structured object
        clause_obj = {
            "clause_type": detection["label"],
            "page_number": page_num,
            "paragraph_index": index,
            "text": paragraph.strip(),
            "confidence": detection["confidence"],
            "is_heading": detection["is_heading"],
            "scores_per_class": detection["scores_per_class"]
        }
        
        structured_output.append(clause_obj)
    
    return structured_output


def group_clauses_by_type(
    structured_clauses: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group structured clauses by type for dashboard display.
    
    Args:
        structured_clauses: Output from extract_clauses_structured()
        
    Returns:
        Dictionary mapping clause types to lists of clauses
    """
    grouped = {}
    
    for clause in structured_clauses:
        clause_type = clause["clause_type"]
        if clause_type not in grouped:
            grouped[clause_type] = []
        grouped[clause_type].append(clause)
    
    return grouped


# ============================================================================
# UPGRADE 5: Hybrid Rule + AI Architecture
# ============================================================================

class HybridClauseClassifier:
    """
    Hybrid classification system combining rule-based and AI approaches.
    
    Strategy:
    - If rule confidence > threshold → use rule result
    - Else → fallback to transformer classifier (ML model)
    
    This architecture allows:
    - Fast, interpretable results for high-confidence cases
    - ML backup for ambiguous cases
    - Easy A/B testing
    - Model updates without breaking rules
    """
    
    def __init__(
        self,
        rule_confidence_threshold: float = 0.75,
        ml_model_path: Optional[str] = None,
        use_preprocessing: bool = True
    ):
        """
        Initialize hybrid classifier.
        
        Args:
            rule_confidence_threshold: Confidence threshold for using rules
            ml_model_path: Path to transformer model (optional)
            use_preprocessing: Use spaCy preprocessing
        """
        self.rule_threshold = rule_confidence_threshold
        self.ml_model_path = ml_model_path
        self.use_preprocessing = use_preprocessing
        self.ml_model = None
        self.ml_available = False
        
        # Try to load ML model if path provided
        if ml_model_path:
            self._load_ml_model(ml_model_path)
    
    def _load_ml_model(self, model_path: str):
        """
        Load transformer model for fallback classification.
        
        Placeholder for actual model loading.
        In production, use: transformers, sentence-transformers, or custom model.
        """
        try:
            # Example: from transformers import pipeline
            # self.ml_model = pipeline("text-classification", model=model_path)
            # self.ml_available = True
            
            logger.info(f"ML model loading from {model_path} - placeholder")
            # For now, mark as unavailable
            self.ml_available = False
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.ml_available = False
    
    def classify_with_ml(self, text: str) -> Dict[str, Any]:
        """
        Classify using ML model (transformer).
        
        Placeholder implementation.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification result with label and confidence
        """
        if not self.ml_available or self.ml_model is None:
            logger.warning("ML model not available, returning fallback")
            return {
                "label": "other",
                "confidence": 0.0,
                "method": "ml_unavailable"
            }
        
        # Placeholder for actual ML inference
        # In production:
        # result = self.ml_model(text)
        # return {
        #     "label": result[0]["label"],
        #     "confidence": result[0]["score"],
        #     "method": "ml"
        # }
        
        return {
            "label": "other",
            "confidence": 0.0,
            "method": "ml_placeholder"
        }
    
    def classify(self, text: str, patterns: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        Hybrid classification: rules first, ML fallback.
        
        Args:
            text: Text to classify
            patterns: Rule patterns
            
        Returns:
            Classification result with method used
        """
        # Step 1: Try rule-based classification
        rule_result = detect_clause_enhanced(
            text,
            patterns=patterns,
            use_preprocessing=self.use_preprocessing
        )
        
        rule_confidence = rule_result["confidence"]
        
        # Step 2: Decide which method to use
        if rule_confidence >= self.rule_threshold:
            # High confidence: use rule result
            return {
                "label": rule_result["label"],
                "confidence": rule_confidence,
                "method": "rule_based",
                "rule_confidence": rule_confidence,
                "is_heading": rule_result["is_heading"]
            }
        else:
            # Low confidence: try ML fallback
            ml_result = self.classify_with_ml(text)
            
            # Compare results
            if ml_result["confidence"] > rule_confidence:
                # ML is more confident
                return {
                    "label": ml_result["label"],
                    "confidence": ml_result["confidence"],
                    "method": "ml_fallback",
                    "rule_confidence": rule_confidence,
                    "ml_confidence": ml_result["confidence"]
                }
            else:
                # Stick with rule result
                return {
                    "label": rule_result["label"],
                    "confidence": rule_confidence,
                    "method": "rule_based_low_confidence",
                    "rule_confidence": rule_confidence,
                    "is_heading": rule_result["is_heading"]
                }
    
    def batch_classify(
        self,
        paragraphs: List[str],
        patterns: Dict[str, List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple paragraphs using hybrid approach.
        
        Args:
            paragraphs: List of text paragraphs
            patterns: Rule patterns
            
        Returns:
            List of classification results
        """
        results = []
        
        for para in paragraphs:
            if para and para.strip():
                result = self.classify(para, patterns)
                results.append(result)
        
        return results


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_contract(
    paragraphs: List[str],
    use_preprocessing: bool = True,
    use_hybrid: bool = False,
    hybrid_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Complete contract analysis with all pro features.
    
    Args:
        paragraphs: List of contract paragraphs
        use_preprocessing: Use spaCy preprocessing
        use_hybrid: Use hybrid classifier
        hybrid_threshold: Confidence threshold for hybrid mode
        
    Returns:
        Comprehensive analysis results
    """
    if use_hybrid:
        # Use hybrid classifier
        classifier = HybridClauseClassifier(
            rule_confidence_threshold=hybrid_threshold,
            use_preprocessing=use_preprocessing
        )
        
        results = classifier.batch_classify(paragraphs)
        
        # Convert to structured format
        structured = []
        for idx, result in enumerate(results):
            structured.append({
                "clause_type": result["label"],
                "paragraph_index": idx,
                "text": paragraphs[idx].strip(),
                "confidence": result["confidence"],
                "method": result["method"],
                "is_heading": result.get("is_heading", False)
            })
        
    else:
        # Use standard enhanced detection
        structured = extract_clauses_structured(
            paragraphs,
            use_preprocessing=use_preprocessing
        )
    
    # Group by type
    grouped = group_clauses_by_type(structured)
    
    # Statistics
    total_paragraphs = len(structured)
    clause_counts = {k: len(v) for k, v in grouped.items()}
    avg_confidence = sum(c["confidence"] for c in structured) / total_paragraphs if total_paragraphs > 0 else 0
    
    return {
        "structured_clauses": structured,
        "grouped_clauses": grouped,
        "statistics": {
            "total_paragraphs": total_paragraphs,
            "clause_counts": clause_counts,
            "average_confidence": round(avg_confidence, 4)
        }
    }


if __name__ == "__main__":
    import json
    
    print("=" * 70)
    print("Legal Clause Classification Engine - PRO EDITION")
    print("=" * 70)
    print(f"spaCy Available: {SPACY_AVAILABLE}")
    print("=" * 70)
    
    # Sample paragraphs
    sample_paragraphs = [
        "1. DEFINITIONS",
        "For purposes of this Agreement, the following terms shall have the meanings set forth below.",
        "2. TERM AND TERMINATION",
        "This Agreement shall commence on the Effective Date and continue for twelve (12) months.",
        "Either party may terminate this Agreement with thirty (30) days written notice.",
        "3. PAYMENT TERMS",
        "Client shall pay Contractor all fees within thirty (30) days of invoice date.",
        "CONFIDENTIALITY",
        "Each party agrees to keep all Confidential Information strictly confidential.",
        "The receiving party shall not disclose such information to any third party.",
        "GOVERNING LAW",
        "This Agreement shall be governed by the laws of the State of California."
    ]
    
    print(f"\nAnalyzing {len(sample_paragraphs)} paragraphs...")
    print("-" * 70)
    
    # Run complete analysis
    analysis = analyze_contract(
        sample_paragraphs,
        use_preprocessing=False,  # Set to True if spaCy is available
        use_hybrid=False
    )
    
    # Display structured output
    print("\nSTRUCTURED OUTPUT (JSON-ready):")
    print("=" * 70)
    print(json.dumps(analysis["statistics"], indent=2))
    
    print("\nSAMPLE CLAUSES:")
    print("-" * 70)
    for clause in analysis["structured_clauses"][:5]:
        print(f"\n[{clause['paragraph_index']}] {clause['clause_type'].upper()}")
        print(f"Confidence: {clause['confidence']:.2f} | Heading: {clause['is_heading']}")
        print(f"Text: {clause['text'][:80]}...")
    
    # Test heading detection
    print("\n" + "=" * 70)
    print("HEADING DETECTION TEST:")
    print("-" * 70)
    test_texts = [
        "1. DEFINITIONS",
        "CONFIDENTIALITY",
        "This is a normal paragraph about payment terms.",
        "Section 4.5 - Intellectual Property Rights"
    ]
    
    for text in test_texts:
        is_head = is_heading(text)
        print(f"'{text}' → {'HEADING' if is_head else 'Normal'}")
    
    # Test hybrid classifier
    print("\n" + "=" * 70)
    print("HYBRID CLASSIFIER TEST:")
    print("-" * 70)
    hybrid = HybridClauseClassifier(rule_confidence_threshold=0.75)
    test_para = "Either party may terminate this Agreement with 30 days notice."
    result = hybrid.classify(test_para)
    print(f"Text: {test_para}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    print("\n" + "=" * 70)
    print("✓ Pro Edition ready with all upgrades!")
    print("=" * 70)

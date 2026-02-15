"""
Legal Clause Classification Engine

Rule-based clause detection using regex patterns and scoring.
"""

import re
import logging
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Step A: Fixed taxonomy of contract clause categories
CLAUSE_TAXONOMY = [
    "definitions",
    "parties",
    "term_duration",
    "termination",
    "payment_fees",
    "intellectual_property",
    "confidentiality",
    "liability_indemnity",
    "warranties",
    "dispute_resolution",
    "governing_law",
    "miscellaneous"
]


# Step B: Regex patterns for each clause type
CLAUSE_PATTERNS = {
    "definitions": [
        r'\bdefinition[s]?\b',
        r'\bmeaning\b',
        r'\bmeans\b',
        r'\bshall mean\b',
        r'\binterpreted as\b',
        r'\brefers? to\b',
        r'\bhereinafter\b',
        r'\bas used (in|herein)\b',
        r'\bfor purposes of\b',
        r'\binclude[s]? but (?:is|are) not limited to\b'
    ],
    
    "parties": [
        r'\bparties\b',
        r'\bparty\b',
        r'\bclient\b',
        r'\bvendor\b',
        r'\bcontractor\b',
        r'\bsupplier\b',
        r'\bcustomer\b',
        r'\bpurchaser\b',
        r'\bseller\b',
        r'\blessor\b',
        r'\blessee\b',
        r'\blicensor\b',
        r'\blicensee\b',
        r'\bbetween.*and\b',
        r'\bhereinafter referred to as\b'
    ],
    
    "term_duration": [
        r'\bterm\b',
        r'\bduration\b',
        r'\beffective date\b',
        r'\bcommencement\b',
        r'\bexpiration\b',
        r'\brenewal\b',
        r'\bextension\b',
        r'\bperiod of\b',
        r'\bshall remain in effect\b',
        r'\bcontinue for\b',
        r'\bvalid for\b',
        r'\byear[s]?\b',
        r'\bmonth[s]?\b'
    ],
    
    "termination": [
        r'\btermination\b',
        r'\bterminate\b',
        r'\bcancellation\b',
        r'\bcancel\b',
        r'\bend this agreement\b',
        r'\bcessation\b',
        r'\bdiscontinue\b',
        r'\bwithout cause\b',
        r'\bfor cause\b',
        r'\bnotice of termination\b',
        r'\bupon termination\b',
        r'\beffect of termination\b',
        r'\bmay terminate\b'
    ],
    
    "payment_fees": [
        r'\bpayment\b',
        r'\bfee[s]?\b',
        r'\bcompensation\b',
        r'\bremuneration\b',
        r'\bcost[s]?\b',
        r'\bprice\b',
        r'\bamount\b',
        r'\binvoice\b',
        r'\bbilling\b',
        r'\bshall pay\b',
        r'\bpayable\b',
        r'\bdue and payable\b',
        r'\bwithin.*days\b',
        r'\blate payment\b',
        r'\binterest\b'
    ],
    
    "intellectual_property": [
        r'\bintellectual property\b',
        r'\bIP rights?\b',
        r'\bcopyright\b',
        r'\btrademark\b',
        r'\bpatent\b',
        r'\bproprietary\b',
        r'\bownership\b',
        r'\blicense\b',
        r'\btrade secret\b',
        r'\bknow-how\b',
        r'\bwork product\b',
        r'\bderivative work\b'
    ],
    
    "confidentiality": [
        r'\bconfidential\b',
        r'\bconfidentiality\b',
        r'\bnon-disclosure\b',
        r'\bproprietary information\b',
        r'\btrade secret\b',
        r'\bsecret\b',
        r'\bdisclose\b',
        r'\bdisclosure\b',
        r'\bnon-public\b',
        r'\bshall not disclose\b',
        r'\bkeep confidential\b',
        r'\bprotect.*information\b'
    ],
    
    "liability_indemnity": [
        r'\bliability\b',
        r'\bliable\b',
        r'\bindemnif(?:y|ication)\b',
        r'\bindemnity\b',
        r'\bhold harmless\b',
        r'\bdamages\b',
        r'\bloss(?:es)?\b',
        r'\bclaim[s]?\b',
        r'\blimitation of liability\b',
        r'\bexclusion of liability\b',
        r'\bshall not be liable\b',
        r'\bresponsible for\b'
    ],
    
    "warranties": [
        r'\bwarrant(?:y|ies)\b',
        r'\brepresentation[s]?\b',
        r'\bguarantee\b',
        r'\brepresents and warrants\b',
        r'\bdisclaimer\b',
        r'\bas is\b',
        r'\bwarranty of merchantability\b',
        r'\bfitness for (?:a )?particular purpose\b',
        r'\bno warranty\b',
        r'\bexpress or implied\b'
    ],
    
    "dispute_resolution": [
        r'\bdispute[s]?\b',
        r'\barbitration\b',
        r'\bmediation\b',
        r'\blitigation\b',
        r'\bresolution\b',
        r'\bcontroversy\b',
        r'\bdisagreement\b',
        r'\bconflict\b',
        r'\barbitrat(?:e|or)\b',
        r'\bbinding arbitration\b',
        r'\bvenue\b',
        r'\bjurisdiction\b'
    ],
    
    "governing_law": [
        r'\bgoverning law\b',
        r'\bapplicable law\b',
        r'\bchoice of law\b',
        r'\bshall be governed\b',
        r'\bconstrued in accordance\b',
        r'\bjurisdiction\b',
        r'\blaws? of\b',
        r'\bstate of\b',
        r'\bfederal law\b'
    ],
    
    "miscellaneous": [
        r'\bmiscellaneous\b',
        r'\bgeneral provisions?\b',
        r'\bentire agreement\b',
        r'\bseverability\b',
        r'\bwaiver\b',
        r'\bamendment\b',
        r'\bmodification\b',
        r'\bassignment\b',
        r'\bforce majeure\b',
        r'\bnotices?\b',
        r'\bcounterparts?\b',
        r'\bheadings?\b'
    ]
}


# Step C & D: Clause detection with scoring
def detect_clause(paragraph: str, patterns: Dict[str, List[str]] = None) -> Dict[str, Any]:
    """
    Detect clause type using regex pattern matching and scoring.
    
    Args:
        paragraph: Contract text paragraph
        patterns: Dictionary of clause patterns (uses default if None)
        
    Returns:
        Dictionary with label and confidence score
    """
    if patterns is None:
        patterns = CLAUSE_PATTERNS
    
    if not paragraph or not paragraph.strip():
        return {"label": "other", "confidence": 0.0}
    
    # Score dictionary for each clause type
    scores = {clause_type: 0 for clause_type in patterns.keys()}
    total_matches = 0
    
    # Test all patterns for all clause types
    for clause_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            try:
                matches = re.findall(pattern, paragraph, re.IGNORECASE)
                match_count = len(matches)
                scores[clause_type] += match_count
                total_matches += match_count
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue
    
    # Find clause type with highest score
    if total_matches == 0:
        return {"label": "other", "confidence": 0.0}
    
    best_clause = max(scores.items(), key=lambda x: x[1])
    clause_label = best_clause[0]
    clause_score = best_clause[1]
    
    # Calculate confidence (0 to 1)
    confidence = clause_score / total_matches if total_matches > 0 else 0.0
    
    # If highest score is 0, return "other"
    if clause_score == 0:
        return {"label": "other", "confidence": 0.0}
    
    return {
        "label": clause_label,
        "confidence": round(confidence, 4)
    }


# Step E: Extraction pipeline
def extract_clauses(paragraphs: List[str], patterns: Dict[str, List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract and classify clauses from list of paragraphs.
    
    Args:
        paragraphs: List of contract paragraphs
        patterns: Dictionary of clause patterns (uses default if None)
        
    Returns:
        Dictionary mapping clause types to list of paragraphs with metadata
    """
    if patterns is None:
        patterns = CLAUSE_PATTERNS
    
    # Initialize result with all clause types
    result = {clause_type: [] for clause_type in patterns.keys()}
    result["other"] = []
    
    # Process each paragraph
    for index, paragraph in enumerate(paragraphs):
        if not paragraph or not paragraph.strip():
            continue
        
        # Detect clause type
        detection = detect_clause(paragraph, patterns)
        clause_label = detection["label"]
        confidence = detection["confidence"]
        
        # Store paragraph with metadata
        clause_data = {
            "text": paragraph.strip(),
            "index": index,
            "confidence": confidence
        }
        
        result[clause_label].append(clause_data)
    
    # Remove empty clause types for cleaner output
    result = {k: v for k, v in result.items() if v}
    
    return result


def get_clause_summary(classified_clauses: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    """
    Generate summary statistics of classified clauses.
    
    Args:
        classified_clauses: Output from extract_clauses()
        
    Returns:
        Dictionary with count per clause type
    """
    return {clause_type: len(items) for clause_type, items in classified_clauses.items()}


if __name__ == "__main__":
    import json
    
    print("=" * 70)
    print("Legal Clause Classification Engine")
    print("=" * 70)
    
    # Sample contract paragraphs
    sample_paragraphs = [
        "1. Definitions. For purposes of this Agreement, the following terms shall have the meanings set forth below.",
        "2. Term. This Agreement shall commence on the Effective Date and continue for a period of twelve (12) months.",
        "3. Payment Terms. Client shall pay Contractor all fees within thirty (30) days of invoice date.",
        "4. Confidentiality. Each party agrees to keep all Confidential Information strictly confidential and not disclose to any third party.",
        "5. Termination. Either party may terminate this Agreement with thirty (30) days written notice.",
        "6. Intellectual Property. All intellectual property rights in the work product shall belong to Client.",
        "7. Limitation of Liability. In no event shall either party be liable for any indirect, incidental, or consequential damages.",
        "8. Governing Law. This Agreement shall be governed by the laws of the State of California.",
        "9. Dispute Resolution. Any disputes arising under this Agreement shall be resolved through binding arbitration.",
        "10. Entire Agreement. This Agreement constitutes the entire agreement between the parties."
    ]
    
    print(f"\nProcessing {len(sample_paragraphs)} sample paragraphs...")
    print("-" * 70)
    
    # Extract and classify clauses
    classified = extract_clauses(sample_paragraphs)
    
    # Display results
    print("\nCLASSIFICATION RESULTS:")
    print("=" * 70)
    
    for clause_type, items in sorted(classified.items()):
        print(f"\n{clause_type.upper()} ({len(items)} paragraph{'s' if len(items) != 1 else ''}):")
        print("-" * 70)
        for item in items:
            print(f"  [{item['index']}] (confidence: {item['confidence']:.2f})")
            print(f"  {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
            print()
    
    # Summary statistics
    summary = get_clause_summary(classified)
    print("\nSUMMARY:")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    
    # Test individual detection
    print("\n" + "=" * 70)
    print("INDIVIDUAL CLAUSE DETECTION TEST:")
    print("=" * 70)
    test_text = "Either party may terminate this Agreement for any reason with 30 days notice."
    result = detect_clause(test_text)
    print(f"Text: {test_text}")
    print(f"Detected: {result['label']} (confidence: {result['confidence']:.2f})")
    
    print("\n" + "=" * 70)
    print("✓ Classification engine ready")
    print("=" * 70)

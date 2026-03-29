"""
Legal AI PDF Text Extraction Module

Production-grade module for extracting text from legal contract PDFs.
Supports both digital PDFs and scanned documents with OCR.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Union
from io import BytesIO

import pdfplumber

# PyMuPDF for robust PDF text extraction and page rendering (no Poppler needed)
try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

# Optional: pytesseract for image-based OCR (needs Tesseract binary)
try:
    import pytesseract
    from PIL import Image
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors."""
    pass


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extract text from a digital PDF using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted raw text as string
        
    Raises:
        PDFExtractionError: If extraction fails
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")
        
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                    logger.debug(f"Extracted text from page {page_num}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
        
        if not text_content:
            logger.warning("No text extracted using pdfplumber")
            return ""
        
        return "\n".join(text_content)
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise PDFExtractionError(f"Failed to extract text from PDF: {e}")


def extract_text_with_ocr(pdf_path: Union[str, Path]) -> str:
    """
    Extract text from a scanned PDF using PyMuPDF (fitz).

    Strategy:
    1. Use PyMuPDF's built-in text extraction (works on most PDFs including
       many scanned ones that have an invisible text layer from the scanner).
    2. If a page yields very little text AND Tesseract is available, render
       the page to a high-resolution image and run pytesseract on it.
    3. If neither PyMuPDF nor Tesseract can extract text, return whatever
       was collected (may be empty).

    This approach removes the dependency on Poppler (pdf2image) entirely.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted raw text as string

    Raises:
        PDFExtractionError: If extraction fails
    """
    if not _FITZ_AVAILABLE:
        raise PDFExtractionError(
            "PyMuPDF (fitz) is required for OCR extraction. "
            "Install with: pip install PyMuPDF"
        )

    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")

        logger.info(f"Opening PDF with PyMuPDF for OCR extraction: {pdf_path}")
        doc = fitz.open(str(pdf_path))
        text_content = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # --- Attempt 1: PyMuPDF built-in text extraction ---------------
            page_text = page.get_text("text")

            if page_text and len(page_text.strip()) > 30:
                # Good enough — use it directly
                text_content.append(page_text)
                logger.debug(f"Page {page_num + 1}: extracted {len(page_text)} chars via PyMuPDF")
                continue

            # --- Attempt 2: render to image + Tesseract OCR ----------------
            if _TESSERACT_AVAILABLE:
                try:
                    logger.debug(f"Page {page_num + 1}: low text, falling back to Tesseract OCR")
                    # Render at 300 DPI for good OCR quality
                    mat = fitz.Matrix(300 / 72, 300 / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img, lang="eng")
                    if ocr_text and ocr_text.strip():
                        text_content.append(ocr_text)
                        logger.debug(f"Page {page_num + 1}: extracted {len(ocr_text)} chars via Tesseract")
                        continue
                except Exception as e:
                    logger.warning(f"Tesseract OCR failed on page {page_num + 1}: {e}")

            # --- Fallback: use whatever PyMuPDF got (even if short) --------
            if page_text and page_text.strip():
                text_content.append(page_text)
                logger.debug(f"Page {page_num + 1}: using {len(page_text)} chars (partial)")
            else:
                logger.warning(f"Page {page_num + 1}: no text could be extracted")

        doc.close()

        if not text_content:
            logger.warning("No text extracted from any page")
            return ""

        return "\n".join(text_content)

    except PDFExtractionError:
        raise
    except Exception as e:
        logger.error(f"Error during OCR extraction: {e}")
        raise PDFExtractionError(f"Failed to extract text with OCR: {e}")


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace, newlines, and special characters.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    # Remove trailing/leading whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove special characters that may interfere with processing
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Normalize unicode characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Final cleanup
    text = text.strip()
    
    return text


def split_paragraphs(text: str) -> List[str]:
    """
    Split text into clean paragraphs.
    
    Args:
        text: Cleaned text
        
    Returns:
        List of paragraph strings
    """
    if not text:
        return []
    
    # Split by double newlines (paragraph breaks)
    paragraphs = text.split('\n\n')
    
    # Clean each paragraph and filter out empty ones
    cleaned_paragraphs = []
    for para in paragraphs:
        # Replace single newlines with spaces within paragraphs
        para = para.replace('\n', ' ')
        # Remove extra spaces
        para = re.sub(r' +', ' ', para)
        para = para.strip()
        
        # Only include non-empty paragraphs with substantial content
        if para and len(para) > 10:
            cleaned_paragraphs.append(para)
    
    return cleaned_paragraphs


def _is_scanned_pdf(pdf_path: Union[str, Path], threshold: float = 0.1) -> bool:
    """
    Detect if a PDF is scanned by checking if text extraction yields minimal text.
    
    Args:
        pdf_path: Path to the PDF file
        threshold: Minimum ratio of text content to consider as digital
        
    Returns:
        True if PDF appears to be scanned, False otherwise
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_chars = 0
            sample_pages = min(3, len(pdf.pages))  # Check first 3 pages
            
            for page in pdf.pages[:sample_pages]:
                text = page.extract_text()
                if text:
                    total_chars += len(text.strip())
            
            # If very little text is extracted, it's likely scanned
            avg_chars_per_page = total_chars / sample_pages if sample_pages > 0 else 0
            
            logger.debug(f"Average characters per page: {avg_chars_per_page}")
            
            # Consider scanned if less than 100 characters per page on average
            return avg_chars_per_page < 100
    
    except Exception as e:
        logger.warning(f"Error detecting PDF type: {e}")
        return False


def _get_page_count(pdf_path: Union[str, Path]) -> int:
    """
    Get the number of pages in a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Number of pages
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        logger.error(f"Error getting page count: {e}")
        return 0


def process_document(pdf_path: Union[str, Path]) -> Dict[str, Union[str, List[str], int]]:
    """
    Main function to process a PDF document and extract structured data.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing:
            - raw_text: Original extracted text
            - clean_text: Cleaned text
            - paragraphs: List of paragraph strings
            - num_pages: Number of pages in PDF
            - source_type: "digital" or "scanned"
            
    Raises:
        PDFExtractionError: If document processing fails
    """
    try:
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF document: {pdf_path}")
        
        # Validate file
        if not pdf_path.exists():
            raise PDFExtractionError(f"File not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise PDFExtractionError(f"Invalid file type: {pdf_path.suffix}. Expected .pdf")
        
        # Get page count
        num_pages = _get_page_count(pdf_path)
        logger.info(f"Document has {num_pages} pages")
        
        # Detect if scanned
        is_scanned = _is_scanned_pdf(pdf_path)
        source_type = "scanned" if is_scanned else "digital"
        logger.info(f"Detected source type: {source_type}")
        
        # Extract text
        if is_scanned:
            logger.info("Using OCR for text extraction")
            raw_text = extract_text_with_ocr(pdf_path)
        else:
            logger.info("Using direct text extraction")
            raw_text = extract_text_from_pdf(pdf_path)
            
            # Fallback to OCR if no text extracted
            if not raw_text or len(raw_text.strip()) < 50:
                logger.warning("Minimal text extracted, falling back to OCR")
                raw_text = extract_text_with_ocr(pdf_path)
                source_type = "scanned"
        
        # Clean text
        logger.info("Cleaning extracted text")
        cleaned_text = clean_text(raw_text)
        
        # Split into paragraphs
        logger.info("Splitting text into paragraphs")
        paragraphs = split_paragraphs(cleaned_text)
        
        result = {
            "raw_text": raw_text,
            "clean_text": cleaned_text,
            "paragraphs": paragraphs,
            "num_pages": num_pages,
            "source_type": source_type
        }
        
        logger.info(
            f"Successfully processed document: {len(paragraphs)} paragraphs, "
            f"{len(cleaned_text)} characters, {num_pages} pages"
        )
        
        return result
    
    except PDFExtractionError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing document: {e}")
        raise PDFExtractionError(f"Failed to process document: {e}")


if __name__ == "__main__":
    import sys
    
    # Example usage
    print("Legal AI PDF Text Extractor")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Use command-line argument if provided
        test_pdf_path = sys.argv[1]
    else:
        # Example path (update with actual PDF path for testing)
        test_pdf_path = "sample_contract.pdf"
        print(f"No PDF path provided. Using default: {test_pdf_path}")
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        print()
    
    try:
        # Process the document
        result = process_document(test_pdf_path)
        
        # Display results
        print(f"\nDocument Analysis Results:")
        print(f"  Source Type: {result['source_type']}")
        print(f"  Number of Pages: {result['num_pages']}")
        print(f"  Total Characters: {len(result['clean_text'])}")
        print(f"  Number of Paragraphs: {len(result['paragraphs'])}")
        print("\nFirst 500 characters of clean text:")
        print("-" * 50)
        print(result['clean_text'][:500])
        print("-" * 50)
        
        if result['paragraphs']:
            print(f"\nFirst paragraph:")
            print("-" * 50)
            print(result['paragraphs'][0])
            print("-" * 50)
        
        print("\n✓ Document processed successfully!")
        
    except PDFExtractionError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)

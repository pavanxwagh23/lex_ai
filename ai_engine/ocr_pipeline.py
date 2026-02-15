"""
Production-Grade OCR Pipeline for Scanned Legal Contracts

Advanced OCR pipeline with image preprocessing to handle:
- Low resolution scans
- Skewed pages
- Noise and artifacts
- Stamps and signatures
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRPipelineError(Exception):
    """Custom exception for OCR pipeline errors."""
    pass


def pdf_to_images(
    pdf_path: Union[str, Path],
    dpi: int = 300,
    fmt: str = 'RGB'
) -> List[np.ndarray]:
    """
    Convert PDF pages to images for OCR processing.
    
    Higher DPI (300-400) provides better OCR accuracy but slower processing.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image conversion (default: 300)
        fmt: Color format - 'RGB' or 'L' (grayscale)
        
    Returns:
        List of images as numpy arrays
        
    Raises:
        OCRPipelineError: If PDF conversion fails
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise OCRPipelineError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Converting PDF to images at {dpi} DPI: {pdf_path}")
        
        # Convert PDF pages to PIL images
        pil_images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt=fmt,
            thread_count=4  # Parallel processing for speed
        )
        
        # Convert PIL images to numpy arrays for OpenCV processing
        np_images = []
        for idx, pil_img in enumerate(pil_images):
            np_img = np.array(pil_img)
            # Convert RGB to BGR for OpenCV if needed
            if len(np_img.shape) == 3 and np_img.shape[2] == 3:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            np_images.append(np_img)
            logger.debug(f"Converted page {idx + 1} to numpy array")
        
        logger.info(f"Successfully converted {len(np_images)} pages")
        return np_images
    
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise OCRPipelineError(f"PDF to image conversion failed: {e}")


def preprocess_image(
    image: np.ndarray,
    deskew: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
    remove_shadows: bool = True,
    binarize: bool = True
) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy.
    
    Preprocessing steps:
    1. Convert to grayscale
    2. Remove shadows and improve lighting
    3. Deskew (correct rotation)
    4. Denoise (remove artifacts)
    5. Enhance contrast
    6. Binarize (convert to black and white)
    
    Args:
        image: Input image as numpy array
        deskew: Apply deskewing to correct rotation
        denoise: Apply noise removal
        enhance_contrast: Enhance image contrast
        remove_shadows: Remove shadows and lighting issues
        binarize: Convert to binary (black/white) image
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        logger.debug("Starting image preprocessing")
        
        # Step 1: Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug("Converted to grayscale")
        else:
            gray = image.copy()
        
        # Step 2: Remove shadows and improve lighting
        if remove_shadows:
            # Dilate the image to remove text temporarily
            dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
            # Blur to get background
            background = cv2.medianBlur(dilated, 21)
            # Subtract background to normalize lighting
            gray = cv2.absdiff(gray, background)
            # Invert back
            gray = 255 - gray
            logger.debug("Removed shadows and normalized lighting")
        
        # Step 3: Deskew image (correct rotation)
        if deskew:
            gray = _deskew_image(gray)
            logger.debug("Applied deskewing")
        
        # Step 4: Denoise
        if denoise:
            # Use Non-Local Means Denoising for better quality
            gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
            logger.debug("Applied denoising")
        
        # Step 5: Enhance contrast
        if enhance_contrast:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            logger.debug("Enhanced contrast with CLAHE")
        
        # Step 6: Binarization (convert to black and white)
        if binarize:
            # Use adaptive thresholding for better results with varying lighting
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            logger.debug("Applied adaptive binarization")
            return binary
        
        return gray
    
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        # Return original image if preprocessing fails
        return image


def _deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew in scanned documents.
    
    Uses Hough Line Transform to detect dominant angle and rotate image.
    
    Args:
        image: Grayscale image
        
    Returns:
        Deskewed image
    """
    try:
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            logger.debug("No lines detected for deskewing")
            return image
        
        # Calculate angles of detected lines
        angles = []
        for line in lines[:50]:  # Use first 50 lines
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        
        # Find median angle (more robust than mean)
        median_angle = np.median(angles)
        
        # Only deskew if angle is significant (> 0.5 degrees)
        if abs(median_angle) < 0.5:
            logger.debug(f"Skew angle too small ({median_angle:.2f}°), skipping deskew")
            return image
        
        logger.debug(f"Detected skew angle: {median_angle:.2f} degrees")
        
        # Rotate image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    except Exception as e:
        logger.warning(f"Deskewing failed, using original image: {e}")
        return image


def run_ocr(
    image: np.ndarray,
    lang: str = 'eng',
    config: Optional[str] = None
) -> str:
    """
    Run OCR on preprocessed image using pytesseract.
    
    Args:
        image: Preprocessed image as numpy array
        lang: Language for OCR (default: 'eng' for English)
        config: Custom tesseract configuration string
        
    Returns:
        Extracted text as string
        
    Raises:
        OCRPipelineError: If OCR fails
    """
    try:
        # Default configuration optimized for legal documents
        if config is None:
            # PSM 3: Fully automatic page segmentation (default)
            # OEM 3: Use both legacy and LSTM OCR engines
            config = '--psm 3 --oem 3'
        
        logger.debug(f"Running OCR with config: {config}")
        
        # Convert numpy array to PIL Image for pytesseract
        pil_image = Image.fromarray(image)
        
        # Run OCR
        text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
        
        logger.debug(f"OCR extracted {len(text)} characters")
        
        return text
    
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise OCRPipelineError(f"OCR processing failed: {e}")


def clean_text(text: str) -> str:
    """
    Clean OCR output by removing noise, symbols, and extra spaces.
    
    Cleaning steps:
    1. Remove null bytes and control characters
    2. Fix common OCR errors (e.g., '|' instead of 'I')
    3. Remove excessive whitespace
    4. Remove standalone special characters
    5. Normalize line breaks
    6. Remove page numbers and headers/footers artifacts
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    logger.debug("Cleaning OCR text")
    
    # Step 1: Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Step 2: Fix common OCR errors
    # Replace common misrecognized characters
    ocr_fixes = {
        '|': 'I',  # Vertical bar to capital I (in context)
        '0': 'O',  # Zero to O (in context of words)
        '§': 'S',
        '¢': 'c',
        '©': '(c)',
    }
    # Apply character fixes cautiously (only in word contexts)
    for wrong, correct in ocr_fixes.items():
        # Only fix if surrounded by letters
        text = re.sub(rf'([a-zA-Z]){re.escape(wrong)}([a-zA-Z])', rf'\1{correct}\2', text)
    
    # Step 3: Remove excessive whitespace
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple tabs with single space
    text = re.sub(r'\t+', ' ', text)
    
    # Step 4: Remove standalone special characters and noise
    # Remove lines with only special characters
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that are only symbols/noise (less than 3 alphanumeric chars)
        if len(re.findall(r'[a-zA-Z0-9]', line)) >= 3:
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    
    # Step 5: Normalize line breaks
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    # Step 6: Remove common OCR artifacts
    # Remove standalone page numbers (e.g., "Page 1", "- 5 -")
    text = re.sub(r'^[\s\-]*(?:Page\s*)?\d+[\s\-]*$', '', text, flags=re.MULTILINE)
    
    # Remove excessive hyphens (scanning artifacts)
    text = re.sub(r'-{3,}', '', text)
    
    # Remove excessive dots (scanning artifacts)
    text = re.sub(r'\.{4,}', '...', text)
    
    # Step 7: Clean up each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # Step 8: Normalize unicode
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Final cleanup
    text = text.strip()
    
    logger.debug(f"Text cleaned, final length: {len(text)} characters")
    
    return text


def split_into_paragraphs(text: str, min_length: int = 20) -> List[str]:
    """
    Split cleaned text into meaningful paragraphs.
    
    Args:
        text: Cleaned text
        min_length: Minimum character length for a paragraph
        
    Returns:
        List of paragraph strings
    """
    if not text:
        return []
    
    logger.debug("Splitting text into paragraphs")
    
    # Split by double newlines (paragraph breaks)
    paragraphs = text.split('\n\n')
    
    # Process each paragraph
    cleaned_paragraphs = []
    for para in paragraphs:
        # Replace single newlines with spaces within paragraphs
        para = para.replace('\n', ' ')
        
        # Remove extra spaces
        para = re.sub(r' +', ' ', para)
        para = para.strip()
        
        # Only include paragraphs with substantial content
        if len(para) >= min_length:
            cleaned_paragraphs.append(para)
    
    logger.debug(f"Created {len(cleaned_paragraphs)} paragraphs")
    
    return cleaned_paragraphs


def extract_text_from_scanned_pdf(
    pdf_path: Union[str, Path],
    dpi: int = 300,
    preprocess: bool = True,
    lang: str = 'eng'
) -> dict:
    """
    Complete OCR pipeline: extract text from scanned PDF with preprocessing.
    
    Pipeline stages:
    1. Convert PDF to images
    2. Preprocess each image (optional)
    3. Run OCR on each page
    4. Clean extracted text
    5. Split into paragraphs
    
    Args:
        pdf_path: Path to scanned PDF file
        dpi: Resolution for PDF to image conversion (300-400 recommended)
        preprocess: Apply image preprocessing for better accuracy
        lang: OCR language ('eng' for English)
        
    Returns:
        Dictionary containing:
            - raw_text: Original OCR output
            - clean_text: Cleaned text
            - paragraphs: List of paragraphs
            - num_pages: Number of pages processed
            - preprocessing_enabled: Whether preprocessing was used
            
    Raises:
        OCRPipelineError: If pipeline fails
    """
    try:
        pdf_path = Path(pdf_path)
        logger.info(f"Starting OCR pipeline for: {pdf_path}")
        logger.info(f"Settings: DPI={dpi}, Preprocessing={preprocess}, Language={lang}")
        
        # Validate input
        if not pdf_path.exists():
            raise OCRPipelineError(f"File not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise OCRPipelineError(f"Invalid file type: {pdf_path.suffix}. Expected .pdf")
        
        # Stage 1: Convert PDF to images
        images = pdf_to_images(pdf_path, dpi=dpi)
        num_pages = len(images)
        logger.info(f"Processing {num_pages} pages")
        
        # Stage 2 & 3: Preprocess and OCR each page
        page_texts = []
        for page_num, image in enumerate(images, start=1):
            logger.info(f"Processing page {page_num}/{num_pages}")
            
            # Preprocess image if enabled
            if preprocess:
                processed_image = preprocess_image(image)
            else:
                # Convert to grayscale at minimum
                if len(image.shape) == 3:
                    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    processed_image = image
            
            # Run OCR
            page_text = run_ocr(processed_image, lang=lang)
            page_texts.append(page_text)
            logger.debug(f"Page {page_num}: Extracted {len(page_text)} characters")
        
        # Combine all pages
        raw_text = '\n\n'.join(page_texts)
        logger.info(f"Total raw text: {len(raw_text)} characters")
        
        # Stage 4: Clean text
        clean_text_output = clean_text(raw_text)
        logger.info(f"Cleaned text: {len(clean_text_output)} characters")
        
        # Stage 5: Split into paragraphs
        paragraphs = split_into_paragraphs(clean_text_output)
        logger.info(f"Created {len(paragraphs)} paragraphs")
        
        # Build result
        result = {
            "raw_text": raw_text,
            "clean_text": clean_text_output,
            "paragraphs": paragraphs,
            "num_pages": num_pages,
            "preprocessing_enabled": preprocess
        }
        
        logger.info("OCR pipeline completed successfully")
        
        return result
    
    except OCRPipelineError:
        raise
    except Exception as e:
        logger.error(f"OCR pipeline failed: {e}")
        raise OCRPipelineError(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    import sys
    import json
    
    print("=" * 60)
    print("Legal Contract OCR Pipeline")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Use command-line argument
        test_pdf_path = sys.argv[1]
    else:
        # Example path
        test_pdf_path = "sample_scanned_contract.pdf"
        print(f"No PDF path provided. Using default: {test_pdf_path}")
        print(f"Usage: python ocr_pipeline.py <path_to_scanned_pdf>")
        print()
    
    try:
        # Run the complete OCR pipeline
        result = extract_text_from_scanned_pdf(
            test_pdf_path,
            dpi=300,
            preprocess=True,
            lang='eng'
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("OCR RESULTS")
        print("=" * 60)
        print(f"Number of Pages: {result['num_pages']}")
        print(f"Preprocessing: {'Enabled' if result['preprocessing_enabled'] else 'Disabled'}")
        print(f"Raw Text Length: {len(result['raw_text'])} characters")
        print(f"Clean Text Length: {len(result['clean_text'])} characters")
        print(f"Number of Paragraphs: {len(result['paragraphs'])}")
        
        print("\n" + "-" * 60)
        print("FIRST 500 CHARACTERS OF CLEAN TEXT:")
        print("-" * 60)
        print(result['clean_text'][:500])
        
        if result['paragraphs']:
            print("\n" + "-" * 60)
            print("FIRST PARAGRAPH:")
            print("-" * 60)
            print(result['paragraphs'][0])
        
        # Save to JSON file
        output_file = Path(test_pdf_path).stem + "_ocr_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print(f"✓ Results saved to: {output_file}")
        print("=" * 60)
        
    except OCRPipelineError as e:
        print(f"\n✗ OCR Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        sys.exit(1)

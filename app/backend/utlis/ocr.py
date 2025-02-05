import cv2
import logging
import numpy as np
import re
from typing import Dict, Optional

from app.backend.utlis.models import OCR_READER

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_image(*, image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image from bytes: convert to grayscale & apply thresholding.

    Args:
        image_bytes (bytes): The raw bytes of an image file.

    Returns:
        np.ndarray: Preprocessed image in NumPy array format.
    """
    # Convert bytes to a NumPy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Error decoding image bytes.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding (to improve contrast)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    return thresh


def pan_ocr(*, image_bytes: bytes) -> Dict[str, Optional[str]]:
    """
    Extract PAN card details from image bytes.

    Args:
        image_bytes (bytes): The raw bytes of an image file.

    Returns:
        Dict[str, Optional[str]]: Extracted details containing PAN number, DOB, Name, and Father's Name.
    """ 

    # Preprocess image
    processed_img = preprocess_image(image_bytes=image_bytes)

    # Extract text from the image
    results = OCR_READER.readtext(processed_img, detail=0)

    extracted_info: Dict[str, Optional[str]] = {
        "ID Number": None,
        "Date of Birth": None,
        "Name": None,
        "Father's Name": None,
    }

    name_found = False  
    father_name_found = False  

    for i, text in enumerate(results):
        text = text.strip()

        
        if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", text):
            extracted_info["ID Number"] = text

        
        elif re.match(r"\d{2}/\d{2}/\d{4}", text):
            extracted_info["Date of Birth"] = text

        
        elif "name" in text.lower() and i + 1 < len(results) and not name_found:
            extracted_info["Name"] = results[i + 1].strip()
            name_found = True

        
        elif "father" in text.lower() and i + 1 < len(results) and not father_name_found:
            extracted_info["Father's Name"] = results[i + 1].strip()
            father_name_found = True

    return extracted_info


def adhar_ocr(*, image_bytes: bytes) -> Dict[str, Optional[str]]:
    """
    Extract important details from an Aadhaar card image using OCR.

    Args:
        image_bytes (bytes): The raw bytes of an image file.

    Returns:
        Dict[str, Optional[str]]: Extracted details containing ADHAAR ID, DOB and Gender.
    """ 

    processed_img = preprocess_image(image_bytes=image_bytes)

    # Extract text using EasyOCR
    results = OCR_READER.readtext(processed_img, detail=0)

    extracted_info = {
        "Aadhaar Number": None,
        "Date of Birth": None,
        "Gender": None,
    }

     
    gender_found = False 
    
    for i, text in enumerate(results):
        text = text.strip()

        if re.match(r"^\d{4}\s?\d{4}\s?\d{4}$", text):
            extracted_info["Aadhaar Number"] = text.replace(" ", "")  # Remove spaces

        match=re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text)
        if match:
            extracted_info["Date of Birth"] = match.group(1)

        if not gender_found:
            if "male" in text.lower():
                extracted_info["Gender"] = "Male"
                gender_found = True
            elif "female" in text.lower():
                extracted_info["Gender"] = "Female"
                gender_found = True

    return extracted_info


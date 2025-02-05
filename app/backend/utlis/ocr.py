import cv2
from app.backend.utlis.logging_config import get_logger
import numpy as np
import re
from typing import Dict, Optional

from app.backend.utlis.models import OCR_READER

# Create a logger object
logger = get_logger(__name__)

def preprocess_image(*, image: np.ndarray) -> np.ndarray:
    """
    Preprocess image: convert to grayscale & apply thresholding.

    Args:
        image (np.ndarray): The image in NumPy array format (cv2 format).

    Returns:
        np.ndarray: Preprocessed image in NumPy array format.
    """

    if image is None:
        logger.error("Error: No image data provided for preprocessing.")
        raise ValueError("Error: No image data provided for preprocessing.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding (to improve contrast)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    logger.info("Image preprocessing completed (grayscale & thresholding).")
    return thresh


def pan_ocr(*, image: np.ndarray) -> Dict[str, Optional[str]]:
    """
    Extract PAN card details from an image.

    Args:
        image (np.ndarray): The image array (cv2 format) to process.

    Returns:
        Dict[str, Optional[str]]: Extracted details containing PAN number, DOB, Name, and Father's Name.
    """ 

    # Preprocess image (image is already an np.ndarray)
    processed_img = preprocess_image(image=image)

    # Extract text from the image using OCR
    results = OCR_READER.readtext(processed_img, detail=0)

    extracted_info: Dict[str, Optional[str]] = {
        "ID Number": None,
        "Date of Birth": None,
        "Name": None,
        "Father Name": None,
    }

    name_found = False  
    father_name_found = False  

    for i, text in enumerate(results):
        text = text.strip()

        if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", text):
            extracted_info["ID Number"] = text
            logger.info(f"PAN ID found: {text}")
        
        elif re.match(r"\d{2}/\d{2}/\d{4}", text):
            extracted_info["Date of Birth"] = text
            logger.info(f"DOB found: {text}")
        
        elif "name" in text.lower() and i + 1 < len(results) and not name_found:
            extracted_info["Name"] = results[i + 1].strip()
            name_found = True
            logger.info(f"Name found: {extracted_info['Name']}")
        
        elif "father" in text.lower() and i + 1 < len(results) and not father_name_found:
            extracted_info["Father Name"] = results[i + 1].strip()
            father_name_found = True
            logger.info(f"Father Name found: {extracted_info['Father Name']}")

    if not any(extracted_info.values()):
        logger.warning("No relevant PAN details found in the image.")

    return extracted_info


def adhar_ocr(*, image: np.ndarray) -> Dict[str, Optional[str]]:
    """
    Extract important details from an Aadhaar card image using OCR.

    Args:
        image (np.ndarray): The image array (cv2 format) to process.

    Returns:
        Dict[str, Optional[str]]: Extracted details containing ADHAAR ID, DOB, and Gender.
    """ 

    # Preprocess image (image is already an np.ndarray)
    processed_img = preprocess_image(image=image)

    # Extract text using OCR
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
            logger.info(f"Aadhaar Number found: {extracted_info['Aadhaar Number']}")

        match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text)
        if match:
            extracted_info["Date of Birth"] = match.group(1)
            logger.info(f"DOB found: {extracted_info['Date of Birth']}")

        if not gender_found:
            if "male" in text.lower():
                extracted_info["Gender"] = "Male"
                gender_found = True
                logger.info("Gender: Male")
            elif "female" in text.lower():
                extracted_info["Gender"] = "Female"
                gender_found = True
                logger.info("Gender: Female")

    if not any(extracted_info.values()):
        logger.warning("No relevant Aadhaar details found in the image.")

    return extracted_info

import cv2
import re
import numpy as np
from typing import Dict, Optional
from logging_config import get_logger
from app.backend.utlis.models import OCR_READER

logger = get_logger(__name__)

def preprocess_image(*, image: np.ndarray) -> np.ndarray:
    if image is None:
        logger.error("No image data provided for preprocessing.")
        raise ValueError("No image data provided for preprocessing.")

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        logger.info("Image preprocessing completed.")
        return thresh
    except Exception as e:
        logger.exception("Error in image preprocessing.")
        raise

def pan_ocr(image: np.ndarray) -> Dict[str, Optional[str]]:
    logger.info("Starting PAN OCR processing...")
    processed_img = preprocess_image(image=image)
    results = OCR_READER.readtext(processed_img, detail=0)

    extracted_info = {
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
        
        elif re.match(r"\d{2}/\d{2}/\d{4}", text):
            extracted_info["Date of Birth"] = text
        
        elif "name" in text.lower() and i + 1 < len(results) and not name_found:
            extracted_info["Name"] = results[i + 1].strip()
            name_found = True
        
        elif "father" in text.lower() and i + 1 < len(results) and not father_name_found:
            extracted_info["Father Name"] = results[i + 1].strip()
            father_name_found = True

    if not any(extracted_info.values()):
        logger.warning("No relevant PAN details found.")

    logger.info("PAN OCR processing completed.")  # No sensitive details logged
    return extracted_info

def adhar_ocr(image: np.ndarray) -> Dict[str, Optional[str]]:
    logger.info("Starting Aadhaar OCR processing...")
    processed_img = preprocess_image(image=image)
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
            extracted_info["Aadhaar Number"] = text.replace(" ", "")

        match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text)
        if match:
            extracted_info["Date of Birth"] = match.group(1)

        if not gender_found:
            if "male" in text.lower():
                extracted_info["Gender"] = "Male"
                gender_found = True
            elif "female" in text.lower():
                extracted_info["Gender"] = "Female"
                gender_found = True

    if not any(extracted_info.values()):
        logger.warning("No relevant Aadhaar details found.")

    logger.info("Aadhaar OCR processing completed.")  # No sensitive details logged
    return extracted_info

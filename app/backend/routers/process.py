from fastapi import APIRouter, UploadFile, File
import asyncio
import io
import cv2
import numpy as np
from PIL import Image, ExifTags
from app.backend.utlis.ocr import pan_ocr, adhar_ocr
from app.backend.utlis.face import extract_face, enhance_image, match_faces_with_facenet
from logging_config import get_logger
from typing import Optional, Dict

router = APIRouter()

logger = get_logger(__name__)

async def read_image(image_data: bytes) -> Optional[np.ndarray]:
    """
    Reads image data, corrects orientation if needed, and converts it into a NumPy array.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        Optional[np.ndarray]: The image as a NumPy array (cv2 format) or None if the data is invalid.
    """
    logger.info(f"Received image data, length: {len(image_data)} bytes")

    if not image_data:
        logger.error("Image data is empty or unreadable")
        return None

    try:
        # Convert bytes to a PIL image
        image = Image.open(io.BytesIO(image_data))

        # Handle EXIF orientation
        try:
            exif = image._getexif()
            if exif:
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == "Orientation":
                        if value == 3:
                            image = image.rotate(180, expand=True)
                        elif value == 6:
                            image = image.rotate(270, expand=True)
                        elif value == 8:
                            image = image.rotate(90, expand=True)
        except Exception as exif_error:
            logger.warning(f"EXIF data not found or could not be processed: {exif_error}")

        # Convert to OpenCV format (BGR)
        image = image.convert("RGB")  # Ensure no alpha channel
        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        logger.info(f"Image processed successfully with shape {cv2_img.shape}")
        return cv2_img

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None


@router.post("/process_documents/")
async def process_documents(pan_image: UploadFile = File(...),adhar_image: UploadFile = File(...),live_image: UploadFile = File(...)) -> Dict[str, dict]:
    """
    Unified API Endpoint (Asynchronous) to process PAN, Aadhar OCR, and Face Matching.

    Args:
        pan_image (UploadFile): The uploaded PAN image.
        adhar_image (UploadFile): The uploaded Aadhar image.
        live_image (UploadFile): The uploaded live image for face matching.

    Returns:
        Dict[str, dict]: The results of OCR and face matching.
    """
    logger.info("Processing documents started")
    
    # Step 1: Read Images in Parallel (Read image content only once)
    try:
        pan_image_data, adhar_image_data, live_image_data = await asyncio.gather(
            pan_image.read(), 
            adhar_image.read(), 
            live_image.read()
        )
    except Exception as e:
        logger.error(f"Error reading images: {e}")
        return {"error": "Failed to read images"}
    
    # Step 2: Pass the image data (in bytes) to the `read_image` function
    try:
        pan_img = await read_image(pan_image_data)
        adhar_img = await read_image(adhar_image_data)
        live_img = await read_image(live_image_data)
    except Exception as e:
        logger.error(f"Error reading image data: {e}")
        return {"error": "Error processing image data"}

    # Step 3: Extract Face from Live Image
    extracted_face = await asyncio.to_thread(extract_face, live_img)
    if extracted_face is None:
        logger.warning("No face detected in live image")
        return {"error": "No face detected in live image"}
    logger.info("Face successfully extracted from live image")


    # Step 5: Run OCR in Parallel
    try:
        pan_text, adhar_text = await asyncio.gather(
            asyncio.to_thread(pan_ocr, pan_img),
            asyncio.to_thread(adhar_ocr, adhar_img)
        )
        logger.info("OCR completed successfully")
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        return {"error": "Failed to perform OCR"}

    # Step 6: Match Extracted Face with Aadhar/PAN Faces in Parallel
    adhar_face, pan_face = await asyncio.gather(
        asyncio.to_thread(extract_face, adhar_img),
        asyncio.to_thread(extract_face, pan_img)
    )

    if adhar_face is not None:
        logger.info("Face extracted from Aadhar image")
    else:
        logger.warning("No face detected in Aadhar image")

    if pan_face is not None:
        logger.info("Face extracted from PAN image")
    else:
        logger.warning("No face detected in PAN image")

    match_adhar, score_adhar = await asyncio.to_thread(
        match_faces_with_facenet, extracted_face, adhar_face
    ) if adhar_face is not None else (False, None)
    match_pan, score_pan = await asyncio.to_thread(
        match_faces_with_facenet, extracted_face, pan_face
    ) if pan_face is not None else (False, None)

    logger.info(f"Face match results - Aadhar: {match_adhar}, Score: {score_adhar}")
    logger.info(f"Face match results - PAN: {match_pan}, Score: {score_pan}")

    # Step 7: Return All Results
    try:
        return {
            "pan_ocr": pan_text,
            "adhar_ocr": adhar_text,
            "face_match_with_adhar": {"match": bool(match_adhar), "score": score_adhar},
            "face_match_with_pan": {"match": bool(match_pan), "score": score_pan},
        }
        logger.info("result sent successfully")
    except Exception as e:
        logger.error(f"Final result processing error: {e}")
        return {"error": "Failed to return the result"}

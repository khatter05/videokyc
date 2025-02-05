from fastapi import APIRouter, UploadFile, File
import asyncio
import io
import cv2
import numpy as np
from PIL import Image
from app.backend.utils.ocr import pan_ocr, adhar_ocr
from app,backend.utils.face import extract_face, enhance_image, match_faces_with_facenet

router = APIRouter()

async def read_image(file: UploadFile):
    """
    Read the uploaded image file asynchronously and convert it to OpenCV format.
    """
    image_bytes = await file.read()  # Read asynchronously
    image = Image.open(io.BytesIO(image_bytes))  # Open with PIL
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

@router.post("/process_documents/")
async def process_documents(pan_image: UploadFile = File(...), adhar_image: UploadFile = File(...), live_image: UploadFile = File(...)):
    """
    Unified API Endpoint (Asynchronous) to process PAN, Aadhar OCR, and Face Matching.
    """
    # 🔹 Step 1: Read Images in Parallel
    pan_img, adhar_img, live_img = await asyncio.gather(
        read_image(pan_image),
        read_image(adhar_image),
        read_image(live_image)
    )

    # 🔹 Step 2: Extract Face from Live Image
    extracted_face = await asyncio.to_thread(extract_face, live_img)  
    if extracted_face is None:
        return {"error": "No face detected in live image"}

    # 🔹 Step 3: Enhance Extracted Face using ESRGAN
    enhanced_face = await asyncio.to_thread(enhance_image, extracted_face)

    # 🔹 Step 4: Run OCR in Parallel
    pan_ocr_task = asyncio.to_thread(pan_ocr, pan_img)
    adhar_ocr_task = asyncio.to_thread(adhar_ocr, adhar_img)
    pan_text, adhar_text = await asyncio.gather(pan_ocr_task, adhar_ocr_task)

    # 🔹 Step 5: Match Extracted Face with Aadhar/PAN Faces in Parallel
    adhar_face_task = asyncio.to_thread(extract_face, adhar_img)
    pan_face_task = asyncio.to_thread(extract_face, pan_img)
    adhar_face, pan_face = await asyncio.gather(adhar_face_task, pan_face_task)

    adhar_match_task = asyncio.to_thread(match_faces_with_facenet, enhanced_face, adhar_face) if adhar_face else None
    pan_match_task = asyncio.to_thread(match_faces_with_facenet, enhanced_face, pan_face) if pan_face else None

    match_adhar, score_adhar = await adhar_match_task if adhar_match_task else (False, None)
    match_pan, score_pan = await pan_match_task if pan_match_task else (False, None)

    # 🔹 Step 6: Return All Results
    return {
        "pan_ocr": pan_text,
        "adhar_ocr": adhar_text,
        "face_match_with_adhar": {"match": match_adhar, "score": score_adhar},
        "face_match_with_pan": {"match": match_pan, "score": score_pan}
    }

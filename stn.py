import streamlit as st
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from logging_config import get_logger

# Initialize Logger
logger = get_logger(__name__)

# FastAPI API URL
API_URL = "http://localhost:8000/results/process_documents/"

# Initialize session state variables
if "captured_image_bytes" not in st.session_state:
    st.session_state.captured_image_bytes = None
if "pan_upload" not in st.session_state:
    st.session_state.pan_upload = None
if "adhar_upload" not in st.session_state:
    st.session_state.adhar_upload = None
if "image_captured" not in st.session_state:
    st.session_state.image_captured = False
if "submitted" not in st.session_state:
    st.session_state.submitted = False  # Prevent multiple submissions

# Function to capture an image from the webcam
def capture_image():
    logger.info("📸 Attempting to capture live image from webcam...")
    st.write("📸 Capturing image from webcam...")
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Webcam not accessible. Please allow camera access.")
        logger.error("❌ Webcam not accessible. Check permissions or connection.")
        return

    ret, frame = cap.read()
    cap.release()

    if ret:
        logger.info("✅ Successfully captured image from webcam.")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(".jpg", frame_rgb)  # Convert to JPEG
        st.session_state.captured_image_bytes = buffer.tobytes()  # Store in session state
        st.session_state.image_captured = True  # Mark as captured

        st.image(frame_rgb, caption="📷 Captured Live Image", use_column_width=True)
        st.success("✅ Image captured successfully!")
    else:
        st.error("❌ Failed to capture image from webcam.")
        logger.error("❌ Failed to capture image from webcam.")

# Function to send images to FastAPI backend (Synchronous)
def send_images_to_api():
    if st.session_state.submitted:
        st.warning("⚠️ Submission already in progress. Please wait...")
        logger.warning("⚠️ Submission already in progress. Preventing duplicate requests.")
        return

    logger.info("🚀 Preparing to send images to API...")

    if not st.session_state.captured_image_bytes or not st.session_state.pan_upload or not st.session_state.adhar_upload:
        st.error("❌ Submission failed - Please upload all required images")
        logger.error("❌ Missing images. Submission aborted.")
        return

    st.session_state.submitted = True  # Prevent multiple submissions

    # Convert images to file-like objects
    files = {
        "pan_image": ("pan.jpg", st.session_state.pan_upload.getvalue(), "image/jpeg"),
        "adhar_image": ("adhar.jpg", st.session_state.adhar_upload.getvalue(), "image/jpeg"),
        "live_image": ("live.jpg", BytesIO(st.session_state.captured_image_bytes), "image/jpeg"),
    }

    logger.info("📤 Sending images to FastAPI endpoint...")

    # Send request to API
    try:
        response = requests.post(API_URL, files=files)
        logger.info(f"✅ API Response Received with status code {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Documents processed successfully!")

            # Display results
            st.subheader("📜 OCR Results")
            st.write(f"**PAN OCR:** {result.get('pan_ocr', 'N/A')}")
            st.write(f"**Aadhar OCR:** {result.get('adhar_ocr', 'N/A')}")

            logger.info("✅ OCR Results displayed.")

            st.subheader("🆔 Face Match Results")
            face_match_adhar = result.get("face_match_with_adhar", {})
            face_match_pan = result.get("face_match_with_pan", {})

            st.write(f"**Face Match with Aadhaar:** {'✅ Matched' if face_match_adhar.get('match') else '❌ Not Matched'}")
            st.write(f"**Match Score (Aadhaar):** {face_match_adhar.get('score', 'N/A')}")

            st.write(f"**Face Match with PAN:** {'✅ Matched' if face_match_pan.get('match') else '❌ Not Matched'}")
            st.write(f"**Match Score (PAN):** {face_match_pan.get('score', 'N/A')}")

            logger.info("✅ Face match results displayed.")

        else:
            st.error(f"❌ Failed to process documents: {response.text}")
            logger.error(f"❌ API Error: {response.text}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        logger.error(f"❌ Exception while sending request to API: {e}")

    st.session_state.submitted = False  # Reset submission state
    logger.info("🔄 Submission state reset.")

# Streamlit UI Layout
st.title("📄 Document Upload Portal")
logger.info("🎬 Streamlit UI loaded.")

# Capture Image Button
if st.button("📸 Capture Live Image"):
    logger.info("🛑 Capture Live Image button clicked.")
    capture_image()

# File uploaders for PAN and Aadhaar
pan_upload = st.file_uploader("🆔 Upload PAN Card", type=["jpg", "png", "jpeg"])
if pan_upload:
    st.session_state.pan_upload = pan_upload
    logger.info("📂 PAN image uploaded.")

adhar_upload = st.file_uploader("🆔 Upload Aadhaar Card", type=["jpg", "png", "jpeg"])
if adhar_upload:
    st.session_state.adhar_upload = adhar_upload
    logger.info("📂 Aadhaar image uploaded.")

# Submit button
if st.button("🚀 Submit to API"):
    logger.info("🛑 Submit button clicked. Processing...")
    st.write("📤 Sending data to API... Please wait.")
    send_images_to_api()  # Synchronous API call

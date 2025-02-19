import streamlit as st
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, Future
from logging_config import get_logger
from app.backend.utlis.liveness_utlis import start_liveness_detection

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
    st.session_state.submitted = False
if "liveness_result" not in st.session_state:
    st.session_state.liveness_result = None
if "liveness_future" not in st.session_state:
    st.session_state.liveness_future = None  # Future object for async execution

# ThreadPoolExecutor for async execution
executor = ThreadPoolExecutor(max_workers=1)

# Streamlit UI Layout
st.title("Document Upload Portal")
logger.info("Streamlit UI loaded.")

# Function to run liveness detection in a separate thread
def run_liveness_detection():
    logger.info("Starting liveness detection asynchronously...")
    future = executor.submit(start_liveness_detection)
    st.session_state.liveness_future = future  # Store the future in session state

# Capture Image using Streamlit's built-in camera
captured_image = st.camera_input("Capture Live Image")
if captured_image:
    st.session_state.captured_image_bytes = captured_image.getvalue()
    st.session_state.image_captured = True
    logger.info("Image captured using Streamlit camera input.")
    run_liveness_detection()  # Start liveness detection asynchronously

# File uploaders for PAN and Aadhaar
pan_upload = st.file_uploader("Upload PAN Card", type=["jpg", "png", "jpeg"])
if pan_upload:
    st.session_state.pan_upload = pan_upload
    logger.info("PAN image uploaded.")

adhar_upload = st.file_uploader("Upload Aadhaar Card", type=["jpg", "png", "jpeg"])
if adhar_upload:
    st.session_state.adhar_upload = adhar_upload
    logger.info("Aadhaar image uploaded.")

# Function to send images to FastAPI backend (Synchronous)
def send_images_to_api():
    if st.session_state.submitted:
        st.warning("Submission already in progress. Please wait...")
        logger.warning("Submission already in progress. Preventing duplicate requests.")
        return

    logger.info("Preparing to send images to API...")

    if not st.session_state.captured_image_bytes or not st.session_state.pan_upload or not st.session_state.adhar_upload:
        st.error("Submission failed - Please upload all required images")
        logger.error("Missing images. Submission aborted.")
        return

    st.session_state.submitted = True  # Prevent multiple submissions

    # Convert images to file-like objects
    files = {
        "pan_image": ("pan.jpg", st.session_state.pan_upload.getvalue(), "image/jpeg"),
        "adhar_image": ("adhar.jpg", st.session_state.adhar_upload.getvalue(), "image/jpeg"),
        "live_image": ("live.jpg", BytesIO(st.session_state.captured_image_bytes), "image/jpeg"),
    }

    logger.info("Sending images to FastAPI endpoint...")

    # Send request to API
    try:
        response = requests.post(API_URL, files=files)
        logger.info(f"API Response Received with status code {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            st.success("Documents processed successfully!")

            # Display results
            st.subheader("OCR Results")
            st.write(f"**PAN OCR:** {result.get('pan_ocr', 'N/A')}")
            st.write(f"**Aadhar OCR:** {result.get('adhar_ocr', 'N/A')}")

            logger.info("OCR Results displayed.")

            st.subheader("Face Match Results")
            face_match_adhar = result.get("face_match_with_adhar", {})
            face_match_pan = result.get("face_match_with_pan", {})

            st.write(f"**Face Match with Aadhaar:** {'Matched' if face_match_adhar.get('match') else 'Not Matched'}")
            st.write(f"**Match Score (Aadhaar):** {face_match_adhar.get('score', 'N/A')}")

            st.write(f"**Face Match with PAN:** {'Matched' if face_match_pan.get('match') else 'Not Matched'}")
            st.write(f"**Match Score (PAN):** {face_match_pan.get('score', 'N/A')}")

            logger.info("Face match results displayed.")

        else:
            st.error(f"Failed to process documents: {response.text}")
            logger.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Error: {e}")
        logger.error(f"Exception while sending request to API: {e}")

    st.session_state.submitted = False  # Reset submission state
    logger.info("Submission state reset.")

# Submit button
if st.button("Submit to API"):
    logger.info("Submit button clicked. Processing...")
    st.write("Sending data to API... Please wait.")
    
    # Wait for liveness detection to finish if it's still running
    if st.session_state.liveness_future is not None:
        if not st.session_state.liveness_future.done():
            st.write("Waiting for liveness detection to complete...")
            st.session_state.liveness_result = st.session_state.liveness_future.result()
        else:
            st.session_state.liveness_result = st.session_state.liveness_future.result()

    # Display liveness detection result
    if st.session_state.liveness_result is not None:
        st.subheader("Liveness Detection Result")
        st.write(st.session_state.liveness_result)
        logger.info("Liveness detection result displayed.")

    send_images_to_api()  # Synchronous API call

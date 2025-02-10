import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import tempfile
import io

# API Endpoint
API_URL = "http://localhost:8000/results/process_documents/"

# Function to capture an image from the webcam
def capture_webcam_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open webcam. Please check your camera.")
        return None

    st.write("Capturing image... Press 'c' to capture.")

    # Live Webcam Feed
    captured_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        # Convert BGR to RGB (Streamlit displays images in RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the image in Streamlit window
        st.image(frame_rgb, caption="Live Camera Feed", use_column_width=True)

        # Wait for the user to press 'c' to capture
        if st.button("Capture Photo"):
            captured_image = frame_rgb
            st.success("Image Captured!")
            break

    cap.release()
    return captured_image

# Function to convert image to bytes
def image_to_bytes(image):
    img = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()

# Home screen
st.title("KYC Verification System")

# KYC Start Button
if "kyc_started" not in st.session_state:
    st.session_state.kyc_started = False

if not st.session_state.kyc_started:
    if st.button("Begin KYC"):
        st.session_state.kyc_started = True
        st.experimental_rerun()

# KYC Process
if st.session_state.kyc_started:
    st.subheader("Step 1: Capture Live Image")

    if "live_image" not in st.session_state:
        st.session_state.live_image = None

    # Open mini window for webcam
    if st.button("Open Camera"):
        st.session_state.live_image = capture_webcam_image()
        st.experimental_rerun()

    if st.session_state.live_image is not None:
        st.image(st.session_state.live_image, caption="Captured Image", use_column_width=True)

    # File uploader for PAN & Aadhaar
    st.subheader("Step 2: Upload PAN & Aadhaar")

    pan_image = st.file_uploader("Upload PAN Card Image", type=["png", "jpg", "jpeg"])
    adhar_image = st.file_uploader("Upload Aadhaar Card Image", type=["png", "jpg", "jpeg"])

    # Submit for verification
    if st.button("Submit for Verification"):
        if st.session_state.live_image is not None and pan_image and adhar_image:
            # Convert captured image to bytes
            live_image_bytes = image_to_bytes(st.session_state.live_image)

            # Convert uploaded images to bytes
            pan_bytes = pan_image.read()
            adhar_bytes = adhar_image.read()

            # Prepare API payload
            files = {
                "live_image": ("live_image.png", live_image_bytes, "image/png"),
                "pan_image": ("pan_image.png", pan_bytes, "image/png"),
                "adhar_image": ("adhar_image.png", adhar_bytes, "image/png"),
            }

            # Send request to API
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success("Verification Complete!")

                # Display API results
                st.subheader("OCR Results:")
                st.write(f"**PAN OCR:** {result['pan_ocr']}")
                st.write(f"**Aadhaar OCR:** {result['adhar_ocr']}")

                st.subheader("Face Match Results:")
                st.write(f"**Face Match with Aadhaar:** {'Match' if result['face_match_with_adhar']['match'] else 'No Match'}")
                st.write(f"**Match Score:** {result['face_match_with_adhar']['score']}")

                st.write(f"**Face Match with PAN:** {'Match' if result['face_match_with_pan']['match'] else 'No Match'}")
                st.write(f"**Match Score:** {result['face_match_with_pan']['score']}")
            else:
                st.error(f"API Error: {response.status_code}")
        else:
            st.warning("Please provide all required images.")

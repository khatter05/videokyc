import streamlit as st
import requests

# API Endpoint (Update if different)
API_URL = "http://localhost:8000/results/process_documents/"

def send_images_to_api(live_image, pan_image, adhar_image):
    """Sends uploaded images to FastAPI endpoint."""
    
    # Prepare files for API request
    files = {
        "live_image": ("live.jpg", live_image, "image/jpeg"),
        "pan_image": ("pan.jpg", pan_image, "image/jpeg"),
        "adhar_image": ("adhar.jpg", adhar_image, "image/jpeg"),
    }

    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API request failed with status code {response.status_code}"}

def main():
    st.title("📄 KYC Verification System")

    # Upload Images
    st.subheader("📤 Upload Required Documents")

    live_image = st.file_uploader("Upload Live Image", type=["jpg", "png", "jpeg"])
    pan_image = st.file_uploader("Upload PAN Card Image", type=["jpg", "png", "jpeg"])
    adhar_image = st.file_uploader("Upload Aadhaar Card Image", type=["jpg", "png", "jpeg"])

    # Submit Button
    if st.button("Submit for Verification"):
        if live_image and pan_image and adhar_image:
            st.write("📡 Sending images to API...")

            # Read image bytes
            response = send_images_to_api(live_image.read(), pan_image.read(), adhar_image.read())

            # Display API response
            st.subheader("✅ API Response:")
            st.json(response)
        else:
            st.error("❌ Please upload all three images before submitting.")

if __name__ == "__main__":
    main()

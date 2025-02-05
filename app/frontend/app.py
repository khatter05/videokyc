import streamlit as st
import requests
import io

# FastAPI endpoint
API_URL = "http://localhost:8000/results/process_documents/"

# Streamlit UI
st.title("📄 Video KYC - OCR & Face Matching")
st.write("Upload PAN, Aadhaar, and Live Image for verification")

# Upload images
pan_file = st.file_uploader("Upload PAN Card Image", type=["jpg", "png", "jpeg"])
adhar_file = st.file_uploader("Upload Aadhaar Card Image", type=["jpg", "png", "jpeg"])
live_file = st.file_uploader("Upload Live Image", type=["jpg", "png", "jpeg"])

if st.button("Submit for Verification") and pan_file and adhar_file and live_file:
    # Convert images to bytes
    files = {
        "pan_image": ("pan.jpg", pan_file, "image/jpeg"),
        "adhar_image": ("adhar.jpg", adhar_file, "image/jpeg"),
        "live_image": ("live.jpg", live_file, "image/jpeg"),
    }

    with st.spinner("Processing..."):
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        data = response.json()
        st.success("✅ Verification Completed!")

        # Display OCR Results
        st.subheader("📌 OCR Results")
        st.write(f"**PAN OCR:** {data['pan_ocr']}")
        st.write(f"**Aadhaar OCR:** {data['adhar_ocr']}")

        # Display Face Match Results
        st.subheader("🤖 Face Matching Results")
        match_adhar = data["face_match_with_adhar"]["match"]
        match_pan = data["face_match_with_pan"]["match"]
        
        st.write(f"**Face Match with Aadhaar:** {'✅ Match' if match_adhar else '❌ No Match'}")
        st.write(f"**Face Match with PAN:** {'✅ Match' if match_pan else '❌ No Match'}")
    else:
        st.error("❌ Error in processing! Please try again.")


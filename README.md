# Video KYC - Face & ID Verification System

## Overview

This project is a **Video KYC (Know Your Customer) system** that verifies a user's face against an identity document in real time. It utilizes **EasyOCR** for text extraction, **ESRGAN** for image enhancement, and **OpenCV** for face detection.

## Project Structure

```
khatter05-videokyc/
├── __init__.py
├── dockerfile
├── logging_config.py
├── requirements.txt
├── stn.py
├── .dockerignore
└── app/
    ├── __init__.py
    └── backend/
        ├── __init__.py
        ├── main.py
        ├── routers/
        │   ├── __init__.py
        │   └── process.py
        └── utlis/
            ├── __init__.py
            ├── face.py
            ├── liveness_utlis.py
            ├── models.py
            └── ocr.py
```

## Features

- **Face Matching**: Compares the user's live face with the photo on their ID card.
- **OCR (Optical Character Recognition)**: Extracts text from identity documents using **EasyOCR**.
- **Liveness Detection**: Ensures the user is present and not using a spoofed image or video.
- **Image Enhancement**: Utilizes **ESRGAN** to improve image quality before OCR processing.

## Installation & Setup

### Prerequisites

- Python 3.8+
- Virtual environment (optional but recommended)
- Dependencies listed in `requirements.txt`

### Steps to Run

```sh
# Clone the repository
git clone https://github.com/khatter05/videokyc.git
cd videokyc

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

#ru the API
uvicorn app.backend.main:app --reload

# Run the application
streamlit run stn.py
```

## Important Note on Docker 🐳

🚨 **This project currently does not work in Docker!** 🚨

While a **Dockerfile** is included, the application **requires OpenCV for webcam access**, and **Docker containers are isolated environments** without direct access to host hardware like cameras.

### Alternative Approach: Using `streamlit-webrtc`

A better way to handle webcam input in a containerized environment is using [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc). However, I encountered issues integrating it. If anyone finds a working solution, feel free to contact me!

## Troubleshooting

- **EasyOCR is slow?** Ensure you're using a GPU. Otherwise, performance will be lower on CPU.
- **Webcam not working?** OpenCV may not have permission to access the camera. Try running with admin privileges.
- **Missing dependencies?** Reinstall them with:
  ```sh
  pip install --no-cache-dir -r requirements.txt
  ```

## Future Enhancements 🚀

- Implement `streamlit-webrtc` for better webcam handling in Docker.
- Optimize liveness detection to reduce false positives.
- Improve OCR accuracy with additional preprocessing techniques.

---

📩 **Found a fix or have suggestions?** Open an issue or reach out! 😊


import cv2
import numpy as np
from io import BytesIO
from typing import Optional
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cosine

from app.backend.utlis.models import mtcnn_detector as detector
from app.backend.utlis.models import srgan_model
from app.backend.utlis.models import facenet 


def extract_face(image_data : bytes, *, padding: float = 0.35) -> Optional[np.ndarray]:
    """
    Detects a face in the uploaded image file, applies padding to the cropped face, and returns the cropped face as a NumPy array.

    Args:
        file (UploadFile): Uploaded image file.
        padding (float, keyword-only): Padding factor (0.1 = 10% extra around the face).

    Returns:
        Optional[np.ndarray]: Cropped face as a NumPy array if face detected, None otherwise.
    """
    # Read image data from the uploaded file into memory
    #image_data = file.file.read()  # Get raw image bytes

    # Convert the bytes into a NumPy array
    image = np.frombuffer(image_data, dtype=np.uint8)

    # Decode the image to an actual image array using OpenCV
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Unable to decode image")
        return None

    # Initialize MTCNN detector
    #detector = MTCNN()

    # Detect faces
    faces = detector.detect_faces(image)

    if not faces:
        print("No face detected.")
        return None

    height, width, _ = image.shape  # Image dimensions

    for i, face in enumerate(faces):  # Loop in case there are multiple faces
        x, y, w, h = face["box"]

        # Compute padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)

        # Expand bounding box
        x1, y1 = max(x - pad_x, 0), max(y - pad_y, 0)
        x2, y2 = min(x + w + pad_x, width), min(y + h + pad_y, height)

        # Crop the padded face
        cropped_face = image[y1:y2, x1:x2]

        print(f"✅ Face extracted with padding {padding*100:.1f}%")

        # Return the cropped face as a NumPy array
        return cropped_face

    return None  # If no face was detected


def enhance_image(image_data: bytes) -> Optional[np.ndarray]:
    """
    Enhances the resolution of the given image using ESRGAN in memory.

    Args:
        image_data (bytes): Raw byte data of the image to enhance.

    Returns:
        Optional[np.ndarray]: Enhanced image as a NumPy array if enhancement is successful, None otherwise.
    """
    # Convert byte data to NumPy array
    image = np.frombuffer(image_data, dtype=np.uint8)

    # Decode the image into an actual image array using OpenCV
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Unable to decode image")
        return None

    # Convert to RGB and normalize (0-1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image_resized = np.expand_dims(image_rgb, axis=0).astype(np.float32)

    # Run the image through ESRGAN model
    enhanced_image = srgan_model(image_resized)[0]  # Get the first output

    # Convert the enhanced image back to a proper format
    enhanced_image = np.array(enhanced_image)
    enhanced_image = (enhanced_image * 255.0).clip(0, 255).astype(np.uint8)

    return enhanced_image


def match_faces_with_facenet(image1_data: bytes, image2_data: bytes) -> tuple:
    """
    Compares two in-memory images and returns whether they match and the similarity score.

    Args:
        image1_data (bytes): Raw byte data of the first image (cropped and enhanced).
        image2_data (bytes): Raw byte data of the second image (cropped and enhanced).

    Returns:
        tuple: A tuple containing:
            - bool: True if the faces match, False otherwise.
            - float: The cosine similarity score between the two faces.
    """
    # Convert byte data to NumPy arrays
    image1 = np.frombuffer(image1_data, dtype=np.uint8)
    image2 = np.frombuffer(image2_data, dtype=np.uint8)

    # Decode images into actual images using OpenCV
    image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)

    if image1 is None:
        raise ValueError("Failed to decode the first image")
    if image2 is None:
        raise ValueError("Failed to decode the second image")

    # Resize the images to the size FaceNet expects (160x160)
    image1 = cv2.resize(image1, (160, 160))
    image2 = cv2.resize(image2, (160, 160))

    # Convert images to RGB (FaceNet expects RGB images, OpenCV loads in BGR by default)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Add an extra batch dimension: from (160, 160, 3) to (1, 160, 160, 3)
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)

    # Initialize the FaceNet model
    #facenet = FaceNet()

    # Extract face embeddings
    embedding1 = facenet.embeddings(image1)[0]
    embedding2 = facenet.embeddings(image2)[0]

    # Compute the cosine distance between the two embeddings
    distance = cosine(embedding1, embedding2)

    # Define a threshold for matching (typically, 0.6 to 0.7 is a common threshold)
    threshold = 0.6

    # Determine if faces match
    faces_match = distance < threshold

    return faces_match, distance

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
from logging_config import get_logger  # Import the global logger

logger=get_logger(__name__)

def extract_face(image_data: np.ndarray, *, padding: float = 0.35) -> Optional[np.ndarray]:
    """
    Detects a face in the uploaded image, applies padding to the cropped face, enhances it, 
    and returns the enhanced cropped face as a NumPy array.

    Args:
        image_data (np.ndarray): Image data as a NumPy array (BGR format).
        padding (float, keyword-only): Padding factor (0.1 = 10% extra around the face).

    Returns:
        Optional[np.ndarray]: Enhanced cropped face as a NumPy array if face detected, None otherwise.
    """
    # Ensure the image is in the correct format (BGR)
    image = image_data

    # Detect faces
    faces = detector.detect_faces(image)

    if not faces:
        logger.warning("No face detected in the image.")
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

        # Call the enhance_image function to enhance the cropped face
        enhanced_face = enhance_image(cropped_face)

        if enhanced_face is None:
            logger.error("Face enhancement failed.")
            return None

        logger.info(f"Face extracted and enhanced with padding {padding*100:.1f}%")
        return enhanced_face

    return None  # If no face was detected


def enhance_image(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Enhances the resolution of the given image using ESRGAN in memory.

    Args:
        image (np.ndarray): Input image as a NumPy array (in BGR format).

    Returns:
        Optional[np.ndarray]: Enhanced image as a NumPy array if enhancement is successful, None otherwise.
    """
    if image is None:
        logger.error("Received None as input image for enhancement.")
        return None

    # Convert to RGB and normalize (0-1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image_resized = np.expand_dims(image_rgb, axis=0).astype(np.float32)

    # Run the image through ESRGAN model
    enhanced_image = srgan_model(image_resized)[0]  # Get the first output

    # Convert the enhanced image back to a proper format
    enhanced_image = np.array(enhanced_image)
    enhanced_image = (enhanced_image * 255.0).clip(0, 255).astype(np.uint8)

    #update
    enhanced_image=cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
    
    logger.info("Image enhancement successful.")
    return enhanced_image


def match_faces_with_facenet(image1: np.ndarray, image2: np.ndarray) -> tuple:
    """
    Compares two in-memory images and returns whether they match and the similarity score.

    Args:
        image1 (np.ndarray): The first image (cropped and enhanced).
        image2 (np.ndarray): The second image (cropped and enhanced).

    Returns:
        tuple: A tuple containing:
            - bool: True if the faces match, False otherwise.
            - float: The cosine similarity score between the two faces.
    """
    # Ensure the images are in BGR format (OpenCV expects BGR by default)
    if image1 is None or image2 is None:
        logger.error("One or both input images are None.")
        raise ValueError("Input images cannot be None.")

    # Resize the images to the size FaceNet expects (160x160)
    image1 = cv2.resize(image1, (160, 160))
    image2 = cv2.resize(image2, (160, 160))

    # Convert images to RGB (FaceNet expects RGB images, OpenCV loads in BGR by default)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Add an extra batch dimension: from (160, 160, 3) to (1, 160, 160, 3)
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)

    # Extract face embeddings
    embedding1 = facenet.embeddings(image1)[0]
    embedding2 = facenet.embeddings(image2)[0]

    # Compute the cosine distance between the two embeddings
    distance = cosine(embedding1, embedding2)

    # Define a threshold for matching (typically, 0.6 to 0.7 is a common threshold)
    threshold = 0.6

    # Determine if faces match
    faces_match = distance < threshold

    logger.info(f"Face matching result: Match = {faces_match}, Similarity Score = {distance}")
    return faces_match, distance

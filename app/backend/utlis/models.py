import easyocr
from keras_facenet import FaceNet
import tensorflow_hub as hub
import cv2
from mtcnn import MTCNN
import mediapipe as mp

from logging_config import get_logger
logger = get_logger(__name__)

try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    logger.info("Mediapipe FaceMesh model initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Mediapipe FaceMesh: {e}")

try:
    mtcnn_detector = MTCNN()
    logger.info("MTCNN face detector initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing MTCNN face detector: {e}")

try:
    facenet = FaceNet()
    logger.info("FaceNet model initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing FaceNet model: {e}")

try:
    srgan_model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
    logger.info("ESRGAN model loaded successfully from TensorFlow Hub.")
except Exception as e:
    logger.error(f"Error loading ESRGAN model: {e}")

try:
    OCR_READER = easyocr.Reader(['en'], gpu = False)
    logger.info("EasyOCR reader initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing EasyOCR reader: {e}")

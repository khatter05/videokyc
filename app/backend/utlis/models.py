import easyocr
from keras_facenet import FaceNet
import tensorflow_hub as hub
import cv2
from mtcnn import MTCNN

from logging_config import get_logger
logger = get_logger(__name__)

mtcnn_detector = MTCNN()


facenet = FaceNet()

srgan_model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

OCR_READER = easyocr.Reader(['en'])

logger.info("models are initialized")
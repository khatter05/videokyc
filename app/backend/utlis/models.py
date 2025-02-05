import easyocr
from keras_facenet import FaceNet
import tensorflow_hub as hub
import cv2
from mtcnn import MTCNN


mtcnn_detector = MTCNN()

facenet = FaceNet()

srgan_model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

easyocr_reader = easyocr.Reader(['en'])


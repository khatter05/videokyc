import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict
import time
from app.backend.utlis.models import face_mesh
from logging_config import get_logger
logger = get_logger(__name__)



# Eye landmarks for Mediapipe
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Detection thresholds
EAR_THRESHOLD = 0.2  # Blink detection threshold
BLINK_FRAMES = 2  # Minimum consecutive frames for a valid blink
HEAD_MOVEMENT_THRESHOLD = 10  # Head movement threshold (degrees)

def initialize_webcam() -> Optional[cv2.VideoCapture]:
    """Initializes the webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam initialization failed.")
        return None
    logger.info("Webcam initialized successfully.")
    return cap

def calculate_ear(landmarks: list, eye_indices: list) -> float:
    """Calculates Eye Aspect Ratio (EAR) for blink detection."""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical_1 = np.linalg.norm([p2.x - p6.x, p2.y - p6.y])
    vertical_2 = np.linalg.norm([p3.x - p5.x, p3.y - p5.y])
    horizontal = np.linalg.norm([p1.x - p4.x, p1.y - p4.y])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    #logger.debug(f"EAR calculated: {ear:.4f}")
    return ear

def estimate_head_movement(landmarks: list, img_shape: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
    """Estimates head movement using facial landmarks."""
    try:
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (-30.0, -30.0, -30.0),
            (30.0, -30.0, -30.0),
            (-40.0, 30.0, -30.0),
            (40.0, 30.0, -30.0),
            (0.0, -50.0, -20.0)
        ], dtype=np.float32)

        image_points = np.array([
            (landmarks[1].x * img_shape[1], landmarks[1].y * img_shape[0]),
            (landmarks[33].x * img_shape[1], landmarks[33].y * img_shape[0]),
            (landmarks[263].x * img_shape[1], landmarks[263].y * img_shape[0]),
            (landmarks[61].x * img_shape[1], landmarks[61].y * img_shape[0]),
            (landmarks[291].x * img_shape[1], landmarks[291].y * img_shape[0]),
            (landmarks[199].x * img_shape[1], landmarks[199].y * img_shape[0])
        ], dtype=np.float32)

        cam_matrix = np.array([
            [img_shape[1], 0, img_shape[1] / 2],
            [0, img_shape[1], img_shape[0] / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, cam_matrix, np.zeros((4, 1), dtype=np.float32))

        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            #logger.debug(f"Head angles (Yaw, Pitch, Roll): {angles}")
            return angles[1], angles[0], angles[2]
    except Exception as e:
        logger.error(f"Error estimating head movement: {e}")
    return None

def process_blink_detection(ear: float, consecutive_frames: int) -> Tuple[bool, int]:
    """Detects blinks based on EAR threshold."""
    if ear < EAR_THRESHOLD:
        return False, consecutive_frames + 1
    if consecutive_frames >= BLINK_FRAMES:
        #logger.info("Blink detected!")
        return True, 0  # Blink detected, reset count
    return False, 0

def process_frame(frame: np.ndarray, consecutive_frames: int, blink_count: int, head_movement_detected: bool) -> Tuple[np.ndarray, int, int, bool]:
    """Processes a frame to detect blinks and head movement."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE)
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE)
            avg_ear = (right_ear + left_ear) / 2.0

            blink_detected, consecutive_frames = process_blink_detection(avg_ear, consecutive_frames)
            if blink_detected:
                blink_count += 1

            head_movement = estimate_head_movement(face_landmarks.landmark, frame.shape)
            if head_movement:
                yaw, pitch, roll = head_movement
                if abs(yaw) > HEAD_MOVEMENT_THRESHOLD or abs(pitch) > HEAD_MOVEMENT_THRESHOLD:
                    head_movement_detected = True
                    #logger.info("Head movement detected.")
    return frame, consecutive_frames, blink_count, head_movement_detected

def start_liveness_detection() -> Optional[Dict[str, int | bool]]:
    """Runs the liveness detection system using webcam feed."""
    cap = initialize_webcam()
    if cap is None:
        return None  # Exit if webcam is unavailable

    blink_count, consecutive_frames, head_movement_detected = 0, 0, False
    start_time, max_duration = time.time(), 30  # Run for 30 seconds

    while time.time() - start_time < max_duration:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from webcam.")
            break

        frame, consecutive_frames, blink_count, head_movement_detected = process_frame(
            frame, consecutive_frames, blink_count, head_movement_detected
        )

    cap.release()
    cv2.destroyAllWindows()

    logger.info(f"Liveness Detection Summary - Blinks: {blink_count}, Head Movement: {head_movement_detected}")
    return {"blink_count": blink_count, "head_movement": head_movement_detected}

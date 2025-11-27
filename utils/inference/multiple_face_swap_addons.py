"""
Helper functions for multiple face swap application
This module contains utility functions that don't directly use Streamlit
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import platform

from utils.inference.image_processing import normalize_and_torch, crop_face as crop_face_util
from utils.inference.realtime_processing import process_single_frame


def ensure_uint8(frame):
    """Ensure frame is in uint8 format"""
    if frame.dtype != np.uint8:
        if frame.dtype in [np.float16, np.float32, np.float64]:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def load_source_face(image, models, crop_size=224):
    """Load and process source face image"""
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Crop face
        face_crops = crop_face_util(img_array, models['app'], crop_size=crop_size)

        if not face_crops or len(face_crops) == 0:
            return None, "No face detected in the source image"

        # Get embedding
        source_normalized = normalize_and_torch(face_crops[0])
        source_embed = models['netArc'](
            F.interpolate(source_normalized, scale_factor=0.5, mode='bilinear', align_corners=True)
        )

        return source_embed, None
    except Exception as e:
        return None, f"Error processing source face: {e}"


def draw_face_bounding_boxes(
    original_frame: np.ndarray,
    result_frame: np.ndarray,
    embedding_map: dict,
    netArc,
    app,
    similarity_threshold: float = 0.25,
    crop_size: int = 224
) -> np.ndarray:
    """
    Draw bounding boxes on detected faces (2x bigger boxes)
    Green box: face was swapped (matched with target)
    Yellow box: face detected but not swapped (no match)

    Args:
        original_frame: Original frame BEFORE swapping (BGR format)
        result_frame: Result frame AFTER swapping (BGR format)
        embedding_map: Pre-computed mapping of target->source embeddings
        netArc: ArcFace model for computing embeddings
        app: Face detection model
        similarity_threshold: Minimum cosine similarity for face matching
        crop_size: Face crop size for alignment

    Returns:
        Result frame with bounding boxes drawn
    """
    from insightface.utils import face_align
    import torch.nn.functional as F_torch

    output_frame = result_frame.copy()

    # Detect all faces in the ORIGINAL frame (before swapping)
    detected_kps = app.get(original_frame, crop_size=crop_size)

    if detected_kps is None or len(detected_kps) == 0:
        return output_frame

    # Process each detected face in the ORIGINAL frame
    for kps in detected_kps:
        # Get bounding box from keypoints
        # Keypoints are 5 facial landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        x_coords = kps[:, 0]
        y_coords = kps[:, 1]

        # Calculate bounding box with padding
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))

        # Add padding around the face (2x bigger boxes)
        width = x_max - x_min
        height = y_max - y_min
        padding_x = int(width * 0.6)   # 2x of 0.3
        padding_y = int(height * 1.0)  # 2x of 0.5

        x_min = max(0, x_min - padding_x)
        x_max = min(original_frame.shape[1], x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(original_frame.shape[0], y_max + padding_y)

        # Compute transformation matrix for face alignment
        M = face_align.estimate_norm(kps, crop_size)

        if M is None:
            continue

        # Crop and align face from ORIGINAL frame
        face_aligned = cv2.warpAffine(original_frame, M, (crop_size, crop_size), borderValue=0.0)

        # Get embedding for this detected face (from original)
        face_112 = cv2.resize(face_aligned, (112, 112))
        face_tensor = normalize_and_torch(face_112)
        face_tensor = face_tensor.cuda()

        with torch.no_grad():
            det_embedding = netArc(face_tensor)

        # Find best matching target face
        best_match_idx = None
        best_similarity = -1

        for map_idx, map_data in embedding_map.items():
            target_embedding = map_data['target_embedding']
            similarity = F_torch.cosine_similarity(det_embedding, target_embedding).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = map_idx

        # Determine box color based on match
        if best_match_idx is not None and best_similarity > similarity_threshold:
            # Green box for swapped faces
            color = (0, 255, 0)
            label = f"Swapped ({best_similarity:.3f})"
        else:
            # Yellow box for detected but not swapped
            color = (0, 255, 255)
            label = f"Detected ({best_similarity:.3f})" if best_match_idx is not None else "Detected"

        # Draw bounding box on the OUTPUT (result) frame
        cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), color, 3)

        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            output_frame,
            (x_min, y_min - label_size[1] - 10),
            (x_min + label_size[0], y_min),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            output_frame,
            label,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return output_frame


def capture_and_swap(models, source_embed, camera_index=0, crop_size=224):
    """Capture single frame and perform face swap"""
    try:
        # Open camera
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            return None, "Failed to open camera"

        # Read frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None, "Failed to capture frame"

        frame = ensure_uint8(frame)

        # Detect face and swap
        result, _, _ = process_single_frame(
            frame=frame,
            source_embed=source_embed,
            netArc=models['netArc'],
            G=models['G'],
            app=models['app'],
            handler=models['handler'],
            bbox=None,
            crop_size=crop_size,
            half=False
        )

        if result is None:
            return frame, "No face detected in camera frame"

        return result, None

    except Exception as e:
        return None, f"Error: {e}"

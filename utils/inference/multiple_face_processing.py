"""
Multiple face processing for video frames
Optimized for processing multiple face swaps in real-time
"""
import time
import numpy as np
import torch
import cv2
from typing import Optional, Tuple, List, Dict
from insightface.utils import face_align

from .image_processing import normalize_and_torch
from .faceshifter_run import faceshifter_batch


def load_face_embeddings(
    source_face_images: List[np.ndarray],
    target_face_images: List[np.ndarray],
    netArc: torch.nn.Module,
    app,
    crop_size: int = 224,
    device: str = 'cuda'
) -> Dict:
    """
    Load and compute embeddings for source and target face pairs

    Args:
        source_face_images: List of source face images (BGR numpy arrays)
        target_face_images: List of target face images (BGR numpy arrays)
        netArc: ArcFace model for computing embeddings
        app: Face detection model
        crop_size: Face crop size for alignment
        device: Device to use for computation

    Returns:
        Dictionary mapping index to embeddings:
        {idx: {'target_embedding': tensor, 'source_embedding': tensor, ...}}
    """
    embedding_map = {}

    for i, (source_img, target_img) in enumerate(zip(source_face_images, target_face_images)):
        # Process target image
        # Convert RGBA to RGB if needed
        if len(target_img.shape) == 3 and target_img.shape[2] == 4:
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGRA2BGR)

        # Detect face in target image
        target_kps = app.get(target_img, crop_size=crop_size)

        if target_kps is None or len(target_kps) == 0:
            print(f"Warning: No face detected in target image {i}")
            # Use whole image as face
            target_face_aligned = cv2.resize(target_img, (crop_size, crop_size))
        else:
            # Get first detected face and align
            kps = target_kps[0]
            M = face_align.estimate_norm(kps, crop_size)
            target_face_aligned = cv2.warpAffine(target_img, M, (crop_size, crop_size), borderValue=0.0)

        # Extract target embedding (ArcFace expects 112x112)
        target_face_112 = cv2.resize(target_face_aligned, (112, 112))
        target_face_tensor = normalize_and_torch(target_face_112)

        if device == 'cuda':
            target_face_tensor = target_face_tensor.cuda()

        with torch.no_grad():
            target_embedding = netArc(target_face_tensor)

        # Process source image
        if len(source_img.shape) == 3 and source_img.shape[2] == 4:
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGRA2BGR)

        # Resize small images for better detection
        source_img_for_detection = source_img.copy()
        if source_img.shape[0] < 400 or source_img.shape[1] < 400:
            scale_factor = max(400 / source_img.shape[0], 400 / source_img.shape[1])
            new_width = int(source_img.shape[1] * scale_factor)
            new_height = int(source_img.shape[0] * scale_factor)
            source_img_for_detection = cv2.resize(source_img, (new_width, new_height))

        source_kps = app.get(source_img_for_detection, crop_size=crop_size)

        if source_kps is None or len(source_kps) == 0:
            # Use whole image as face
            source_face_aligned = cv2.resize(source_img, (crop_size, crop_size))
        else:
            # Get source face keypoints and crop
            source_kps_original = source_kps[0]
            if source_img_for_detection.shape != source_img.shape:
                scale_x = source_img.shape[1] / source_img_for_detection.shape[1]
                scale_y = source_img.shape[0] / source_img_for_detection.shape[0]
                source_kps_original = source_kps[0] * np.array([scale_x, scale_y])

            source_M = face_align.estimate_norm(source_kps_original, crop_size)
            source_face_aligned = cv2.warpAffine(source_img, source_M, (crop_size, crop_size), borderValue=0.0)

        # Get source embedding
        source_face_112 = cv2.resize(source_face_aligned, (112, 112))
        source_tensor = normalize_and_torch(source_face_112)

        if device == 'cuda':
            source_tensor = source_tensor.cuda()

        with torch.no_grad():
            source_embedding = netArc(source_tensor)

        # Store in embedding map
        embedding_map[i] = {
            'target_embedding': target_embedding,
            'source_embedding': source_embedding,
            'target_image': target_img,
            'source_image': source_img
        }

    return embedding_map


def process_single_frame_multiple(
    frame: np.ndarray,
    embedding_map: Dict,
    netArc: torch.nn.Module,
    G: torch.nn.Module,
    app,
    similarity_threshold: float = 0.076,
    crop_size: int = 224,
    generator_input_size: int = 256,
    device: str = 'cuda'
) -> Tuple[Optional[np.ndarray], float, float, int]:
    """
    Process a single frame for real-time multiple face swapping

    Args:
        frame: Input frame (BGR format numpy array)
        embedding_map: Pre-computed mapping of target->source embeddings
        netArc: ArcFace model for computing embeddings
        G: Generator model (AEI_Net)
        app: Face detection model
        similarity_threshold: Minimum cosine similarity for face matching
        crop_size: Face crop size for alignment
        generator_input_size: Input size for generator model
        device: Device to use for computation

    Returns:
        Tuple of (swapped_frame, detection_time_ms, generator_time_ms, num_faces_swapped)
        Returns (frame, 0, 0, 0) if no faces detected or processing fails
    """
    detection_start = time.time()

    try:
        # Make a copy of the frame
        result_frame = frame.copy()

        # Detect all faces in the frame
        detected_kps = app.get(frame, crop_size=crop_size)

        if detected_kps is None or len(detected_kps) == 0:
            return frame, 0, 0, 0

        detection_time = (time.time() - detection_start) * 1000
        generator_start = time.time()

        num_faces_swapped = 0

        # Process each detected face
        for kps in detected_kps:
            # Compute transformation matrix for face alignment
            M = face_align.estimate_norm(kps, crop_size)

            if M is None:
                continue

            # Crop and align face
            face_aligned = cv2.warpAffine(result_frame, M, (crop_size, crop_size), borderValue=0.0)

            # Get embedding for this detected face (ArcFace expects 112x112)
            face_112 = cv2.resize(face_aligned, (112, 112))
            face_tensor = normalize_and_torch(face_112)

            if device == 'cuda':
                face_tensor = face_tensor.cuda()

            with torch.no_grad():
                det_embedding = netArc(face_tensor)

            # Find best matching target face using cosine similarity
            best_match_idx = None
            best_similarity = -1

            for map_idx, map_data in embedding_map.items():
                target_embedding = map_data['target_embedding']
                similarity = torch.nn.functional.cosine_similarity(det_embedding, target_embedding).item()

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = map_idx

            # Only swap if similarity exceeds threshold
            if best_match_idx is None or best_similarity <= similarity_threshold:
                continue

            # Get pre-computed source embedding
            source_embedding = embedding_map[best_match_idx]['source_embedding']

            # Prepare target face for generator
            target_face_resized = cv2.resize(face_aligned, (generator_input_size, generator_input_size))
            target_face_batch = np.expand_dims(target_face_resized, axis=0)

            # Convert to torch tensor with proper normalization
            if device == 'cuda':
                target_tensor = torch.from_numpy(target_face_batch.copy()).cuda()
            else:
                target_tensor = torch.from_numpy(target_face_batch.copy())

            target_tensor = target_tensor[:, :, :, [2, 1, 0]] / 255.0  # BGR to RGB
            target_tensor = target_tensor.float()
            target_tensor = (target_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            target_tensor = target_tensor.permute(0, 3, 1, 2)  # (1, 3, H, W)

            # Run generator
            with torch.no_grad():
                swapped_face = faceshifter_batch(source_embedding, target_tensor, G)

            # Convert output to numpy
            swapped_face_np = swapped_face[0]

            # Resize back to crop size for blending
            swapped_face_resized = cv2.resize(swapped_face_np, (crop_size, crop_size))

            # Warp the swapped face back to the original frame
            mat_rev = cv2.invertAffineTransform(M)
            swap_warped = cv2.warpAffine(
                swapped_face_resized,
                mat_rev,
                (result_frame.shape[1], result_frame.shape[0]),
                borderMode=cv2.BORDER_REPLICATE
            )

            # Create blending mask
            h, w = swapped_face_resized.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            center = (w // 2, h // 2)
            radius = min(w, h) // 2 - 2
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            mask = (dist_from_center <= radius).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (21, 21), 11)

            # Warp mask back to original frame
            mask_warped = cv2.warpAffine(
                mask,
                mat_rev,
                (result_frame.shape[1], result_frame.shape[0])
            )

            # Blend the swapped face with the result frame
            mask_3d = mask_warped[:, :, np.newaxis]
            result_frame = (swap_warped * mask_3d + result_frame * (1 - mask_3d)).astype(np.uint8)

            num_faces_swapped += 1

        generator_time = (time.time() - generator_start) * 1000

        return result_frame, detection_time, generator_time, num_faces_swapped

    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return frame, 0, 0, 0

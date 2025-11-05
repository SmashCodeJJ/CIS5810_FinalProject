"""
Real-time single-frame processing for face swapping
Optimized for low latency and high FPS
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Optional, Tuple, List
from insightface.utils import face_align

from .image_processing import normalize_and_torch
from .faceshifter_run import faceshifter_batch
from .masks import face_mask_static


def process_single_frame(
    frame: np.ndarray,
    source_embed: torch.Tensor,
    netArc: torch.nn.Module,
    G: torch.nn.Module,
    app,
    handler,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    crop_size: int = 224,
    half: bool = False  # Changed default to False for stability
) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Process a single frame for real-time face swapping
    
    Args:
        frame: Input frame (BGR format)
        source_embed: Pre-computed source face embedding
        netArc: ArcFace model
        G: Generator model
        app: Face detection model
        handler: Landmark detection handler
        bbox: Optional bounding box from tracker (x, y, w, h)
        crop_size: Face crop size
        half: Use FP16 precision
    
    Returns:
        (swapped_frame, detection_time_ms, generator_time_ms)
        Returns (None, 0, 0) if no face detected
    """
    detection_start = time.time()
    
    try:
        # Get face keypoints - always use full detection for accuracy
        # The bbox is only used to speed up detection by providing a hint
        # But we still need keypoints for proper alignment
        kps = app.get(frame, crop_size)
        if not kps or len(kps) == 0:
            return None, 0, 0
        
        detection_time = (time.time() - detection_start) * 1000
        
        # Get first keypoint and validate
        first_kp = kps[0]
        if first_kp is None:
            return None, 0, 0
        
        # Compute transformation matrix
        result = face_align.estimate_norm(first_kp, crop_size)
        if isinstance(result, tuple):
            M, _ = result
        else:
            M = result
        
        if M is None:
            return None, 0, 0
        
        # Ensure correct shape
        if M.shape != (2, 3):
            if M.size == 6:
                if M.shape == (3, 2):
                    M = M.T
                elif len(M.shape) == 1 or M.shape == (1, 6):
                    M = M.reshape(2, 3)
            else:
                return None, 0, 0
        
        if M.dtype != np.float32:
            M = M.astype(np.float32)
        
        # Crop and align face
        crop_face = cv2.warpAffine(frame, M, (crop_size, crop_size), borderValue=0.0)
        
        # Get target embedding (use original crop size)
        target_norm = normalize_and_torch(crop_face)
        target_embed = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear', align_corners=True))
        
        # Resize to 256x256 for generator (generator expects this size, not crop_size)
        # Use INTER_LINEAR for consistent resizing (same as resize_frames function)
        generator_input_size = 256
        
        # Debug: print original crop_face shape
        print(f"DEBUG: crop_face shape before resize: {crop_face.shape}")
        
        crop_face_resized = cv2.resize(
            crop_face, 
            (generator_input_size, generator_input_size),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Debug: verify resized shape
        print(f"DEBUG: crop_face_resized shape: {crop_face_resized.shape}")
        assert crop_face_resized.shape == (256, 256, 3), f"Resize failed: {crop_face_resized.shape}"
        
        # Transform to torch tensor - match transform_target_to_torch exactly
        # This ensures exact same preprocessing and normalization as working inference code
        # Input: (H, W, C) = (256, 256, 3) numpy array
        crop_face_batch = np.expand_dims(crop_face_resized, axis=0)  # Add batch dim: (1, 256, 256, 3)
        
        print(f"DEBUG: crop_face_batch shape: {crop_face_batch.shape}")
        
        # Convert to torch tensor and normalize (same as transform_target_to_torch)
        target_tensor = torch.from_numpy(crop_face_batch.copy()).cuda()
        target_tensor = target_tensor[:, :, :, [2, 1, 0]] / 255.0  # BGR to RGB, normalize to [0, 1]
        
        # Always use FP32 for stability (FP16 causes dimension mismatch errors)
        target_tensor = target_tensor.float()
        
        target_tensor = (target_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        target_tensor = target_tensor.permute(0, 3, 1, 2)  # (1, 3, 256, 256)
        
        # Verify final shape and print
        print(f"DEBUG: target_tensor shape before G: {target_tensor.shape}")
        assert target_tensor.shape == (1, 3, generator_input_size, generator_input_size), \
            f"Expected shape (1, 3, {generator_input_size}, {generator_input_size}), got {target_tensor.shape}"
        
        # Prepare source embed - always use FP32 for stability
        source_embed_half = source_embed.float()
        
        print(f"DEBUG: source_embed_half shape: {source_embed_half.shape}")
        print(f"DEBUG: Generator dtype: {next(G.parameters()).dtype}")
        
        generator_start = time.time()
        
        # Run generator (single frame)
        print("DEBUG: Calling faceshifter_batch...")
        swapped_face = faceshifter_batch(source_embed_half, target_tensor, G)
        print("DEBUG: faceshifter_batch completed successfully")
        generator_time = (time.time() - generator_start) * 1000
        
        # Convert to numpy (faceshifter_batch already returns uint8 BGR format)
        # swapped_face shape is (1, H, W, C) from faceshifter_batch
        swapped_face_np = swapped_face[0]  # Already uint8 BGR format from faceshifter_batch
        
        # Resize to crop size for blending (generator outputs 256x256, resize to original crop_size)
        swapped_face_resized = cv2.resize(swapped_face_np, (crop_size, crop_size))
        
        # Get landmarks for blending
        landmarks = handler.get_without_detection_without_transform(swapped_face_resized)
        landmarks_tgt = handler.get_without_detection_without_transform(crop_face)
        
        # Generate mask
        mask, _ = face_mask_static(crop_face, landmarks, landmarks_tgt, None)
        
        # Fallback mask if needed
        if mask.max() == 0:
            h, w = mask.shape
            center = (w // 2, h // 2)
            radius = min(w, h) // 2 - 10
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            mask = (dist_from_center <= radius).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (15, 15), 10)
        
        # Inverse transform
        mat_rev = cv2.invertAffineTransform(M)
        
        # Warp swapped face and mask back to original frame
        swap_warped = cv2.warpAffine(
            swapped_face_resized, 
            mat_rev, 
            (frame.shape[1], frame.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )
        mask_warped = cv2.warpAffine(
            mask, 
            mat_rev, 
            (frame.shape[1], frame.shape[0])
        )
        
        # Blend
        mask_3d = mask_warped[:, :, np.newaxis]
        result = (swap_warped * mask_3d + frame * (1 - mask_3d)).astype(np.uint8)
        
        return result, detection_time, generator_time
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0


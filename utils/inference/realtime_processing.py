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
    half: bool = True
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
        
        # Get target embedding
        target_norm = normalize_and_torch(crop_face)
        target_embed = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear', align_corners=True))
        
        # Normalize and prepare for generator
        target_tensor = torch.from_numpy(crop_face.copy()).cuda()
        target_tensor = target_tensor[:, :, [2, 1, 0]] / 255.0
        
        if half:
            target_tensor = target_tensor.half()
        else:
            target_tensor = target_tensor.float()
        
        target_tensor = (target_tensor - 0.5) / 0.5
        target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Prepare source embed
        if half:
            source_embed_half = source_embed.half()
        else:
            source_embed_half = source_embed.float()
        
        generator_start = time.time()
        
        # Run generator (single frame)
        swapped_face = faceshifter_batch(source_embed_half, target_tensor, G)
        generator_time = (time.time() - generator_start) * 1000
        
        # Convert to numpy
        swapped_face_np = swapped_face[0].cpu().numpy().transpose(1, 2, 0)
        swapped_face_np = ((swapped_face_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        
        # Resize to crop size
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


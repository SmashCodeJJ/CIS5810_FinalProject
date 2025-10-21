import base64
from io import BytesIO
from typing import Callable, List

import numpy as np
import torch
import cv2
from .masks import face_mask_static 
from matplotlib import pyplot as plt
from insightface.utils import face_align


def crop_face(image_full: np.ndarray, app: Callable, crop_size: int) -> np.ndarray:
    """
    Crop face from image and resize
    """
    kps = app.get(image_full, crop_size)
    if not kps or len(kps) == 0:
        print(f"Warning: No face keypoints detected. Image shape: {image_full.shape}")
        return []
    
    # Get first keypoint and validate
    first_kp = kps[0]
    if first_kp is None:
        return []
    
    try:
        # estimate_norm expects (kps, image_size) where image_size is an integer
        # It returns (M, pose_index) where M should be shape (2, 3)
        result = face_align.estimate_norm(first_kp, crop_size)
        
        # Handle different return formats
        if isinstance(result, tuple):
            M, pose_index = result
        else:
            M = result
        
        if M is None:
            return []
        
        # Ensure M has correct shape (2, 3)
        if M.shape != (2, 3):
            print(f"Warning: M has incorrect shape {M.shape}, expected (2, 3)")
            if M.size == 6:
                # If it's a flat array or (3, 2), try to get correct shape
                if M.shape == (3, 2):
                    M = M.T  # Transpose
                    print(f"Debug: Transposed M to {M.shape}")
                elif M.shape == (6,) or M.shape == (1, 6):
                    M = M.reshape(2, 3)
                    print(f"Debug: Reshaped flat array to {M.shape}")
                else:
                    print(f"Debug: Unhandled M shape: {M.shape}")
                    return []
            else:
                print(f"Debug: Cannot fix M with size {M.size} (expected 6)")
                return []
        
        # Convert to float32 if needed
        if M.dtype != np.float32:
            M = M.astype(np.float32)
        
        align_img = cv2.warpAffine(image_full, M, (crop_size, crop_size), borderValue=0.0)         
        return [align_img]
    except Exception as e:
        print(f"Warning: Error during face alignment: {e}")
        import traceback
        traceback.print_exc()
        return []


def normalize_and_torch(image: np.ndarray) -> torch.tensor:
    """
    Normalize image and transform to torch
    """
    image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    if image.max() > 1.:
        image = image/255.
    
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image


def normalize_and_torch_batch(frames: np.ndarray) -> torch.tensor:
    """
    Normalize batch images and transform to torch
    """
    batch_frames = torch.from_numpy(frames.copy()).cuda()
    if batch_frames.max() > 1.:
        batch_frames = batch_frames/255.
    
    batch_frames = batch_frames.permute(0, 3, 1, 2)
    batch_frames = (batch_frames - 0.5)/0.5

    return batch_frames


def get_final_image(final_frames: List[np.ndarray],
                    crop_frames: List[np.ndarray],
                    full_frame: np.ndarray,
                    tfm_arrays: List[np.ndarray],
                    handler) -> None:
    """
    Create final video from frames
    """
    print(f"[DEBUG get_final_image] final_frames: {len(final_frames) if final_frames else 0}, crop_frames: {len(crop_frames) if crop_frames else 0}")
    if final_frames:
        print(f"[DEBUG get_final_image] final_frames[0] length: {len(final_frames[0]) if final_frames[0] else 0}")
    
    final = full_frame.copy()
    
    # Check if we have any frames to process
    if not final_frames or len(final_frames) == 0:
        print("Warning: No faces detected in target image, returning original image")
        return final
    
    # Check if first list is empty
    if not final_frames[0] or len(final_frames[0]) == 0:
        print("Warning: No valid face swap results, returning original image")
        return final
    
    print(f"[DEBUG get_final_image] Processing {len(final_frames)} faces for blending")
    params = [None for i in range(len(final_frames))]
    
    for i in range(len(final_frames)):
        print(f"[DEBUG blend loop] Processing face {i}")
        # Skip if no frame data
        if not final_frames[i] or len(final_frames[i]) == 0:
            print(f"[DEBUG blend loop] Skipping {i}: no final_frames")
            continue
        if not crop_frames[i] or len(crop_frames[i]) == 0:
            print(f"[DEBUG blend loop] Skipping {i}: no crop_frames")
            continue
        
        print(f"[DEBUG blend loop] final_frames[{i}][0] shape: {final_frames[i][0].shape}")
        print(f"[DEBUG blend loop] crop_frames[{i}][0] shape: {crop_frames[i][0].shape}")
        print(f"[DEBUG blend loop] tfm_arrays[{i}][0] shape: {tfm_arrays[i][0].shape}")
            
        frame = cv2.resize(final_frames[i][0], (224, 224))
        print(f"[DEBUG blend loop] Resized frame shape: {frame.shape}, min/max: {frame.min()}/{frame.max()}")
        
        landmarks = handler.get_without_detection_without_transform(frame)     
        landmarks_tgt = handler.get_without_detection_without_transform(crop_frames[i][0])
        print(f"[DEBUG blend loop] Got landmarks")

        mask, _ = face_mask_static(crop_frames[i][0], landmarks, landmarks_tgt, params[i])
        print(f"[DEBUG blend loop] Mask shape: {mask.shape}, min/max: {mask.min():.2f}/{mask.max():.2f}")
        
        # FIX: If mask is all zeros, create a simple circular mask
        if mask.max() == 0:
            print(f"[DEBUG blend loop] Mask is all zeros! Creating fallback circular mask")
            h, w = mask.shape
            center = (w // 2, h // 2)
            radius = min(w, h) // 2 - 10
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            mask = (dist_from_center <= radius).astype(np.float32)
            # Smooth the edges
            mask = cv2.GaussianBlur(mask, (15, 15), 10)
            print(f"[DEBUG blend loop] Created fallback mask, min/max: {mask.min():.2f}/{mask.max():.2f}")
        
        mat_rev = cv2.invertAffineTransform(tfm_arrays[i][0])
        print(f"[DEBUG blend loop] Inverted transform matrix:\n{mat_rev}")

        swap_t = cv2.warpAffine(frame, mat_rev, (full_frame.shape[1], full_frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        print(f"[DEBUG blend loop] Warped swap_t shape: {swap_t.shape}, min/max: {swap_t.min()}/{swap_t.max()}")
        
        mask_t = cv2.warpAffine(mask, mat_rev, (full_frame.shape[1], full_frame.shape[0]))
        mask_t = np.expand_dims(mask_t, 2)
        print(f"[DEBUG blend loop] Warped mask_t shape: {mask_t.shape}, min/max: {mask_t.min():.2f}/{mask_t.max():.2f}")

        final = mask_t*swap_t + (1-mask_t)*final
        print(f"[DEBUG blend loop] Blended! Final min/max: {final.min()}/{final.max()}")
    final = np.array(final, dtype='uint8')
    return final


def show_images(images: List[np.ndarray], 
                titles=None, 
                figsize=(20, 5), 
                fontsize=15):
    if titles:
        assert len(titles) == len(images), "Amount of images should be the same as the amount of titles"
    
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for idx, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image[:, :, ::-1])
        if titles:
            ax.set_title(titles[idx], fontsize=fontsize)
        ax.axis("off")

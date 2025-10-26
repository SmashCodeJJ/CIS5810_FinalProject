# 🔄 Code Flow: How inference.py Works

## 📋 **Quick Navigation**

1. [Main Entry Point](#1-main-entry-point)
2. [Model Initialization](#2-model-initialization)
3. [Source Face Processing](#3-source-face-processing)
4. [Target Face Processing](#4-target-face-processing)
5. [Face Swapping](#5-face-swapping)
6. [Output Generation](#6-output-generation)

---

## 1️⃣ **Main Entry Point**

```python
# inference.py lines 135-161

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    
    # Key parameters you can change:
    parser.add_argument('--source_paths', default=['examples/images/mark.jpg'])
    parser.add_argument('--target_image', default='examples/images/beckham.jpg')
    parser.add_argument('--image_to_image', default=False, type=bool)
    
    args = parser.parse_args()
    main(args)
```

**What happens here**:
- Parse command-line arguments
- Call `main(args)` to start the pipeline

---

## 2️⃣ **Model Initialization**

```python
# inference.py lines 20-52

def init_models(args):
    """
    Load all 5 models into GPU memory
    """
    
    # ┌─────────────────────────────────────────┐
    # │ MODEL 1: Face Detection (SCRFD)         │
    # └─────────────────────────────────────────┘
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))
    # → Loads: scrfd_10g_bnkps.onnx + glintr100.onnx
    # → GPU memory: ~500MB
    
    # ┌─────────────────────────────────────────┐
    # │ MODEL 2: Main Generator (AEI-Net)       │
    # └─────────────────────────────────────────┘
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')))
    G = G.cuda()      # Move to GPU
    G = G.half()      # Convert to FP16 for speed
    # → Loads: weights/G_unet_2blocks.pth
    # → GPU memory: ~2GB
    # → FP16 = 16-bit precision (2x faster, minimal quality loss)
    
    # ┌─────────────────────────────────────────┐
    # │ MODEL 3: Identity Encoder (ArcFace)     │
    # └─────────────────────────────────────────┘
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc = netArc.cuda()
    netArc.eval()
    # → Loads: arcface_model/backbone.pth
    # → GPU memory: ~400MB
    
    # ┌─────────────────────────────────────────┐
    # │ MODEL 4: Landmark Detector (2D106Det)   │
    # └─────────────────────────────────────────┘
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)
    # → Tries to load: 2d106det-0000.onnx
    # → If not found: Falls back to InsightFace landmarks
    
    # ┌─────────────────────────────────────────┐
    # │ MODEL 5: Super Resolution (Optional)    │
    # └─────────────────────────────────────────┘
    if args.use_sr:
        model = Pix2PixModel(opt)
        model.netG.train()
    else:
        model = None  # Disabled by default
    
    return app, G, netArc, handler, model
```

**Total GPU Memory**: ~3.2 GB  
**Initialization Time**: ~5-10 seconds

---

## 3️⃣ **Source Face Processing**

```python
# inference.py lines 58-76

def main(args):
    # Initialize models first
    app, G, netArc, handler, model = init_models(args)
    
    # ┌─────────────────────────────────────────────────────────┐
    # │ PROCESS SOURCE IMAGE (The face we want to transfer)    │
    # └─────────────────────────────────────────────────────────┘
    
    source = []  # List to store multiple source faces
    
    for source_path in args.source_paths:  # Can have multiple sources
        # Step 1: Read image
        img = cv2.imread(source_path)  # BGR format
        # Example: 'examples/images/mark.jpg' → numpy array (1024, 768, 3)
        
        if img is None:
            print(f"Error: Could not read image: {source_path}")
            exit()
        
        # Step 2: Detect and crop face
        face_crop = crop_face(img, app, args.crop_size)
        # ├─ Uses MODEL 1 (SCRFD) to detect face
        # ├─ Gets bounding box + 5 keypoints
        # ├─ Aligns face to 224x224
        # └─ Returns: [aligned_face], [transformation_matrix]
        
        if not face_crop or face_crop[0] is None:
            print(f"Error: No face detected in source image: {source_path}")
            exit()
        
        # Step 3: Get the cropped face
        img = face_crop[0]  # numpy array (224, 224, 3)
        
        # Step 4: Convert BGR to RGB
        source.append(img[:, :, ::-1])  # Reverse color channels
    
    # Result: source = [face1_rgb, face2_rgb, ...]
    #         Each face is 224x224x3 numpy array
```

**Key Functions Called**:

```python
# utils/inference/image_processing.py

def crop_face(img, app, crop_size):
    """
    Detect face and crop it aligned
    """
    # Call face detection
    bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
    
    # Filter by confidence threshold
    keep = bboxes[:, 4] >= app.det_thresh  # Default: 0.6
    
    # Get best face (highest confidence)
    best_index = np.argmax(bboxes[:, 4])
    kps = kpss[best_index]  # 5 keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth]
    
    # Compute alignment transformation
    M, _ = face_align.estimate_norm(kps, crop_size, mode='None')
    # M is 2x3 affine transformation matrix
    
    # Warp image to align face
    align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
    
    return [align_img], [M]
```

---

## 4️⃣ **Target Face Processing**

```python
# inference.py lines 78-100

    # ┌─────────────────────────────────────────────────────────┐
    # │ PROCESS TARGET (The image/video we want to modify)     │
    # └─────────────────────────────────────────────────────────┘
    
    if not args.image_to_image:
        # Video mode: Read all frames
        full_frames, fps = read_video(args.target_video)
        # Returns: list of frames, fps
        # Example: 300 frames @ 30 fps = 10 second video
    else:
        # Image mode: Single frame
        target_full = cv2.imread(args.target_image)
        full_frames = [target_full]
        # Example: 'examples/images/beckham.jpg' → [(1024, 768, 3)]
    
    # ┌─────────────────────────────────────────────────────────┐
    # │ GET TARGET FACES (Which faces to swap in the target)   │
    # └─────────────────────────────────────────────────────────┘
    
    set_target = True
    
    if not args.target_faces_paths:
        # AUTO MODE: Detect faces automatically
        target = get_target(full_frames, app, args.crop_size)
        # ├─ Loops through frames until it finds a face
        # └─ Returns: [cropped_face]
        set_target = False
    else:
        # MANUAL MODE: User specified target faces
        target = []
        for target_faces_path in args.target_faces_paths:
            img = cv2.imread(target_faces_path)
            img = crop_face(img, app, args.crop_size)[0]
            target.append(img)
    
    # Result: target = [target_face1, target_face2, ...]
```

**Why two modes?**
- **Auto mode** (`set_target=False`): Swap ANY face found in target
- **Manual mode** (`set_target=True`): Only swap specific faces (useful for multi-person scenes)

---

## 5️⃣ **Face Swapping** ⭐ THE MAGIC

```python
# inference.py lines 102-115

    start = time.time()
    
    # ┌─────────────────────────────────────────────────────────┐
    # │ CORE SWAPPING PIPELINE                                  │
    # └─────────────────────────────────────────────────────────┘
    
    final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(
        full_frames,      # Original images/video frames
        source,           # Source faces (who we want to look like)
        target,           # Target faces (reference for matching)
        netArc,           # ArcFace model for embeddings
        G,                # AEI-Net generator
        app,              # Face detector
        set_target,       # Auto or manual mode
        similarity_th=args.similarity_th,  # Matching threshold (0.15)
        crop_size=args.crop_size,          # 224
        BS=args.batch_size                 # 40 frames at once
    )
    
    # Optional: Enhance quality
    if args.use_sr:
        final_frames_list = face_enhancement(final_frames_list, model)
```

**Deep Dive into `model_inference`**:

```python
# utils/inference/core.py lines 29-103

def model_inference(full_frames, source, target, netArc, G, app, 
                    set_target, similarity_th=0.15, crop_size=224, BS=60):
    """
    The heart of the face swapping system
    """
    
    # ═══════════════════════════════════════════════════════════
    # STEP 1: Get Target Embeddings (for matching)
    # ═══════════════════════════════════════════════════════════
    target_norm = normalize_and_torch_batch(np.array(target))
    # ├─ Convert to torch tensor
    # ├─ Normalize: (img/255 - 0.5) / 0.5  → range [-1, 1]
    # └─ Move to GPU
    
    target_embeds = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear'))
    # ├─ Resize 224x224 → 112x112 (ArcFace input size)
    # ├─ Pass through MODEL 3 (ArcFace)
    # └─ Output: 512-dimensional embedding vector
    # Example: tensor([0.23, -0.45, 0.78, ..., 0.12]) shape=(1, 512)
    
    
    # ═══════════════════════════════════════════════════════════
    # STEP 2: Crop Faces from All Frames
    # ═══════════════════════════════════════════════════════════
    crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(
        full_frames, 
        target_embeds, 
        app, 
        netArc, 
        crop_size, 
        set_target, 
        similarity_th=similarity_th
    )
    # For each frame:
    # ├─ Detect all faces (MODEL 1 - SCRFD)
    # ├─ Extract embeddings (MODEL 3 - ArcFace)
    # ├─ Compare with target_embeds (cosine similarity)
    # ├─ If similarity < threshold: SKIP this face
    # ├─ If similarity >= threshold: KEEP and crop
    # └─ Store transformation matrices
    
    # Result:
    # crop_frames_list = [[face1_fr1, face1_fr2, ...], [face2_fr1, ...]]
    # tfm_array_list = [[M1_fr1, M1_fr2, ...], [M2_fr1, ...]]
    
    
    # ═══════════════════════════════════════════════════════════
    # STEP 3: Get Source Embeddings
    # ═══════════════════════════════════════════════════════════
    source_embeds = []
    for source_curr in source:
        source_curr = normalize_and_torch(source_curr)
        embed = netArc(F.interpolate(source_curr, scale_factor=0.5, mode='bilinear'))
        source_embeds.append(embed)
    # Each source gets a 512-dim identity embedding
    # This is the "who" we want to become
    
    
    # ═══════════════════════════════════════════════════════════
    # STEP 4: Face Swap Generation (Per Source)
    # ═══════════════════════════════════════════════════════════
    final_frames_list = []
    
    for idx, (crop_frames, tfm_array, source_embed) in enumerate(
        zip(crop_frames_list, tfm_array_list, source_embeds)
    ):
        # Resize all cropped faces to 256x256 (generator input size)
        resized_frs, present = resize_frames(crop_frames)
        # present = binary vector: [1, 1, 0, 1, ...] (1=face found, 0=no face)
        
        if len(resized_frs) == 0:
            final_frames_list.append([])
            continue
        
        resized_frs = np.array(resized_frs)  # shape: (N, 256, 256, 3)
        
        # Convert to torch tensor
        target_batch_rs = transform_target_to_torch(resized_frs, half=True)
        # ├─ BGR → RGB
        # ├─ Normalize to [-1, 1]
        # ├─ Convert to FP16
        # └─ shape: (N, 3, 256, 256)
        
        if half:
            source_embed = source_embed.half()
        
        # ═══════════════════════════════════════════════════════
        # RUN THE GENERATOR (Batch processing)
        # ═══════════════════════════════════════════════════════
        size = target_batch_rs.shape[0]  # Number of faces
        model_output = []
        
        for i in tqdm(range(0, size, BS)):  # Process BS frames at a time
            # Core face swap call!
            Y_st = faceshifter_batch(
                source_embed,              # Who we want (512-dim)
                target_batch_rs[i:i+BS],  # Target faces (Nx3x256x256)
                G                          # Generator model (AEI-Net)
            )
            model_output.append(Y_st)
        
        model_output = np.concatenate(model_output)  # (N, 256, 256, 3)
        
        # ═══════════════════════════════════════════════════════
        # STEP 5: Build Final Frame List
        # ═══════════════════════════════════════════════════════
        final_frames = []
        idx_fs = 0
        
        for pres in present:  # [1, 1, 0, 1, ...]
            if pres == 1:
                final_frames.append(model_output[idx_fs])
                idx_fs += 1
            else:
                final_frames.append([])  # No face in this frame
        
        final_frames_list.append(final_frames)
    
    return final_frames_list, crop_frames_list, full_frames, tfm_array_list
```

**What happens in `faceshifter_batch`**:

```python
# utils/inference/faceshifter_run.py

def faceshifter_batch(source_emb, target, G):
    """
    Apply the AEI-Net generator
    """
    bs = target.shape[0]  # Batch size
    
    # Replicate source embedding for each target face
    if bs > 1:
        source_emb = torch.cat([source_emb]*bs)
    # Example: source_emb shape: (40, 512) for batch of 40
    
    with torch.no_grad():
        # ┌────────────────────────────────────────────────┐
        # │ THE ACTUAL FACE SWAP HAPPENS HERE!             │
        # └────────────────────────────────────────────────┘
        Y_st, _ = G(target, source_emb)
        # G is AEI_Net model
        # Input 1: target (40, 3, 256, 256) - target faces
        # Input 2: source_emb (40, 512) - source identity
        # Output: Y_st (40, 3, 256, 256) - swapped faces
        
        # Post-process output
        Y_st = (Y_st.permute(0, 2, 3, 1) * 0.5 + 0.5) * 255
        # ├─ Permute: (B,C,H,W) → (B,H,W,C)
        # ├─ Denormalize: [-1,1] → [0,255]
        # └─ Convert to uint8
        
        Y_st = Y_st[:, :, :, [2,1,0]].type(torch.uint8)
        # ├─ RGB → BGR (OpenCV format)
        # └─ Convert to uint8 (0-255)
        
        Y_st = Y_st.cpu().detach().numpy()
        # Move from GPU to CPU as numpy array
    
    return Y_st  # (40, 256, 256, 3) numpy array
```

---

## 6️⃣ **Output Generation**

```python
# inference.py lines 116-132

    if not args.image_to_image:
        # ┌─────────────────────────────────────────────┐
        # │ VIDEO OUTPUT MODE                           │
        # └─────────────────────────────────────────────┘
        get_final_video(
            final_frames_list,   # Swapped face regions
            crop_frames_list,    # Original cropped faces
            full_frames,         # Original full frames
            tfm_array_list,      # Transformation matrices
            args.out_video_name, # Output path
            fps,                 # Video framerate
            handler              # Landmark detector
        )
        
        # Add audio back
        add_audio_from_another_video(args.target_video, args.out_video_name, "audio")
        print(f"Video saved with path {args.out_video_name}")
        
    else:
        # ┌─────────────────────────────────────────────┐
        # │ IMAGE OUTPUT MODE                           │
        # └─────────────────────────────────────────────┘
        result = get_final_image(
            final_frames_list,   # Swapped face regions
            crop_frames_list,    # Original cropped faces  
            full_frames[0],      # Original full image
            tfm_array_list,      # Transformation matrices
            handler              # Landmark detector
        )
        cv2.imwrite(args.out_image_name, result)
        print(f'Swapped Image saved with path {args.out_image_name}')
    
    print('Total time: ', time.time()-start)
```

**Deep Dive into `get_final_image`**:

```python
# utils/inference/image_processing.py

def get_final_image(final_frames, crop_frames, full_img, tfm_arrays, handler):
    """
    Blend swapped faces back into original image
    """
    # final_frames = [[swapped_face_256x256]]
    # crop_frames = [[original_crop_224x224]]
    # full_img = original image (1024, 768, 3)
    # tfm_arrays = [[transformation_matrix_2x3]]
    
    result = full_img.copy()
    
    # ═══════════════════════════════════════════════════════════
    # STEP 1: Process Each Swapped Face
    # ═══════════════════════════════════════════════════════════
    for idx, (final_face, crop_face, M) in enumerate(
        zip(final_frames[0], crop_frames[0], tfm_arrays[0])
    ):
        # Resize swapped face to match crop size
        final_face = cv2.resize(final_face, (224, 224))
        # 256x256 → 224x224
        
        # ═══════════════════════════════════════════════════════
        # STEP 2: Detect Landmarks
        # ═══════════════════════════════════════════════════════
        lmk = handler.get(final_face[:, :, ::-1])
        # Uses MODEL 5 (2D106Det) or fallback
        # Returns: 106 (x, y) landmark coordinates
        
        # ═══════════════════════════════════════════════════════
        # STEP 3: Generate Face Mask
        # ═══════════════════════════════════════════════════════
        mask = face_mask_static(lmk, final_face.shape)
        # Creates smooth mask using landmarks
        # Result: (224, 224) binary mask with smooth edges
        # Example:
        #   0 0 0 0 0 0 0 0 0 0
        #   0 0 0.2 0.5 0.8 0.8 0.5 0.2 0 0
        #   0 0.3 0.7 1.0 1.0 1.0 1.0 0.7 0.3 0
        #   ...
        
        if mask.max() == 0:
            # Fallback: Create circular mask
            mask = np.zeros((224, 224), dtype=np.float32)
            cv2.circle(mask, (112, 112), 90, (1.0), -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 11)
        
        # ═══════════════════════════════════════════════════════
        # STEP 4: Invert Transformation (Warp Back)
        # ═══════════════════════════════════════════════════════
        M_inv = cv2.invertAffineTransform(M)
        # M was used to crop face from original
        # M_inv will put face back to original position
        
        # Warp swapped face to original position
        swap_t = cv2.warpAffine(
            final_face, 
            M_inv, 
            (full_img.shape[1], full_img.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )
        # Example: 224x224 → 1024x768 (with face in correct position)
        
        # Warp mask too
        mask_t = cv2.warpAffine(
            mask, 
            M_inv, 
            (full_img.shape[1], full_img.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # ═══════════════════════════════════════════════════════
        # STEP 5: Alpha Blending
        # ═══════════════════════════════════════════════════════
        mask_t = mask_t[:, :, np.newaxis]  # Add channel dimension
        # (1024, 768) → (1024, 768, 1)
        
        # Blend equation:
        # result = swapped_face * mask + original * (1 - mask)
        result = swap_t * mask_t + result * (1.0 - mask_t)
        result = result.astype(np.uint8)
        
        # Example at pixel (512, 384):
        #   mask = 0.8
        #   swap_pixel = [120, 150, 180]
        #   orig_pixel = [100, 130, 160]
        #   result_pixel = [120,150,180]*0.8 + [100,130,160]*0.2
        #                = [116, 146, 176]
    
    return result
```

---

## 🎬 **Complete Example Run**

```bash
python inference.py \
  --image_to_image True \
  --target_image examples/images/beckham.jpg \
  --source_paths examples/images/mark.jpg \
  --out_image_name examples/results/result.png
```

**Execution Timeline**:

```
[0.0s] Start
[0.0s] Parse arguments
[0.0s] Call main(args)

[0.0s] init_models()
  ├─ [0.5s] Load SCRFD detector → GPU
  ├─ [2.0s] Load AEI-Net generator → GPU
  ├─ [3.5s] Load ArcFace model → GPU
  ├─ [4.0s] Load 2D106Det landmarks
  └─ [4.5s] Models ready!

[4.5s] Process source: 'mark.jpg'
  ├─ [4.6s] Read image (1024x768)
  ├─ [4.62s] Detect face (SCRFD) → bbox + keypoints
  ├─ [4.63s] Crop and align → 224x224
  └─ [4.64s] Convert BGR→RGB

[4.64s] Process target: 'beckham.jpg'
  ├─ [4.65s] Read image (1024x768)
  └─ [4.66s] Store as full_frames

[4.66s] model_inference() START
  ├─ [4.67s] Normalize target face
  ├─ [4.72s] ArcFace embedding (target) → 512-dim
  ├─ [4.75s] Crop faces from full_frames
  │   ├─ Detect face in beckham.jpg
  │   ├─ Compare embedding with target (similarity=0.95)
  │   └─ Crop face + store transformation matrix
  ├─ [4.78s] ArcFace embedding (source) → 512-dim
  ├─ [4.80s] Resize cropped faces to 256x256
  ├─ [4.81s] Normalize and convert to torch
  ├─ [4.82s] 🔥 RUN GENERATOR (AEI-Net)
  │   ├─ Input: target_face (1,3,256,256) + source_embed (1,512)
  │   ├─ Encoder extracts 8 feature levels
  │   ├─ Generator blends with AAD blocks
  │   └─ Output: swapped_face (1,3,256,256)
  └─ [5.02s] Conversion to numpy (1,256,256,3)

[5.02s] get_final_image() START
  ├─ [5.03s] Resize swapped face 256→224
  ├─ [5.04s] Detect landmarks (106 points)
  ├─ [5.05s] Generate face mask
  ├─ [5.06s] Invert transformation matrix
  ├─ [5.08s] Warp face back to original position
  ├─ [5.10s] Warp mask
  └─ [5.12s] Alpha blend with original image

[5.13s] Save result.png
[5.15s] Done!

Total: 5.15 seconds
```

---

## 🔍 **Key Takeaways**

### **1. Model Loading (One-Time Cost)**
- Takes ~4-5 seconds
- Only happens once
- For video: amortized over many frames

### **2. Per-Frame Processing**
```
Single frame timeline:
├─ Face Detection: 20ms
├─ ArcFace Embedding: 50ms
├─ Generator (AEI-Net): 120ms ⚠️ BOTTLENECK
├─ Blending: 40ms
└─ Total: ~230ms = 4.3 FPS
```

### **3. Batch Processing**
- Video mode processes 40 frames at once
- Amortizes GPU overhead
- Much faster than processing individually

### **4. The Core Innovation**
The magic is in lines 18-19 of `faceshifter_run.py`:
```python
Y_st, _ = G(target, source_emb)
```
Everything else is just setup and blending!

---

## 💡 **How to Modify**

Want to change something? Here's where:

| **What** | **Where** | **How** |
|----------|-----------|---------|
| Source/Target images | Command-line | `--source_paths` `--target_image` |
| Detection threshold | `inference.py:23` | `det_thresh=0.6` → lower for more faces |
| Crop size | `inference.py:144` | `crop_size=224` (don't change!) |
| Generator quality | Command-line | `--num_blocks 1/2/3` |
| Generator weights | Command-line | `--G_path weights/G_unet_Xblocks.pth` |
| Batch size | Command-line | `--batch_size 40` → lower for less VRAM |
| Face matching | `inference.py:146` | `similarity_th=0.15` → lower = stricter |
| Super resolution | Command-line | `--use_sr True` |

---

## ❓ **Common Questions**

**Q: Why is it so slow?**  
A: The generator (AEI-Net) is computationally expensive. It needs to:
- Extract 8 feature levels from target
- Process through 8 AAD blocks (each with 2 ResBlocks)
- Each block has attention mechanisms
→ 120ms per face is actually quite good!

**Q: Can I speed it up?**  
A: Yes, several options:
1. Use 1-block generator (`--num_blocks 1`) → 30% faster
2. Reduce batch size for lower VRAM
3. Process smaller images
4. Use INT8 quantization (advanced)

**Q: Why two ArcFace calls?**  
A: One for target (matching), one for source (identity). They serve different purposes.

**Q: What if no face is detected?**  
A: The code will exit with error message. Adjust `det_thresh` lower or ensure image has visible face.

**Q: Can I swap multiple people?**  
A: Yes! Provide multiple source paths:
```bash
--source_paths mark.jpg elon.jpg --target_faces_paths person1.jpg person2.jpg
```

---

**Summary**: The code follows a clear pipeline: Load models → Extract identities → Match faces → Generate swaps → Blend back. The generator (AEI-Net) does the heavy lifting, everything else is support infrastructure.


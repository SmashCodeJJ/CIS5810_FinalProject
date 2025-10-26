# 🚀 Quick Reference: Models & Pipeline

## 📌 **TL;DR - What Does Each Model Do?**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUR FACE SWAP PIPELINE                          │
├──────────┬──────────────┬─────────────┬───────────┬────────────────┤
│          │              │             │           │                │
│  MODEL 1 │   MODEL 2    │   MODEL 3   │  MODEL 4  │   MODEL 5      │
│  SCRFD   │  GLintr100   │   ArcFace   │  AEI-Net  │  2D106Det      │
│          │              │             │           │  (Optional)    │
├──────────┼──────────────┼─────────────┼───────────┼────────────────┤
│          │              │             │           │                │
│ "Where   │ "Is this the │ "Extract    │ "Swap the │ "Where are     │
│  are     │  target      │  identity   │  face!"   │  facial        │
│  faces?" │  person?"    │  features"  │           │  landmarks?"   │
│          │              │             │           │                │
├──────────┼──────────────┼─────────────┼───────────┼────────────────┤
│          │              │             │           │                │
│ Input:   │ Input:       │ Input:      │ Input:    │ Input:         │
│ Any img  │ 112x112      │ 112x112     │ 256x256 + │ 192x192        │
│          │              │             │ 512-dim   │                │
│          │              │             │           │                │
│ Output:  │ Output:      │ Output:     │ Output:   │ Output:        │
│ Bboxes + │ 512-dim      │ 512-dim     │ Swapped   │ 106 landmarks  │
│ 5 points │ embedding    │ embedding   │ face      │                │
│          │              │             │ 256x256   │                │
├──────────┼──────────────┼─────────────┼───────────┼────────────────┤
│          │              │             │           │                │
│ Speed:   │ Speed:       │ Speed:      │ Speed:    │ Speed:         │
│ ~20ms    │ ~15ms        │ ~50ms       │ ~120ms ⚠️ │ ~10ms          │
│          │              │             │           │                │
│ VRAM:    │ VRAM:        │ VRAM:       │ VRAM:     │ VRAM:          │
│ 300MB    │ 200MB        │ 400MB       │ 2000MB ⚠️ │ 150MB          │
│          │              │             │           │                │
└──────────┴──────────────┴─────────────┴───────────┴────────────────┘

                     Total: ~230ms/frame = 4.3 FPS
                     Total VRAM: ~3.2 GB / 16 GB
```

---

## 🎯 **Visual Pipeline Flow**

```
SOURCE IMAGE                        TARGET IMAGE
(mark.jpg)                         (beckham.jpg)
     │                                    │
     │                                    │
     ▼                                    ▼
┌─────────┐                          ┌─────────┐
│ MODEL 1 │ SCRFD                    │ MODEL 1 │ SCRFD
│ Detect  │ Find face                │ Detect  │ Find face
└────┬────┘                          └────┬────┘
     │                                    │
     │ bbox + 5 keypoints                │ bbox + 5 keypoints
     ▼                                    ▼
┌─────────┐                          ┌─────────┐
│ Crop &  │                          │ Crop &  │
│ Align   │ 224x224                  │ Align   │ 224x224
└────┬────┘                          └────┬────┘
     │                                    │
     │                                    │
     ▼                                    ▼
┌─────────┐                          ┌─────────┐
│ MODEL 3 │ ArcFace                  │ MODEL 3 │ ArcFace
│ Extract │                          │ Extract │
│Identity │                          │Identity │
└────┬────┘                          └────┬────┘
     │                                    │
     │ 512-dim                            │ 512-dim
     │ "Who is Mark?"                     │ "Who is Beckham?"
     │                                    │
     └────────────┬───────────────────────┘
                  │
                  │ Both embeddings
                  ▼
         ┌────────────────────┐
         │ MODEL 2 (GLintr100)│
         │ Compare Embeddings │
         └────────┬───────────┘
                  │
                  │ Similarity score
                  │ (Are they the target person?)
                  ▼
         ┌────────────────────┐
         │ If similar enough: │
         │ Proceed to swap    │
         └────────┬───────────┘
                  │
                  │
                  ▼
         ┌─────────────────────────────────┐
         │     MODEL 4: AEI-Net (MAGIC!)   │
         ├─────────────────────────────────┤
         │                                 │
         │  Input 1: Target face (Beckham) │
         │           256x256, RGB          │
         │           ↓                     │
         │       ┌──────────┐              │
         │       │ Encoder  │              │
         │       │ (U-Net)  │              │
         │       └────┬─────┘              │
         │            │ 8 feature levels   │
         │            ↓                     │
         │  Input 2: Source identity (Mark)│
         │           512-dim embedding     │
         │            ↓                     │
         │       ┌──────────┐              │
         │       │   AAD    │              │
         │       │Generator │              │
         │       │(8 blocks)│              │
         │       └────┬─────┘              │
         │            │                     │
         │            ↓                     │
         │  Output: Swapped face           │
         │          (Mark's face with      │
         │           Beckham's pose)       │
         │          256x256                │
         └─────────────┬───────────────────┘
                       │
                       │ Swapped face (256x256)
                       ▼
              ┌────────────────┐
              │   MODEL 5      │
              │   2D106Det     │
              │Get 106 landmarks│
              └────────┬───────┘
                       │
                       │ 106 (x,y) points
                       ▼
              ┌────────────────┐
              │  Generate Mask │
              │  (smooth edges)│
              └────────┬───────┘
                       │
                       │ Face mask (224x224)
                       ▼
              ┌────────────────────┐
              │  Warp back to      │
              │  original position │
              │  (using matrix M)  │
              └────────┬───────────┘
                       │
                       ▼
              ┌────────────────────┐
              │   Alpha Blend      │
              │  swapped * mask +  │
              │  original * (1-mask)│
              └────────┬───────────┘
                       │
                       ▼
                  FINAL RESULT
            (Beckham with Mark's face)
```

---

## 🗂️ **File Locations**

```
sber-swap/
│
├── 📁 insightface_func/models/antelope/
│   ├── scrfd_10g_bnkps.onnx        ← MODEL 1 (Face Detection)
│   └── glintr100.onnx              ← MODEL 2 (Face Recognition)
│
├── 📁 arcface_model/
│   └── backbone.pth                ← MODEL 3 (Identity Encoder)
│
├── 📁 weights/
│   ├── G_unet_1block.pth           ← MODEL 4 option 1 (Fast)
│   ├── G_unet_2blocks.pth          ← MODEL 4 option 2 (DEFAULT)
│   └── G_unet_3blocks.pth          ← MODEL 4 option 3 (Best quality)
│
├── 📁 coordinate_reg/model/
│   └── 2d106det-0000.onnx          ← MODEL 5 (Landmarks, optional)
│
└── 📄 inference.py                 ← Main script
```

---

## ⚙️ **How to Use**

### **Basic Image Swap**
```bash
python inference.py \
  --image_to_image True \
  --target_image examples/images/beckham.jpg \
  --source_paths examples/images/mark.jpg \
  --out_image_name examples/results/result.png
```

### **Video Swap**
```bash
python inference.py \
  --image_to_image False \
  --target_video examples/videos/nggyup.mp4 \
  --source_paths examples/images/mark.jpg \
  --out_video_name examples/results/result.mp4
```

### **Multiple Source Faces**
```bash
python inference.py \
  --source_paths mark.jpg elon.jpg \
  --target_faces_paths person1.jpg person2.jpg \
  --target_video video.mp4
```

---

## 🎛️ **Key Parameters**

| Parameter | Default | What It Does | Recommendation |
|-----------|---------|--------------|----------------|
| `--G_path` | `G_unet_2blocks.pth` | Generator weights | Use 2blocks for balance |
| `--num_blocks` | `2` | AAD block count | 1=fast, 2=balanced, 3=quality |
| `--crop_size` | `224` | Face crop size | **DON'T CHANGE** |
| `--det_thresh` | `0.6` | Detection confidence | Lower = more faces detected |
| `--similarity_th` | `0.15` | Face matching threshold | Lower = stricter matching |
| `--batch_size` | `40` | Frames per batch | Lower if out of VRAM |
| `--use_sr` | `False` | Super resolution | True = better quality, slower |

---

## 🔧 **Troubleshooting**

### **Problem: "No face detected"**
```bash
# Solution 1: Lower detection threshold
--det_thresh 0.3

# Solution 2: Check if face is clearly visible
# - Front-facing preferred
# - Good lighting
# - Not too small
```

### **Problem: "CUDA out of memory"**
```bash
# Solution 1: Lower batch size
--batch_size 20

# Solution 2: Use smaller generator
--G_path weights/G_unet_1block.pth

# Solution 3: Process fewer frames at once
```

### **Problem: "Wrong person swapped"**
```bash
# Solution: Stricter matching
--similarity_th 0.08

# Or specify exact target faces
--target_faces_paths face_to_swap.jpg
```

### **Problem: "Visible seams/edges"**
```bash
# This is a limitation of the blending
# MODEL 5 (landmarks) should help
# Ensure 2d106det-0000.onnx exists
# Otherwise falls back to 5-point landmarks
```

---

## 📊 **Performance Expectations**

### **Image Mode** (Single face):
```
Model Loading:     ~5 seconds  (one-time)
Source Processing: ~0.1 seconds
Target Processing: ~0.1 seconds
Face Swapping:     ~0.2 seconds
Blending:          ~0.05 seconds
──────────────────────────────────
Total:             ~5.5 seconds
```

### **Video Mode** (100 frames, 1 face per frame):
```
Model Loading:     ~5 seconds  (one-time)
Source Processing: ~0.1 seconds
Face Detection:    ~2 seconds   (100 frames × 20ms)
Face Swapping:     ~12 seconds  (100 frames × 120ms)
Blending:          ~5 seconds   (100 frames × 50ms)
──────────────────────────────────
Total:             ~24 seconds   = 4.2 FPS
```

---

## 🚀 **Optimization Tips**

### **For Speed**:
1. Use 1-block generator (`--num_blocks 1`)
2. Increase batch size (`--batch_size 60`)
3. Skip super resolution (`--use_sr False`)
4. Lower detection resolution (modify code: `det_size=(320,320)`)

### **For Quality**:
1. Use 3-block generator (`--num_blocks 3`)
2. Enable super resolution (`--use_sr True`)
3. Stricter face matching (`--similarity_th 0.08`)
4. Higher detection resolution (keep `det_size=(640,640)`)

---

## 💡 **Common Use Cases**

### **1. Swap Your Face onto a Celebrity**
```bash
python inference.py \
  --image_to_image True \
  --source_paths my_face.jpg \
  --target_image celebrity.jpg \
  --out_image_name me_as_celeb.png
```

### **2. Face Swap in a Group Photo**
```bash
python inference.py \
  --image_to_image True \
  --source_paths new_person.jpg \
  --target_faces_paths person_to_replace.jpg \
  --target_image group_photo.jpg \
  --out_image_name swapped_group.png
```

### **3. Create Deepfake Video**
```bash
python inference.py \
  --image_to_image False \
  --source_paths actor_face.jpg \
  --target_video original_movie.mp4 \
  --out_video_name deepfake_movie.mp4
```

---

## ⚠️ **Current Limitations**

1. **Speed**: ~4 FPS (not real-time)
   - Real-time needs 15-30 FPS
   - See real-time implementation plan

2. **Profile Faces**: Works best with frontal faces
   - Side profiles may have artifacts
   - Extreme angles challenging

3. **Lighting**: Large differences cause color mismatch
   - Source in bright light, target in shadow = noticeable
   - Post-processing can help

4. **Multiple Faces**: Can swap multiple, but slower
   - Each face adds ~230ms
   - 3 faces per frame = ~1.3 FPS

5. **Resolution**: Limited to 256x256 generator
   - Upscaling helps but limited
   - Higher res requires retraining

---

## 🎓 **Understanding the Output**

When you run inference, you'll see:

```
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
→ Using GPU acceleration ✅

find model: ./insightface_func/models/antelope/glintr100.onnx recognition
→ MODEL 2 loaded ✅

find model: ./insightface_func/models/antelope/scrfd_10g_bnkps.onnx detection
→ MODEL 1 loaded ✅

set det-size: (640, 640)
→ Detection resolution set ✅

loading ./coordinate_reg/model/2d106det 0
→ MODEL 5 loading...

ONNX model not found, will use alternative methods for landmarks
→ MODEL 5 not found, using fallback (5-point landmarks) ⚠️

List of source paths: ['examples/images/mark.jpg']
→ Source face identified ✅

100% 1/1 [00:00<00:00, 38.30it/s]
→ Processing frames (MODEL 4 running) ⚡

Swapped Image saved with path examples/results/result.png
→ Done! ✅

Total time: 1.25 seconds
→ Performance metric
```

---

## 📚 **Further Reading**

- **MODEL_ARCHITECTURE_EXPLAINED.md** - Deep dive into each model
- **CODE_FLOW_EXPLAINED.md** - Step-by-step code walkthrough
- **Original repo**: https://github.com/sberbank-ai/sber-swap

---

## 🤝 **Quick Help**

**Q: Which model does the actual face swapping?**  
A: MODEL 4 (AEI-Net) - the generator

**Q: Can I use this on CPU?**  
A: Technically yes, but 10-20× slower

**Q: Is this real-time?**  
A: No, currently ~4 FPS. Real-time needs optimization (see implementation plan)

**Q: Which model is the bottleneck?**  
A: MODEL 4 (AEI-Net generator) at ~120ms per face

**Q: Can I train my own models?**  
A: Yes, use `train.py`, but needs large dataset + GPU cluster

---

**Summary**: You're using 5 models in a pipeline. SCRFD finds faces, ArcFace extracts identity, AEI-Net does the swap, and landmarks help blend it back. Each model has a specific job, and together they create realistic face swaps.


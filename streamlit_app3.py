# =======================================================================
#
# Demo the face swap with Multiple Face Swap capability
# Launch the web interface through: streamlit run streamlit_app3.py
# 1) load model (left panel)
# 2) load the source face and process it
# 3) single image capture by webcam and swap the face using the source image
# 4) swap the face directly in the live video
# 5) swap multiple faces in an image / video
#
# =======================================================================

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import time
import os
import sys
from pathlib import Path
import pandas as pd
import tempfile
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(project_root))

# Load models
from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from utils.inference.image_processing import normalize_and_torch
from utils.inference.realtime_processing import process_single_frame
from utils.inference.image_processing import crop_face as crop_face_util
from utils.inference.multiple_face_processing import load_face_embeddings, process_single_frame_multiple
import platform

# Import helper functions from addon module
from utils.inference.multiple_face_swap_addons import (
    ensure_uint8,
    load_source_face,
    draw_face_bounding_boxes,
    capture_and_swap
)

# Page configuration
st.set_page_config(
    page_title="Face Swap with Multiple Face Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Image constants
camera_index = 0
crop_size = 224
show_bbox = True
show_stats = False
det_threshold = 0.6
det_size = (640, 640)
target_fps = 30


# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'source_embed' not in st.session_state:
    st.session_state.source_embed = None
if 'source_image' not in st.session_state:
    st.session_state.source_image = None
if 'video_running' not in st.session_state:
    st.session_state.video_running = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
# Multiple face swap session state
if 'multi_embedding_map' not in st.session_state:
    st.session_state.multi_embedding_map = None
if 'multi_source_faces' not in st.session_state:
    st.session_state.multi_source_faces = []
if 'multi_target_faces' not in st.session_state:
    st.session_state.multi_target_faces = []
if 'processed_video_bytes' not in st.session_state:
    st.session_state.processed_video_bytes = None
if 'processed_video_result_path' not in st.session_state:
    st.session_state.processed_video_result_path = None
if 'original_video_bytes' not in st.session_state:
    st.session_state.original_video_bytes = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None


@st.cache_resource
def load_models(det_threshold=0.6, det_size=(640, 640)):
    """Load all models (cached to avoid reloading)"""
    try:
        # Initialize face detection model
        app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        app.prepare(ctx_id=0, det_thresh=det_threshold, det_size=det_size)

        # Initialize generator model
        G = AEI_Net('unet', num_blocks=2, c_id=512)
        G.eval()
        G.load_state_dict(torch.load('weights/G_unet_2blocks.pth', map_location=torch.device('cpu')))
        G = G.cuda()
        G = G.float()  # Use FP32

        # Initialize ArcFace model
        netArc = iresnet100(fp16=False)
        netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
        netArc = netArc.cuda()
        netArc.eval()

        # Initialize landmark handler
        handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

        return {
            'app': app,
            'G': G,
            'netArc': netArc,
            'handler': handler
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def process_uploaded_video(
    video_path: str,
    output_path: str,
    embedding_map: dict,
    models: dict,
    similarity_threshold: float = 0.076,
    progress_placeholder=None,
    status_placeholder=None
):
    """Process uploaded video with multiple face swapping"""
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file"

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Try different codecs 
        codecs_to_try = [
            ('mp4v', '.mp4'),  # MPEG-4 - works and creates MP4
            ('XVID', '.avi'),  # XVID - reliable fallback
        ]

        out = None
        temp_output = None

        for codec, ext in codecs_to_try:
            temp_output = output_path.replace('.mp4', f'_temp{ext}')
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                if test_out.isOpened():
                    out = test_out
                    if status_placeholder:
                        status_placeholder.text(f"Using {codec} codec for video encoding")
                    break
                else:
                    test_out.release()
            except:
                pass

        if out is None:
            return False, "Could not initialize video writer with any codec"

        # Process frames
        frame_count = 0
        total_faces_swapped = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result_frame, det_time, gen_time, num_swapped = process_single_frame_multiple(
                frame=frame,
                embedding_map=embedding_map,
                netArc=models['netArc'],
                G=models['G'],
                app=models['app'],
                similarity_threshold=similarity_threshold,
                crop_size=crop_size,
                device='cuda'
            )

            # Write frame
            out.write(result_frame)

            frame_count += 1
            total_faces_swapped += num_swapped

            # Update progress
            if progress_placeholder:
                progress = frame_count / total_frames
                progress_placeholder.progress(progress, text=f"Processing: {frame_count}/{total_frames} frames")

            if status_placeholder and frame_count % 10 == 0:
                status_placeholder.text(f"Processed {frame_count}/{total_frames} frames | Swapped {total_faces_swapped} faces")

        cap.release()
        out.release()

        # Convert through ffmpeg to ensure browser compatibility
        # Even if we have MP4, we need to add faststart flag for streaming
        if status_placeholder:
            status_placeholder.text("Converting video to browser-compatible format...")

        # Convert to browser-compatible MP4 using ffmpeg
        try:
            import subprocess

            # Use ffmpeg with web-optimized settings for browser playback
            # Key flags for web compatibility:
            # -movflags +faststart: enables progressive streaming in browsers
            # -pix_fmt yuv420p: ensures compatibility with all browsers
            # -vf scale=...: ensures even dimensions (required for some codecs)
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', temp_output,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
                '-movflags', '+faststart',  # Enable progressive streaming for browsers
                '-profile:v', 'baseline',   # Use baseline profile for max compatibility
                '-level', '3.0',
                output_path
            ]

            if status_placeholder:
                status_placeholder.text("Converting to browser-compatible MP4 format...")

            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
            )

            if result.returncode == 0:
                # Verify output file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    # Remove temp file
                    try:
                        os.unlink(temp_output)
                    except:
                        pass

                    if status_placeholder:
                        status_placeholder.success(f"Complete! Processed {frame_count} frames, swapped {total_faces_swapped} faces")
                    return True, output_path
                else:
                    if status_placeholder:
                        status_placeholder.error("FFmpeg conversion produced invalid file")
                    return False, "FFmpeg conversion failed - output file is empty"
            else:
                # ffmpeg failed, show error
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                if status_placeholder:
                    status_placeholder.error(f"FFmpeg conversion failed: {error_msg[:200]}")
                # Return temp file as fallback
                return True, temp_output

        except subprocess.TimeoutExpired:
            if status_placeholder:
                status_placeholder.error("Video conversion timed out")
            return True, temp_output
        except FileNotFoundError:
            if status_placeholder:
                status_placeholder.warning("FFmpeg not found - using original format")
            return True, temp_output
        except Exception as e:
            if status_placeholder:
                status_placeholder.warning(f"Conversion error: {str(e)[:100]}")
            return True, temp_output

    except Exception as e:
        return False, f"Error processing video: {e}"


st.markdown('<h2 class="main-header">Face Swap with Multiple Face Support</h2>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:

    # Load models
    if st.button("Load/Reload Models", type="primary"):
        with st.spinner("Loading models... This may take a minute..."):
            models = load_models(det_threshold, det_size)
            if models:
                st.session_state.models = models
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models")

if not st.session_state.models_loaded:
    st.warning("Please load models from the sidebar first")
    st.stop()

# Create tabs
tab_source, tab_frame, tab_video, tab_multi = st.tabs([
    "Upload Source Face",
    "Single Frame Swap",
    "Live Video Swap",
    "Multiple Face Swap"
])

# Upload Source Face
with tab_source:
    st.header("Upload Source Face Image")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Select Source")
        uploaded_file = st.file_uploader(
            "Choose a source face image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a source image face"
        )

        if uploaded_file:
            source_image = Image.open(uploaded_file)
            st.session_state.source_image = source_image


        if st.session_state.source_image:
            st.image(st.session_state.source_image, caption="Source Face", width="stretch")

            if st.button("Process Source Face", type="primary"):
                with st.spinner("Processing source face..."):
                    source_embed, error = load_source_face(
                        st.session_state.source_image,
                        st.session_state.models,
                        crop_size
                    )

                    if error:
                        st.error(f"{error}")
                    else:
                        st.session_state.source_embed = source_embed
                        st.success("Source face processed successfully!")

    with col2:
        st.subheader("Status")
        if st.session_state.source_embed is not None:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("**Source face ready!**")
            st.markdown("You can now proceed to the other tabs for face swapping.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**No source face loaded**")
            st.markdown("Please upload and process a source image first.")
            st.markdown('</div>', unsafe_allow_html=True)

# Single Frame Swap
with tab_frame:
    st.header("Capture and Swap Single Frame")

    if st.session_state.source_embed is None:
        st.warning("Please upload and process a source face in the first tab")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Camera Capture")

            if st.button("Capture & Swap", type="primary"):
                with st.spinner("Capturing and processing..."):
                    result_frame, error = capture_and_swap(
                        st.session_state.models,
                        st.session_state.source_embed,
                        camera_index,
                        crop_size
                    )

                    if error:
                        st.error(f"{error}")
                        if result_frame is not None:
                            # Show original frame even if swap failed
                            st.image(
                                cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                caption="Original Frame (no face detected)",
                                width="stretch"
                            )
                    else:
                        st.success("Face swap successful!")
                        st.session_state.last_capture = result_frame

        with col2:
            st.subheader("Result")
            if 'last_capture' in st.session_state and st.session_state.last_capture is not None:
                st.image(
                    cv2.cvtColor(st.session_state.last_capture, cv2.COLOR_BGR2RGB),
                    caption="Swapped Face Result",
                    width="stretch"
                )

                # Download button
                result_pil = Image.fromarray(cv2.cvtColor(st.session_state.last_capture, cv2.COLOR_BGR2RGB))
                import io
                buf = io.BytesIO()
                result_pil.save(buf, format='PNG')
                st.download_button(
                    label="Download Result",
                    data=buf.getvalue(),
                    file_name="face_swap_result.png",
                    mime="image/png"
                )
            else:
                st.info("Click 'Capture & Swap' to see result here")

# Live Video Swap
with tab_video:
    st.header("Live Video Face Swap")

    if st.session_state.source_embed is None:
        st.warning("Please upload and process a source face in the first tab")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            Click 'Stop Video' button to end
            """)
            st.markdown('</div>', unsafe_allow_html=True)

            # Initialize video control states
            if 'video_active' not in st.session_state:
                st.session_state.video_active = False
            if 'stop_video' not in st.session_state:
                st.session_state.stop_video = False

            if st.button("Start Video", type="primary", disabled=st.session_state.video_active):
                st.session_state.video_active = True
                st.session_state.stop_video = False
                st.rerun()

            if st.button("Stop Video", type="secondary", disabled=not st.session_state.video_active):
                st.session_state.stop_video = True
                st.session_state.video_active = False
                st.rerun()

        with col2:
            st.subheader("Live Video Feed")
            video_placeholder = st.empty()
            status_placeholder = st.empty()


        st.subheader("Statistics")
        stats_table = st.empty()

        # Run video if active
        if st.session_state.video_active and not st.session_state.stop_video:
            # Import required for live video
            from collections import deque

            # Performance monitor
            class PerfMonitor:
                def __init__(self, window=30):
                    self.window = window
                    self.frame_start_time = None
                    self.processing_ms = deque(maxlen=window)
                    self.det_ms = deque(maxlen=window)
                    self.gen_ms = deque(maxlen=window)

                def start_frame(self):
                    self.frame_start_time = time.time()

                def end_frame(self):
                    if self.frame_start_time is not None:
                        elapsed_ms = (time.time() - self.frame_start_time) * 1000.0
                        self.processing_ms.append(elapsed_ms)
                        self.frame_start_time = None

                def record_detection_time(self, ms):
                    if ms is None: return
                    self.det_ms.append(float(ms))

                def record_generator_time(self, ms):
                    if ms is None: return
                    self.gen_ms.append(float(ms))

                def record_processing_time(self, ms):
                    if ms is None: return
                    self.processing_ms.append(float(ms))

                def get_stats(self):
                    def avg(q):
                        return float(sum(q)/len(q)) if len(q) else 0.0
                    avg_proc = avg(self.processing_ms)
                    fps = (1000.0 / avg_proc) if avg_proc > 0 else 0.0
                    return {
                        "fps": fps,
                        "avg_latency_ms": avg_proc,
                        "avg_detection_ms": avg(self.det_ms),
                        "avg_generator_ms": avg(self.gen_ms),
                    }

            monitor = PerfMonitor(window=60)

            # Open camera
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(camera_index)

            if not cap.isOpened():
                video_placeholder.error("Failed to open camera")
                st.session_state.video_active = False
                st.stop()

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            frame_count = 0
            min_frame_time = 1.0 / target_fps if target_fps > 0 else 0

            status_placeholder.info("Video stream active - Click 'Stop Video' to end")

            try:
                while st.session_state.video_active and not st.session_state.stop_video:
                    frame_start = time.time()

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    frame = ensure_uint8(frame)
                    monitor.start_frame()

                    # Process frame
                    result, det_time, gen_time = process_single_frame(
                        frame=frame,
                        source_embed=st.session_state.source_embed,
                        netArc=st.session_state.models['netArc'],
                        G=st.session_state.models['G'],
                        app=st.session_state.models['app'],
                        handler=st.session_state.models['handler'],
                        bbox=None,
                        crop_size=crop_size,
                        half=False
                    )

                    monitor.record_detection_time(det_time)
                    monitor.record_generator_time(gen_time)
                    total_time = (time.time() - monitor.frame_start_time) * 1000 if monitor.frame_start_time else 0
                    monitor.record_processing_time(total_time)
                    monitor.end_frame()

                    # Display frame
                    display_frame = result if result is not None else frame

                    # Add bounding box if face detected and enabled
                    if show_bbox and result is not None:
                        # Add green border to indicate successful swap
                        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1),
                                    (0, 255, 0), 3)

                    # Convert BGR to RGB for Streamlit display
                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    # Display in webpage
                    video_placeholder.image(display_frame_rgb, channels="RGB", width="stretch")

                    frame_count += 1

                    # Update stats
                    stats = monitor.get_stats()
                    status_text = "Face Swap Active" if result is not None else "No Face Detected"

                    stats_data = pd.DataFrame({
                        "Metric": ["FPS", "Latency", "Detection Time", "Generation Time", "Frames Processed", "Status"],
                        "Value": [
                            f"{stats['fps']:.1f}",
                            f"{stats['avg_latency_ms']:.0f} ms",
                            f"{stats['avg_detection_ms']:.0f} ms",
                            f"{stats['avg_generator_ms']:.0f} ms",
                            str(frame_count),
                            status_text
                        ]
                    })
                    stats_table.dataframe(stats_data, hide_index=True, width="stretch")

                    # FPS limiting
                    elapsed = time.time() - frame_start
                    if elapsed < min_frame_time:
                        time.sleep(min_frame_time - elapsed)

            except Exception as e:
                video_placeholder.error(f"Error: {e}")
            finally:
                cap.release()

                # Final stats
                if frame_count > 0:
                    stats = monitor.get_stats()
                    status_placeholder.success(f"Video stopped | Processed {frame_count} frames | Avg FPS: {stats['fps']:.1f}")
                else:
                    status_placeholder.info("Video stopped")

                st.session_state.video_active = False

        elif not st.session_state.video_active:
            # Show placeholder when video is not active
            video_placeholder.info("Click 'Start Video' to begin live face swapping")

            default_stats = pd.DataFrame({
                "Metric": ["FPS", "Latency", "Detection Time", "Generation Time", "Frames Processed", "Status"],
                "Value": ["0.0", "0 ms", "0 ms", "0 ms", "0", "Video Inactive"]
            })
            stats_table.dataframe(default_stats, hide_index=True, width="stretch")


# Multiple Face Swap Tab
with tab_multi:
    st.header("Multiple Face Swap for Image/Video")
    st.markdown("Upload 2 source faces, 2 target faces, and an image or video to swap multiple faces")

    # Mode selection
    mode = st.radio("Select Mode", ["Image", "Video"], horizontal=True, key='multi_mode')

    # Initialize default files in session state
    if 'default_files_loaded' not in st.session_state:
        st.session_state.default_files_loaded = False
        # Video mode defaults
        st.session_state.default_video_source1 = None
        st.session_state.default_video_source2 = None
        st.session_state.default_video_target1 = None
        st.session_state.default_video_target2 = None
        st.session_state.default_video = None
        # Image mode defaults
        st.session_state.default_image_source1 = None
        st.session_state.default_image_source2 = None
        st.session_state.default_image_target1 = None
        st.session_state.default_image_target2 = None
        # default_image_target_image will be loaded from file

    # Load default files
    if not st.session_state.default_files_loaded:
        # Video mode default paths
        video_default_paths = {
            'source1': 'examples/images/trump.png',
            'source2': 'examples/images/trump_wife.png',
            'target1': 'examples/images/tgt1.png',
            'target2': 'examples/images/tgt2.png',
            'video': 'examples/videos/dirtydancing.mp4'
        }

        # Image mode default paths
        image_default_paths = {
            'source1': 'examples/images/elon_musk.jpg',
            'source2': 'examples/images/trump_wife.png',
            'target1': 'examples/images/man.png',
            'target2': 'examples/images/trump.png',
            'target_image': 'examples/images/target_3.png'
        }

        # Load video mode defaults
        for key, path in video_default_paths.items():
            if os.path.exists(path):
                try:
                    if key == 'video':
                        with open(path, 'rb') as f:
                            st.session_state.default_video = f.read()
                    else:
                        st.session_state[f'default_video_{key}'] = Image.open(path)
                except Exception as e:
                    st.warning(f"Could not load video default {key}: {e}")

        # Load image mode defaults
        for key, path in image_default_paths.items():
            if os.path.exists(path):
                try:
                    st.session_state[f'default_image_{key}'] = Image.open(path)
                except Exception as e:
                    st.warning(f"Could not load image default {key}: {e}")

        st.session_state.default_files_loaded = True

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Faces")

        # Select appropriate defaults based on mode
        if mode == "Image":
            default_source1 = st.session_state.default_image_source1
            default_source2 = st.session_state.default_image_source2
            default_target1 = st.session_state.default_image_target1
            default_target2 = st.session_state.default_image_target2
            # Note: target_image is stored as default_image_target_image
            default_target_img = st.session_state.get('default_image_target_image', None)
        else:  # Video mode
            default_source1 = st.session_state.default_video_source1
            default_source2 = st.session_state.default_video_source2
            default_target1 = st.session_state.default_video_target1
            default_target2 = st.session_state.default_video_target2

        st.markdown("**Source Faces** (Face will be displayed in the final image/video)")
        source1 = st.file_uploader("Source Face 1", type=['jpg', 'jpeg', 'png'], key='source1')
        source2 = st.file_uploader("Source Face 2", type=['jpg', 'jpeg', 'png'], key='source2')

        st.markdown("**Target Faces** (Face will be replaced in the existing image/video)")
        target1 = st.file_uploader("Target Face 1", type=['jpg', 'jpeg', 'png'], key='target1')
        target2 = st.file_uploader("Target Face 2", type=['jpg', 'jpeg', 'png'], key='target2')

        # Use uploaded files or defaults
        source1_img = Image.open(source1) if source1 else default_source1
        source2_img = Image.open(source2) if source2 else default_source2
        target1_img = Image.open(target1) if target1 else default_target1
        target2_img = Image.open(target2) if target2 else default_target2

        # Display faces (uploaded or default)
        if source1_img and source2_img and target1_img and target2_img:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                caption1 = "Source 1" + (" (default)" if not source1 else " (uploaded)")
                st.image(source1_img, caption=caption1, width="stretch")
            with col_s2:
                caption2 = "Source 2" + (" (default)" if not source2 else " (uploaded)")
                st.image(source2_img, caption=caption2, width="stretch")

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                caption3 = "Target 1" + (" (default)" if not target1 else " (uploaded)")
                st.image(target1_img, caption=caption3, width="stretch")
            with col_t2:
                caption4 = "Target 2" + (" (default)" if not target2 else " (uploaded)")
                st.image(target2_img, caption=caption4, width="stretch")

            # Process faces button
            if st.button("Process Face Embeddings", type="primary"):
                with st.spinner("Computing face embeddings..."):
                    # Convert PIL images to numpy arrays (use displayed images)
                    source_faces = []
                    target_faces = []

                    for img in [source1_img, source2_img]:
                        img_array = np.array(img)
                        if len(img_array.shape) == 2:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                        elif img_array.shape[2] == 4:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                        else:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        source_faces.append(img_array)

                    for img in [target1_img, target2_img]:
                        img_array = np.array(img)
                        if len(img_array.shape) == 2:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                        elif img_array.shape[2] == 4:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                        else:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        target_faces.append(img_array)

                    # Load embeddings
                    try:
                        embedding_map = load_face_embeddings(
                            source_face_images=source_faces,
                            target_face_images=target_faces,
                            netArc=st.session_state.models['netArc'],
                            app=st.session_state.models['app'],
                            crop_size=crop_size,
                            device='cuda'
                        )

                        st.session_state.multi_embedding_map = embedding_map
                        st.success(f"Face embeddings computed! Found {len(embedding_map)} face pairs")
                    except Exception as e:
                        st.error(f"Error computing embeddings: {e}")

    with col2:
        st.subheader(f"Upload {mode}")

        # Image mode
        if mode == "Image":
            image_file = st.file_uploader("Upload image file", type=['jpg', 'jpeg', 'png'], key='image_upload')

            # Use uploaded image or default
            if image_file:
                original_img = Image.open(image_file)
                st.session_state.original_image = original_img
                image_caption = f"Uploaded: {image_file.name}"
            else:
                # Use default target image
                original_img = st.session_state.get('default_image_target_image', None)
                st.session_state.original_image = original_img
                image_caption = "Default Image: target_3.png"

            if original_img:
                # Display original image
                st.markdown("**Original Image:**")
                st.image(original_img, caption=image_caption, width="stretch")

                # Process button
                if st.session_state.multi_embedding_map is not None:
                    similarity_threshold_img = st.slider(
                        "Similarity Threshold (Image)",
                        min_value=0.0,
                        max_value=0.3,
                        value=0.25,
                        step=0.01,
                        help="Lower = more permissive matching, Higher = stricter matching",
                        key='similarity_img'
                    )

                    if st.button("Swap Faces in Image", type="primary", key='process_image'):
                        with st.spinner("Processing image..."):
                            # Convert PIL to OpenCV format
                            img_array = np.array(original_img)
                            if len(img_array.shape) == 2:  # Grayscale
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                            elif img_array.shape[2] == 4:  # RGBA
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                            else:  # RGB
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                            # Process the image with multiple face swap
                            result_frame, det_time, gen_time, num_swapped = process_single_frame_multiple(
                                frame=img_array,
                                embedding_map=st.session_state.multi_embedding_map,
                                netArc=st.session_state.models['netArc'],
                                G=st.session_state.models['G'],
                                app=st.session_state.models['app'],
                                similarity_threshold=similarity_threshold_img,
                                crop_size=crop_size,
                                device='cuda'
                            )

                            # Draw bounding boxes (2x bigger)
                            result_with_boxes = draw_face_bounding_boxes(
                                original_frame=img_array,
                                result_frame=result_frame,
                                embedding_map=st.session_state.multi_embedding_map,
                                netArc=st.session_state.models['netArc'],
                                app=st.session_state.models['app'],
                                similarity_threshold=similarity_threshold_img,
                                crop_size=crop_size
                            )

                            # Convert back to RGB for display
                            result_rgb = cv2.cvtColor(result_with_boxes, cv2.COLOR_BGR2RGB)
                            st.session_state.processed_image = result_rgb

                            st.success(f"Image processing complete! Swapped {num_swapped} faces")

                    # Display processed image if available
                    if st.session_state.processed_image is not None:
                        st.markdown("**Processed Image:**")
                        st.image(st.session_state.processed_image, caption="Face Swapped Result", width="stretch")

                        # Download button
                        result_pil = Image.fromarray(st.session_state.processed_image)
                        import io
                        buf = io.BytesIO()
                        result_pil.save(buf, format='PNG')
                        st.download_button(
                            label="Download Processed Image",
                            data=buf.getvalue(),
                            file_name="face_swapped_image.png",
                            mime="image/png",
                            type="secondary"
                        )
                else:
                    st.warning("Please process face embeddings first")
            else:
                st.warning("Default target image not found. Please upload an image file.")

        # Video mode
        elif mode == "Video":
            video_file = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov', 'mkv'], key='video_upload')

            # Use uploaded video or default
            video_to_use = video_file
            use_default_video = False

            if not video_file and st.session_state.default_video:
                use_default_video = True

            if video_file or use_default_video:
                # Store original video bytes in session state
                if 'replay_counter' not in st.session_state:
                    st.session_state.replay_counter = 0

                # Read and store original video (uploaded or default)
                if use_default_video:
                    st.session_state.original_video_bytes = st.session_state.default_video
                    video_caption = "Default Video: dirtydancing.mp4"
                else:
                    video_file.seek(0)
                    st.session_state.original_video_bytes = video_file.read()
                    video_caption = f"Uploaded Video: {video_file.name}"

                # Display caption
                st.caption(video_caption)

                # Use a placeholder that gets cleared and recreated on replay
                original_video_placeholder = st.empty()

                # with original_video_placeholder.container():
                #     # Force recreation by changing content
                #     st.markdown(f"**Original Video** (Replay: {st.session_state.replay_counter})")
                #     st.video(st.session_state.original_video_bytes, start_time=0, key=f"original_video_{st.session_state.replay_counter}")
                with st.container(key=f"original_video_container_{st.session_state.replay_counter}"):
                    st.markdown(f"**Original Video** (Replay: {st.session_state.replay_counter})")
                    st.video(st.session_state.original_video_bytes, start_time=0)


                # Placeholder for processed video (will appear above similarity threshold)
                processed_video_placeholder = st.empty()

                # Similarity threshold slider
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=0.3,
                    value=0.078,
                    step=0.01,
                    help="Lower = more permissive matching, Higher = stricter matching"
                )

                # Process video button, download button, and replay button side by side
                if st.session_state.multi_embedding_map is not None:
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

                    with col_btn1:
                        process_button = st.button("Swap Faces in Video", type="primary", key='process_video')

                    with col_btn2:
                        # Download button placeholder (will be enabled after processing)
                        download_placeholder = st.empty()

                    with col_btn3:
                        # Replay button placeholder (will be enabled after processing)
                        replay_placeholder = st.empty()

                    # Show download and replay buttons if video was already processed
                    if st.session_state.processed_video_bytes is not None:
                        # Display processed video from session state
                        with processed_video_placeholder.container():
                            st.markdown("**Processed Video:**")
                            # Use hidden marker to force reload on replay
                            st.markdown(f"<!-- processed_replay_{st.session_state.replay_counter} -->", unsafe_allow_html=True)
                            st.video(st.session_state.processed_video_bytes,
                                     start_time=0)

                        # Determine file info
                        result = st.session_state.processed_video_result_path
                        if result and result.endswith('.mp4'):
                            file_name = "face_swapped_video.mp4"
                            mime_type = "video/mp4"
                        else:
                            file_name = "face_swapped_video.avi"
                            mime_type = "video/x-msvideo"

                        # Download button (download only, keeps videos loaded)
                        with download_placeholder:
                            st.download_button(
                                label="Download",
                                data=st.session_state.processed_video_bytes,
                                file_name=file_name,
                                mime=mime_type,
                                type="secondary",
                                key='download_existing',
                                help="Download processed video file"
                            )

                        # Replay button
                        # with replay_placeholder:
                        #     if st.button("Replay Both", type="secondary", key='replay_existing', help="Restart both videos from beginning"):
                        #         st.session_state.replay_counter += 1

                    if process_button:
                        # Save uploaded video to temp file (use stored bytes)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                            tmp_input.write(st.session_state.original_video_bytes)
                            input_path = tmp_input.name

                        # Create output path
                        fd, output_path = tempfile.mkstemp(suffix='_swapped.mp4')
                        os.close(fd)

                        # Process video
                        progress_bar = st.progress(0, text="Starting video processing...")
                        status_text = st.empty()

                        success, result = process_uploaded_video(
                            video_path=input_path,
                            output_path=output_path,
                            embedding_map=st.session_state.multi_embedding_map,
                            models=st.session_state.models,
                            similarity_threshold=similarity_threshold,
                            progress_placeholder=progress_bar,
                            status_placeholder=status_text
                        )

                        if success:
                            st.success("Video processing complete!")

                            # Verify file exists and is readable
                            if not os.path.exists(result):
                                st.error(f"Output file not found: {result}")
                            else:
                                file_size = os.path.getsize(result)
                                st.info(f"Video file: {os.path.basename(result)} ({file_size / 1024 / 1024:.2f} MB)")

                                # Read processed video
                                try:
                                    with open(result, 'rb') as f:
                                        video_bytes = f.read()

                                    # Store in session state
                                    st.session_state.processed_video_bytes = video_bytes
                                    st.session_state.processed_video_result_path = result

                                    # Display processed video in the placeholder (above similarity threshold)
                                    with processed_video_placeholder.container():
                                        st.markdown("**Processed Video:**")
                                        # Use hidden marker to force reload on replay
                                        st.markdown(f"<!-- processed_replay_{st.session_state.replay_counter} -->", unsafe_allow_html=True)
                                        try:
                                            # Method 1: Direct video display with bytes
                                            st.video(video_bytes, start_time=0)
                                        except Exception as e:
                                            st.warning(f"Direct playback failed: {e}")
                                            try:
                                                # Method 2: Display from file path
                                                st.video(result, start_time=0)
                                            except Exception as e2:
                                                st.error(f"File path playback also failed: {e2}")
                                                st.info("Please download the video to play it locally")

                                    # Determine file extension and mime type
                                    if result.endswith('.mp4'):
                                        file_name = "face_swapped_video.mp4"
                                        mime_type = "video/mp4"
                                    else:
                                        file_name = "face_swapped_video.avi"
                                        mime_type = "video/x-msvideo"

                                    # Download button in the placeholder (download only, keeps videos loaded)
                                    with download_placeholder:
                                        st.download_button(
                                            label="Download",
                                            data=video_bytes,
                                            file_name=file_name,
                                            mime=mime_type,
                                            type="secondary",
                                            key='download_new',
                                            help="Download processed video file"
                                        )

                                    # Replay button
                                    # with replay_placeholder:
                                    #     if st.button("Replay Both", type="secondary", key='replay_videos', help="Restart both videos from beginning"):
                                    #         # Increment counter to force reload
                                    #         st.session_state.replay_counter += 1
                                    #         # st.rerun()

                                except Exception as e:
                                    st.error(f"Error reading video file: {e}")

                                # Cleanup temp files - do this AFTER displaying the video
                                # Keep the output file for now, delete input and temp files
                                try:
                                    if os.path.exists(input_path):
                                        os.unlink(input_path)
                                    # Cleanup temp files but keep result for display
                                    temp_avi = result.replace('.mp4', '_temp.avi')
                                    if os.path.exists(temp_avi):
                                        os.unlink(temp_avi)
                                    temp_mp4 = result.replace('.mp4', '_temp.mp4')
                                    if os.path.exists(temp_mp4) and temp_mp4 != result:
                                        os.unlink(temp_mp4)
                                except Exception as e:
                                    st.warning(f"Cleanup warning: {e}")

                        else:
                            st.error(f"Error: {result}")
                else:
                    st.warning("Please process face embeddings first")
            else:
                st.info("Upload a video file to begin")


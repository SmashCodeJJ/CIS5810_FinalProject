"""
Flask Web Server for Real-Time Face Swapping on EC2
Provides continuous streaming without permission delays
"""
from flask import Flask, render_template_string, Response, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
import sys
import torch
import time
from io import BytesIO
from PIL import Image

# Add project to path
sys.path.insert(0, '.')

from inference_realtime import init_models, load_source_face
from utils.inference.realtime_processing import process_single_frame
from utils.realtime.face_tracker import FaceTracker
from utils.realtime.performance_monitor import PerformanceMonitor

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Global variables for models
models_loaded = False
app_model = None
G = None
netArc = None
handler = None
source_embed = None
tracker = None
monitor = None

def load_models_once():
    """Load models only once at startup"""
    global app_model, G, netArc, handler, source_embed, tracker, monitor, models_loaded
    
    if models_loaded:
        return True
    
    try:
        print("Loading models... This may take 1-2 minutes...")
        
        class Args:
            def __init__(self):
                self.G_path = 'weights/G_unet_2blocks.pth'
                self.backbone = 'unet'
                self.num_blocks = 2
                self.fast_mode = True
                self.crop_size = 224
                self.detect_interval = 5
                self.tracker_type = 'CSRT'
        
        args = Args()
        app_model, G, netArc, handler = init_models(args)
        
        # Load source face (default, can be changed via API)
        source_path = 'examples/images/mark.jpg'
        source_embed = load_source_face(source_path, app_model, netArc, args.crop_size)
        
        # Initialize tracker and monitor
        tracker = FaceTracker(
            detector=app_model,
            detect_interval=args.detect_interval,
            tracker_type=args.tracker_type,
            confidence_threshold=0.6
        )
        
        monitor = PerformanceMonitor(window_size=30)
        
        models_loaded = True
        print("✅ Models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def base64_to_image(data):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in data:
            data = data.split(',')[1]
        
        # Decode base64
        img_bytes = base64.b64decode(data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def image_to_base64(img):
    """Convert OpenCV image to base64 string"""
    try:
        # Encode image
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

@app.route('/')
def index():
    """Main web interface"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Face Swap - EC2</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .video-container {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin: 20px 0;
        }
        video, canvas {
            border: 3px solid #333;
            border-radius: 5px;
            max-width: 640px;
            height: auto;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            background: #4CAF50;
            color: white;
        }
        button:hover {
            background: #45a049;
        }
        button#stop {
            background: #f44336;
        }
        button#stop:hover {
            background: #da190b;
        }
        .stats {
            text-align: center;
            margin: 20px 0;
            font-size: 18px;
            color: #666;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            text-align: center;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Real-Time Face Swapping</h1>
        <div class="status" id="status">Ready to start</div>
        
        <div class="video-container">
            <div>
                <h3>Camera Input</h3>
                <video id="video" width="640" height="480" autoplay></video>
            </div>
            <div>
                <h3>Swapped Output</h3>
                <canvas id="canvas" width="640" height="480"></canvas>
            </div>
        </div>
        
        <div class="controls">
            <button id="start">▶ Start Streaming</button>
            <button id="stop">⏹ Stop</button>
        </div>
        
        <div class="stats">
            <div>FPS: <span id="fps">0</span></div>
            <div>Latency: <span id="latency">0</span> ms</div>
            <div>Frames Processed: <span id="frames">0</span></div>
        </div>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let streaming = false;
        let frameCount = 0;
        let startTime = Date.now();
        let totalLatency = 0;
        
        function updateStatus(message, isError = false) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + (isError ? 'error' : 'success');
        }
        
        async function startStream() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({video: true});
                video.srcObject = stream;
                streaming = true;
                updateStatus('Streaming active');
                frameCount = 0;
                startTime = Date.now();
                totalLatency = 0;
                processFrames();
            } catch (error) {
                updateStatus('Error accessing camera: ' + error.message, true);
                console.error(error);
            }
        }
        
        async function processFrames() {
            if (!streaming) return;
            
            const frameStart = Date.now();
            
            // Capture frame
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to server
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({frame: frameData})
                });
                
                if (!response.ok) {
                    throw new Error('Server error');
                }
                
                const result = await response.json();
                
                if (result.error) {
                    throw new Error(result.error);
                }
                
                // Display result
                const img = new Image();
                img.onload = () => {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = result.result;
                
                // Update stats
                const latency = Date.now() - frameStart;
                totalLatency += latency;
                frameCount++;
                
                if (frameCount % 10 === 0) {
                    const fps = (frameCount * 1000) / (Date.now() - startTime);
                    const avgLatency = totalLatency / frameCount;
                    document.getElementById('fps').textContent = fps.toFixed(1);
                    document.getElementById('latency').textContent = avgLatency.toFixed(1);
                    document.getElementById('frames').textContent = frameCount;
                }
                
            } catch (error) {
                console.error('Processing error:', error);
                updateStatus('Error: ' + error.message, true);
            }
            
            // Continue processing (target ~30 FPS)
            setTimeout(processFrames, 33);
        }
        
        function stopStream() {
            streaming = false;
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            updateStatus('Streaming stopped');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        
        document.getElementById('start').onclick = startStream;
        document.getElementById('stop').onclick = stopStream;
    </script>
</body>
</html>
    ''')

@app.route('/process', methods=['POST'])
def process_frame():
    """Process a single frame from the client"""
    global tracker, monitor
    
    try:
        # Ensure models are loaded
        if not models_loaded:
            if not load_models_once():
                return jsonify({'error': 'Failed to load models'}), 500
        
        # Get frame from request
        data = request.json
        if 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Decode frame
        frame = base64_to_image(data['frame'])
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # Start monitoring
        monitor.start_frame()
        
        # Update tracker
        bbox = tracker.update(frame)
        
        # Process frame
        det_time = 0
        gen_time = 0
        result = None
        
        if bbox is not None:
            result, det_time, gen_time = process_single_frame(
                frame=frame,
                source_embed=source_embed,
                netArc=netArc,
                G=G,
                app=app_model,
                handler=handler,
                bbox=bbox,
                crop_size=224,
                half=True
            )
        
        if result is None:
            result = frame
        
        # Draw bbox if face detected
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Record metrics
        monitor.record_detection_time(det_time)
        monitor.record_generator_time(gen_time)
        total_time = (time.time() - monitor.frame_start_time) * 1000 if monitor.frame_start_time else 0
        monitor.record_processing_time(total_time)
        monitor.end_frame()
        
        # Encode result
        result_base64 = image_to_base64(result)
        if result_base64 is None:
            return jsonify({'error': 'Failed to encode result'}), 500
        
        # Return result
        return jsonify({
            'result': result_base64,
            'stats': {
                'fps': monitor.get_fps(),
                'latency': monitor.get_avg_latency(),
                'detection_time': monitor.get_avg_detection_time(),
                'generator_time': monitor.get_avg_generator_time()
            }
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get performance statistics"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    
    stats = monitor.get_stats()
    return jsonify(stats)

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Face Swap Web Server")
    print("="*60)
    
    # Load models on startup
    if load_models_once():
        print("\n✅ Server ready!")
        print("📡 Access at: http://0.0.0.0:5000")
        print("⚠️  Make sure to configure EC2 security group to allow port 5000")
        print("="*60)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("\n❌ Failed to load models. Exiting.")
        sys.exit(1)


# ☁️ EC2 Deployment for Real-Time Face Swapping

## ✅ Yes, EC2 Would Work Better Than Colab!

**EC2 advantages:**
- ✅ No browser security restrictions
- ✅ Can run continuous webcam streaming
- ✅ True real-time performance
- ✅ Full Linux environment
- ✅ Can create web interface

**EC2 considerations:**
- ⚠️ EC2 instances don't have cameras directly
- Need to stream video from client or use external camera
- Requires network setup (port forwarding, security groups)

---

## 🏗️ EC2 Deployment Architectures

### Option 1: Web Interface with Video Streaming (Recommended)

**Architecture:**
```
Browser (Camera) → WebRTC/HTTP → EC2 Server → Face Swap → Stream Back to Browser
```

**Implementation:**
- Flask/FastAPI web server on EC2
- Receives video frames from browser
- Processes frames with face swapping
- Streams results back to browser
- **True continuous streaming** (no permission delays)

**Example Code Structure:**
```python
# server.py (runs on EC2)
from flask import Flask, render_template, Response, request
import cv2
import base64
import numpy as np

app = Flask(__name__)

# Load models once at startup
app_model, G, netArc, handler = init_models(args)
source_embed = load_source_face(source_path, app_model, netArc)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Receive frame from browser
    frame_data = request.json['frame']
    frame = decode_base64_frame(frame_data)
    
    # Process frame
    result = process_single_frame(frame, ...)
    
    # Return result
    result_data = encode_frame_to_base64(result)
    return {'result': result_data}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

### Option 2: Video File Upload and Processing

**Architecture:**
```
Browser → Upload Video → EC2 → Process → Download Result
```

**Implementation:**
- Upload video file to EC2
- Process entire video with face swapping
- Download result video
- Better for batch processing

**Advantages:**
- ✅ No real-time streaming needed
- ✅ Process high-quality videos
- ✅ Batch multiple videos
- ✅ Simpler implementation

---

### Option 3: RTSP/WebRTC Streaming from External Camera

**Architecture:**
```
External Camera → RTSP Stream → EC2 → Face Swap → Stream Out
```

**Implementation:**
- Connect external camera/IP camera
- Stream via RTSP to EC2
- Process stream in real-time
- Output to web interface or save

**Use Cases:**
- Security cameras
- IP cameras
- External USB cameras (if EC2 has USB access)

---

## 🚀 Quick Setup Guide for EC2

### Step 1: Launch EC2 Instance

**Recommended Instance:**
- **Type**: `g4dn.xlarge` or better (GPU instance)
- **GPU**: NVIDIA T4 (similar to Colab)
- **OS**: Ubuntu 22.04 LTS
- **Storage**: 50GB+ (for models and data)

**Key Settings:**
```bash
# Security Group: Open ports
- Port 22 (SSH)
- Port 5000 (Flask/Web server)
- Port 8080 (Alternative web port)
```

---

### Step 2: Setup Environment on EC2

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install -y python3-pip git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python flask flask-cors numpy onnxruntime-gpu insightface
```

---

### Step 3: Deploy Face Swapping Code

```bash
# Clone repository
git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git
cd CIS5810_FinalProject

# Download models
bash download_models.sh

# Create web server
cat > server.py << 'EOF'
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
import sys
import torch
from inference_realtime import init_models, load_source_face
from utils.inference.realtime_processing import process_single_frame

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Initialize models
print("Loading models...")
class Args:
    G_path = 'weights/G_unet_2blocks.pth'
    backbone = 'unet'
    num_blocks = 2
    fast_mode = True
    crop_size = 224
    detect_interval = 5
    tracker_type = 'CSRT'

args = Args()
app_model, G, netArc, handler = init_models(args)
source_embed = load_source_face('examples/images/mark.jpg', app_model, netArc)

def base64_to_image(data):
    """Convert base64 string to OpenCV image"""
    data = data.split(',')[1] if ',' in data else data
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def image_to_base64(img):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Face Swap</title>
        <style>
            body { font-family: Arial; text-align: center; }
            video, canvas { border: 2px solid #333; margin: 10px; }
            button { padding: 10px 20px; font-size: 16px; margin: 5px; }
        </style>
    </head>
    <body>
        <h1>Real-Time Face Swapping</h1>
        <video id="video" width="640" height="480" autoplay></video>
        <canvas id="canvas" width="640" height="480"></canvas><br>
        <button id="start">Start Streaming</button>
        <button id="stop">Stop</button>
        <div id="fps">FPS: 0</div>
        
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            let streaming = false;
            let frameCount = 0;
            let startTime = Date.now();
            
            async function startStream() {
                const stream = await navigator.mediaDevices.getUserMedia({video: true});
                video.srcObject = stream;
                streaming = true;
                processFrames();
            }
            
            async function processFrames() {
                if (!streaming) return;
                
                // Capture frame
                ctx.drawImage(video, 0, 0);
                const frameData = canvas.toDataURL('image/jpeg', 0.8);
                
                // Send to server
                try {
                    const response = await fetch('/process', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({frame: frameData})
                    });
                    const result = await response.json();
                    
                    // Display result
                    const img = new Image();
                    img.onload = () => {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0);
                    };
                    img.src = result.result;
                    
                    // Update FPS
                    frameCount++;
                    if (frameCount % 30 === 0) {
                        const fps = 30000 / (Date.now() - startTime);
                        document.getElementById('fps').textContent = `FPS: ${fps.toFixed(1)}`;
                    }
                } catch (e) {
                    console.error(e);
                }
                
                // Continue processing
                setTimeout(processFrames, 33); // ~30 FPS
            }
            
            document.getElementById('start').onclick = startStream;
            document.getElementById('stop').onclick = () => {
                streaming = false;
                video.srcObject.getTracks().forEach(track => track.stop());
            };
        </script>
    </body>
    </html>
    '''
    
@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        frame = base64_to_image(data['frame'])
        
        # Process frame
        from utils.realtime.face_tracker import FaceTracker
        tracker = FaceTracker(detector=app_model, detect_interval=5)
        bbox = tracker.update(frame)
        
        if bbox is not None:
            result, _, _ = process_single_frame(
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
        else:
            result = frame
        
        # Return result
        result_base64 = image_to_base64(result)
        return jsonify({'result': result_base64})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF

# Run server
python3 server.py
```

---

## 📊 EC2 vs Colab Comparison

| Feature | Colab | EC2 |
|---------|-------|-----|
| **Camera Access** | ⚠️ Browser security (permission each time) | ✅ Direct access (if camera available) |
| **Real-Time Streaming** | ⚠️ Limited by browser | ✅ True continuous streaming |
| **Web Interface** | ❌ Not possible | ✅ Can create web app |
| **GPU Access** | ✅ Free T4 (limited) | ✅ Paid GPU instances |
| **Cost** | ✅ Free (with limits) | ⚠️ Pay per hour |
| **Setup Complexity** | ✅ Simple | ⚠️ Requires configuration |
| **Continuous Operation** | ❌ Time limits | ✅ 24/7 if needed |
| **No Permission Delays** | ❌ Every frame | ✅ One-time or none |

---

## 💰 Cost Considerations

### EC2 GPU Instance Costs (Approximate):
- **g4dn.xlarge** (1x T4 GPU): ~$0.50/hour
- **g4dn.2xlarge** (1x T4 GPU): ~$0.75/hour
- **g4dn.4xlarge** (1x T4 GPU): ~$1.20/hour

### Colab:
- **Free**: T4 GPU (with usage limits)
- **Pro**: $10/month (better GPU access)

**Recommendation**: 
- For development/testing: Use Colab (free)
- For production/demos: Use EC2 (reliable, no limits)

---

## 🎯 Best Use Cases for EC2

### ✅ Use EC2 When:
- Need **true continuous streaming** (no permission delays)
- Want **web interface** for remote access
- Need **24/7 operation**
- Processing **large batches** of videos
- **Production deployment**

### ✅ Use Colab When:
- **Quick testing** and development
- **Learning** and experimentation
- **Free GPU access**
- **One-time processing**

---

## 🚀 Quick Start for EC2 Web Interface

I can create a complete web server implementation if you want. It would include:
- Flask/FastAPI server
- HTML/JavaScript client
- Real-time frame streaming
- No permission delays
- True continuous processing

**Would you like me to create the complete EC2 deployment package?**

---

## 📝 Summary

**Q: Will it work in EC2?**  
**A: Yes! Even better than Colab for real-time streaming.**

**Key Advantages:**
- ✅ No browser permission issues
- ✅ True continuous streaming
- ✅ Can create web interface
- ✅ 24/7 operation possible

**Main Challenge:**
- ⚠️ EC2 instances don't have cameras - need to stream from client or use video files

**Solution:**
- Web interface that streams frames from browser to EC2
- Or process uploaded video files
- Or connect external camera via network

**I can create a complete EC2 deployment setup if you're interested!**


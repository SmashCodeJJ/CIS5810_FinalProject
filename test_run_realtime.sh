#!/bin/bash
# Test run of real-time face swapping (with timeout for testing)
# This will run for 10 seconds then stop

cd "/Users/apple/Documents/Upenn Academy/CIS5810/Final Project/sber-swap"

echo "🚀 Starting real-time face swapping test..."
echo "⚠️  This will run for 10 seconds, then stop automatically"
echo ""

# Run with timeout (10 seconds)
timeout 10s python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --camera_id 0 \
    --fast_mode \
    2>&1 || echo "Test completed or timeout reached"

echo ""
echo "✅ Test finished!"


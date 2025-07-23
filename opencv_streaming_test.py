import cv2

# Load the face detection classifier from local directory
face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')

# Check if classifier loaded successfully
if face_cascade.empty():
    print("Error: Could not load face cascade classifier")
    print("Make sure 'haarcascade_frontalface_default.xml' is in your directory")
    exit()
else:
    print("Face cascade loaded successfully")

# Try different camera backends for better compatibility
backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
cap = None

for backend in backends_to_try:
    print(f"Trying camera with backend: {backend}")
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        # Test if we can actually read a frame
        ret, test_frame = cap.read()
        if ret:
            print(f"Camera opened successfully with backend: {backend}")
            break
        else:
            cap.release()
            cap = None
    else:
        if cap:
            cap.release()
        cap = None

if cap is None or not cap.isOpened():
    print("Error: Could not open video capture with any backend")
    exit()

# Set resolution and framerate (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Verify actual resolution
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")

# GStreamer pipeline to send video via UDP to host PC
host_ip = "192.168.1.100"  # replace with your host PC's IP
gst_str = (
    f'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=1000 speed-preset=superfast '
    f'! rtph264pay config-interval=1 pt=96 ! udpsink host={host_ip} port=5000'
)

# OpenCV VideoWriter for sending via GStreamer
out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (int(actual_width), int(actual_height)))

if not out.isOpened():
    print("Error: Could not open video writer")
    exit()

print("Streaming with face detection... Press Ctrl+C to stop")
frame_count = 0
detection_interval = 1  # Only run detection every N frames
last_faces = []  # Store last detected faces for non-detection frames

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print status every 30 frames
            print(f"Processing frame {frame_count}")
        
        # Only run face detection every N frames for better performance
        if frame_count % detection_interval == 0:
            # Resize frame for faster detection (smaller = faster)
            detection_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces on smaller frame
            faces_small = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,  # Slightly larger steps for speed
                minNeighbors=3,   # Reduced for speed
                minSize=(20, 20), # Smaller minimum size due to resize
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Scale face coordinates back to original frame size
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240
            last_faces = [(int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)) 
                         for (x, y, w, h) in faces_small]
            
            # Debug: Print number of faces detected
            if len(last_faces) > 0:
                print(f"Detected {len(last_faces)} face(s) in frame {frame_count}")
        
        # Always draw rectangles using last detected faces
        for (x, y, w, h) in last_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Send the frame with face detection overlay to the stream
        out.write(frame)

except KeyboardInterrupt:
    print("Streaming stopped")
finally:
    cap.release()
    out.release()
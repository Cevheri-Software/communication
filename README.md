# Jetson Xavier NX → Host PC Video Streaming Setup (via Wireless Bridge)
---

## 🧩 Hardware Setup

### 1. Camera Connection
- Connect your **USB camera** (Logitech, Osmo Action) to the **Jetson Xavier NX** using a standard **USB-A to USB cable**.
  - ✅ Plug the USB cable into one of the **USB 3.0 (blue)** ports of the Jetson for better bandwidth.
  - If using a USB-C camera (like DJI Osmo Action 3), use a **USB-C to USB-A adapter**.

### 2. Optional PC Receiver
- Ensure the **Jetson** and **PC** are connected to the **same local network** (Wi-Fi or Ethernet).
- For lower latency and better throughput, **wired Ethernet is preferred**.

---

## 📡 Network Configuration

| Device              | IP Address     |
|---------------------|----------------|
| Host PC             | 192.168.1.100  |
| UBNT LAP-120 (AP)   | 192.168.1.101  |
| AirMax Bullet AC    | 192.168.1.102  |
| Jetson Xavier NX    | 192.168.1.103  |

- **Jetson Xavier NX** is connected to the **AirMax Bullet AC** (station mode).
- **Host PC** is connected to the **UBNT LAP-120** (access point mode).
- These two devices communicate wirelessly over the bridge.

## 🔌 Physical Wiring

Will be added..

## 🎥 Video Streaming Setup (RAW)

```bash
sudo apt update
sudo apt install v4l-utils
```

To list all connected video devices: ```bash v4l2-ctl --list-devices ```
Usually your camera will appear as /dev/video0 or /dev/video1.
To test the camera locally:

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! "image/jpeg,width=640,height=480,framerate=30/1" ! jpegdec ! autovideosink
```
Tp check supported formats and frame sizes:
```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

## USB Webcam
Captures video from a USB webcam and streams over UDP in H.264 format:

### On Jetson Xavier NX (192.168.1.103)
```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! 'image/jpeg,width=640,height=480,framerate=30/1' ! jpegdec ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=192.168.1.100 port=5000
```
### On Host PC (192.168.1.100) 

```bash
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264" ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink 
```

## DJI Osmo Action Cam
Captures video from DJI Cam and streams over UDP in H.264 format:

### On Jetson Xavier NX (192.168.1.103)
```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
'image/jpeg, width=1280, height=720, framerate=30/1' ! \
jpegdec ! videoconvert ! x264enc tune=zerolatency bitrate=1000 speed-preset=superfast ! \
rtph264pay ! udpsink host=192.168.1.100 port=5000
```

### On Host PC (192.168.1.100) 
```bash
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264" ! \
rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

### Optimization Tip (Skip JPEG decode & H.264 re-encode)

Since your camera already supports H.264, you can avoid decoding MJPEG and re-encoding to H.264 — this reduces CPU usage and latency.

Try this zero-copy pipeline (on Jetson):

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
'video/x-h264, width=1280, height=720, framerate=30/1' ! \
h264parse ! rtph264pay config-interval=1 pt=96 ! \
udpsink host=192.168.1.100 port=5000
```
This directly streams the H.264 from the webcam.

Receiver remains the same.

### opencv_streaming_test.py 

Run the ```opencv_streaming_test.py``` and then run this script in your HOST PC:
```bash
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

"""
Optimized Real-Time Detection System for Jetson Xavier NX
Triple model approach: YOLOv8 (vehicles) + gun_detecktor.pt (guns) + license_plate_detector.pt (plates)
Minimal logging for maximum performance
"""

from ultralytics import YOLO
import cv2
import torch
import numpy as np
import time
import threading
import queue
import re
from collections import defaultdict, deque

class OptimizedTracker:
    """Lightweight object tracker optimized for Jetson"""
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())

            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > 50:  # Distance threshold
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col])

        # Return tracking results
        tracking_results = {}
        for object_id, centroid in self.objects.items():
            # Find the corresponding rectangle
            for i, (x1, y1, x2, y2) in enumerate(rects):
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                if abs(cx - centroid[0]) < 5 and abs(cy - centroid[1]) < 5:
                    tracking_results[object_id] = (x1, y1, x2, y2)
                    break
        
        return tracking_results

class DroneDetectionSystem:
    def __init__(self, 
                 source=0,
                 output_ip="192.168.1.100", 
                 output_port=5000,
                 vehicle_model_path='yolov8n.pt',
                 gun_model_path='gun_detecktor.pt',
                 plate_model_path='license_plate_detector.pt',
                 input_size=640,
                 confidence_threshold=0.6,
                 gun_confidence_threshold=0.4,
                 plate_confidence_threshold=0.5):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Performance settings
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.gun_confidence_threshold = gun_confidence_threshold
        self.plate_confidence_threshold = plate_confidence_threshold
        self.skip_frames = 3  # Process plates every 4th frame
        self.frame_count = 0
        
        # Detection classes (COCO)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Load all three models
        self.load_models(vehicle_model_path, gun_model_path, plate_model_path)
        
        # Initialize trackers
        self.vehicle_tracker = OptimizedTracker(max_disappeared=15)
        self.gun_tracker = OptimizedTracker(max_disappeared=5)
        self.plate_tracker = OptimizedTracker(max_disappeared=8)
        
        # Results storage (in-memory only)
        self.detected_plates = set()
        
        # Threading setup
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Setup video input/output
        self.setup_input(source)
        self.setup_output(output_ip, output_port)
        
        # OCR setup
        self.setup_ocr()
    
    def load_models(self, vehicle_model_path, gun_model_path, plate_model_path):
        """Load all three YOLO models"""
        try:
            # Load vehicle detection model
            self.vehicle_model = YOLO(vehicle_model_path)
            self.vehicle_model.to(self.device)
            self.vehicle_model.fuse()
            
            # Load gun detection model
            self.gun_model = YOLO(gun_model_path)
            self.gun_model.to(self.device)
            self.gun_model.fuse()
            
            # Load license plate detection model
            self.plate_model = YOLO(plate_model_path)
            self.plate_model.to(self.device)
            self.plate_model.fuse()
            
            # Optimize all models for inference
            if self.device == 'cuda':
                self.vehicle_model.model.half()
                self.gun_model.model.half()
                self.plate_model.model.half()
                torch.backends.cudnn.benchmark = True
            
            print("✅ All three models loaded successfully")
            
        except Exception as e:
            print(f"Failed to load models: {e}")
            raise
    
    def setup_input(self, source):
        """Setup video input with optimization for Jetson"""
        if isinstance(source, str) and source.startswith('rtsp://'):
            # Optimized RTSP pipeline for Jetson
            pipeline = (
                f"rtspsrc location={source} latency=0 drop-on-latency=true ! "
                "rtph264depay ! h264parse ! nvv4l2decoder ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # Try different camera backends
            backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
            self.cap = None
            
            for backend in backends_to_try:
                self.cap = cv2.VideoCapture(source, backend)
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret:
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    if self.cap:
                        self.cap.release()
                    self.cap = None
            
            if self.cap is None or not self.cap.isOpened():
                raise Exception(f"Could not open video source: {source}")
            
            # Optimize camera settings for Jetson
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual resolution
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open video source: {source}")
    
    def setup_output(self, ip, port):
        """Setup GStreamer output"""
        # Hardware encoding pipeline
        gst_str = (
            f'appsrc ! videoconvert ! nvvidconv ! '
            f'nvv4l2h264enc bitrate=2000000 preset-level=1 ! '
            f'h264parse ! rtph264pay config-interval=1 pt=96 ! '
            f'udpsink host={ip} port={port} sync=false'
        )
        
        try:
            self.out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 30, (self.actual_width, self.actual_height))
            
            if not self.out.isOpened():
                # Fallback to software encoding
                gst_str_fallback = (
                    f'appsrc ! videoconvert ! '
                    f'x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! '
                    f'rtph264pay ! '
                    f'udpsink host={ip} port={port} sync=false'
                )
                self.out = cv2.VideoWriter(gst_str_fallback, cv2.CAP_GSTREAMER, 30, (self.actual_width, self.actual_height))
                
                if not self.out.isOpened():
                    self.out = None
                    
        except Exception as e:
            self.out = None
    
    def setup_ocr(self):
        """Setup EasyOCR for license plates"""
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        except:
            self.ocr_reader = None
    
    def detect_objects(self, frame):
        """Run triple YOLO detection - vehicles, guns, and license plates"""
        try:
            # Resize frame for faster inference
            inference_frame = cv2.resize(frame, (416, 416))
            
            vehicles = []
            guns = []
            plates = []
            
            # Vehicle detection
            vehicle_results = self.vehicle_model(inference_frame, 
                                               imgsz=416,
                                               conf=self.confidence_threshold,
                                               iou=0.4,
                                               device=self.device,
                                               verbose=False)[0]
            
            # Gun detection
            gun_results = self.gun_model(inference_frame,
                                       imgsz=416,
                                       conf=self.gun_confidence_threshold,
                                       iou=0.3,
                                       device=self.device,
                                       verbose=False)[0]
            
            # License plate detection (every few frames)
            if self.frame_count % self.skip_frames == 0:
                plate_results = self.plate_model(inference_frame,
                                               imgsz=416,
                                               conf=self.plate_confidence_threshold,
                                               iou=0.4,
                                               device=self.device,
                                               verbose=False)[0]
            else:
                plate_results = None
            
            # Scale factors for coordinate conversion
            scale_x = frame.shape[1] / 416
            scale_y = frame.shape[0] / 416
            
            # Process vehicle detections
            if vehicle_results.boxes is not None:
                boxes = vehicle_results.boxes.xyxy.cpu().numpy()
                confidences = vehicle_results.boxes.conf.cpu().numpy()
                classes = vehicle_results.boxes.cls.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = confidences[i]
                    cls = int(classes[i])
                    
                    # Scale coordinates back to original frame
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(frame.shape[1], x1))
                    y1 = max(0, min(frame.shape[0], y1))
                    x2 = max(0, min(frame.shape[1], x2))
                    y2 = max(0, min(frame.shape[0], y2))
                    
                    # Filter for vehicle classes
                    if cls in self.vehicle_classes:
                        # Skip small detections
                        if x2 - x1 < 80 or y2 - y1 < 60:
                            continue
                        vehicles.append((x1, y1, x2, y2, conf, cls))
            
            # Process gun detections
            if gun_results.boxes is not None:
                gun_boxes = gun_results.boxes.xyxy.cpu().numpy()
                gun_confidences = gun_results.boxes.conf.cpu().numpy()
                
                for i, box in enumerate(gun_boxes):
                    x1, y1, x2, y2 = box
                    conf = gun_confidences[i]
                    
                    # Scale coordinates back to original frame
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(frame.shape[1], x1))
                    y1 = max(0, min(frame.shape[0], y1))
                    x2 = max(0, min(frame.shape[1], x2))
                    y2 = max(0, min(frame.shape[0], y2))
                    
                    # Filter by area for guns
                    area = (x2 - x1) * (y2 - y1)
                    if 500 < area < 50000:
                        guns.append((x1, y1, x2, y2, conf))
            
            # Process license plate detections
            if plate_results is not None and plate_results.boxes is not None:
                plate_boxes = plate_results.boxes.xyxy.cpu().numpy()
                plate_confidences = plate_results.boxes.conf.cpu().numpy()
                
                for i, box in enumerate(plate_boxes):
                    x1, y1, x2, y2 = box
                    conf = plate_confidences[i]
                    
                    # Scale coordinates back to original frame
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(frame.shape[1], x1))
                    y1 = max(0, min(frame.shape[0], y1))
                    x2 = max(0, min(frame.shape[1], x2))
                    y2 = max(0, min(frame.shape[0], y2))
                    
                    # Filter by size and aspect ratio for license plates
                    w, h = x2 - x1, y2 - y1
                    aspect_ratio = w / h if h > 0 else 0
                    area = w * h
                    
                    if (1.5 < aspect_ratio < 8.0 and 
                        800 < area < 20000 and
                        w > 40 and h > 10):
                        plates.append((x1, y1, x2, y2, conf))
            
            return vehicles, guns, plates
            
        except Exception as e:
            return [], [], []
    
    def extract_license_plate_text(self, plate_crop):
        """Extract text from license plate crop using EasyOCR"""
        if self.ocr_reader is None or plate_crop.size == 0:
            return None, 0
        
        try:
            # Preprocess the crop
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            results = self.ocr_reader.readtext(thresh, detail=1)
            
            if results:
                # Get best result
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].upper()
                confidence = best_result[2]
                
                # Clean text (remove non-alphanumeric)
                text = re.sub(r'[^A-Z0-9]', '', text)
                
                if len(text) >= 4 and confidence > 0.3:
                    return text, confidence
            
            return None, 0
            
        except:
            return None, 0
    
    def process_frame(self, frame):
        """Main frame processing function"""
        self.frame_count += 1
        
        # Detection with triple models
        vehicles, guns, plates = self.detect_objects(frame)
        
        # Vehicle tracking
        vehicle_rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf, cls in vehicles]
        tracked_vehicles = self.vehicle_tracker.update(vehicle_rects)
        
        # Gun tracking
        gun_rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf in guns]
        tracked_guns = self.gun_tracker.update(gun_rects)
        
        # License plate tracking and OCR
        plate_rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf in plates]
        tracked_plates = self.plate_tracker.update(plate_rects)
        
        # Process detected plates with OCR
        for x1, y1, x2, y2, conf in plates:
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size > 0:
                text, ocr_conf = self.extract_license_plate_text(plate_crop)
                if text and text not in self.detected_plates:
                    self.detected_plates.add(text)
        
        # Draw all detections
        self.draw_detections(frame, tracked_vehicles, tracked_guns, tracked_plates, vehicles, guns, plates)
        
        return frame
    
    def draw_detections(self, frame, tracked_vehicles, tracked_guns, tracked_plates, vehicles, guns, plates):
        """Draw all detection results on frame"""
        # Draw tracked vehicles
        for track_id, (x1, y1, x2, y2) in tracked_vehicles.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'V{track_id}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw tracked guns with enhanced visibility
        for track_id, (x1, y1, x2, y2) in tracked_guns.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            text = f'ARMED HUMAN{track_id}'
            cv2.putText(frame, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw license plates and their text
        for x1, y1, x2, y2, conf in plates:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Try to extract and display text
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size > 0:
                text, ocr_conf = self.extract_license_plate_text(plate_crop)
                if text:
                    cv2.putText(frame, text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Info overlay
        cv2.putText(frame, f'V:{len(tracked_vehicles)} G:{len(tracked_guns)} P:{len(self.detected_plates)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Gun alert overlay
        if len(tracked_guns) > 0:
            cv2.putText(frame, '🚨 WEAPON ALERT 🚨', 
                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    def capture_thread(self):
        """Thread for capturing frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            else:
                break
    
    def process_thread(self):
        """Thread for processing frames"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                processed_frame = self.process_frame(frame)
                
                if not self.result_queue.full():
                    self.result_queue.put(processed_frame)
                
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except:
                pass
    
    def run(self):
        """Main execution loop"""
        print("🚀 TRIPLE-MODEL DETECTION SYSTEM STARTED")
        print("🚗 Vehicles: YOLOv8 | 🔫 Guns: gun_detecktor.pt | 🚙 Plates: license_plate_detector.pt")
        
        self.running = True
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_thread)
        process_thread = threading.Thread(target=self.process_thread)
        
        capture_thread.start()
        process_thread.start()
        
        frame_count = 0
        start_time = time.time()
        last_fps_time = time.time()
        
        try:
            while self.running:
                try:
                    # Get processed frame
                    processed_frame = self.result_queue.get(timeout=1)
                    frame_count += 1
                    
                    # Send to output stream
                    if self.out is not None:
                        self.out.write(processed_frame)
                    
                    # Display FPS every 5 seconds
                    current_time = time.time()
                    if current_time - last_fps_time >= 5.0:
                        fps = frame_count / (current_time - start_time)
                        guns_detected = len(self.gun_tracker.objects)
                        if guns_detected > 0:
                            print(f"🚨 ALERT: {guns_detected} GUNS | FPS: {fps:.1f}")
                        else:
                            print(f"FPS: {fps:.1f} | V:{len(self.vehicle_tracker.objects)} | P:{len(self.detected_plates)}")
                        last_fps_time = current_time
                    
                    self.result_queue.task_done()
                    
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
        
        except Exception as e:
            print(f"System error: {e}")
        finally:
            self.running = False
            capture_thread.join()
            process_thread.join()
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Triple-Model Detection System')
    parser.add_argument('--source', default=0, help='Video source')
    parser.add_argument('--output-ip', default='192.168.1.100', help='Output IP')
    parser.add_argument('--output-port', type=int, default=5000, help='Output port')
    parser.add_argument('--vehicle-model', default='yolov8n.pt', help='Vehicle model')
    parser.add_argument('--gun-model', default='gun_detecktor.pt', help='Gun model')
    parser.add_argument('--plate-model', default='license_plate_detector.pt', help='License plate model')
    parser.add_argument('--confidence', type=float, default=0.6, help='Vehicle confidence')
    parser.add_argument('--gun-confidence', type=float, default=0.4, help='Gun confidence')
    parser.add_argument('--plate-confidence', type=float, default=0.5, help='Plate confidence')
    
    args = parser.parse_args()
    
    try:
        system = DroneDetectionSystem(
            source=args.source,
            output_ip=args.output_ip,
            output_port=args.output_port,
            vehicle_model_path=args.vehicle_model,
            gun_model_path=args.gun_model,
            plate_model_path=args.plate_model,
            confidence_threshold=args.confidence,
            gun_confidence_threshold=args.gun_confidence,
            plate_confidence_threshold=args.plate_confidence
        )
        system.run()
        
    except Exception as e:
        print(f"System failed: {e}")

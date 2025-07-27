"""
Fixed Real-Time Detection System with GUI Support
Includes local display window and improved streaming
Works on both Jetson Xavier NX and regular PCs
"""

from ultralytics import YOLO
import cv2
import torch
import numpy as np
import time
import threading
import queue
import re
import logging
from collections import defaultdict, deque

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                 model_path='yolov8n.pt',
                 input_size=640,
                 confidence_threshold=0.6,
                 show_gui=True,
                 enable_streaming=True):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # GUI and streaming options
        self.show_gui = show_gui
        self.enable_streaming = enable_streaming
        
        # Performance settings
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.skip_frames = 2  # Process every 2nd frame for better performance
        self.frame_count = 0
        
        # Detection classes (COCO)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Load model
        self.load_model(model_path)
        
        # Initialize tracker
        self.vehicle_tracker = OptimizedTracker(max_disappeared=15)
        
        # Results storage (in-memory only)
        self.detected_plates = set()
        
        # Threading setup
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Setup video input
        self.setup_input(source)
        
        # Setup output stream if enabled
        if self.enable_streaming:
            self.setup_output(output_ip, output_port)
        else:
            self.out = None
        
        # OCR setup
        self.setup_ocr()
    
    def load_model(self, model_path):
        """Load optimized YOLO model"""
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Optimize model
            self.model.fuse()  # Fuse layers for faster inference
            
            # Additional optimizations for inference
            if self.device == 'cuda':
                try:
                    self.model.model.half()  # Use FP16 for speed
                    torch.backends.cudnn.benchmark = True
                except:
                    logger.warning("Could not enable FP16, using FP32")
            
            logger.info(f"Model loaded and optimized: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_input(self, source):
        """Setup video input with fallback options"""
        logger.info(f"Setting up video input: {source}")
        
        if isinstance(source, str) and source.startswith('rtsp://'):
            # RTSP input
            self.cap = cv2.VideoCapture(source)
        else:
            # Camera input - try multiple backends
            backends_to_try = [cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY]
            self.cap = None
            
            for backend in backends_to_try:
                logger.info(f"Trying camera with backend: {backend}")
                try:
                    self.cap = cv2.VideoCapture(source, backend)
                    if self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"Camera opened successfully with backend: {backend}")
                            break
                        else:
                            self.cap.release()
                            self.cap = None
                    else:
                        if self.cap:
                            self.cap.release()
                        self.cap = None
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
            
            if self.cap is None or not self.cap.isOpened():
                raise Exception(f"Could not open video source with any backend: {source}")
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual resolution
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera resolution: {self.actual_width}x{self.actual_height}")
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open video source: {source}")
        
        logger.info("Video input ready")
    
    def setup_output(self, ip, port):
        """Setup video output with multiple fallback options"""
        logger.info(f"Setting up stream output: {ip}:{port}")
        
        # Try multiple encoding pipelines
        pipelines_to_try = [
            # Hardware encoding (Jetson)
            (f'appsrc ! videoconvert ! nvvidconv ! '
             f'nvv4l2h264enc bitrate=2000000 preset-level=1 ! '
             f'h264parse ! rtph264pay config-interval=1 pt=96 ! '
             f'udpsink host={ip} port={port} sync=false', "Hardware (Jetson)"),
            
            # Software encoding with x264
            (f'appsrc ! videoconvert ! '
             f'x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! '
             f'rtph264pay config-interval=1 pt=96 ! '
             f'udpsink host={ip} port={port} sync=false', "Software (x264)"),
            
            # Basic software encoding
            (f'appsrc ! videoconvert ! '
             f'avenc_h264_omx bitrate=2000000 ! '
             f'rtph264pay ! '
             f'udpsink host={ip} port={port} sync=false', "OMX Hardware"),
            
            # Fallback - very basic
            (f'appsrc ! videoconvert ! '
             f'theoraenc ! oggmux ! '
             f'udpsink host={ip} port={port} sync=false', "Theora fallback")
        ]
        
        self.out = None
        
        for pipeline, name in pipelines_to_try:
            try:
                logger.info(f"Trying {name} encoding...")
                self.out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, 30, 
                                         (self.actual_width, self.actual_height))
                
                if self.out.isOpened():
                    logger.info(f"✅ Video output stream ready with {name}")
                    break
                else:
                    if self.out:
                        self.out.release()
                    self.out = None
                    
            except Exception as e:
                logger.warning(f"{name} failed: {e}")
                if self.out:
                    self.out.release()
                self.out = None
        
        if self.out is None:
            logger.warning("❌ All streaming methods failed - continuing without streaming")
    
    def setup_ocr(self):
        """Setup OCR for license plates"""
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            logger.info("OCR initialized with EasyOCR")
        except ImportError:
            logger.warning("EasyOCR not available - install with: pip install easyocr")
            self.ocr_reader = None
            # Try alternative OCR
            try:
                import pytesseract
                self.ocr_reader = "tesseract"
                logger.info("Using Tesseract as OCR fallback")
            except ImportError:
                logger.warning("No OCR available - license plate text won't be extracted")
                self.ocr_reader = None
        except Exception as e:
            logger.error(f"OCR setup failed: {e}")
            self.ocr_reader = None
    
    def detect_objects(self, frame):
        """Run YOLO detection on frame with optimizations"""
        try:
            # Resize frame for faster inference
            inference_frame = cv2.resize(frame, (416, 416))
            
            # Run inference with optimized parameters
            results = self.model(inference_frame, 
                               imgsz=416,
                               conf=self.confidence_threshold,
                               iou=0.4,
                               device=self.device,
                               verbose=False)[0]
            
            vehicles = []
            
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                # Scale boxes back to original frame size
                scale_x = frame.shape[1] / 416
                scale_y = frame.shape[0] / 416
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = confidences[i]
                    cls = int(classes[i])
                    
                    # Scale coordinates back
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(frame.shape[1], x1))
                    y1 = max(0, min(frame.shape[0], y1))
                    x2 = max(0, min(frame.shape[1], x2))
                    y2 = max(0, min(frame.shape[0], y2))
                    
                    # Filter detections - only vehicles
                    if cls in self.vehicle_classes:
                        # Skip small detections
                        if x2 - x1 < 60 or y2 - y1 < 40:
                            continue
                        vehicles.append((x1, y1, x2, y2, conf, cls))
            
            return vehicles
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def extract_license_plate_text(self, plate_crop):
        """Extract license plate text using available OCR"""
        if self.ocr_reader is None or plate_crop.size == 0:
            return None, 0
        
        try:
            # Preprocess image
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            # Resize if too small
            h, w = gray.shape
            if h < 40 or w < 100:
                scale_factor = max(40/h, 100/w)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Apply filters
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            
            if self.ocr_reader == "tesseract":
                # Use Tesseract
                import pytesseract
                
                # Try different preprocessing
                methods = [
                    gray,
                    cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                    cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                ]
                
                best_text = None
                best_confidence = 0
                
                for processed_img in methods:
                    try:
                        text = pytesseract.image_to_string(processed_img, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        text = text.strip().upper()
                        text = re.sub(r'[^A-Z0-9]', '', text)
                        
                        if self.is_valid_license_plate(text):
                            best_text = text
                            best_confidence = 0.8  # Assume good confidence for tesseract
                            break
                    except:
                        continue
                
                return best_text, best_confidence
            
            else:
                # Use EasyOCR
                results = self.ocr_reader.readtext(gray, 
                                                 detail=1,
                                                 paragraph=False,
                                                 width_ths=0.7,
                                                 height_ths=0.7)
                
                if results:
                    for result in results:
                        text = result[1].upper().strip()
                        confidence = result[2]
                        
                        # Clean text
                        text = re.sub(r'[^A-Z0-9\s\-]', '', text)
                        text = re.sub(r'\s+', '', text)
                        
                        if self.is_valid_license_plate(text) and confidence > 0.3:
                            return text, confidence
                
                return None, 0
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None, 0
    
    def is_valid_license_plate(self, text):
        """Validate if text looks like a license plate"""
        if not text or len(text) < 4:
            return False
        
        # Remove common OCR errors
        text = text.replace('O', '0').replace('I', '1').replace('S', '5')
        
        # Check for reasonable license plate patterns
        if 4 <= len(text) <= 8:
            has_letter = any(c.isalpha() for c in text)
            has_number = any(c.isdigit() for c in text)
            
            if has_letter and has_number:
                return True
            elif text.isdigit() and len(text) >= 5:
                return True
        
        return False
    
    def process_license_plates(self, frame, vehicles, tracked_vehicles):
        """Process license plates within detected vehicles"""
        for x1, y1, x2, y2, conf, cls in vehicles:
            # Focus on bottom area of vehicle where plates are typically located
            vehicle_h = y2 - y1
            
            # Define search area (bottom portion of vehicle)
            search_y1 = max(0, y1 + int(vehicle_h * 0.6))
            search_y2 = min(frame.shape[0], y2 + 20)
            search_x1 = max(0, x1 - 10)
            search_x2 = min(frame.shape[1], x2 + 10)
            
            # Extract search region
            search_crop = frame[search_y1:search_y2, search_x1:search_x2]
            
            if search_crop.size > 0:
                self.detect_license_plate_regions(frame, search_crop, search_x1, search_y1)
    
    def detect_license_plate_regions(self, frame, vehicle_crop, offset_x, offset_y):
        """Detect license plate regions using contour analysis"""
        try:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 30, 100, apertureSize=3)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:5]:  # Check top 5 contours
                x, y, w, h = cv2.boundingRect(contour)
                
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # License plate criteria
                if (2.0 < aspect_ratio < 6.0 and
                    500 < area < 15000 and
                    w > 60 and h > 15 and
                    w < vehicle_crop.shape[1] * 0.9 and
                    h < vehicle_crop.shape[0] * 0.7):
                    
                    # Extract potential license plate
                    plate_crop = vehicle_crop[y:y+h, x:x+w]
                    
                    if plate_crop.size > 0:
                        # Try to read text
                        text, confidence = self.extract_license_plate_text(plate_crop)
                        
                        if text and text not in self.detected_plates and confidence > 0.3:
                            self.detected_plates.add(text)
                            
                            # Draw on frame
                            abs_x1 = offset_x + x
                            abs_y1 = offset_y + y
                            abs_x2 = abs_x1 + w
                            abs_y2 = abs_y1 + h
                            
                            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
                            cv2.putText(frame, f'{text} ({confidence:.2f})', (abs_x1, abs_y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            
                            logger.info(f"License plate detected: {text} (confidence: {confidence:.2f})")
        
        except Exception as e:
            logger.error(f"License plate detection error: {e}")
    
    def draw_detections(self, frame, tracked_vehicles, vehicles):
        """Draw detection results on frame"""
        # Draw tracked vehicles
        for track_id, (x1, y1, x2, y2) in tracked_vehicles.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Vehicle {track_id}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add info overlay
        cv2.putText(frame, f'Vehicles: {len(tracked_vehicles)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'License Plates: {len(self.detected_plates)}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add controls info
        cv2.putText(frame, 'Press Q to quit', 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        """Main frame processing function"""
        self.frame_count += 1
        
        # Detection
        vehicles = self.detect_objects(frame)
        
        # Vehicle tracking
        vehicle_rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf, cls in vehicles]
        tracked_vehicles = self.vehicle_tracker.update(vehicle_rects)
        
        # Draw detections
        self.draw_detections(frame, tracked_vehicles, vehicles)
        
        # Process license plates less frequently for performance
        if self.frame_count % self.skip_frames == 0:
            self.process_license_plates(frame, vehicles, tracked_vehicles)
        
        return frame
    
    def run(self):
        """Main execution loop with GUI support"""
        logger.info("=" * 60)
        logger.info("🚗 VEHICLE & LICENSE PLATE DETECTION SYSTEM STARTED")
        logger.info("=" * 60)
        
        if self.show_gui:
            logger.info("GUI Mode: Press 'Q' to quit")
        if self.enable_streaming and self.out:
            logger.info("Streaming enabled")
        
        frame_count = 0
        start_time = time.time()
        last_fps_time = time.time()
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                frame_count += 1
                
                # Show GUI if enabled
                if self.show_gui:
                    cv2.imshow('Vehicle & License Plate Detection', processed_frame)
                    
                    # Handle key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        logger.info("Quit requested by user")
                        break
                
                # Send to output stream if enabled
                if self.enable_streaming and self.out is not None:
                    try:
                        self.out.write(processed_frame)
                    except Exception as e:
                        logger.warning(f"Streaming error: {e}")
                
                # Calculate and display FPS every 5 seconds
                current_time = time.time()
                if current_time - last_fps_time >= 5.0:
                    fps = frame_count / (current_time - start_time)
                    logger.info(f"FPS: {fps:.1f} | Frames: {frame_count} | "
                              f"Vehicles: {len(self.vehicle_tracker.objects)} | "
                              f"License Plates: {len(self.detected_plates)}")
                    last_fps_time = current_time
        
        except KeyboardInterrupt:
            logger.info("Stopping system...")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()
        
        if self.show_gui:
            cv2.destroyAllWindows()
        
        logger.info("System cleanup complete")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vehicle & License Plate Detection System')
    parser.add_argument('--source', default=0, help='Video source (camera ID or RTSP URL)')
    parser.add_argument('--output-ip', default='192.168.1.100', help='Output stream IP')
    parser.add_argument('--output-port', type=int, default=5000, help='Output stream port')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    parser.add_argument('--confidence', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI display')
    parser.add_argument('--no-streaming', action='store_true', help='Disable video streaming')
    
    args = parser.parse_args()
    
    try:
        # Create and run detection system
        system = DroneDetectionSystem(
            source=args.source,
            output_ip=args.output_ip,
            output_port=args.output_port,
            model_path=args.model,
            input_size=args.input_size,
            confidence_threshold=args.confidence,
            show_gui=not args.no_gui,
            enable_streaming=not args.no_streaming
        )
        system.run()
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        logger.info("Make sure all dependencies are installed and camera is connected")

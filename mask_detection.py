from ultralytics import YOLO
import cv2
import time
import numpy as np
from typing import Tuple, List
import logging

class MaskDetector:
    def __init__(self, model_path: str, camera_id: int = 0, width: int = 640, height: int = 480):
        """Initialize the mask detection system."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Constants
        self.CONFIDENCE_THRESHOLD = 0.25
        self.CLASS_NAMES = ["masked", "unmasked"]
        self.COLORS = {
            "masked": (0, 255, 0),    # Green
            "unmasked": (0, 0, 255),  # Red
            "fps": (0, 255, 0),       # Green
            "text": (255, 255, 255)   # White
        }
        
        # Initialize components
        self._setup_camera(camera_id, width, height)
        self._load_model(model_path)
        
        # FPS calculation variables
        self.prev_time = time.time()
        self.frame_count = 0

    def _setup_camera(self, camera_id: int, width: int, height: int) -> None:
        """Initialize and configure the webcam."""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to initialize webcam")

    def _load_model(self, model_path: str) -> None:
        """Load the YOLO model."""
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _calculate_fps(self) -> float:
        """Calculate the current FPS."""
        self.frame_count += 1
        curr_time = time.time()
        fps = self.frame_count / (curr_time - self.prev_time)
        return fps

    def _draw_detection(self, img: np.ndarray, bbox: Tuple[int, int, int, int], 
                       label: str, confidence: float) -> np.ndarray:
        """Draw bounding box and label on the image."""
        x1, y1, x2, y2 = bbox
        color = self.COLORS[label]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background and text
        text = f"{label} ({confidence:.2f}%)"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, self.COLORS["text"], 2)
        
        return img

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Process a single frame and return the annotated image and detections."""
        results = self.model.predict(frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                try:
                    # Get detection details
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0]) * 100
                    cls = int(box.cls[0])
                    
                    if cls >= len(self.CLASS_NAMES):
                        self.logger.warning(f"Detected class index {cls} is out of range!")
                        continue

                    label = self.CLASS_NAMES[cls]
                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2)
                    })
                    
                    # Draw detection on frame
                    frame = self._draw_detection(frame, (x1, y1, x2, y2), label, confidence)
                    
                except Exception as e:
                    self.logger.error(f"Error processing detection: {e}")
                    continue

        return frame, detections

    def run(self):
        """Run the mask detection system."""
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    self.logger.error("Failed to read from webcam")
                    break

                # Process frame
                frame, detections = self.process_frame(frame)
                
                # Calculate and display FPS
                fps = self._calculate_fps()
                cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS["fps"], 2)

                # Display results
                cv2.imshow("Mask Detection", frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = MaskDetector(
            model_path=r'C:\Users\amank\Downloads\mask_detection\detect\train\weights\best.pt'
        )
        detector.run()
    except Exception as e:
        logging.error(f"Application error: {e}")
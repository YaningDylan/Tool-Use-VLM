"""YOLO object detection tool"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from ultralytics import YOLO
from ...base.tool_interface import VisionTool
from ...base.tool_result import ToolResult

@dataclass
class YOLODetection:
    """YOLO detection result"""
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (cx, cy)
    area: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'class_name': self.class_name,
            'class_id': self.class_id,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area
        }

class YOLODetector(VisionTool):
    """YOLO-based object detection tool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="yolo_detector",
            description="YOLO-based object detection tool",
            config=config
        )
        self.model = None
        self.model_path = self.config.get('model_path', 'yolov8n.pt')
        self.device = self.config.get('device', 'cpu')
    
    def initialize(self) -> bool:
        """Initialize YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.is_initialized = True
            print(f"YOLO model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
    
    def execute(self, image: np.ndarray, confidence_threshold: float = 0.5, 
                class_filter: List[str] = None) -> ToolResult:
        """
        Execute object detection
        
        Args:
            image: Input image array
            confidence_threshold: Detection confidence threshold
            class_filter: Filter by specific class names
            
        Returns:
            ToolResult containing detection results
        """
        if not self.is_initialized:
            return ToolResult(
                success=False,
                data=[],
                message="YOLO model not initialized",
                tool_name=self.name
            )
        
        if not self.validate_image(image):
            return ToolResult(
                success=False,
                data=[],
                message="Invalid input image",
                tool_name=self.name
            )
        
        try:
            # Run detection
            results = self.model(image, conf=confidence_threshold, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Apply class filter
                        if class_filter and class_name not in class_filter:
                            continue
                        
                        # Calculate center and area
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        area = (x2 - x1) * (y2 - y1)
                        
                        detection = YOLODetection(
                            class_name=class_name,
                            class_id=class_id,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            center=(center_x, center_y),
                            area=area
                        )
                        detections.append(detection)
            
            # Sort by confidence
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
            metadata = {
                'total_detections': len(detections),
                'unique_classes': len(set(d.class_name for d in detections)),
                'confidence_threshold': confidence_threshold,
                'image_shape': image.shape
            }
            
            return ToolResult(
                success=True,
                data=detections,
                message=f"Detected {len(detections)} objects",
                tool_name=self.name,
                metadata=metadata
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                message=f"YOLO detection failed: {str(e)}",
                tool_name=self.name
            )
    
    def detect_objects(self, image: np.ndarray, **kwargs) -> ToolResult:
        """Simplified detection interface"""
        return self.execute(image, **kwargs)
    
    def get_class_names(self) -> List[str]:
        """Get supported class names"""
        if self.model:
            return list(self.model.names.values())
        return []
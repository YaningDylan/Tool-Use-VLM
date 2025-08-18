"""Enhanced toolbase integrating universal tools with env-specific tools"""

import numpy as np
from typing import Dict, List, Any, Tuple

try:
    from vagen.toolbase import YOLODetector, ToolResult
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ YOLO tools not available: {e}")
    print("To enable YOLO: pip install ultralytics opencv-python")
    YOLO_AVAILABLE = False
    from dataclasses import dataclass
    @dataclass
    class ToolResult:
        success: bool
        data: Any
        message: str = ""
        tool_name: str = ""
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}

from .toolbase import ManiSkillTools, ToolCallParser

class EnhancedManiSkillTools(ManiSkillTools):
    """Enhanced ManiSkill tools with optional YOLO integration"""
    
    def __init__(self, env, config: Dict[str, Any] = None):
        super().__init__(env)
        
        self.config = config or {}
        self.yolo_detector = None
        self.yolo_enabled = YOLO_AVAILABLE and self.config.get('enable_yolo', True)
        
        # Initialize YOLO only if available and enabled
        if self.yolo_enabled:
            self._init_yolo()
        else:
            if not YOLO_AVAILABLE:
                print("⚠️ YOLO not available - enhanced vision tools disabled")
            else:
                print("ℹ️ YOLO disabled in configuration")
    
    def _init_yolo(self):
        """Initialize YOLO detector safely"""
        if not YOLO_AVAILABLE:
            return
            
        try:
            yolo_config = self.config.get('yolo', {})
            yolo_config.setdefault('model_path', 'yolov8n.pt')
            yolo_config.setdefault('device', 'cpu')
            
            from vagen.toolbase import YOLODetector
            self.yolo_detector = YOLODetector(yolo_config)
            success = self.yolo_detector.initialize()
            
            if success:
                print(f"✓ YOLO detector initialized")
            else:
                print(f"❌ YOLO detector failed to initialize")
                self.yolo_detector = None
                self.yolo_enabled = False
        except Exception as e:
            print(f"❌ YOLO initialization error: {e}")
            self.yolo_detector = None
            self.yolo_enabled = False
    
    def detect_scene_objects(self, confidence_threshold: float = 0.5) -> ToolResult:
        """Use YOLO to detect all objects in current scene"""
        if not self.yolo_enabled or not self.yolo_detector:
            return ToolResult(
                success=False,
                data={},
                message="YOLO detector not available. Install with: pip install ultralytics opencv-python",
                tool_name="detect_scene_objects"
            )
        
        try:
            # Get scene image from environment
            image = self._get_scene_image()
            if image is None:
                return ToolResult(
                    success=False,
                    data={},
                    message="Cannot get scene image from environment",
                    tool_name="detect_scene_objects"
                )
            
            # Run YOLO detection
            yolo_result = self.yolo_detector.detect_objects(image, confidence_threshold=confidence_threshold)
            
            if not yolo_result.success:
                return yolo_result
            
            # Enhance with environment information
            enhanced_data = self._enhance_with_env_info(yolo_result.data)
            
            return ToolResult(
                success=True,
                data=enhanced_data,
                message=f"YOLO detected {len(yolo_result.data)} objects: {self._get_class_summary(yolo_result.data)}",
                tool_name="detect_scene_objects",
                metadata=yolo_result.metadata
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Scene detection failed: {str(e)}",
                tool_name="detect_scene_objects"
            )
    
    def find_object_by_description(self, description: str) -> ToolResult:
        """Find object by natural language description"""
        if not self.yolo_enabled:
            return ToolResult(
                success=False,
                data=None,
                message="YOLO-based object finding not available. Install ultralytics package.",
                tool_name="find_object_by_description"
            )
        
        try:
            # First get YOLO detections
            scene_result = self.detect_scene_objects()
            if not scene_result.success:
                return scene_result
            
            # Simple description matching (can be enhanced with NLP)
            matches = []
            for item in scene_result.data['enhanced_detections']:
                if self._matches_description(description, item):
                    matches.append(item)
            
            if matches:
                # Sort by confidence and return best match
                best_match = max(matches, key=lambda x: x['yolo_detection']['confidence'])
                return ToolResult(
                    success=True,
                    data=best_match,
                    message=f"Found object matching '{description}': {best_match['yolo_detection']['class_name']}",
                    tool_name="find_object_by_description"
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"No object found matching '{description}'",
                    tool_name="find_object_by_description"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Object search failed: {str(e)}",
                tool_name="find_object_by_description"
            )
    
    def get_spatial_relationships(self) -> ToolResult:
        """Analyze spatial relationships between detected objects"""
        if not self.yolo_enabled:
            return ToolResult(
                success=False,
                data={'relationships': []},
                message="YOLO-based spatial analysis not available. Install ultralytics package.",
                tool_name="get_spatial_relationships"
            )
        
        try:
            scene_result = self.detect_scene_objects()
            if not scene_result.success:
                return scene_result
            
            detections = scene_result.data['enhanced_detections']
            if len(detections) < 2:
                return ToolResult(
                    success=True,
                    data={'relationships': []},
                    message="Need at least 2 objects for spatial analysis",
                    tool_name="get_spatial_relationships"
                )
            
            relationships = []
            for i, obj1 in enumerate(detections):
                for j, obj2 in enumerate(detections):
                    if i >= j:
                        continue
                    
                    relationship = self._analyze_relationship(obj1, obj2)
                    if relationship:
                        relationships.append(relationship)
            
            return ToolResult(
                success=True,
                data={'relationships': relationships},
                message=f"Analyzed {len(relationships)} spatial relationships",
                tool_name="get_spatial_relationships"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={'relationships': []},
                message=f"Spatial analysis failed: {str(e)}",
                tool_name="get_spatial_relationships"
            )
    
    def _get_scene_image(self) -> np.ndarray:
        try:
            if hasattr(self.env, 'env') and hasattr(self.env.env, 'render'):
                return self.env.env.render()
            return None
        except Exception as e:
            print(f"Warning: Could not get scene image: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _enhance_with_env_info(self, yolo_detections: List) -> Dict:
        """Enhance YOLO detections with environment position info"""
        env_state = self.env.get_env_state()
        
        enhanced_detections = []
        for detection in yolo_detections:
            env_position = self._find_matching_env_position(detection, env_state)
            
            enhanced_item = {
                'yolo_detection': detection.to_dict(),
                'env_position': env_position,
                'spatial_description': self._get_spatial_description(detection),
                'combined_info': {
                    'has_env_position': env_position is not None,
                    'pixel_center': detection.center,
                    'bbox_area': detection.area
                }
            }
            enhanced_detections.append(enhanced_item)
        
        return {
            'enhanced_detections': enhanced_detections,
            'env_state': env_state,
            'summary': {
                'total_objects': len(enhanced_detections),
                'with_env_positions': sum(1 for x in enhanced_detections if x['env_position'] is not None),
                'unique_classes': len(set(x['yolo_detection']['class_name'] for x in enhanced_detections))
            }
        }
    
    def _find_matching_env_position(self, detection, env_state: Dict) -> Tuple[int, int, int]:
        """Try to match YOLO detection with environment position"""
        class_name = detection.class_name.lower()
        
        for key, position in env_state.items():
            if any(keyword in key.lower() for keyword in [class_name, 'cube', 'apple', 'target']):
                if class_name in key.lower() or ('cube' in class_name and 'cube' in key.lower()):
                    try:
                        return tuple(int(x) for x in position)
                    except:
                        continue
        return None
    
    def _get_spatial_description(self, detection) -> str:
        """Generate spatial description from pixel coordinates"""
        cx, cy = detection.center
        img_width, img_height = 640, 480
        
        if cx < img_width * 0.33:
            horizontal = "left"
        elif cx < img_width * 0.67:
            horizontal = "center"
        else:
            horizontal = "right"
        
        if cy < img_height * 0.33:
            vertical = "top"
        elif cy < img_height * 0.67:
            vertical = "middle"
        else:
            vertical = "bottom"
        
        return f"{horizontal}-{vertical}"
    
    def _matches_description(self, description: str, enhanced_item: Dict) -> bool:
        """Check if object matches natural language description"""
        desc_lower = description.lower()
        yolo_data = enhanced_item['yolo_detection']
        spatial_desc = enhanced_item['spatial_description']
        
        if yolo_data['class_name'].lower() in desc_lower:
            return True
        
        colors = ['red', 'green', 'blue', 'yellow', 'white', 'black']
        for color in colors:
            if color in desc_lower and color in yolo_data['class_name'].lower():
                return True
        
        spatial_keywords = ['left', 'right', 'top', 'bottom', 'center', 'middle']
        for keyword in spatial_keywords:
            if keyword in desc_lower and keyword in spatial_desc:
                return True
        
        return False
    
    def _analyze_relationship(self, obj1: Dict, obj2: Dict) -> str:
        """Analyze spatial relationship between two objects"""
        center1 = obj1['yolo_detection']['center']
        center2 = obj2['yolo_detection']['center']
        
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        if abs(dx) > abs(dy):
            relation = "right of" if dx > 0 else "left of"
        else:
            relation = "below" if dy > 0 else "above"
        
        return f"{obj2['yolo_detection']['class_name']} is {relation} {obj1['yolo_detection']['class_name']}"
    
    def _get_class_summary(self, detections: List) -> str:
        """Get summary of detected classes"""
        class_counts = {}
        for detection in detections:
            class_name = detection.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary_parts = []
        for class_name, count in class_counts.items():
            if count == 1:
                summary_parts.append(class_name)
            else:
                summary_parts.append(f"{count} {class_name}s")
        
        return ", ".join(summary_parts)

class EnhancedToolCallParser(ToolCallParser):
    """Enhanced parser supporting both original and YOLO tools"""
    
    @staticmethod
    def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
        """Parse tool calls including enhanced YOLO tools"""
        # Enhanced tool mapping including original + YOLO tools
        tool_mapping = {
            # Original tools
            'get_pos': 'get_object_position_by_color',
            'get_object_position_by_color': 'get_object_position_by_color',
            'get_targets': 'get_target_positions',
            'get_target_positions': 'get_target_positions', 
            'get_objects': 'get_all_objects',
            'get_all_objects': 'get_all_objects',
            'get_workspace': 'get_workspace_limits',
            'get_workspace_limits': 'get_workspace_limits',
            
            # Enhanced YOLO tools
            'detect_scene': 'detect_scene_objects',
            'detect_scene_objects': 'detect_scene_objects',
            'find_object': 'find_object_by_description',
            'find_object_by_description': 'find_object_by_description',
            'get_spatial_relations': 'get_spatial_relationships',
            'get_spatial_relationships': 'get_spatial_relationships'
        }
        
        # Use parent class parsing logic with enhanced mapping
        import re
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, response)
        
        tool_calls = []
        for func_name, args_str in matches:
            if func_name in tool_mapping:
                args = []
                if args_str.strip():
                    args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
                
                tool_calls.append({
                    'tool_name': tool_mapping[func_name],
                    'arguments': args,
                    'original_call': f"{func_name}({args_str})"
                })
        
        return tool_calls

# 使用原有的ToolExecutor，避免导入冲突
from .toolbase import ToolExecutor

class EnhancedToolExecutor(ToolExecutor):
    """Enhanced executor supporting both original and YOLO tools"""
    
    def __init__(self, tools: EnhancedManiSkillTools):
        """Initialize with enhanced tools"""
        self.tools = tools

def get_enhanced_tools_description() -> str:
    """Get description of available tools"""
    base_description = """Available Tools:

Basic Environment Tools:
- get_objects(): List all visible objects in the scene
- get_pos(color): Get position of object by color (e.g., get_pos(red))
- get_targets(): Get all target positions in the scene
- get_workspace(): Get robot workspace boundaries"""
    
    if YOLO_AVAILABLE:
        enhanced_description = """

Enhanced YOLO Vision Tools:
- detect_scene(): Use YOLO to detect all objects with detailed visual information
- find_object(description): Find objects by natural language description
- get_spatial_relations(): Analyze spatial relationships between detected objects

Tool Usage Examples:
- detect_scene() → Returns comprehensive YOLO detection results
- find_object("red cube") → Returns specific object matching description
- get_spatial_relations() → Returns object relationships"""
        return base_description + enhanced_description
    else:
        return base_description + "\n\nNote: Enhanced YOLO tools disabled (install ultralytics for full functionality)"
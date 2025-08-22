import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re

@dataclass 
class ObjectInfo:
    """Complete information about a detected object"""
    object_id: int
    object_name: str
    object_type: str  # cube, target, robot_part, etc.
    color: Optional[str]  # red, green, blue, etc.
    position_3d: Tuple[float, float, float]  # in mm
    position_2d: Tuple[int, int]  # pixel center
    bounding_box_2d: Tuple[int, int, int, int]  # x, y, w, h
    pixel_count: int
    confidence: float = 1.0

@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    data: Any
    message: str = ""
    tool_name: str = ""

class SegmentationTools:
    """
    New segmentation-based tool system with optional SAM enhancement
    Replaces all old tools with visual understanding
    """
    
    def __init__(self, env, enable_sam=False, sam_config=None):
        """
        Initialize with environment reference and optional SAM support
        
        Args:
            env: PrimitiveSkillEnv instance
            enable_sam: Whether to enable SAM enhancement
            sam_config: Configuration for SAM tool
        """
        self.env = env
        self.debug = True
        
        # Cache for avoiding repeated analysis
        self._last_segmentation = None
        self._last_objects = None
        self._analysis_cache_valid = False
        
        # SAM integration
        self.enable_sam = enable_sam
        self.sam_tool = None
        self.sam_matcher = None
        
        if enable_sam:
            try:
                from .sam_integration import SAMTool, SAMEnvironmentMatcher
                
                # Default SAM config
                default_sam_config = {
                    'sam_checkpoint_path': 'sam_vit_h_4b8939.pth',
                    'model_type': 'vit_h',
                    'device': 'cpu'
                }
                
                if sam_config:
                    default_sam_config.update(sam_config)
                
                self.sam_tool = SAMTool(**default_sam_config)
                self.sam_matcher = SAMEnvironmentMatcher()
                
                if self.debug:
                    print(" SAM integration enabled")
                    
            except Exception as e:
                print(f" SAM initialization failed: {e}")
                print(f" Terminating because enable_sam=True but SAM is not available")
                raise RuntimeError(f"SAM initialization failed: {e}. Cannot proceed with enable_sam=True")
    
    def _invalidate_cache(self):
        """Invalidate analysis cache when environment changes"""
        self._analysis_cache_valid = False
        self._last_segmentation = None
        self._last_objects = None
    
    def _analyze_scene(self, force_refresh: bool = False) -> List[ObjectInfo]:
        """
        Comprehensive scene analysis using segmentation data
        
        Args:
            force_refresh: Force re-analysis even if cache is valid
            
        Returns:
            List of ObjectInfo for all detected objects
        """
        if self._analysis_cache_valid and not force_refresh and self._last_objects:
            return self._last_objects
        
        # Get fresh segmentation data
        segmentation, segmentation_id_map, object_positions = self.env.get_segmentation_data()
        
        if segmentation is None:
            return []
        
        self._last_segmentation = segmentation
        objects = []
        
        # Analyze each unique object ID
        unique_ids = np.unique(segmentation)
        
        for obj_id in unique_ids:
            if obj_id == 0:  # Skip background
                continue
            
            # Get object mask and basic info
            mask = (segmentation == obj_id)
            pixel_count = mask.sum()
            
            if pixel_count < 10:  # Skip tiny objects
                continue
            
            # Calculate 2D properties
            coords = np.where(mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            center_2d = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            bbox_2d = (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
            
            # Extract object name and parse properties
            raw_name = self._get_object_name_from_id(obj_id, segmentation_id_map)
            parsed_info = self._parse_object_name(raw_name)
            
            # Try to find 3D position
            position_3d = self._find_3d_position(parsed_info['clean_name'], object_positions)
            
            if position_3d:
                # Create ObjectInfo
                obj_info = ObjectInfo(
                    object_id=obj_id,
                    object_name=parsed_info['clean_name'],
                    object_type=parsed_info['type'],
                    color=parsed_info['color'],
                    position_3d=position_3d,
                    position_2d=center_2d,
                    bounding_box_2d=bbox_2d,
                    pixel_count=int(pixel_count),
                    confidence=1.0
                )
                objects.append(obj_info)
                
                if self.debug:
                    print(f"✓ Found {obj_info.object_type} '{obj_info.object_name}' "
                          f"(color: {obj_info.color}) at 3D: {obj_info.position_3d}")
        
        # Cache results
        self._last_objects = objects
        self._analysis_cache_valid = True
        
        return objects
    
    def _get_object_name_from_id(self, obj_id: int, segmentation_id_map: Dict) -> str:
        """Extract clean object name from segmentation ID map"""
        if obj_id in segmentation_id_map:
            obj_info = segmentation_id_map[obj_id]
            try:
                obj_str = str(obj_info)
                # Extract name from format like "<cube: struct of type...>"
                if '<' in obj_str and ':' in obj_str:
                    name = obj_str.split('<')[1].split(':')[0].strip()
                    return name
                return f"object_{obj_id}"
            except:
                return f"object_{obj_id}"
        return f"unknown_{obj_id}"
    
    def _parse_object_name(self, raw_name: str) -> Dict[str, str]:
        """
        Parse object name to extract type, color, and other properties
        
        Args:
            raw_name: Raw object name from environment
            
        Returns:
            Dict with parsed information
        """
        name = raw_name.lower()
        
        # Define patterns
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan']
        types = {
            'cube': ['cube', 'block', 'box'],
            'target': ['target', 'goal', 'region'],
            'robot': ['panda', 'link', 'hand', 'finger', 'tcp'],
            'table': ['table', 'surface', 'workspace'],
            'ground': ['ground', 'floor'],
            'apple': ['apple', 'fruit'],
            'drawer': ['drawer', 'cabinet']
        }
        
        # Extract color
        detected_color = None
        for color in colors:
            if color in name:
                detected_color = color
                break
        
        # Extract type
        detected_type = 'object'  # default
        for obj_type, keywords in types.items():
            for keyword in keywords:
                if keyword in name:
                    detected_type = obj_type
                    break
            if detected_type != 'object':
                break
        
        # Clean name (remove common prefixes/suffixes)
        clean_name = raw_name
        for prefix in ['panda_', 'goal_', '_pad']:
            clean_name = clean_name.replace(prefix, '')
        
        return {
            'clean_name': clean_name,
            'type': detected_type,
            'color': detected_color,
            'original': raw_name
        }
    
    def _find_3d_position(self, obj_name: str, object_positions: Dict) -> Optional[Tuple[float, float, float]]:
        """Find 3D position for object name"""
        # Try exact match
        if obj_name in object_positions:
            return object_positions[obj_name]
        
        # Try with _position suffix
        position_key = f"{obj_name}_position"
        if position_key in object_positions:
            return object_positions[position_key]
        
        # Try fuzzy matching
        for key, pos in object_positions.items():
            if obj_name.lower() in key.lower() or key.lower() in obj_name.lower():
                return pos
        
        return None
    
    def find_object_with_sam(self, description: str) -> ToolResult:
        """
        SAM-enhanced object finding with better semantic understanding
        
        Args:
            description: Natural language description of target object
            
        Returns:
            ToolResult with object information from SAM + environment matching
        """
        if not self.enable_sam or not self.sam_tool:
            # No fallback - fail hard if SAM is expected but not available
            return ToolResult(
                success=False,
                data=None,
                message="SAM is not available but was requested. Check SAM initialization.",
                tool_name="find_object_with_sam"
            )
        
        try:
            # Step 1: Get current image from environment
            current_image = self.env.get_current_image()
            if current_image is None:
                return ToolResult(
                    success=False,
                    data=None,
                    message="Could not get current image from environment",
                    tool_name="find_object_with_sam"
                )
            
            if self.debug:
                print(f"🔍 Using SAM to find: '{description}'")
                print(f"📷 Image shape: {current_image.shape}")
            
            # Step 2: Use SAM to segment the target object
            sam_result = self.sam_tool.segment_from_image_array(
                image=current_image,
                text_description=description,
            )
            
            if sam_result is None:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"SAM could not locate '{description}' in the image",
                    tool_name="find_object_with_sam"
                )
            
            if self.debug:
                print(f"✅ SAM found object: center={sam_result.center_2d}, score={sam_result.score:.3f}")
            
            # Step 3: Get environment segmentation data
            env_segmentation, env_seg_map, env_positions = self.env.get_segmentation_data()
            
            if env_segmentation is None:
                return ToolResult(
                    success=False,
                    data=None,
                    message="Could not get environment segmentation data",
                    tool_name="find_object_with_sam"
                )
            
            # Step 4: Match SAM result with environment objects
            matched_objects = self.sam_matcher.match_sam_to_environment(
                sam_results=[sam_result],
                env_segmentation=env_segmentation,
                env_segmentation_map=env_seg_map,
                env_positions=env_positions,
                min_overlap_threshold=0.2
            )
            
            if not matched_objects:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"No environment match found for SAM result of '{description}'",
                    tool_name="find_object_with_sam"
                )
            
            # Step 5: Return best match
            best_match = matched_objects[0]  # SAM matcher returns sorted by confidence
            
            result_data = {
                'name': best_match.env_object_name,
                'type': self._parse_object_name(best_match.env_object_name)['type'],
                'color': self._parse_object_name(best_match.env_object_name)['color'],
                'position_2d': sam_result.center_2d,  # Use SAM's 2D position
                'position_3d': best_match.env_position_3d,  # Use environment's 3D position
                'bounding_box_2d': sam_result.bounding_box_2d,  # Use SAM's bounding box
                'sam_score': sam_result.score,
                'match_confidence': best_match.match_confidence,
                'overlap_score': best_match.overlap_score
            }
            
            color_str = f"{result_data['color']} " if result_data['color'] else ""
            message = f"SAM found {color_str}{result_data['type']} '{result_data['name']}' at 2D:{result_data['position_2d']} 3D:{result_data['position_3d']} (confidence: {best_match.match_confidence:.3f})"
            
            if self.debug:
                print(f"✅ {message}")
            
            return ToolResult(
                success=True,
                data=result_data,
                message=message,
                tool_name="find_object_with_sam"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"SAM processing failed: {str(e)}",
                tool_name="find_object_with_sam"
            )
    
    def get_all_objects(self) -> ToolResult:
        """
        Get all objects in the scene with their properties
        
        Returns:
            ToolResult with list of all objects including both 2D and 3D coordinates
        """
        try:
            objects = self._analyze_scene()
            
            object_summary = []
            message_parts = []
            
            for obj in objects:
                summary = {
                    'name': obj.object_name,
                    'type': obj.object_type,
                    'color': obj.color,
                    'position_2d': obj.position_2d,  # 像素坐标
                    'position_3d': obj.position_3d,  # 世界坐标（毫米）
                    'bounding_box_2d': obj.bounding_box_2d,
                    'pixel_count': obj.pixel_count
                }
                object_summary.append(summary)
                
                # 为VLM创建描述
                color_str = f"{obj.color} " if obj.color else ""
                message_parts.append(f"{color_str}{obj.object_type} at 2D:{obj.position_2d} 3D:{obj.position_3d}")
            
            message = f"Found {len(objects)} objects: " + "; ".join(message_parts)
            
            return ToolResult(
                success=True,
                data=object_summary,
                message=message,
                tool_name="get_all_objects"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                message=f"Error analyzing scene: {str(e)}",
                tool_name="get_all_objects"
            )
    
    def find_object_by_description(self, description: str) -> ToolResult:
        """
        Find object by natural language description using basic segmentation
        
        Args:
            description: Natural language description like "red cube", "left target", etc.
            
        Returns:
            ToolResult with matching object information including both 2D and 3D coordinates
        """
        # If SAM is enabled, enforce its usage instead of basic method
        if self.enable_sam:
            return ToolResult(
                success=False,
                data=None,
                message=f"SAM is enabled. Use find_object_with_sam() instead of basic find_object()",
                tool_name="find_object_by_description"
            )
        
        try:
            objects = self._analyze_scene()
            desc_lower = description.lower()
            
            # Score each object based on description match
            matches = []
            
            for obj in objects:
                score = 0
                
                # Check color match
                if obj.color and obj.color in desc_lower:
                    score += 3
                
                # Check type match
                if obj.object_type in desc_lower:
                    score += 3
                
                # Check name match
                if obj.object_name.lower() in desc_lower or desc_lower in obj.object_name.lower():
                    score += 2
                
                # Position-based matching
                if 'left' in desc_lower and obj.position_2d[0] < 150:  # Left side of image
                    score += 1
                if 'right' in desc_lower and obj.position_2d[0] > 150:  # Right side of image
                    score += 1
                
                if score > 0:
                    matches.append((obj, score))
            
            if not matches:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"No object found matching '{description}'",
                    tool_name="find_object_by_description"
                )
            
            # Get best match
            best_match = max(matches, key=lambda x: x[1])[0]
            
            result_data = {
                'name': best_match.object_name,
                'type': best_match.object_type,
                'color': best_match.color,
                'position_2d': best_match.position_2d,  # 像素坐标
                'position_3d': best_match.position_3d,  # 世界坐标（毫米）
                'bounding_box_2d': best_match.bounding_box_2d,
                'pixel_count': best_match.pixel_count
            }
            
            color_str = f"{best_match.color} " if best_match.color else ""
            message = f"Found {color_str}{best_match.object_type} '{best_match.object_name}' at 2D:{best_match.position_2d} 3D:{best_match.position_3d}"
            
            return ToolResult(
                success=True,
                data=result_data,
                message=message,
                tool_name="find_object_by_description"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Error finding object: {str(e)}",
                tool_name="find_object_by_description"
            )
    
    def get_object_position(self, description: str) -> ToolResult:
        """
        Get both 2D and 3D positions of object by description
        
        Args:
            description: Object description
            
        Returns:
            ToolResult with both 2D and 3D positions
        """
        result = self.find_object_by_description(description)
        if result.success:
            position_data = {
                'position_2d': result.data['position_2d'],
                'position_3d': result.data['position_3d'],
                'bounding_box_2d': result.data['bounding_box_2d']
            }
            return ToolResult(
                success=True,
                data=position_data,
                message=f"Position of {description}: 2D:{result.data['position_2d']} 3D:{result.data['position_3d']}",
                tool_name="get_object_position"
            )
        else:
            return ToolResult(
                success=False,
                data=None,
                message=result.message,
                tool_name="get_object_position"
            )
    
    def get_scene_summary(self) -> ToolResult:
        """
        Get a comprehensive scene summary for VLM with both 2D and 3D information
        
        Returns:
            ToolResult with scene description including coordinates
        """
        try:
            objects = self._analyze_scene()
            
            # Group objects by type
            cubes = [obj for obj in objects if obj.object_type == 'cube']
            targets = [obj for obj in objects if obj.object_type == 'target']
            others = [obj for obj in objects if obj.object_type not in ['cube', 'target', 'robot']]
            
            summary = []
            detailed_data = {
                'cubes': [],
                'targets': [],
                'others': []
            }
            
            if cubes:
                cube_desc = []
                for cube in cubes:
                    color_str = f"{cube.color} " if cube.color else ""
                    cube_desc.append(f"{color_str}cube at 2D:{cube.position_2d} 3D:{cube.position_3d}")
                    detailed_data['cubes'].append({
                        'name': cube.object_name,
                        'color': cube.color,
                        'position_2d': cube.position_2d,
                        'position_3d': cube.position_3d,
                        'bounding_box_2d': cube.bounding_box_2d
                    })
                summary.append(f"Cubes: {'; '.join(cube_desc)}")
            
            if targets:
                target_desc = []
                for target in targets:
                    pos_2d = target.position_2d
                    side = "left" if pos_2d[0] < 150 else "right"
                    target_desc.append(f"{side} target at 2D:{target.position_2d} 3D:{target.position_3d}")
                    detailed_data['targets'].append({
                        'name': target.object_name,
                        'side': side,
                        'position_2d': target.position_2d,
                        'position_3d': target.position_3d,
                        'bounding_box_2d': target.bounding_box_2d
                    })
                summary.append(f"Targets: {'; '.join(target_desc)}")
            
            if others:
                other_desc = []
                for obj in others:
                    other_desc.append(f"{obj.object_type} at 2D:{obj.position_2d} 3D:{obj.position_3d}")
                    detailed_data['others'].append({
                        'name': obj.object_name,
                        'type': obj.object_type,
                        'position_2d': obj.position_2d,
                        'position_3d': obj.position_3d,
                        'bounding_box_2d': obj.bounding_box_2d
                    })
                summary.append(f"Other objects: {'; '.join(other_desc)}")
            
            scene_description = "; ".join(summary)
            
            # Add summary statistics
            detailed_data.update({
                'total_cubes': len(cubes),
                'total_targets': len(targets),
                'total_objects': len(objects),
                'description': scene_description
            })
            
            return ToolResult(
                success=True,
                data=detailed_data,
                message=scene_description,
                tool_name="get_scene_summary"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Error creating scene summary: {str(e)}",
                tool_name="get_scene_summary"
            )

class SegmentationToolExecutor:
    """
    Executes segmentation-based tool calls
    """
    
    def __init__(self, tools: SegmentationTools):
        self.tools = tools
    
    def execute_tool_calls(self, response: str) -> Tuple[List[ToolResult], str]:
        """
        Execute tool calls from VLM response
        
        Args:
            response: VLM response containing tool calls
            
        Returns:
            Tuple of (tool results, formatted results string)
        """
        # Parse tool calls
        tool_calls = self._parse_tool_calls(response)
        
        if not tool_calls:
            return [], ""
        
        # Execute each tool call
        results = []
        result_strings = []
        
        for tool_call in tool_calls:
            result = self._execute_single_tool_call(tool_call)
            results.append(result)
            
            # Format result for VLM
            if result.success:
                result_strings.append(f"✓ {result.tool_name}: {result.message}")
            else:
                result_strings.append(f"✗ {result.tool_name}: {result.message}")
        
        return results, "\n".join(result_strings)
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from VLM response"""
        tool_calls = []
        
        # Pattern to match function calls
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, response)
        
        tool_mapping = {
            'get_objects': ('get_all_objects', []),
            'get_all_objects': ('get_all_objects', []),
            'find_object': ('find_object_by_description', ['description']),
            'find_object_by_description': ('find_object_by_description', ['description']),
            'find_object_sam': ('find_object_with_sam', ['description']),
            'find_object_with_sam': ('find_object_with_sam', ['description']),
            'get_position': ('get_object_position', ['description']),
            'get_object_position': ('get_object_position', ['description']),
            'get_scene': ('get_scene_summary', []),
            'get_scene_summary': ('get_scene_summary', []),
        }
        
        for func_name, args_str in matches:
            if func_name in tool_mapping:
                method_name, param_names = tool_mapping[func_name]
                
                # Parse arguments
                args = []
                if args_str.strip():
                    args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
                
                tool_calls.append({
                    'method_name': method_name,
                    'arguments': args,
                    'original_call': f"{func_name}({args_str})"
                })
        
        return tool_calls
    
    def _execute_single_tool_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a single tool call"""
        method_name = tool_call['method_name']
        arguments = tool_call['arguments']
        
        try:
            if hasattr(self.tools, method_name):
                method = getattr(self.tools, method_name)
                
                if len(arguments) == 0:
                    return method()
                elif len(arguments) == 1:
                    return method(arguments[0])
                else:
                    return method(*arguments)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"Unknown tool: {method_name}",
                    tool_name=method_name
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Tool execution error: {str(e)}",
                tool_name=method_name
            )

def get_segmentation_tools_description(enable_sam=False) -> str:
    """Get description of segmentation tools for VLM prompt"""
    base_description = """Available Scene Understanding Tools:

Visual Analysis Tools:
- get_objects(): Get all objects with both 2D (pixel) and 3D (world) coordinates
- find_object(description): Find specific object by description, returns both coordinate systems
- get_position(description): Get both 2D and 3D coordinates of specific object
- get_scene(): Get comprehensive scene summary with all coordinate information"""

    sam_description = """
- find_object_sam(description): Enhanced object finding with SAM for complex descriptions"""

    coordinate_info = """

Coordinate Systems:
- 2D coordinates: (x, y) pixel positions in the camera image (300x300)
- 3D coordinates: (x, y, z) real-world positions in millimeters for robot actions

Tool Usage Examples:
- get_objects() → Returns all objects with 2D:(x,y) 3D:(x,y,z) positions
- find_object("red cube") → Returns red cube at 2D:(150,120) 3D:(-16,-77,20)
- get_position("green cube") → Returns both pixel and world coordinates
- get_scene() → Returns complete scene layout with all coordinates"""

    sam_examples = """
- find_object_sam("the larger red block") → Enhanced semantic understanding for complex descriptions"""

    usage_info = """

Use 3D coordinates for robot pick/place actions. Use 2D coordinates to understand spatial relationships in the image."""

    if enable_sam:
        return base_description + sam_description + coordinate_info + sam_examples + usage_info
    else:
        return base_description + coordinate_info + usage_info
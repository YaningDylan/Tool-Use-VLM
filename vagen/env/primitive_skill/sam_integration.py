import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    OWLVIT_AVAILABLE = True
except ImportError:
    OWLVIT_AVAILABLE = False

@dataclass
class SAMResult:
    """SAM segmentation result"""
    mask: np.ndarray
    score: float
    center_2d: Tuple[int, int]
    bounding_box_2d: Tuple[int, int, int, int]  # x, y, w, h
    description: str
    detection_confidence: float

@dataclass
class MatchedObject:
    """Matched object with both SAM and environment info"""
    sam_result: SAMResult
    env_object_id: int
    env_object_name: str
    env_position_3d: Tuple[float, float, float]  # x, y, z in mm
    match_confidence: float
    overlap_score: float

class SAMEnvironmentMatcher:
    """
    Matches SAM segmentation results with ManiSkill environment objects
    """
    
    def __init__(self):
        self.debug = True
        
    def calculate_mask_overlap(self, sam_mask: np.ndarray, env_mask: np.ndarray) -> float:
        """
        Calculate IoU (Intersection over Union) between SAM mask and environment mask
        
        Args:
            sam_mask: Binary mask from SAM (H, W)
            env_mask: Binary mask from environment segmentation (H, W)
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        # Ensure both masks are binary
        sam_binary = (sam_mask > 0).astype(np.uint8)
        env_binary = (env_mask > 0).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.logical_and(sam_binary, env_binary).sum()
        union = np.logical_or(sam_binary, env_binary).sum()
        
        if union == 0:
            return 0.0
            
        iou = intersection / union
        return float(iou)
    
    def get_mask_from_segmentation(self, segmentation: np.ndarray, object_id: int) -> np.ndarray:
        """
        Extract binary mask for specific object from environment segmentation
        
        Args:
            segmentation: Environment segmentation map (H, W)
            object_id: Target object ID
            
        Returns:
            Binary mask for the object
        """
        return (segmentation == object_id).astype(np.uint8)
    
    def match_sam_to_environment(self, 
                                sam_results: List[SAMResult], 
                                env_segmentation: np.ndarray,
                                env_segmentation_map: Dict[int, Any],
                                env_positions: Dict[str, Tuple[float, float, float]],
                                min_overlap_threshold: float = 0.3) -> List[MatchedObject]:
        """
        Match SAM results with environment objects
        
        Args:
            sam_results: List of SAM segmentation results
            env_segmentation: Environment segmentation map (H, W)
            env_segmentation_map: Mapping from object_id to object info
            env_positions: Mapping from object_name to 3D position
            min_overlap_threshold: Minimum IoU threshold for valid match
            
        Returns:
            List of matched objects
        """
        matched_objects = []
        
        for sam_result in sam_results:
            best_match = None
            best_overlap = 0.0
            
            # Try to match with each environment object
            for obj_id, obj_info in env_segmentation_map.items():
                if obj_id == 0:  # Skip background
                    continue
                
                # Get object name and position first
                obj_name = self._extract_object_name(obj_info)
                obj_position = self._get_object_position(obj_name, env_positions)
                
                if obj_position is None:
                    if self.debug:
                        print(f"    Skipping {obj_name} (no position found)")
                    continue
                    
                # Get environment object mask
                env_mask = self.get_mask_from_segmentation(env_segmentation, obj_id)
                
                # Skip if environment mask is empty
                if env_mask.sum() == 0:
                    if self.debug:
                        print(f"    Skipping {obj_name} (empty mask)")
                    continue
                
                # Calculate overlap
                overlap_score = self.calculate_mask_overlap(sam_result.mask, env_mask)
                
                if self.debug:
                    print(f"    Checking {obj_name} at {obj_position}: IoU = {overlap_score:.3f}")
                
                # Update best match if this is better
                if overlap_score > best_overlap and overlap_score >= min_overlap_threshold:
                    best_match = MatchedObject(
                        sam_result=sam_result,
                        env_object_id=obj_id,
                        env_object_name=obj_name,
                        env_position_3d=obj_position,
                        match_confidence=sam_result.detection_confidence * overlap_score,
                        overlap_score=overlap_score
                    )
                    best_overlap = overlap_score
            
            if best_match is not None:
                matched_objects.append(best_match)
                if self.debug:
                    print(f"✓ Matched '{sam_result.description}' to '{best_match.env_object_name}' "
                          f"at {best_match.env_position_3d} (IoU: {best_overlap:.3f})")
            else:
                if self.debug:
                    print(f"✗ No match found for '{sam_result.description}' "
                          f"(best IoU: {best_overlap:.3f}, threshold: {min_overlap_threshold})")
        
        return matched_objects
    
    def _extract_object_name(self, obj_info) -> str:
        """Extract object name from environment object info"""
        try:
            # Handle different object info formats
            if hasattr(obj_info, '__str__'):
                obj_str = str(obj_info)
                # Extract name from format like "<cube: struct of type...>"
                if '<' in obj_str and ':' in obj_str:
                    name = obj_str.split('<')[1].split(':')[0].strip()
                    return name
            return f"object_{id(obj_info)}"
        except:
            return f"unknown_object"
    
    def _get_object_position(self, obj_name: str, env_positions: Dict[str, Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        """Get 3D position for object name"""
        # Try exact match first
        if obj_name in env_positions:
            return env_positions[obj_name]
        
        # Try with _position suffix
        position_key = f"{obj_name}_position"
        if position_key in env_positions:
            return env_positions[position_key]
        
        # Try fuzzy matching
        for key, pos in env_positions.items():
            if obj_name.lower() in key.lower() or key.lower() in obj_name.lower():
                return pos
        
        if self.debug:
            print(f"    ⚠ No position found for object '{obj_name}' in {list(env_positions.keys())}")
        return None

class SAMTool:
    """
    Enhanced SAM tool integrated with ManiSkill environment
    """
    
    def __init__(self, 
                 sam_checkpoint_path="sam_vit_h_4b8939.pth", 
                 model_type="vit_h", 
                 device="cuda",
                 confidence_threshold=0.1):
        """
        Initialize SAM tool
        
        Args:
            sam_checkpoint_path: Path to SAM model checkpoint
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            device: Device to use ("cuda" or "cpu")
            confidence_threshold: Minimum confidence threshold for detection
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        print(f"SAM using device: {self.device}")
        print(f"SAM confidence threshold: {self.confidence_threshold}")
        
        # Initialize SAM
        self._init_sam(sam_checkpoint_path, model_type)
        
        # Initialize semantic understanding if available
        self._init_owlvit()
        
        # Initialize matcher
        self.matcher = SAMEnvironmentMatcher()
        
    def _init_sam(self, sam_checkpoint_path: str, model_type: str):
        """Initialize SAM model"""
        try:
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            print("✓ SAM model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load SAM model: {e}")
            raise
    
    def _init_owlvit(self):
        """Initialize OWL-ViT for semantic understanding"""
        if not OWLVIT_AVAILABLE:
            print("⚠ OWL-ViT not available - only point-based segmentation supported")
            self.owlvit_processor = None
            self.owlvit_model = None
            return
            
        try:
            self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self.owlvit_model.to(self.device)
            print("✓ OWL-ViT model loaded successfully")
        except Exception as e:
            print(f"⚠ Failed to load OWL-ViT model: {e}")
            self.owlvit_processor = None
            self.owlvit_model = None
    
    def segment_from_image_array(self, 
                                image: np.ndarray, 
                                text_description: str) -> Optional[SAMResult]:
        """
        Segment object from numpy image array using text description
        
        Args:
            image: RGB image array (H, W, 3)
            text_description: Description of target object
            
        Returns:
            SAMResult or None if segmentation fails
        """
        if image is None or image.size == 0:
            print("✗ Invalid image provided")
            return None
            
        print(f"🔍 Segmenting '{text_description}' from image shape {image.shape}")
        
        # Save the input image for debugging
        try:
            import os
            from PIL import Image as PILImage
            
            # Create debug folder if it doesn't exist
            debug_folder = "sam_debug"
            os.makedirs(debug_folder, exist_ok=True)
            
            # Save the image
            safe_description = text_description.replace(" ", "_").replace("/", "_")
            image_filename = f"{debug_folder}/sam_input_{safe_description}.png"
            
            # Convert numpy array to PIL Image and save
            if image.dtype != np.uint8:
                # Normalize to 0-255 if needed
                if image.max() <= 1.0:
                    image_to_save = (image * 255).astype(np.uint8)
                else:
                    image_to_save = image.astype(np.uint8)
            else:
                image_to_save = image
                
            pil_image = PILImage.fromarray(image_to_save)
            pil_image.save(image_filename)
            print(f"📸 Saved input image to: {image_filename}")
            
        except Exception as e:
            print(f"⚠ Failed to save debug image: {e}")
        
        # Detect object using semantic understanding
        detection_result = self._detect_object_semantic(image, text_description)
        
        if detection_result is None:
            print(f"✗ No object detected for '{text_description}'")
            return None
        
        center_point, bounding_box, detection_confidence = detection_result
        
        # Perform SAM segmentation
        self.predictor.set_image(image)
        
        try:
            # Use both point and box prompts for better segmentation
            masks, scores, logits = self.predictor.predict(
                point_coords=center_point.reshape(1, -1),
                point_labels=np.array([1]),  # foreground point
                box=bounding_box.reshape(1, -1) if bounding_box is not None else None,
                multimask_output=True,
            )
            
            # Select best mask
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            # Calculate bounding box from mask
            mask_bbox = self._get_mask_bounding_box(best_mask)
            
            result = SAMResult(
                mask=best_mask,
                score=best_score,
                center_2d=tuple(center_point.astype(int).tolist()),
                bounding_box_2d=mask_bbox,
                description=text_description,
                detection_confidence=detection_confidence
            )
            
            # Save segmentation result for debugging
            try:
                # Save mask overlay
                mask_filename = f"{debug_folder}/sam_mask_{safe_description}.png"
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image
                axes[0].imshow(image_to_save)
                axes[0].set_title(f"Original: {text_description}")
                axes[0].axis('off')
                
                # Mask overlay
                axes[1].imshow(image_to_save)
                axes[1].imshow(best_mask, alpha=0.5, cmap='jet')
                x, y, w, h = mask_bbox
                rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
                axes[1].add_patch(rect)
                axes[1].plot(center_point[0], center_point[1], 'r*', markersize=15)
                axes[1].set_title(f"Segmentation (score: {best_score:.3f})")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(mask_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"🎭 Saved segmentation result to: {mask_filename}")
                
            except Exception as e:
                print(f"⚠ Failed to save segmentation debug image: {e}")
            
            print(f"✓ Segmentation successful: score={best_score:.3f}, confidence={detection_confidence:.3f}")
            print(f"📍 SAM found object at:")
            print(f"   - Center 2D: {center_point}")
            print(f"   - Bounding box: {mask_bbox} (x, y, width, height)")
            print(f"   - Mask covers {best_mask.sum()} pixels")
            print(f"   - Detection confidence: {detection_confidence:.3f}")
            return result
            
        except Exception as e:
            print(f"✗ SAM segmentation failed: {e}")
            return None
    
    def _detect_object_semantic(self, 
                               image: np.ndarray, 
                               text_description: str) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Detect object using OWL-ViT semantic understanding"""
        if self.owlvit_processor is None or self.owlvit_model is None:
            print("⚠ Semantic detection not available - using center point")
            # Fallback: use image center
            h, w = image.shape[:2]
            center_point = np.array([w//2, h//2])
            return center_point, None, 1.0
        
        try:
            # Convert to PIL image
            pil_image = Image.fromarray(image)
            
            # Process inputs
            inputs = self.owlvit_processor(text=[text_description], images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.owlvit_model(**inputs)
            
            # Process results
            target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)  # (height, width)
            
            # Try new method first, fallback to old method
            try:
                results = self.owlvit_processor.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
                )
            except:
                results = self.owlvit_processor.post_process_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
                )
            
            boxes = results[0]["boxes"].cpu().numpy()
            scores = results[0]["scores"].cpu().numpy()
            
            if len(boxes) == 0:
                return None
            
            # Select best detection
            best_idx = np.argmax(scores)
            best_box = boxes[best_idx]
            best_score = scores[best_idx]
            
            # Calculate center point
            x1, y1, x2, y2 = best_box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            return np.array([center_x, center_y]), best_box, best_score
            
        except Exception as e:
            print(f"Semantic detection failed: {e}")
            return None
    
    def _get_mask_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box (x, y, w, h) from binary mask"""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return (0, 0, 0, 0)
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
    
    def segment_and_match_objects(self, 
                                 image: np.ndarray,
                                 descriptions: List[str],
                                 env_segmentation: np.ndarray,
                                 env_segmentation_map: Dict[int, Any],
                                 env_positions: Dict[str, Tuple[float, float, float]]) -> List[MatchedObject]:
        """
        Segment multiple objects and match them with environment
        
        Args:
            image: RGB image array
            descriptions: List of object descriptions to segment
            env_segmentation: Environment segmentation map
            env_segmentation_map: Environment object mapping
            env_positions: Environment object positions
            
        Returns:
            List of matched objects with 3D positions
        """
        sam_results = []
        
        # Segment each description
        for desc in descriptions:
            result = self.segment_from_image_array(image, desc)
            if result is not None:
                sam_results.append(result)
        
        if not sam_results:
            print("✗ No successful segmentations")
            return []
        
        # Use distance-based matching - find closest object to SAM center
        matched_objects = []
        
        for sam_result in sam_results:
            sam_center_x, sam_center_y = sam_result.center_2d
            
            print(f"\n🎯 Distance-based matching for '{sam_result.description}':")
            print(f"   SAM center: ({sam_center_x}, {sam_center_y})")
            
            best_match = None
            min_distance = float('inf')
            
            # Check each object ID in the filtered segmentation
            for obj_id in np.unique(env_segmentation):
                if obj_id == 0:  # Skip background
                    continue
                
                if obj_id in env_segmentation_map:
                    # Find all pixels belonging to this object
                    object_mask = (env_segmentation == obj_id)
                    object_pixels = np.where(object_mask)
                    
                    if len(object_pixels[0]) > 0:
                        # Calculate object center
                        obj_center_y = int(object_pixels[0].mean())
                        obj_center_x = int(object_pixels[1].mean())
                        
                        # Calculate distance from SAM center to object center
                        distance = np.sqrt((sam_center_x - obj_center_x)**2 + (sam_center_y - obj_center_y)**2)
                        
                        # Get object info
                        obj_name = self.matcher._extract_object_name(env_segmentation_map[obj_id])
                        obj_position = self.matcher._get_object_position(obj_name, env_positions)
                        
                        print(f"   Object {obj_id} ({obj_name}) center: ({obj_center_x}, {obj_center_y}), distance: {distance:.1f}")
                        
                        if obj_position is not None and distance < min_distance:
                            min_distance = distance
                            best_match = {
                                'obj_id': obj_id,
                                'obj_name': obj_name,
                                'obj_position': obj_position,
                                'obj_center': (obj_center_x, obj_center_y),
                                'distance': distance
                            }
            
            if best_match is not None:
                matched_obj = MatchedObject(
                    sam_result=sam_result,
                    env_object_id=best_match['obj_id'],
                    env_object_name=best_match['obj_name'],
                    env_position_3d=best_match['obj_position'],
                    match_confidence=sam_result.detection_confidence,
                    overlap_score=1.0  # Perfect match since we use distance
                )
                matched_objects.append(matched_obj)
                
                print(f"   ✅ Best match: {best_match['obj_name']} at distance {min_distance:.1f} pixels")
            else:
                print(f"   ❌ No valid objects found for matching")
        
        return matched_objects
        
        return matched_objects
    
    def visualize_results(self, 
                         image: np.ndarray, 
                         matched_objects: List[MatchedObject], 
                         output_path: str = None):
        """Visualize segmentation and matching results"""
        if not matched_objects:
            print("No matched objects to visualize")
            return
        
        fig, axes = plt.subplots(2, len(matched_objects), figsize=(5*len(matched_objects), 10))
        if len(matched_objects) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, matched_obj in enumerate(matched_objects):
            sam_result = matched_obj.sam_result
            
            # Top row: Original image with detection
            axes[0, i].imshow(image)
            x, y, w, h = sam_result.bounding_box_2d
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            axes[0, i].add_patch(rect)
            axes[0, i].plot(sam_result.center_2d[0], sam_result.center_2d[1], 'r*', markersize=15)
            axes[0, i].set_title(f"'{sam_result.description}'\n→ {matched_obj.env_object_name}\nIoU: {matched_obj.overlap_score:.3f}")
            axes[0, i].axis('off')
            
            # Bottom row: Mask overlay
            axes[1, i].imshow(image)
            axes[1, i].imshow(sam_result.mask, alpha=0.5, cmap='jet')
            axes[1, i].set_title(f"3D Pos: {matched_obj.env_position_3d}\nScore: {sam_result.score:.3f}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        
        plt.show()
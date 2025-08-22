import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class ObjectInfo:
    """Information about a detected object"""
    object_id: int
    object_name: str
    pixel_count: int
    bounding_box_2d: Tuple[int, int, int, int]  # x, y, w, h
    center_2d: Tuple[int, int]
    env_position_3d: Optional[Tuple[float, float, float]] = None

class EnvironmentSegmentationAnalyzer:
    """
    Analyzes ManiSkill3 environment segmentation data
    """
    
    def __init__(self, env):
        """
        Initialize with ManiSkill environment
        
        Args:
            env: PrimitiveSkillEnv instance
        """
        self.env = env
        self.debug = True
    
    def get_current_segmentation_data(self) -> Tuple[np.ndarray, Dict[int, Any], Dict[str, Tuple[float, float, float]]]:
        """
        Get current segmentation data from environment
        
        Returns:
            Tuple of (segmentation_map, segmentation_id_map, object_positions)
        """
        # Reset environment to get fresh observation
        obs, _ = self.env.reset()
        
        if self.debug:
            print(f"Reset completed, obs keys: {list(obs.keys()) if hasattr(obs, 'keys') else type(obs)}")
        
        # Use the environment's new segmentation access method
        segmentation, segmentation_id_map, object_positions = self.env.get_segmentation_data()
        
        if self.debug:
            print(f"Segmentation shape: {segmentation.shape if segmentation is not None else None}")
            print(f"ID map entries: {len(segmentation_id_map) if segmentation_id_map else 0}")
            print(f"Position entries: {len(object_positions) if object_positions else 0}")
        
        return segmentation, segmentation_id_map, object_positions
    

    

    
    def analyze_segmentation(self, 
                           segmentation: np.ndarray, 
                           segmentation_id_map: Dict[int, Any], 
                           object_positions: Dict[str, Tuple[float, float, float]]) -> List[ObjectInfo]:
        """
        Analyze segmentation data and extract object information
        
        Args:
            segmentation: Segmentation map array
            segmentation_id_map: Mapping from object_id to object info
            object_positions: Object positions from environment
            
        Returns:
            List of ObjectInfo with detected objects
        """
        objects = []
        unique_ids = np.unique(segmentation)
        
        print(f"\n🔍 Analyzing segmentation with shape {segmentation.shape}")
        print(f"Found {len(unique_ids)} unique IDs: {unique_ids}")
        
        for obj_id in unique_ids:
            if obj_id == 0:  # Skip background
                continue
            
            # Get mask for this object
            mask = (segmentation == obj_id)
            pixel_count = mask.sum()
            
            if pixel_count == 0:
                continue
            
            # Calculate bounding box and center
            coords = np.where(mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            bbox_2d = (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
            center_2d = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            
            # Get object name from segmentation map
            object_name = self._extract_object_name_from_id(obj_id, segmentation_id_map)
            
            # Try to find 3D position
            env_position_3d = self._find_3d_position(object_name, object_positions)
            
            obj_info = ObjectInfo(
                object_id=obj_id,
                object_name=object_name,
                pixel_count=int(pixel_count),
                bounding_box_2d=bbox_2d,
                center_2d=center_2d,
                env_position_3d=env_position_3d
            )
            
            objects.append(obj_info)
            
            if self.debug:
                print(f"  Object {obj_id} ({object_name}):")
                print(f"    Pixels: {pixel_count}")
                print(f"    2D BBox: {bbox_2d}")
                print(f"    2D Center: {center_2d}")
                print(f"    3D Position: {env_position_3d}")
        
        return objects
    
    def _extract_object_name_from_id(self, obj_id: int, segmentation_id_map: Dict[int, Any]) -> str:
        """Extract object name from segmentation ID map"""
        if obj_id in segmentation_id_map:
            obj_info = segmentation_id_map[obj_id]
            try:
                # Handle different object info formats
                if hasattr(obj_info, '__str__'):
                    obj_str = str(obj_info)
                    # Extract name from format like "<cube: struct of type...>"
                    if '<' in obj_str and ':' in obj_str:
                        name = obj_str.split('<')[1].split(':')[0].strip()
                        return name
                return f"object_{obj_id}"
            except:
                return f"object_{obj_id}"
        else:
            return f"unknown_{obj_id}"
    
    def _find_3d_position(self, object_name: str, object_positions: Dict[str, Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        """Find 3D position for object name"""
        # Try exact match
        if object_name in object_positions:
            return object_positions[object_name]
        
        # Try with _position suffix
        position_key = f"{object_name}_position"
        if position_key in object_positions:
            return object_positions[position_key]
        
        # Try fuzzy matching
        for key, pos in object_positions.items():
            if object_name.lower() in key.lower() or key.lower() in object_name.lower():
                return pos
        
        return None
    
    def visualize_segmentation(self, 
                             segmentation: np.ndarray, 
                             objects: List[ObjectInfo], 
                             save_path: str = None):
        """Visualize segmentation with object information"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original segmentation
        axes[0].imshow(segmentation, cmap='tab20')
        axes[0].set_title(f'Segmentation Map\n{len(objects)} objects detected')
        axes[0].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label('Object ID')
        
        # Segmentation with annotations
        axes[1].imshow(segmentation, cmap='tab20')
        
        # Annotate each object
        for obj in objects:
            x, y, w, h = obj.bounding_box_2d
            center_x, center_y = obj.center_2d
            
            # Draw bounding box
            rect = plt.Rectangle((x, y), w, h, fill=False, color='white', linewidth=2)
            axes[1].add_patch(rect)
            
            # Add text annotation
            axes[1].text(center_x, center_y, f"{obj.object_id}\n{obj.object_name}", 
                        ha='center', va='center', color='white', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        axes[1].set_title('Annotated Objects')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📷 Visualization saved to: {save_path}")
        
        plt.show()
    
    def print_analysis_summary(self, objects: List[ObjectInfo]):
        """Print detailed analysis summary"""
        print(f"\n📊 SEGMENTATION ANALYSIS SUMMARY")
        print(f"=" * 50)
        print(f"Total objects detected: {len(objects)}")
        
        # Group by object type
        object_types = {}
        for obj in objects:
            obj_type = obj.object_name.split('_')[0] if '_' in obj.object_name else obj.object_name
            if obj_type not in object_types:
                object_types[obj_type] = []
            object_types[obj_type].append(obj)
        
        print(f"Object types: {list(object_types.keys())}")
        
        # Objects with and without 3D positions
        with_3d = [obj for obj in objects if obj.env_position_3d is not None]
        without_3d = [obj for obj in objects if obj.env_position_3d is None]
        
        print(f"Objects with 3D positions: {len(with_3d)}")
        print(f"Objects without 3D positions: {len(without_3d)}")
        
        print(f"\n📍 OBJECTS WITH 3D POSITIONS:")
        for obj in with_3d:
            print(f"  {obj.object_id:2d} | {obj.object_name:20s} | 2D: {obj.center_2d} | 3D: {obj.env_position_3d}")
        
        if without_3d:
            print(f"\n❓ OBJECTS WITHOUT 3D POSITIONS:")
            for obj in without_3d:
                print(f"  {obj.object_id:2d} | {obj.object_name:20s} | 2D: {obj.center_2d} | Pixels: {obj.pixel_count}")
        
        print(f"=" * 50)

def test_environment_segmentation(env_id="PushCube-v1"):
    """
    Test environment segmentation analysis
    
    Args:
        env_id: Environment ID to test
    """
    print(f"🧪 Testing Environment Segmentation Analysis")
    print(f"Environment: {env_id}")
    
    try:
        # Import your environment (adjust import path as needed)
        import sys
        sys.path.append('.')  # Add current directory to path
        
        from vagen.env.primitive_skill.env import PrimitiveSkillEnv
        from vagen.env.primitive_skill.env_config import PrimitiveSkillEnvConfig
        
        # Create environment config
        config = PrimitiveSkillEnvConfig(
            env_id=env_id,
            render_mode="vision",  # Important: we need visual data
        )
        
        # Initialize environment
        env = PrimitiveSkillEnv(config)
        print("✅ Environment initialized successfully")
        
        # Initialize analyzer
        analyzer = EnvironmentSegmentationAnalyzer(env)
        
        # Test segmentation access directly
        print("\n🔧 Testing segmentation access...")
        segmentation, seg_id_map, positions = env.get_segmentation_data()
        
        if segmentation is not None:
            print(f"✅ Segmentation access successful: {segmentation.shape}")
            print(f"✅ Found {len(seg_id_map)} segmentation IDs")
            print(f"✅ Found {len(positions)} object positions")
        else:
            print("❌ Failed to get segmentation data")
            return None
        
        # Get segmentation data
        print("\n🔍 Extracting segmentation data...")
        segmentation, segmentation_id_map, object_positions = analyzer.get_current_segmentation_data()
        
        if segmentation is None:
            print("❌ Failed to extract segmentation data")
            return
        
        print(f"✅ Segmentation extracted: shape {segmentation.shape}")
        print(f"✅ ID map has {len(segmentation_id_map)} entries")
        print(f"✅ Position map has {len(object_positions)} entries")
        
        # Analyze segmentation
        objects = analyzer.analyze_segmentation(segmentation, segmentation_id_map, object_positions)
        
        # Print summary
        analyzer.print_analysis_summary(objects)
        
        # Visualize (optional)
        try:
            analyzer.visualize_segmentation(segmentation, objects, f"segmentation_analysis_{env_id}.png")
        except Exception as e:
            print(f"⚠ Visualization failed: {e}")
        
        # Clean up
        env.close()
        print("\n✅ Test completed successfully!")
        
        return objects, segmentation, segmentation_id_map, object_positions
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with different environments
    test_environments = ["PlaceTwoCube", "StackThreeCube"]
    
    for env_id in test_environments:
        print(f"\n{'='*60}")
        try:
            result = test_environment_segmentation(env_id)
            if result:
                print(f"✅ {env_id} test passed")
            else:
                print(f"❌ {env_id} test failed")
        except Exception as e:
            print(f"❌ {env_id} test failed with error: {e}")
        print(f"{'='*60}")
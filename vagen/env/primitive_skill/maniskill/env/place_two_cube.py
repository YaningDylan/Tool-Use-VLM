from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common
import time
from transforms3d.euler import euler2quat
from collections import defaultdict
@register_env("PlaceTwoCube", max_episode_steps=2e3)
class PlaceTwoCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]
    skill_config=None
    vlm_info_keys=[]
    state_keys=["red_cube_position", "green_cube_position","left_target_position","right_target_position"]

    def __init__(self, stage=0,*args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.stage=stage
        self.workspace_x=[-0.10, 0.15]
        self.workspace_y=[-0.2, 0.2]
        self.workspace_z=[0.01, 0.2]
        self.region = np.array([[-0.1, -0.1], [0, 0.1]])
        self.robot_init_qpos_noise = robot_init_qpos_noise
                
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def task_skill_indices(self):
        return {
        0 : "pick",
        1 : "place",
        2 : "push",
    }

    def instruction(self):
        return "Please place red cube at left target and green cube at right target."
        
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[1, 0.0, 0.6], target=[-0.2, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 300, 300, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1, 0.0, 0.6], [-0.2, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 300,300, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create two cubes
        self.red_cube = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="red_cube"
        )
        self.green_cube = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="green_cube"
        )
        
        # Create two target areas
        self.goal_radius = 0.05
        self.goal_region_A = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region_A",
            add_collision=False,
            body_type="kinematic",
        )
        self.goal_region_B = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region_B",
            add_collision=False,
            body_type="kinematic",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Place cubes at random positions
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            region =self.region
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.02

            red_cube_xy = sampler.sample(radius, 100)
            green_cube_xy = sampler.sample(radius, 100, verbose=False)

            # Set initial positions for cubes
            xyz[:, :2] = red_cube_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.red_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = green_cube_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.green_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Set fixed positions for target areas
            left_target_position = torch.tensor([0.08, -0.1, 0.0])
            right_target_position = torch.tensor([0.08, 0.1, 0.0])
        
            self.goal_region_A.set_pose(Pose.create_from_pq(
                p=left_target_position,
                q=euler2quat(0, np.pi / 2, 0),
            ))
            self.goal_region_B.set_pose(Pose.create_from_pq(
                p=right_target_position,
                q=euler2quat(0, np.pi / 2, 0),
            ))

    def is_cube_in_goal(self, cube_position, goal_position):
        distance = torch.norm(cube_position[..., :2] - goal_position[...,:2], dim=-1)
        return distance <= self.goal_radius

    def _get_obs_extra(self, info: Dict):
        if "state" in self.obs_mode or "segmentation" in self.obs_mode:
            obs = dict(
                red_cube_position=info["red_cube_position"],
                green_cube_position=info["green_cube_position"],
                is_red_cube_grasped=info["is_red_cube_grasped"],
                is_green_cube_grasped=info["is_green_cube_grasped"],
                left_target_position=self.goal_region_A.pose.p,
                right_target_position=self.goal_region_B.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros_like(info["success"],dtype=torch.float32,device=self.device)
        

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 80.0
    
    
    def task_fail(self, info: Dict):
        # if cube position is out of workspace return true
        for cube in ["red_cube", "green_cube"]:
            if info[f"{cube}_position"][0] < self.workspace_x[0] or info[f"{cube}_position"][0] > self.workspace_x[1]:
                return True
            if info[f"{cube}_position"][1] < self.workspace_y[0] or info[f"{cube}_position"][1] > self.workspace_y[1]:
                return True
            if info[f"{cube}_position"][2] < 0:
                return True
        return False
    
    def evaluate(self):
        pos_A = self.red_cube.pose.p
        pos_B = self.green_cube.pose.p

        is_red_cube_grasped = self.agent.is_grasping(self.red_cube)
        is_green_cube_grasped = self.agent.is_grasping(self.green_cube)

        is_red_cube_in_goal = self.is_cube_in_goal(pos_A, self.goal_region_A.pose.p)
        is_green_cube_in_goal = self.is_cube_in_goal(pos_B, self.goal_region_B.pose.p)

        stage0_success = is_red_cube_grasped
        stage1_success = is_red_cube_in_goal & (~is_red_cube_grasped)
        stage2_success = is_red_cube_in_goal & is_green_cube_grasped
        stage3_success = is_red_cube_in_goal & is_green_cube_in_goal & (~is_green_cube_grasped) & (~is_red_cube_grasped)

        
        
        success = stage3_success
        


        info = {
            "left_target_position":self.goal_region_A.pose.p,
            "right_target_position":self.goal_region_B.pose.p,
            "is_red_cube_grasped": is_red_cube_grasped,
            "is_green_cube_grasped": is_green_cube_grasped,
            "red_cube_position": pos_A,
            "green_cube_position": pos_B,
            "is_red_cube_in_goal": is_red_cube_in_goal,
            "is_green_cube_in_goal": is_green_cube_in_goal,
            "stage0_success": stage0_success.bool(),
            "stage1_success": stage1_success.bool(),
            "stage2_success": stage2_success.bool(),
            "success": success.bool(),
        }
        return info

    def skill_reward(self, prev_info, cur_info, action, **kwargs):
        return 0.0

            
            

    def reset(self, **kwargs):
        # reset reward components to 0
        return super().reset(**kwargs)
    
    def get_segmentation_data(self):
        """
        Get filtered segmentation data for PlaceTwoCube task with consistent name mapping
        
        Returns:
            tuple: (segmentation_array, segmentation_id_map, object_positions)
        """
        # Get raw observation with segmentation
        obs = self.get_obs()
        
        if 'sensor_data' not in obs:
            return None, None, None
        
        sensor_data = obs['sensor_data']
        if 'base_camera' not in sensor_data:
            return None, None, None
        
        camera_data = sensor_data['base_camera']
        if 'segmentation' not in camera_data:
            return None, None, None
        
        # Extract segmentation data
        seg_data = camera_data['segmentation']
        if hasattr(seg_data, 'cpu'):
            seg_data = seg_data.cpu().numpy()
        
        # Handle dimensions
        if len(seg_data.shape) == 4:
            seg_data = seg_data[0]
        if len(seg_data.shape) == 3 and seg_data.shape[-1] == 1:
            seg_data = seg_data.squeeze(-1)
        
        segmentation = seg_data.astype(np.int16)
        
        # Get full segmentation ID map
        full_seg_map = {}
        if hasattr(self, 'segmentation_id_map'):
            full_seg_map = self.segmentation_id_map
        elif hasattr(self, 'unwrapped') and hasattr(self.unwrapped, 'segmentation_id_map'):
            full_seg_map = self.unwrapped.segmentation_id_map
        
        # Define name mapping for SAM understanding
        name_mapping = {
            'goal_region_A': 'left_target',
            'goal_region_B': 'right_target'
            # Add more mappings as needed for other objects
        }
        
        # Task objects we want to keep
        task_objects = ['red_cube', 'green_cube', 'left_target', 'right_target']
        
        # Filter and remap objects
        filtered_seg_map = {}
        new_id_mapping = {}
        new_id = 1
        
        for obj_id, obj_info in full_seg_map.items():
            if obj_id == 0:
                continue
            
            # Extract original object name
            obj_str = str(obj_info)
            if '<' in obj_str and ':' in obj_str:
                original_name = obj_str.split('<')[1].split(':')[0].strip()
            else:
                original_name = f"object_{obj_id}"
            
            # Apply name mapping
            mapped_name = name_mapping.get(original_name, original_name)
            
            # Check if this object should be included
            if mapped_name in task_objects or any(target in mapped_name for target in task_objects):
                # Create mapped object info
                mapped_obj_info = self._create_mapped_object_info(mapped_name, obj_info)
                filtered_seg_map[new_id] = mapped_obj_info
                new_id_mapping[obj_id] = new_id
                new_id += 1
        
        # Create filtered segmentation array - only mark task objects, everything else is 0
        filtered_segmentation = np.zeros_like(segmentation, dtype=np.int16)
        for old_id, new_id in new_id_mapping.items():
            mask = (segmentation == old_id)
            filtered_segmentation[mask] = new_id
        
        # Create consistent object_positions using mapped names + _position suffix
        object_positions = {
            'red_cube_position': (self.red_cube.pose.p.cpu().numpy() * 1000).tolist(),
            'green_cube_position': (self.green_cube.pose.p.cpu().numpy() * 1000).tolist(),
            'left_target_position': (self.goal_region_A.pose.p.cpu().numpy() * 1000).tolist(),
            'right_target_position': (self.goal_region_B.pose.p.cpu().numpy() * 1000).tolist(),
        }
        
        return filtered_segmentation, filtered_seg_map, object_positions

    def _create_mapped_object_info(self, mapped_name, original_obj_info):
        """
        Create object info with mapped name for SAM understanding
        
        Args:
            mapped_name: The mapped name (e.g., 'left_target')
            original_obj_info: Original object info from environment
        
        Returns:
            Object info with mapped name
        """
        # Create a simple object info that SAM matcher can understand
        # The exact format depends on how your SAM matcher extracts names
        class MappedObjectInfo:
            def __init__(self, name):
                self.name = name
            
            def __str__(self):
                return f"<{self.name}: mapped object>"
        
        return MappedObjectInfo(mapped_name)
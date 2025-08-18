#!/usr/bin/env python3

import gymnasium as gym
import rlbench
import numpy as np
import time
from typing import Dict, Optional, List, Tuple

class InteractiveRLBenchController:
    def __init__(self, task_name: str = 'reach_target'):
        self.task_name = task_name
        self.env = None
        self.rlbench_env = None
        self._current_obs = None
        self.total_attempts = 0
        self.successful_attempts = 0
        
    def initialize_env(self):
        """Initialize the RLBench environment"""
        try:
            # Set environment for headless mode
            import os
            os.environ['DISPLAY'] = ':99'
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            
            # Import RLBench components for headless setup
            from rlbench.environment import Environment
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import JointVelocity
            from rlbench.action_modes.gripper_action_modes import Discrete
            from rlbench.observation_config import ObservationConfig
            from rlbench.utils import name_to_task_class
            
            # Setup observation config for vision mode
            obs_config = ObservationConfig()
            obs_config.set_all_high_dim(True)
            obs_config.set_all_low_dim(True)
            
            # Setup action mode
            action_mode = MoveArmThenGripper(
                arm_action_mode=JointVelocity(),
                gripper_action_mode=Discrete()
            )
            
            # Create environment directly (headless)
            self.rlbench_env = Environment(
                action_mode=action_mode,
                obs_config=obs_config,
                headless=True  # This is the key for headless mode
            )
            
            # Launch environment
            self.rlbench_env.launch()
            
            # Get task class
            task_class = name_to_task_class(self.task_name)
            self.task = self.rlbench_env.get_task(task_class)
            
            print(f"✓ Environment '{self.task_name}' initialized successfully in headless mode!")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize environment: {e}")
            print(f"Make sure you have:")
            print(f"1. Started X server: X :99 &")
            print(f"2. Set DISPLAY=:99")
            print(f"Available tasks include: reach_target, pick_and_lift, stack_blocks, open_drawer, close_jar, etc.")
            return False
    
    def reset_episode(self):
        """Reset the environment for a new episode"""
        try:
            _, self._current_obs = self.task.reset()
            print("\n🔄 Episode reset - new scenario loaded!")
        except Exception as e:
            print(f"Error resetting episode: {e}")
            return False
        return True
        
    def get_all_objects_positions(self) -> Dict[str, np.ndarray]:
        """Get positions of all objects in the scene"""
        positions = {}
        try:
            scene_objects = self.task.get_scene_objects()
            
            for obj in scene_objects:
                name = obj.get_name()
                pos = obj.get_position()
                positions[name] = np.array(pos)
                
        except Exception as e:
            print(f"Error getting object positions: {e}")
        
        return positions
    
    def get_robot_state(self) -> Dict:
        """Get current robot state"""
        try:
            robot = self.rlbench_env.robot
            return {
                'joint_positions': robot.arm.get_joint_positions(),
                'end_effector_pose': robot.arm.get_tip().get_pose()[:3],
                'gripper_open': robot.gripper.get_open_amount()[0] > 0.5
            }
        except Exception as e:
            print(f"Error getting robot state: {e}")
            return {}
    
    def move_to_position(self, target_position: np.ndarray, max_steps: int = 100) -> bool:
        """Move robot end-effector to target position"""
        print(f"🤖 Moving to position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        
        for step in range(max_steps):
            try:
                robot = self.rlbench_env.robot
                current_pose = robot.arm.get_tip().get_pose()
                current_position = np.array(current_pose[:3])
                
                # Calculate position difference
                position_diff = target_position - current_position
                distance = np.linalg.norm(position_diff)
                
                # Check if close enough
                if distance < 0.02:  # 2cm precision
                    print(f"✓ Reached target position (distance: {distance:.4f}m)")
                    return True
                
                # Calculate movement direction and speed
                if distance > 0.001:  # Avoid division by zero
                    direction = position_diff / distance
                    speed = min(0.05, distance * 2)  # Adaptive speed
                    
                    # Simple proportional control for joint velocities
                    joint_velocities = direction * speed * 0.5  # Scale factor
                    
                    # Create action array
                    action = np.concatenate([joint_velocities, [0]])  # Add gripper action
                    
                    # Step the environment
                    self._current_obs, reward, terminate = self.rlbench_env.step(action)
                    
                    if step % 20 == 0:  # Progress update
                        print(f"  Step {step}: distance = {distance:.4f}m")
                        
                else:
                    break
                    
            except Exception as e:
                print(f"Error during movement: {e}")
                return False
        
        print(f"✗ Failed to reach target position after {max_steps} steps")
        return False
    
    def control_gripper(self, open_gripper: bool, steps: int = 20):
        """Control gripper open/close"""
        action_value = 1.0 if open_gripper else 0.0
        action_name = "Opening" if open_gripper else "Closing"
        
        print(f"🦾 {action_name} gripper...")
        
        for _ in range(steps):
            # Create action with zero joint velocities and gripper control
            action = np.concatenate([np.zeros(7), [action_value]])
            self._current_obs, reward, terminate = self.rlbench_env.step(action)
            time.sleep(0.01)  # Small delay for realistic movement
    
    def grasp_object_at_position(self, target_position: np.ndarray) -> bool:
        """Execute full grasp sequence at target position"""
        approach_height = 0.05  # 5cm above target
        
        # Step 1: Move to approach position
        approach_pos = target_position.copy()
        approach_pos[2] += approach_height
        
        print("\n📍 Step 1: Moving to approach position...")
        if not self.move_to_position(approach_pos):
            return False
        
        # Step 2: Open gripper
        print("📍 Step 2: Opening gripper...")
        self.control_gripper(open_gripper=True)
        
        # Step 3: Move down to target
        print("📍 Step 3: Moving down to target...")
        if not self.move_to_position(target_position):
            return False
        
        # Step 4: Close gripper
        print("📍 Step 4: Closing gripper...")
        self.control_gripper(open_gripper=False)
        
        # Step 5: Lift up
        lift_pos = target_position.copy()
        lift_pos[2] += approach_height
        print("📍 Step 5: Lifting object...")
        success = self.move_to_position(lift_pos)
        
        if success:
            print("✓ Grasp sequence completed!")
        else:
            print("✗ Grasp sequence failed!")
            
        return success
    
    def check_task_success(self) -> bool:
        """Check if the task was completed successfully"""
        try:
            return self.task.success()
        except:
            return False
    
    def display_objects(self, positions: Dict[str, np.ndarray]):
        """Display all objects and their positions"""
        print("\n📦 Objects in scene:")
        print("-" * 50)
        for i, (name, pos) in enumerate(positions.items(), 1):
            print(f"{i:2d}. {name:<25} | Position: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
        print("-" * 50)
    
    def get_user_choice(self, positions: Dict[str, np.ndarray]) -> Optional[Tuple[str, np.ndarray]]:
        """Get user's choice for which object to grasp"""
        object_list = list(positions.items())
        
        while True:
            try:
                choice = input(f"\nEnter object number to grasp (1-{len(object_list)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                idx = int(choice) - 1
                if 0 <= idx < len(object_list):
                    return object_list[idx]
                else:
                    print(f"Please enter a number between 1 and {len(object_list)}")
                    
            except ValueError:
                print("Please enter a valid number or 'q'")
    
    def display_stats(self):
        """Display success statistics"""
        if self.total_attempts > 0:
            success_rate = (self.successful_attempts / self.total_attempts) * 100
            print(f"\n📊 Statistics:")
            print(f"   Total attempts: {self.total_attempts}")
            print(f"   Successful: {self.successful_attempts}")
            print(f"   Success rate: {success_rate:.1f}%")
        else:
            print("\n📊 No attempts made yet.")
    
    def run_interactive_session(self):
        """Main interactive loop"""
        print("=" * 60)
        print("🤖 RLBench Interactive Controller")
        print("=" * 60)
        
        if not self.initialize_env():
            return
        
        print("\nCommands:")
        print("  - Press Enter to continue")
        print("  - Type 'q' to quit")
        print("  - Follow prompts for object selection")
        
        while True:
            try:
                # Reset episode
                if not self.reset_episode():
                    print("Failed to reset episode, trying again...")
                    continue
                
                # Get object positions
                print("\n🔍 Scanning for objects...")
                positions = self.get_all_objects_positions()
                
                if not positions:
                    print("No objects found in scene!")
                    input("Press Enter to try again or Ctrl+C to quit...")
                    continue
                
                # Display objects
                self.display_objects(positions)
                
                # Get robot state
                robot_state = self.get_robot_state()
                print(f"\n🤖 Robot status:")
                print(f"   End effector: [{robot_state['end_effector_pose'][0]:.3f}, {robot_state['end_effector_pose'][1]:.3f}, {robot_state['end_effector_pose'][2]:.3f}]")
                print(f"   Gripper: {'Open' if robot_state['gripper_open'] else 'Closed'}")
                
                # Get user choice
                choice = self.get_user_choice(positions)
                if choice is None:
                    break
                
                object_name, target_pos = choice
                print(f"\n🎯 Selected: {object_name}")
                print(f"   Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                
                # Confirm action
                confirm = input("\nProceed with grasp? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
                
                # Execute grasp
                print(f"\n🚀 Executing grasp on '{object_name}'...")
                self.total_attempts += 1
                
                success = self.grasp_object_at_position(target_pos)
                
                # Check task completion
                task_success = self.check_task_success()
                
                if success and task_success:
                    print("🎉 SUCCESS! Task completed successfully!")
                    self.successful_attempts += 1
                elif success:
                    print("⚠️  Grasp successful, but task not completed")
                else:
                    print("❌ Grasp failed")
                
                # Display stats
                self.display_stats()
                
                # Continue or quit
                print("\n" + "="*60)
                continue_choice = input("Continue with new episode? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                continue
        
        # Final stats
        print("\n" + "="*60)
        print("🏁 Session completed!")
        self.display_stats()
        
        if self.rlbench_env:
            self.rlbench_env.shutdown()
            print("Environment closed.")

if __name__ == "__main__":
    # You can change the task here - try: reach_target, stack_blocks, open_drawer, close_jar
    controller = InteractiveRLBenchController('reach_target')
    controller.run_interactive_session()
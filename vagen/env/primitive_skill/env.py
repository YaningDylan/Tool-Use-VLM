from vagen.env.base.base_env import BaseEnv
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any
from gymnasium.utils import seeding
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import PrimitiveSkillEnvConfig
from .maniskill.utils import build_env, handle_info, get_workspace_limits
from .prompt import system_prompt, init_observation_template, action_template, format_prompt
from .toolbase import ManiSkillTools, ToolExecutor, ToolCallParser, get_available_tools_description
import vagen.env.primitive_skill.maniskill.env
import random
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper

class PrimitiveSkillEnv(BaseEnv):
    def __init__(self, config: PrimitiveSkillEnvConfig):
        """
        Initialize the PrimitiveSkill environment with optional enhanced tools.
        
        Args:
            config (PrimitiveSkillEnvConfig): Configuration parameters for the environment
        """
        BaseEnv.__init__(self)
        self.config = config
        if self.config.record_video:
            record_dir = self.config.video_record_dir
        else:
            record_dir = None
        self.env = build_env(config.env_id, record_dir=record_dir)
        
        # Store the format prompt function for later use based on the configuration
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        # Define the state keys for the environment
        self.state_keys = self.env.state_keys
        self.last_info = None
        
        # Initialize tools based on configuration
        if self.config.enable_enhanced_tools:
            # Use enhanced tools with YOLO integration
            from .enhanced_toolbase import EnhancedManiSkillTools, EnhancedToolExecutor, get_enhanced_tools_description
            self.tools = EnhancedManiSkillTools(self, self.config.toolbase_config)
            self.tool_executor = EnhancedToolExecutor(self.tools)
            self.tools_description = get_enhanced_tools_description()
            print(" Enhanced tools with YOLO enabled")
        else:
            # Use original tools only
            from .toolbase import ManiSkillTools, ToolExecutor, get_available_tools_description
            self.tools = ManiSkillTools(self)
            self.tool_executor = ToolExecutor(self.tools)
            self.tools_description = get_available_tools_description()
            print(" Basic tools enabled")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed (Optional[int]): Random seed for environment generation
                                  If None, a random seed is used
        
        Returns:
            Tuple[Dict, Dict]: 
                - obs: Dictionary containing observation string and optional image data
                - info: Empty dictionary for initial state
        """
        _, info = self.env.reset(seed=seed)
        self.last_info = info
        obs = self._render(init_obs=True)
        self.initial_reward = self._compute_reward()
        self.total_reward = 0
        self.steps = 0
        return obs, {}
    
    def _is_tool_call(self, action_str: str) -> bool:
        """
        Check if the action string contains tool calls
        
        Args:
            action_str: Input string from VLM
            
        Returns:
            True if contains tool calls, False if contains robot actions
        """
        # Check for common tool patterns
        tool_patterns = [
            r'get_pos\s*\(',
            r'get_targets\s*\(',
            r'get_objects\s*\(',
            r'get_workspace\s*\(',
            r'get_object_position_by_color\s*\(',
            r'get_target_positions\s*\(',
            r'get_all_objects\s*\(',
            r'get_workspace_limits\s*\(',
            r'detect_scene\s*\(',
            r'find_object\s*\(',
            r'get_spatial_relations\s*\(',
        ]
        
        import re
        for pattern in tool_patterns:
            if re.search(pattern, action_str, re.IGNORECASE):
                return True
        
        return False
    
    def _handle_tool_call(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Handle tool calls from VLM
        
        Args:
            action_str: String containing tool calls
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Execute tool calls
        tool_results, formatted_results = self.tool_executor.execute_tool_calls(action_str)
        
        # Create observation with tool results
        obs = self._render_with_tools(action_str, formatted_results)
        
        # Tool calls don't change environment state, so no reward and not done
        reward = 0
        done = False
        
        # Add tool execution info
        info = {
            "tool_calls_executed": len(tool_results),
            "tool_results": tool_results,
            "tool_response": formatted_results,
            "is_tool_step": True,
            "metrics": {
                "turn_metrics": {
                    "action_is_valid": all(r.success for r in tool_results),
                },
                "traj_metrics": {
                    "success": False,
                },
            }
        }
        
        return obs, reward, done, info
    
    @env_state_reward_wrapper
    def step(self, action_str):
        """
        Take a step in the environment based on the agent's action.
        
        Args:
            action_str (str): Raw string from LLM containing actions or tool calls
        
        Returns:
            Tuple[Dict, float, bool, Dict]:
                - obs: Dictionary with observation string and optional image data
                - reward: Numeric reward for the step
                - done: Boolean indicating if episode is complete
                - info: Dictionary containing metrics and parsed action data
        """
        # Check if this is a tool call or robot action
        if self._is_tool_call(action_str):
            return self._handle_tool_call(action_str)
        
        # Original robot action handling
        reward = 0
        rst = self.parse_func(response=action_str,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step)
        
        output_info = {}
        output_info.update(rst)
        valid_actions = []
        metrics = {
            "turn_metrics": {
                "action_is_valid": False,  # True if at least one valid action was parsed
            },
            "traj_metrics": {
                "success": False,  # Will be set to True if agent reaches goal
            },
        }
        
        info = self.last_info
        terminated, truncated = False, False
        
        # Execute each action in the list
        for action in rst['actions']:
            parsed_action = self._parse_action(action)
            if parsed_action is not None:
                _, _, terminated, truncated, info = self.env.step(parsed_action)
                valid_actions.append(action)
                self.last_info = info
                self.steps += 1
            else:
                info = self.last_info
                terminated, truncated = False, False
                break
            if truncated or terminated:
                break
        
        # Check if actions were valid and format was correct
        metrics["turn_metrics"]['action_is_valid'] = len(valid_actions) > 0 and len(valid_actions) == len(rst['actions'])
        if metrics["turn_metrics"]['action_is_valid'] and rst["format_correct"]:
            reward += self.config.format_reward
            output_info["is_format_rewarded"] = True
        else:
            output_info["is_format_rewarded"] = False
        
        if info.get('is_success', False):
            metrics["traj_metrics"]['success'] = True
        
        done = terminated or truncated
        
        obs = self._render(init_obs=False, valid_actions=valid_actions)
        output_info["metrics"] = metrics
        output_info["is_tool_step"] = False
        
        self.total_reward += reward
        if isinstance(done, np.ndarray):
            done = done.item()
            
        return obs, reward, done, output_info
    
    def system_prompt(self):
        """
        Get the system prompt for the environment.
        
        Returns:
            str: System prompt string with environment description and instructions
        """
        # Get format prompt with examples for system prompt
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            state_keys=self.state_keys,
            add_example=True,  # Always true for system prompt
            is_tool_turn=True  # System prompt should show tool examples
        )
        
        # Import and use the new system_prompt function
        from .prompt import system_prompt
        base_prompt = system_prompt()
        
        # Use appropriate tools description based on configuration
        tools_description = self.tools_description
        
        return base_prompt + '\n\n' + tools_description + '\n\n' + format_prompt_text
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        self.env.close()
    
    def _compute_reward(self):
        """
        Calculate the reward based on environment state.
        
        Returns:
            float: Computed reward value
        """
        if self.last_info.get("success", False):
            return 10
        
        # Find the highest successful stage
        max_stage = -1
        for key in self.last_info.keys():
            if key.startswith("stage_") and key.endswith("_success"):
                try:
                    # Extract the stage number
                    stage_num = int(key.split("_")[1])
                    # Check if this stage is successful
                    if self.last_info[key]:
                        max_stage = max(max_stage, stage_num)
                except (ValueError, IndexError):
                    # Skip keys that don't follow the expected format
                    continue
        return (max_stage + 1) * 2
    
    def compute_reward(self):
        """
        Get the cumulative reward for the episode.
        
        Returns:
            float: Total reward accumulated during the current episode
        """
        return self._compute_reward() - self.initial_reward - self.steps * 0.1

    def _render(self, init_obs=True, valid_actions=None, seed=42):
        """
        Render the environment as an observation.
        
        Args:
            init_obs (bool): If True, create initial observation
            valid_actions (list): List of valid actions executed (for step observations)
        
        Returns:
            Dict: Observation dictionary containing observation string and optional image data
        """
        info = self.last_info.copy()
        new_info = handle_info(info, state_keys=self.state_keys, mask_success=self.config.mask_success, env=self.env)
        
        instruction = self.env.instruction()
        img_placeholder = self.config.image_placeholder
        x_workspace, y_workspace, z_workspace = get_workspace_limits(self.env)
        
        # Get format prompt without examples for action/init templates
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            state_keys=self.state_keys,
            add_example=False,  # No examples for action and init obs
            is_tool_turn=init_obs  # True for first turn, False for subsequent turns
        )
        
        if init_obs:
            # Use the init_observation_template from prompt.py
            from .prompt import init_observation_template
            obs_str = init_observation_template(
                observation=img_placeholder,
                instruction=instruction,
                x_workspace=x_workspace,
                y_workspace=y_workspace,
                z_workspace=z_workspace,
                other_information="No other information needed"
            )
            obs_str += "\n" + format_prompt_text
        else:
            # Use the action_template from prompt.py
            from .prompt import action_template
            obs_str = action_template(
                observation=img_placeholder,
                instruction=instruction,
                x_workspace=x_workspace,
                y_workspace=y_workspace,
                z_workspace=z_workspace,
                valid_actions=valid_actions,
                other_information="No other information needed",
                tool_results=""  # No tool results in regular step
            )
            obs_str += "\n" + format_prompt_text
        
        multi_modal_data = None
        if self.config.render_mode == "vision":
            img = self.env.render()
            multi_modal_data = {
                img_placeholder: [convert_numpy_to_PIL(img)]
            }
        
        # Return observation dictionary with appropriate fields
        if multi_modal_data is not None:
            return {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal_data,
            }
        else:
            return {
                "obs_str": obs_str,
            }

    def _render_with_tools(self, tool_calls: str, tool_results: str):
        """
        Render observation after tool calls
        
        Args:
            tool_calls: Original tool call string
            tool_results: Formatted tool results
            
        Returns:
            Observation dictionary
        """
        info = self.last_info.copy()
        new_info = handle_info(info, state_keys=self.state_keys, mask_success=self.config.mask_success, env=self.env)
        
        instruction = self.env.instruction()
        img_placeholder = self.config.image_placeholder
        x_workspace, y_workspace, z_workspace = get_workspace_limits(self.env)
        
        # Use the tool_response_template from prompt.py
        from .prompt import tool_response_template
        obs_str = tool_response_template(
            instruction=instruction,
            x_workspace=x_workspace,
            y_workspace=y_workspace,
            z_workspace=z_workspace,
            tool_results=tool_results,
            other_information="No other information needed"
        )
        
        # Add format prompt for actions (not tool turn)
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            state_keys=self.state_keys,
            add_example=False,
            is_tool_turn=False  # This is for robot actions
        )
        obs_str += "\n" + format_prompt_text
        
        multi_modal_data = None
        if self.config.render_mode == "vision":
            img = self.env.render()
            multi_modal_data = {
                img_placeholder: [convert_numpy_to_PIL(img)]
            }
        
        if multi_modal_data is not None:
            return {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal_data,
            }
        else:
            return {
                "obs_str": obs_str,
            }
    
    def get_env_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            dict: Dictionary representation of the environment state
        """
        rst = handle_info(self.last_info, state_keys=self.state_keys, mask_success=self.config.mask_success, env=self.env)
        return rst["obj_positions"]
    
    def _parse_action(self, action_str):
        """
        Parse a single action string into an action array.
        
        Args:
            action_str (str): Action string to parse
            
        Returns:
            np.array: Parsed action array or None if invalid
        """
        # Initialize empty 9-dim array (3 for action type, 6 for coordinates)
        action_array = np.zeros(9)
        
        # Workspace boundaries
        workspace_x, workspace_y, workspace_z = get_workspace_limits(self.env)
        
        # Check if the string is empty or None
        if not action_str:
            return None
        
        try:
            # Extract action name and parameters
            action_name = action_str.split('(')[0].strip().lower()
            
            # Set the action type
            if action_name == "pick":
                action_array[0] = 1
            elif action_name == "place":
                action_array[1] = 1
            elif action_name == "push":
                action_array[2] = 1
            else:
                # Invalid action name
                return None
            
            # Extract parameters
            params_str = action_str.split('(')[1].split(')')[0]
            params = [float(p.strip()) for p in params_str.split(',')]
            
            # Check if we have the correct number of parameters
            if action_name in ["pick", "place"] and len(params) != 3:
                return None
            elif action_name == "push" and len(params) != 6:
                return None
            
            # Apply workspace constraints and scale
            # First point (x,y,z)
            params[0] = np.clip(params[0], workspace_x[0], workspace_x[1])
            params[1] = np.clip(params[1], workspace_y[0], workspace_y[1])
            params[2] = np.clip(params[2], workspace_z[0], workspace_z[1])
            
            # Second point (x1,y1,z1) if it exists (for push)
            if action_name == "push":
                params[3] = np.clip(params[3], workspace_x[0], workspace_x[1])
                params[4] = np.clip(params[4], workspace_y[0], workspace_y[1])
                params[5] = np.clip(params[5], workspace_z[0], workspace_z[1])
            
            # Fill the coordinate dimensions (after dividing by 1000 as in your modified function)
            for i in range(len(params)):
                action_array[i+3] = params[i]/1000.0
            
            return action_array
        
        except (IndexError, ValueError):
            # If any parsing error occurs, return None
            return None

if __name__ == "__main__":
    """
    Interactive test for enhanced PrimitiveSkillEnv with YOLO support
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Test PrimitiveSkill Environment with Enhanced Tools")
    parser.add_argument(
        "--env_id",
        type=str,
        default="PlaceTwoCube",
        choices=["AlignTwoCube", "PlaceTwoCube", "PutAppleInDrawer", "StackThreeCube"],
        help="Environment to test"
    )
    parser.add_argument(
        "--save_images",
        action="store_true", 
        help="Save images during interaction"
    )
    parser.add_argument(
        "--enable_yolo",
        action="store_true",
        default=True,
        help="Enable YOLO enhanced tools (default: True)"
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLO model to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for YOLO inference"
    )
    
    args = parser.parse_args()
    
    # Create environment config with enhanced tools
    config = PrimitiveSkillEnvConfig(
        env_id=args.env_id,
        render_mode="vision",
        record_video=False,
        mask_success=True,
        enable_enhanced_tools=args.enable_yolo,
        toolbase_config={
            'enable_yolo': args.enable_yolo,
            'yolo': {
                'model_path': args.yolo_model,
                'device': args.device,
                'confidence_threshold': 0.5
            }
        }
    )
    
    # Initialize environment
    print(f"🔧 Initializing {args.env_id} environment...")
    if args.enable_yolo:
        print(f"🤖 Enhanced tools enabled with YOLO {args.yolo_model} on {args.device}")
    else:
        print(f"🛠️ Basic tools only (YOLO disabled)")
    
    env = PrimitiveSkillEnv(config)
    
    # Save directory for images
    if args.save_images:
        save_dir = f"./test_images_{args.env_id}_{'yolo' if args.enable_yolo else 'basic'}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Images will be saved to: {save_dir}")
    
    print("✓ Environment initialized!")
    
    def save_image(obs, step, suffix=""):
        """Save current image"""
        if args.save_images and 'multi_modal_data' in obs:
            img = obs['multi_modal_data'][config.image_placeholder][0]
            filename = f"step_{step:02d}{suffix}.png"
            filepath = os.path.join(save_dir, filename)
            img.save(filepath)
            print(f"📷 Saved: {filename}")
    
    def print_system_prompt():
        """Print the system prompt"""
        print("\n" + "="*80)
        print("📋 SYSTEM PROMPT:")
        print("="*80)
        print(env.system_prompt())
        print("="*80)
    
    def print_available_tools():
        """Print available tools based on configuration"""
        print("\n💡 Available Commands:")
        print("  Basic Tools:")
        print("    get_objects() - List all visible objects")
        print("    get_pos(color) - Get position by color (e.g., get_pos(red))")
        print("    get_targets() - Get all target positions")
        print("    get_workspace() - Get workspace boundaries")
        
        if args.enable_yolo:
            print("  Enhanced YOLO Tools:")
            print("    detect_scene() - YOLO detection of all objects")
            print("    find_object(description) - Find by description (e.g., find_object(\"red cube\"))")
            print("    get_spatial_relations() - Analyze spatial relationships")
        
        print("  Robot Actions:")
        print("    pick(x,y,z) - Grasp object at position")
        print("    place(x,y,z) - Place object at position")
        print("    push(x1,y1,z1,x2,y2,z2) - Push object")
        
        print("  Control Commands:")
        print("    reset - Reset environment")
        print("    prompt - Show system prompt")
        print("    tools - Show this tool list")
        print("    save - Save current image")
        print("    quit/q/exit - Exit")
    
    def test_yolo_tools():
        """Test YOLO tools functionality"""
        if not args.enable_yolo:
            print("⚠️ YOLO tools not enabled")
            return
            
        print("\n🧪 Testing YOLO tools...")
        try:
            # Test detect_scene
            print("Testing detect_scene()...")
            result = env.tools.detect_scene_objects()
            if result.success:
                print(f"✓ detect_scene: {result.message}")
                print(f"  Detected {len(result.data.get('enhanced_detections', []))} objects")
            else:
                print(f"❌ detect_scene failed: {result.message}")
            
            # Test find_object if objects detected
            if result.success and result.data.get('enhanced_detections'):
                detected_classes = [item['yolo_detection']['class_name'] 
                                  for item in result.data['enhanced_detections']]
                if detected_classes:
                    test_class = detected_classes[0]
                    print(f"Testing find_object(\"{test_class}\")...")
                    find_result = env.tools.find_object_by_description(test_class)
                    if find_result.success:
                        print(f"✓ find_object: Found {test_class}")
                    else:
                        print(f"❌ find_object failed: {find_result.message}")
            
            # Test spatial relations
            print("Testing get_spatial_relations()...")
            spatial_result = env.tools.get_spatial_relationships()
            if spatial_result.success:
                relations = spatial_result.data.get('relationships', [])
                print(f"✓ get_spatial_relations: Found {len(relations)} relationships")
                for relation in relations[:3]:  # Show first 3
                    print(f"  - {relation}")
            else:
                print(f"❌ get_spatial_relations failed: {spatial_result.message}")
                
        except Exception as e:
            print(f"❌ YOLO test failed: {e}")
    
    try:
        print("\n🎮 Starting Interactive Session")
        print(f"🎯 Task: {args.env_id}")
        
        # Show system prompt
        print_system_prompt()
        
        # Reset environment
        print("\n🔄 Resetting environment...")
        obs, info = env.reset()
        step = 0
        
        # Save initial image
        save_image(obs, step, "_initial")
        
        # Show initial observation
        print("\n📖 INITIAL OBSERVATION:")
        print("-" * 60)
        print(obs["obs_str"])
        print("-" * 60)
        
        # Show available tools
        print_available_tools()
        
        # Test YOLO tools if enabled
        if args.enable_yolo:
            test_yolo_tools()
        
        # Interactive loop
        while True:
            try:
                user_input = input(f"\n[Step {step}]> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'q', 'exit']:
                    print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == 'reset':
                    print("🔄 Resetting environment...")
                    obs, info = env.reset()
                    step = 0
                    save_image(obs, step, "_reset")
                    print("✓ Environment reset!")
                    continue
                    
                elif user_input.lower() == 'prompt':
                    print_system_prompt()
                    continue
                    
                elif user_input.lower() == 'tools':
                    print_available_tools()
                    continue
                    
                elif user_input.lower() == 'test' and args.enable_yolo:
                    test_yolo_tools()
                    continue
                    
                elif user_input.lower() == 'save':
                    save_image(obs, step, "_manual")
                    continue
                
                # Execute user input as action/tool call
                print(f"🤖 Executing: {user_input}")
                
                try:
                    obs, reward, done, info = env.step(user_input)
                    step += 1
                    
                    # Save image after action
                    if info.get('is_tool_step', False):
                        save_image(obs, step, "_tool")
                    else:
                        save_image(obs, step, "_action")
                    
                    # Show results
                    print(f"✓ Executed successfully!")
                    print(f"📊 Reward: {reward:.2f}")
                    print(f"🏁 Done: {done}")
                    
                    if info.get('is_tool_step', False):
                        print(f"🛠️ Tool calls executed: {info.get('tool_calls_executed', 0)}")
                        if 'tool_response' in info:
                            print(f"🔧 Tool results:")
                            print(info['tool_response'])
                    else:
                        valid_actions = info.get('metrics', {}).get('turn_metrics', {}).get('action_is_valid', False)
                        print(f"🎯 Action valid: {valid_actions}")
                    
                    # Show new observation
                    print("\n📖 NEW OBSERVATION:")
                    print("-" * 60)
                    print(obs["obs_str"])
                    print("-" * 60)
                    
                    if done:
                        success = info.get('metrics', {}).get('traj_metrics', {}).get('success', False)
                        if success:
                            print("🎉 Task completed successfully!")
                        else:
                            print("❌ Episode ended (failed or timeout)")
                        
                        choice = input("\nReset for another attempt? (y/n): ").strip().lower()
                        if choice == 'y':
                            obs, info = env.reset()
                            step = 0
                            save_image(obs, step, "_new_episode")
                            print("✓ Environment reset for new episode!")
                        else:
                            break
                
                except Exception as e:
                    print(f"❌ Error executing action: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
    
    finally:
        # Cleanup
        env.close()
        print("🔧 Environment closed")
        if args.save_images:
            print(f"📁 Images saved in: {save_dir}")
        
        # Show final statistics
        total_reward = env.compute_reward()
        print(f"📊 Final total reward: {total_reward:.2f}")
        
        # Show tool usage summary
        if args.enable_yolo:
            yolo_status = "✓ Available" if hasattr(env.tools, 'yolo_detector') and env.tools.yolo_detector else "❌ Failed"
            print(f"🤖 YOLO Status: {yolo_status}")
        
        print("\n🎯 Usage Examples:")
        print("  Basic: python env.py --env_id PlaceTwoCube")
        print("  With YOLO: python env.py --env_id PlaceTwoCube --enable_yolo --yolo_model yolov8s.pt")
        print("  GPU: python env.py --enable_yolo --device cuda")
        print("  Save images: python env.py --save_images")
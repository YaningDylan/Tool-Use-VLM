import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    data: Any
    message: str = ""
    tool_name: str = ""

class ManiSkillTools:
    """
    Tool collection for ManiSkill environments
    Provides environment-specific tools for object detection and scene understanding
    """
    
    def __init__(self, env):
        """
        Initialize tools with reference to the environment
        
        Args:
            env: PrimitiveSkillEnv instance
        """
        self.env = env
        
    def get_object_position_by_color(self, color: str) -> ToolResult:
        """
        Get object position by color name
        
        Args:
            color: Color name (e.g., 'red', 'green', 'blue', 'yellow', 'purple')
            
        Returns:
            ToolResult with position tuple (x, y, z) in mm or error
        """
        try:
            # Get current environment state
            env_state = self.env.get_env_state()
            
            # Map color names to position keys
            color_mapping = {
                'red': 'red_cube_position',
                'green': 'green_cube_position', 
                'blue': 'blue_cube_position',
                'yellow': 'yellow_cube_position',
                'purple': 'purple_cube_position',
                'apple': 'apple_position'
            }
            
            position_key = color_mapping.get(color.lower())
            if position_key and position_key in env_state:
                pos = env_state[position_key]
                position = tuple(int(x) for x in pos)
                return ToolResult(
                    success=True,
                    data=position,
                    message=f"Found {color} object at position {position}",
                    tool_name="get_object_position_by_color"
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"Object with color '{color}' not found in scene",
                    tool_name="get_object_position_by_color"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Error getting object position: {str(e)}",
                tool_name="get_object_position_by_color"
            )
    
    def get_target_positions(self) -> ToolResult:
        """
        Get all target/goal positions in the scene
        
        Returns:
            ToolResult with dictionary mapping target names to positions
        """
        try:
            env_state = self.env.get_env_state()
            targets = {}
            
            # Look for target positions
            for key, pos in env_state.items():
                if 'target' in key.lower():
                    target_name = key.replace('_position', '')
                    targets[target_name] = tuple(int(x) for x in pos)
            
            if targets:
                return ToolResult(
                    success=True,
                    data=targets,
                    message=f"Found {len(targets)} target(s): {list(targets.keys())}",
                    tool_name="get_target_positions"
                )
            else:
                return ToolResult(
                    success=True,
                    data={},
                    message="No targets found in this environment",
                    tool_name="get_target_positions"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Error getting target positions: {str(e)}",
                tool_name="get_target_positions"
            )
    
    def get_all_objects(self) -> ToolResult:
        """
        Get all visible objects in the scene
        
        Returns:
            ToolResult with list of object names
        """
        try:
            env_state = self.env.get_env_state()
            objects = []
            
            for key in env_state.keys():
                if key.endswith('_position'):
                    obj_name = key.replace('_position', '')
                    objects.append(obj_name)
            
            return ToolResult(
                success=True,
                data=objects,
                message=f"Found {len(objects)} object(s): {objects}",
                tool_name="get_all_objects"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                message=f"Error getting objects: {str(e)}",
                tool_name="get_all_objects"
            )
    
    def get_workspace_limits(self) -> ToolResult:
        """
        Get robot workspace boundaries
        
        Returns:
            ToolResult with workspace limits
        """
        try:
            from .maniskill.utils import get_workspace_limits
            x_workspace, y_workspace, z_workspace = get_workspace_limits(self.env.env)
            
            limits = {
                'x_limit': x_workspace,
                'y_limit': y_workspace, 
                'z_limit': z_workspace
            }
            
            return ToolResult(
                success=True,
                data=limits,
                message=f"Workspace limits: X{x_workspace}, Y{y_workspace}, Z{z_workspace}",
                tool_name="get_workspace_limits"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Error getting workspace limits: {str(e)}",
                tool_name="get_workspace_limits"
            )

class ToolCallParser:
    """
    Parser for tool calls in VLM responses
    Supports multiple formats for tool calling
    """
    
    @staticmethod
    def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from VLM response
        
        Supported formats:
        - get_pos(red)
        - get_targets()
        - get_objects()
        
        Args:
            response: VLM response text
            
        Returns:
            List of parsed tool calls
        """
        tool_calls = []
        
        # Pattern to match function calls: function_name(args)
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, response)
        
        for func_name, args_str in matches:
            # Map function names to tool methods
            tool_mapping = {
                'get_pos': 'get_object_position_by_color',
                'get_object_position_by_color': 'get_object_position_by_color',
                'get_targets': 'get_target_positions',
                'get_target_positions': 'get_target_positions', 
                'get_objects': 'get_all_objects',
                'get_all_objects': 'get_all_objects',
                'get_workspace': 'get_workspace_limits',
                'get_workspace_limits': 'get_workspace_limits'
            }
            
            if func_name in tool_mapping:
                # Parse arguments
                args = []
                if args_str.strip():
                    # Split by comma and clean up
                    args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
                
                tool_calls.append({
                    'tool_name': tool_mapping[func_name],
                    'arguments': args,
                    'original_call': f"{func_name}({args_str})"
                })
        
        return tool_calls
    
    @staticmethod
    def format_tool_result(result: ToolResult) -> str:
        """
        Format tool execution result for VLM
        
        Args:
            result: ToolResult to format
            
        Returns:
            Formatted string for VLM consumption
        """
        if result.success:
            if isinstance(result.data, dict):
                data_str = ", ".join([f"{k}: {v}" for k, v in result.data.items()])
            elif isinstance(result.data, (list, tuple)):
                data_str = str(result.data)
            else:
                data_str = str(result.data)
            
            return f"✓ {result.tool_name}: {data_str}"
        else:
            return f"✗ {result.tool_name}: {result.message}"

class ToolExecutor:
    """
    Executes tool calls and manages tool execution
    """
    
    def __init__(self, tools: ManiSkillTools):
        """
        Initialize executor with tools instance
        
        Args:
            tools: ManiSkillTools instance
        """
        self.tools = tools
        
    def execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        """
        Execute a single tool call
        
        Args:
            tool_call: Parsed tool call dictionary
            
        Returns:
            ToolResult from execution
        """
        tool_name = tool_call['tool_name']
        arguments = tool_call['arguments']
        
        try:
            # Get the tool method
            if hasattr(self.tools, tool_name):
                method = getattr(self.tools, tool_name)
                
                # Call with appropriate arguments
                if len(arguments) == 0:
                    result = method()
                elif len(arguments) == 1:
                    result = method(arguments[0])
                else:
                    result = method(*arguments)
                
                return result
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"Unknown tool: {tool_name}",
                    tool_name=tool_name
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Tool execution error: {str(e)}",
                tool_name=tool_name
            )
    
    def execute_tool_calls(self, response: str) -> Tuple[List[ToolResult], str]:
        """
        Execute all tool calls found in response
        
        Args:
            response: VLM response containing tool calls
            
        Returns:
            Tuple of (tool results, formatted results string)
        """
        # Parse tool calls
        tool_calls = ToolCallParser.parse_tool_calls(response)
        
        if not tool_calls:
            return [], ""
        
        # Execute each tool call
        results = []
        result_strings = []
        
        for tool_call in tool_calls:
            result = self.execute_tool_call(tool_call)
            results.append(result)
            
            # Format result for VLM
            formatted_result = ToolCallParser.format_tool_result(result)
            result_strings.append(formatted_result)
        
        # Combine all results
        combined_results = "\n".join(result_strings)
        
        return results, combined_results

def get_available_tools_description() -> str:
    """
    Get description of all available tools for VLM prompt
    
    Returns:
        String description of available tools
    """
    return """Available Tools:
- get_pos(color): Get position of object by color (e.g., get_pos(red))
- get_targets(): Get all target positions in the scene
- get_objects(): List all visible objects in the scene  
- get_workspace(): Get robot workspace boundaries

Tool Usage Examples:
- get_pos(red) → Returns position like (100, 50, 20)
- get_targets() → Returns target positions like {left_target: (80, -100, 0)}
- get_objects() → Returns object list like ['red_cube', 'green_cube']"""

if __name__ == "__main__":
    # Test tool call parsing
    test_response = """
    I need to understand the scene first.
    get_objects()
    Now let me find the red cube position:
    get_pos(red)
    And check target locations:
    get_targets()
    """
    
    print("Testing tool call parsing:")
    tool_calls = ToolCallParser.parse_tool_calls(test_response)
    for call in tool_calls:
        print(f"- {call}")
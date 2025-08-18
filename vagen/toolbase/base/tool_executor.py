"""Tool executor for managing and running tools"""

from typing import Dict, List, Any
from .tool_interface import BaseTool
from .tool_result import ToolResult

class ToolExecutor:
    """Execute and manage multiple tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool) -> bool:
        """Register a tool"""
        if not tool.is_initialized:
            if not tool.initialize():
                return False
        
        self.tools[tool.name] = tool
        return True
    
    def execute_tool(self, tool_name: str, *args, **kwargs) -> ToolResult:
        """Execute a specific tool"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                message=f"Tool '{tool_name}' not found",
                tool_name=tool_name
            )
        
        try:
            return self.tools[tool_name].execute(*args, **kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Tool execution failed: {str(e)}",
                tool_name=tool_name
            )
    
    def get_available_tools(self) -> List[str]:
        """Get list of registered tools"""
        return list(self.tools.keys())
    
    def cleanup_all(self):
        """Clean up all registered tools"""
        for tool in self.tools.values():
            tool.cleanup()

__all__ = ['ToolExecutor']
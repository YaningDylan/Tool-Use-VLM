"""Universal toolbase for VLM agents"""

from .base.tool_interface import BaseTool, VisionTool, ToolResult
from .vision.yolo.detector import YOLODetector

AVAILABLE_TOOLS = {
    'yolo_detector': YOLODetector,
}

def create_tool(tool_name: str, config: dict = None) -> BaseTool:
    """Create tool instance by name"""
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool_class = AVAILABLE_TOOLS[tool_name]
    return tool_class(config)

def get_available_tools() -> list:
    """Get list of available tools"""
    return list(AVAILABLE_TOOLS.keys())

__all__ = [
    'BaseTool', 'VisionTool', 'ToolResult', 
    'YOLODetector', 'create_tool', 'get_available_tools'
]
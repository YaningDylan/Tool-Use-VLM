"""Tool execution result data class"""

from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class ToolResult:
    """Unified tool execution result"""
    success: bool
    data: Any
    message: str = ""
    tool_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'data': self.data,
            'message': self.message,
            'tool_name': self.tool_name,
            'metadata': self.metadata
        }
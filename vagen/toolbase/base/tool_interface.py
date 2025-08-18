"""Base tool interfaces and abstract classes"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
from .tool_result import ToolResult

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str, config: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize tool resources"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> ToolResult:
        """Execute tool with given parameters"""
        pass
    
    def cleanup(self):
        """Clean up tool resources"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": self.name,
            "description": self.description,
            "is_initialized": self.is_initialized,
            "config": self.config
        }

class VisionTool(BaseTool):
    """Base class for vision-related tools"""
    
    def __init__(self, name: str, description: str, config: Dict[str, Any] = None):
        super().__init__(name, description, config)
        self.requires_image = True
    
    def validate_image(self, image: np.ndarray) -> bool:
        """Validate input image format"""
        if not isinstance(image, np.ndarray):
            return False
        if len(image.shape) not in [2, 3]:
            return False
        if image.shape[0] == 0 or image.shape[1] == 0:
            return False
        return True
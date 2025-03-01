import inspect
from typing import Any, Callable, Dict, List, Optional
from .schema import ToolCall, ToolResult

class Tool:
    def __init__(self, fn: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or fn.__doc__ or "No description provided."
        self.parameters = self._generate_parameters_schema()

    def _generate_parameters_schema(self) -> Dict[str, Any]:
        sig = inspect.signature(self.fn)
        properties = {}
        required = []
        for name, param in sig.parameters.items():
            param_type = "string" 
            if param.annotation == int:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            
            properties[name] = {
                "type": param_type,
                "description": f"The {name} parameter"
            }
            if param.default == inspect.Parameter.empty:
                required.append(name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def __call__(self, **kwargs) -> Any:
        return self.fn(**kwargs)

    def to_gemini_schema(self) -> Dict[str, Any]:
        """Convert to Gemini/OpenAI compatible tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, name: Optional[str] = None, description: Optional[str] = None):
        def decorator(fn: Callable):
            tool = Tool(fn, name, description)
            self.tools[tool.name] = tool
            return fn
        return decorator

    def execute(self, tool_call: ToolCall) -> ToolResult:
        if tool_call.name not in self.tools:
            return ToolResult(call_id=tool_call.id, output=f"Error: Tool {tool_call.name} not found.")
        
        try:
            tool = self.tools[tool_call.name]
            # No type casting in initial version (bug)
            output = tool(**tool_call.args)
            return ToolResult(call_id=tool_call.id, output=output)
        except Exception as e:
            return ToolResult(call_id=tool_call.id, output=f"Error executing tool: {str(e)}")

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_gemini_schema() for tool in self.tools.values()]

registry = ToolRegistry()
tool = registry.register

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    id: str

@dataclass
class ToolResult:
    call_id: str
    output: Any

@dataclass
class Message:
    role: Role
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None

@dataclass
class AgentState:
    history: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

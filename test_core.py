from agent_pure.tools import registry, tool
from agent_pure.schema import ToolCall

@tool(name="test_tool", description="A test tool")
def my_tool(x: int, y: int) -> int:
    return x + y

def test_registry():
    schemas = registry.get_schemas()
    assert len(schemas) > 0
    assert schemas[0]["name"] in ["calculator", "get_weather", "test_tool"]
    print("Registry schemas:", schemas)

def test_execution():
    call = ToolCall(name="test_tool", args={"x": 5, "y": 10}, id="123")
    result = registry.execute(call)
    assert result.output == 15
    print("Execution result:", result.output)

if __name__ == "__main__":
    test_registry()
    test_execution()
    print("Core logic verification successful!")

from .schema import Message, Role, AgentState, ToolCall, ToolResult
from .tools import registry
from .llm import LLM

class Agent:
    def __init__(self, llm: LLM, system_prompt: str = "You are a helpful AI Agent."):
        self.llm = llm
        self.system_prompt = system_prompt
        self.state = AgentState()
        self.state.history.append(Message(role=Role.SYSTEM, content=self.system_prompt))

    def run(self, user_input: str, max_iterations: int = 5):
        print(f"\n--- Agent Starting: {user_input} ---")
        self.state.history.append(Message(role=Role.USER, content=user_input))
        
        for i in range(max_iterations):
            print(f"\n[Iteration {i+1}] Thinking...")
            
            response = self.llm.generate(
                self.state.history, 
                tools=registry.get_schemas()
            )
            self.state.history.append(response)
            
            if response.content:
                print(f"[Thought/Response]: {response.content}")
            
            if not response.tool_calls:
                print("\n--- Agent Finished ---")
                return response.content
            
            print(f"[Action]: Calling tools: {[tc.name for tc in response.tool_calls]}")
            
            # Step 4: Execute tools (Observe)
            tool_results = []
            for tc in response.tool_calls:
                result = registry.execute(tc)
                tool_results.append(result)
                print(f"[Observation]: Result from {tc.name}: {result.output}")
            
            self.state.history.append(Message(
                role=Role.TOOL,
                content="", 
                tool_results=tool_results
            ))
            
        print("\n--- Reached max iterations ---")
        return "I'm sorry, I reached my iteration limit without a final answer."

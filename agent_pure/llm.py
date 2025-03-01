import os
from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
from .schema import Message, Role, ToolCall, ToolResult
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self, model_name: str = "gemini-1.5-flash-8b"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _convert_messages(self, messages: List[Message]) -> List[types.Content]:
        gemini_messages = []
        for msg in messages:
            role = "user" if msg.role == Role.USER else "model"
            if msg.role == Role.SYSTEM:
                role = "user" 
            
            parts = []
            if msg.content:
                parts.append(types.Part(text=msg.content))
            
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(types.Part(
                        function_call=types.FunctionCall(
                            name=tc.name,
                            args=tc.args
                        )
                    ))
            
            if msg.tool_results:
                for tr in msg.tool_results:
                    parts.append(types.Part(
                        function_response=types.FunctionResponse(
                            name="unknown", 
                            response={"result": tr.output}
                        )
                    ))

            gemini_messages.append(types.Content(role=role, parts=parts))
        
        # Fixing the "unknown" names for tool results by looking at the previous message
        for i, content in enumerate(gemini_messages):
            if any(p.function_response for p in content.parts):
                # Look at the previous model message for the function call names
                if i > 0:
                    prev_content = gemini_messages[i-1]
                    call_names = [p.function_call.name for p in prev_content.parts if p.function_call]
                    resp_idx = 0
                    for p in content.parts:
                        if p.function_response and resp_idx < len(call_names):
                            p.function_response.name = call_names[resp_idx]
                            resp_idx += 1

        return gemini_messages

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> Message:
        contents = self._convert_messages(messages)
        
        # Prepare tools
        genai_tools = []
        if tools:
            decls = []
            for t in tools:
                decls.append(types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t["parameters"]
                ))
            genai_tools.append(types.Tool(function_declarations=decls))

        # We disable automatic function calling to keep the "Action" step in our ReAct loop
        config = types.GenerateContentConfig(
            tools=genai_tools if genai_tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )

        # Parse output
        result_content = ""
        result_tool_calls = []
        
        # Check if we have candidates
        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if part.text:
                    result_content += part.text
                if part.function_call:
                    result_tool_calls.append(ToolCall(
                        name=part.function_call.name,
                        args=dict(part.function_call.args),
                        id=None
                    ))

        return Message(
            role=Role.ASSISTNT,
            content=result_content,
            tool_calls=result_tool_calls if result_tool_calls else None
        )

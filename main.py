from agent_pure.tools import tool
from agent_pure.llm import LLM
from agent_pure.agent import Agent

# Define some tools using our decorator
@tool()
def calculator(a: float, b: float, op: str) -> float:
    """Perform a simple mathematical operation."""
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": 
        if b == 0: return 0.0 # avoid div by zero
        return a / b
    return 0.0

@tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Mock weather data
    weather_data = {
        "New York": "Cloudy, 15°C",
        "London": "Rainy, 10°C",
        "Paris": "Sunny, 20°C",
        "Tokyo": "Windy, 18°C"
    }
    return weather_data.get(city, "Weather data not available for this city.")

def main():
    # Initialize the LLM (ensure GOOGLE_API_KEY is in .env)
    llm = LLM(model_name="gemini-1.5-flash-8b")
    
    # Initialize the Agent
    agent = Agent(llm, system_prompt="You are a smart assistant that can use tools to answer questions. Always think step-by-step.")
    
    # Run the agent with a multi-step task
    query = "What is the weather in Paris, and what is 20 multiplied by 5?"
    result = agent.run(query)
    
    print("\n[Final Answer]:", result)

if __name__ == "__main__":
    main()

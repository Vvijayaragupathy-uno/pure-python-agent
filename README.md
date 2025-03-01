# 🤖 Pure Python Agent - ReAct


## 🧠 Core Engineering

- **The Orchestrator (ReAct Loop)**: A decoupled Reasoning-Acting loop that manages state without hidden magic.
- **Dynamic Tooling**: Leverages Python's `inspect` module for automated schema generation directly from function signatures—mapping professional Python type-hints to Gemini's tool specification.
- **Type Safety Shield**: An integrated validation layer that casts LLM-generated JSON into strictly typed Python primitives, mitigating the common "string-injection" errors seen in early-stage agent development.

## 🛠 Modular Structure

- `agent_pure/`: A self-contained package housing the Schema, Tools, LLM Wrapper, and Agent logic.
- `main.py`: The entry point for testing the framework's multi-step reasoning capabilities.

## 🧪 Technical Reflection

This project reinforces a core engineering principle: to truly master a technology, you must understand the primitives it is built upon. By building this "from scratch,"

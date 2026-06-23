from llm.gemini import Gemini
from tools.tools import run_tool


def run_agent(message):
    react_prompt = ""
    with open("prompts/react_prompt.txt") as f:
        react_prompt = f.read()
    messages = [
        {"role": "system", "content": react_prompt},
        {"role": "user", "content": message},
    ]
    while True:
        response = Gemini.generate_structured(messages)
        messages.append({"role": "assistant", "content": response.content})
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            result = run_tool(tool_call)
            messages.append({"role": "tool", "content": result})

        content = response.content

        if content and "Final Answer:" in content:
            return content
        else:
            print(content)

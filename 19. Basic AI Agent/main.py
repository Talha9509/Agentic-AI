import os
from groq import Groq
from dotenv import load_dotenv
from tavily import TavilyClient
import json

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> str:
    """Search the web for current information."""
    try:
        response = tavily.search(
            query=query,
            max_results=3,
            search_depth="basic"
        )
        # print(response)
        
        results = []
        
        for result in response["results"]:
            results.append(
                f"Title: {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Content: {result['content']}\n"
            )
        return "\n\n".join(results)
    
    except Exception as e:
        return f"search failed: {e}" 
    
# web_search("who is ronaldo?")

def calculate(expression: str) -> str:
    """Perform mathematical calculation"""
    try: 
        return eval(expression)
    except Exception as e:
        return f"Calculation failed: {e}"
    
    
AVAILABLE_TOOLS = {
    "web_search": web_search,
    "calculate": calculate,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, "
                "news, recent events, prices, statistics, "
                "or facts that may have changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A specific and concise "
                            "web search query."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform mathematical calculations "
                "such as percentages, multiplication, "
                "division, or arithmetic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A mathematical expression "
                            "such as '2500 * 0.15'."
                        )
                    }
                }
            },
            "required": ["expression"]
        }
    }
]


SYSTEM_PROMPT = """
You are a research assistant.

Rules:

1. Use web_search for current or changing information.
2. Use calculate for mathematical calculations.
3. You may use tools multiple times.
4. When you have enough information, answer the user.
5. Include useful source URLs from web search results.
"""

def run_agent(question, max_iterations = 10):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }, 
        {
            "role": "user",
            "content": question
        }
    ]
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        print(response.choices[0].message)
        messages.append(message)
        
        if not message.tool_calls:
            print("\nFINAL ANSWER:")
            print(message.content)

            return message.content
        
        print(f"LLM requested {len(message.tool_calls)} tool(s)")
        
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"Tool: {tool_name}")
            print(f"Arguments: {arguments}")

            if tool_name not in AVAILABLE_TOOLS:
                result = (f"Unknown tool: {tool_name}")
            else:
                result = AVAILABLE_TOOLS[tool_name](**arguments)
                print("Tool result received.")
            print("result")
            print(type(result))
            print(result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    print("\nMaximum iterations reached.")

    return (
        "I couldn't complete the task "
        "within the iteration limit."
    )

                

question = input("Ask me anything: ")
run_agent(question, 5)
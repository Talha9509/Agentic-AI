from typing import TypedDict, Literal
import random
import os
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, START, END

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
groq = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"

Menu = {
    "Chicken Biryani": 2,
    "Pasta": 4
}

class State(TypedDict):
    dish: str
    quantity: int
    status: Literal['orderfail', 'confirm', 'not available', 'partial', 'ready', 'cookfail', 'complete', 'servefail']
    user_message: str
    order_attempts: int
    cook_attempts: int
    serve_attempts: int
    result: Literal['complete', 'incomplete']
    llm_response: str

def order_confirm(state: State) -> dict:
    print("confirming order")
    dish = state.get("dish", "")
    quantity = state.get("quantity", 0)
    attempts = state.get("order_attempts", 3)

    if attempts <= 0:
        return {"status": "orderfail"}

    available = Menu.get(dish, 0)
    if available == 0:
        return {"status": "not available"}
    if available < quantity:
        return {"status": "partial"}

    Menu[dish] = available - quantity
    return {"status": "confirm", "order_attempts": attempts - 1}

def cook(state: State) -> dict:
    print("cooking order")
    attempts = state.get("cook_attempts", 2)
    if attempts <= 0:
        return {"status": "cookfail"}

    if random.randint(1, 10) < 5:
        return {"status": "cookfail", "cook_attempts": attempts - 1}

    return {"status": "ready", "cook_attempts": attempts - 1}

def serve(state: State) -> dict:
    print("serving order")
    attempts = state.get("serve_attempts", 2)
    if attempts <= 0:
        return {"status": "servefail"}

    if random.randint(1, 10) < 5:
        return {"status": "servefail", "serve_attempts": attempts - 1}

    return {"status": "complete", "result": "complete", "serve_attempts": attempts - 1}

def llm_apologize_and_refund(state: State) -> dict:
    status = state.get("status", "")
    dish = state.get("dish", "")
    qty = state.get("quantity", 0)

    if status in ["complete"]:
        prompt = f"The order for {qty} x {dish} was cooked and served successfully. Thank the customer."
    else:
        prompt = f"The order for {qty} x {dish} failed at stage status '{status}'. Apologize politely and state that a full refund has been issued."

    response = groq.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a customer service assistant for a restaurant."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content
    print(f"\n--- LLM Response ---\n{content}\n")
    return {"llm_response": content}

# Conditional Routing Functions
def route_order(state: State) -> Literal["cook", "llm_apologize_and_refund"]:
    if state.get("status") == "confirm":
        return "cook"
    if state.get("status") == "partial":
        return "order_confirm"
    return "llm_apologize_and_refund"

def route_cook(state: State) -> Literal["serve", "cook", "llm_apologize_and_refund"]:
    if state.get("status") == "ready":
        return "serve"
    if state.get("cook_attempts", 0) > 0:
        return "cook"
    return "llm_apologize_and_refund"

def route_serve(state: State) -> Literal["llm_apologize_and_refund", "serve"]:
    if state.get("status") == "complete":
        return "llm_apologize_and_refund"
    if state.get("serve_attempts", 0) > 0:
        return "serve"
    return "llm_apologize_and_refund"

builder = StateGraph(State)

builder.add_node("order_confirm", order_confirm)
builder.add_node("cook", cook)
builder.add_node("serve", serve)
builder.add_node("llm_apologize_and_refund", llm_apologize_and_refund)

builder.add_edge(START, "order_confirm")

builder.add_conditional_edges("order_confirm", route_order)
builder.add_conditional_edges("cook", route_cook)
builder.add_conditional_edges("serve", route_serve)

builder.add_edge("llm_apologize_and_refund", END)

graph = builder.compile()

if __name__ == "__main__":
    initial_state = {
        "dish": "Chicken Biryani",
        "quantity": 10,
        "user_message": "I'd like 1 Chicken Biryani please.",
        "order_attempts": 3,
        "cook_attempts": 2,
        "serve_attempts": 2,
        "status": "not available",
        "result": "incomplete"
    }
    
    final_state = graph.invoke(initial_state)
    print("Final State:", final_state)
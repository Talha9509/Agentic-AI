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
    print("\n[Node: order_confirm]")
    dish = state.get("dish", "")
    quantity = state.get("quantity", 0)
    attempts = state.get("order_attempts", 3)

    if attempts <= 0:
        return {"status": "orderfail"}

    available = Menu.get(dish, 0)
    if available == 0:
        return {"status": "not available"}

    if available < quantity:
        print(f"Partial availability: Requested {quantity}, only {available} left. Remaining attempts: {attempts - 1}")
        return {
            "status": "partial",
            "order_attempts": attempts - 1
        }

    Menu[dish] = available - quantity
    return {
        "status": "confirm",
        "order_attempts": attempts - 1
    }

def ask_llm(state: State) -> dict:
    print("\n[Node: ask_llm]")
    dish = state.get("dish", "")
    quantity = state.get("quantity", 0)
    available = Menu.get(dish, 0)

    prompt = f"The customer ordered {quantity} x {dish}, but only {available} are available. Generate a message informing the user of the available quantity."
    
    response = groq.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a polite restaurant assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content
    print(f"LLM Response to User:\n{content}")

    # Adjust requested quantity to available stock for re-evaluation
    return {
        "user_message": content,
        "quantity": available
    }

def cook(state: State) -> dict:
    print("\n[Node: cook]")
    attempts = state.get("cook_attempts", 2)
    if attempts <= 0:
        return {"status": "cookfail"}

    if random.randint(1, 10) < 5:
        return {"status": "cookfail", "cook_attempts": attempts - 1}

    return {"status": "ready", "cook_attempts": attempts - 1}

def serve(state: State) -> dict:
    print("\n[Node: serve]")
    attempts = state.get("serve_attempts", 2)
    if attempts <= 0:
        return {"status": "servefail"}

    if random.randint(1, 10) < 5:
        return {"status": "servefail", "serve_attempts": attempts - 1}

    return {"status": "complete", "result": "complete", "serve_attempts": attempts - 1}

def llm_apologize_and_refund(state: State) -> dict:
    print("\n[Node: llm_apologize_and_refund]")
    status = state.get("status", "")
    dish = state.get("dish", "")
    qty = state.get("quantity", 0)

    if status == "complete":
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
    print(f"Final Message:\n{content}\n")
    return {"llm_response": content}

# Conditional Routing Functions
def route_order(state: State) -> Literal["cook", "ask_llm", "llm_apologize_and_refund"]:
    status = state.get("status")
    attempts = state.get("order_attempts", 0)

    if status == "confirm":
        return "cook"
    if status == "partial" and attempts > 0:
        return "ask_llm"
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
builder.add_node("ask_llm", ask_llm)
builder.add_node("cook", cook)
builder.add_node("serve", serve)
builder.add_node("llm_apologize_and_refund", llm_apologize_and_refund)

builder.add_edge(START, "order_confirm")

builder.add_conditional_edges("order_confirm", route_order)
builder.add_edge("ask_llm", "order_confirm")
builder.add_conditional_edges("cook", route_cook)
builder.add_conditional_edges("serve", route_serve)

builder.add_edge("llm_apologize_and_refund", END)

graph = builder.compile()

if __name__ == "__main__":
    initial_state = {
        "dish": "Chicken Biryani",
        "quantity": 10,
        "user_message": "I'd like 10 Chicken Biryani please.",
        "order_attempts": 3,
        "cook_attempts": 2,
        "serve_attempts": 2,
        "status": "not available",
        "result": "incomplete"
    }

    final_state = graph.invoke(initial_state)
    print("Final State:", final_state)
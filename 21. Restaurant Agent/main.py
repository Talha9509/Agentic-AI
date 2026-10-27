from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Literal
from groq import Groq
from dotenv import load_dotenv
import os
import random

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
groq = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"

class State(TypedDict):
    dish: str
    quantity: int
    # available: str
    status: Literal['orderfail', 'confirm', 'not available', 'partial', 'ready', 'cookfail', 'complete', 'servefail']
    user_message: str
    order_attempts: int
    cook_attempts: int
    serve_attempts: int
    result: Literal['complete', 'incomplete']
    llm_response: str

Menu = {
    "Chicken Biryani": 2,
    "Pasta": 4
}

def order_confirm(state: State, Menu) -> dict:
    order_attempts = state['order_attempts']
    if(order_attempts <= 0):
        return { "status": "orderfail" }
    if(not Menu[state['dish']] or Menu[state['dish']] == 0):
        # not available
        return { "status": "not available" }
    if(Menu.get(state['dish']) < state['quantity']):
        # partially available
        return { "status": "partial" }
    
    order_attempts = order_attempts - 1
    available = Menu.get(state['dish'])
    deducted_avail = available - state['quantity']
    Menu[state['dish']] = deducted_avail
        
    # if amount and dish available, then confirm
    return { "status": "confirm", "order_attempts": order_attempts }
    
def cook(state: State):
    cook_attempts = state['cook_attempts']
    if(cook_attempts <= 0):
        return { "status": "cookfail" }
    randomno = random.randint(1,10)
    if(randomno < 5):
        return { "status": "cookfail" }
    
    cook_attempts = cook_attempts - 1
    
    return { "status": "ready", "cook_attempts": cook_attempts }
    
def serve(state: State):
    serve_attempts = state['serve_attempts']
    if(serve_attempts <= 0):
        return { "status": "servefail" }
    randomno = random.randint(1,10)
    if(randomno < 5):
        return { "status": "servefail" }
    
    serve_attempts = serve_attempts - 1
    
    return { "status": "complete", "result": "complete", "serve_attempts": serve_attempts }
    

def ask_llm(query, state: State):
    system_message = {
        "role": "system",
        "content": ""
    }
    message = {
        "role": "user",
        "content": state['user_message']
    }
    messages = [message]

    response = groq.chat.completions.create(model=model, messages=messages)
    print(response.choices[0].message.content)

builder = StateGraph(State)
builder.add_node("order_confirm", order_confirm)
builder.add_node("cook", cook)
builder.add_node("serve", serve)

builder.set_entry_point("order_confirm")

builder.add_edge(START, "order_confirm")
builder.add_edge("order_confirm", "cook")
builder.add_edge("cook", "serve")
builder.add_edge("serve", END)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({
    "dish": "Biryani",
    "quantity": 1,
    # status: Literal['orderfail', 'confirm', 'not available', 'partial', 'ready', 'cookfail', 'complete', 'servefail']
    "user_message": ""
    # result: Literal['complete', 'incomplete']
    })
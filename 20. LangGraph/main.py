from typing import TypedDict
from langgraph.graph import END, StateGraph

class State(TypedDict):
    num: int
    
def double(state: State) -> dict:
    boardnum = state['num']
    newnum = 2 * boardnum
    print("in double")
    print(newnum)
    return { "num": newnum }

def finish(state: State) -> dict:
    boardnum = state['num']
    print("finish")
    print(boardnum)
    return { "num": boardnum }

def decision(state: State) -> str:
    if state['num'] < 100:
        return "double"
    else:
        return "finish"
    
builder = StateGraph(State)
builder.add_node("double", double)
builder.add_node("finish", finish)

builder.set_entry_point("double")

builder.add_conditional_edges(
    "double",
    decision,
    { "double": "double", "finish": "finish" }
)

builder.add_edge("finish", END)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"num": 5})
    

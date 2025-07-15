from typing import Annotated,TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph.message import AnyMessage
from langgraph.graph import START,END,StateGraph
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

def chat_node(state:State)->State:
    model = ChatOpenAI(model_name="gpt-4o",temperature=0)
    state["messages"] = model.invoke(state["messages"])
    return state
    
graph_builder=StateGraph(State)
graph_builder.add_node("chat_node",chat_node)
graph_builder.add_edge(START,"chat_node")
graph_builder.add_edge("chat_node",END)
graph = graph_builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    config={"configurable":{"thread_id":"test"}}
    response = graph.invoke({"messages":[{"role":"user","content":"Hello, how are you?"}]},config=config)
    print(response)






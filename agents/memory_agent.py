import os
from dotenv import load_dotenv
from typing import Annotated, Literal, Union
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

class StateAgent(TypedDict):
    messages: list[Union[HumanMessage, AIMessage]]


llm = init_chat_model(model="anthropic:claude-3-5-sonnet-latest")

def process(state: StateAgent) -> StateAgent:
     response = llm.invoke(state["messages"])

     state["messages"].append(AIMessage(content=response.content))

     print(f"\nAI: {response.content}")
     #print("current state:", state["messages"])

     return state

graph = StateGraph(StateAgent)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

conversation_history = []

user_input = input("Enter:")

while user_input != "exit":
    conversation_history.append(HumanMessage(content = user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter:")

with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")

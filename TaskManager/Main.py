from openai import OpenAI
import streamlit as st
import json
import CommonUtils
from UIManager import DisplayCurrentTasks, DisplayTaskManager
from datetime import datetime

client = OpenAI(api_key=st.secrets["OPEN_AI_API_KEY"])
if __name__ == "__main__":
    st.title("Task Manager :u7533:")

    DisplayTaskManager()

if "messages" not in st.session_state:
    st.session_state.messages = []


def handle_tool(tool_call):
    fn_name = tool_call.name
    args = json.loads(tool_call.arguments)
    print("Function Name: ", fn_name, " Time: ", datetime.now())
    result = CommonUtils.function_map[fn_name](**args)

    # append model's function call message
    st.session_state.messages.append(tool_call)
    # append result message
    st.session_state.messages.append({  
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(result)
    })

def call_llm(user_input):
    st.session_state.messages.append({"role": "system", "content": "You are a task planner that decomposes user requests into multiple function calls."})
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = client.responses.create(
        model="gpt-4-1106-preview",
        input=st.session_state.messages,
        tools=CommonUtils.function_schemas,
        tool_choice="auto"
    )
    
    print("Response: ", response.output)
    tool_calls=[]
    for resp in response.output:
        if resp.type != "function_call":
            continue
        tool_calls.append(resp)
    
    for tool_call in tool_calls:
        handle_tool(tool_call)

    response_2 = client.responses.create(
            model="gpt-4-1106-preview",
            input=st.session_state.messages,
            tools=CommonUtils.function_schemas,
        )
    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_2.output_text
        }
    )

    print("Response2", response_2.output_text)

    DisplayCurrentTasks(st.session_state.tasks)
    for message in st.session_state.messages:
        # Handle standard message types (user/assistant)
        if isinstance(message, dict) and "role" in message:
            if message["role"] != "system":    
                with st.chat_message(message["role"]):
                    st.markdown(message['content'])
            # Log any other message formats
            else:
                print(f"Unhandled message format: {type(message)}")
from openai import OpenAI
import streamlit as st
import asyncio
import json
import CommonUtils
from UIManager import DisplayCurrentTasks, DisplayTaskManager

client = OpenAI(api_key=st.secrets["OPEN_AI_API_KEY"])
if __name__ == "__main__":
    st.title("Task Manager :u7533:")

    DisplayTaskManager()

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

async def call_llm(user_input):
    st.session_state.messages.append({"role": "system", "content": "You are a task planner that decomposes user requests into multiple function calls."})
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    response = client.responses.create(
        model="gpt-4.1",
        input=st.session_state.messages,
        tools=CommonUtils.function_schemas,
        tool_choice="auto"
    )
    

    tool_calls = response.output or []
    print(response.output)
    # Parallel execution of tool calls
    async def handle_tool(tool_call):
        fn_name = tool_call.name
        args = json.loads(tool_call.arguments)
        print("Args: ", args)
        CommonUtils.function_map[fn_name](**args)

    await asyncio.gather(*[handle_tool(tc) for tc in tool_calls])
    DisplayCurrentTasks(st.session_state.tasks)
    print("Tasks: ", st.session_state.tasks)
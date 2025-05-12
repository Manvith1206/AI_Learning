import streamlit as st
from openai import OpenAI
import json
import asyncio

client = OpenAI(api_key=st.secrets["OPEN_AI_API_KEY"])  

# -----------------------------
# Streamlit UI State
# -----------------------------
if "tasks" not in st.session_state:
    st.session_state.tasks = []

# -----------------------------
# Function Definitions (Tools)
# -----------------------------
functions = [
    {
        "type": "function",
        "name": "add_task",
        "description": "Add a new task to the todo list",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task description"},
            },
            "required": ["task"]
        },
    },
    {
        "type": "function",
        "name": "complete_task",
        "description": "Mark a task as completed",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task to mark as done"},
            },
            "required": ["task"]
        },
    },
    {
        "type": "function",
        "name": "add_subtasks",
        "description": "Add subtasks to an existing task",
        "parameters": {
            "type": "object",
            "properties": {
                "parent_task": {"type": "string"},
                "subtasks": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["parent_task", "subtasks"]
        }
    },
    {
        "type": "function",
        "name": "tag_tasks",
        "description": "Tag tasks with categories",
        "parameters": {
            "type": "object",
            "properties": {
                "tasks_tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "tag": {"type": "string"}
                        },
                        "required": ["task", "tag"]
                    }
                }
            },
            "required": ["tasks_tags"]
        }
    }
]

# -----------------------------
# Function Logic
# -----------------------------
def add_task(task):
    st.session_state.tasks.append({"task": task, "done": False, "subtasks": [], "tags": []})

def complete_task(task):
    for t in st.session_state.tasks:
        if t["task"] == task:
            t["done"] = True
            break

def add_subtasks(parent_task, subtasks):
    for t in st.session_state.tasks:
        if t["task"] == parent_task:
            t["subtasks"].extend(subtasks)
            break

def tag_tasks(tasks_tags):
    for item in tasks_tags:
        for t in st.session_state.tasks:
            if t["task"] == item["task"]:
                t["tags"].append(item["tag"])

function_map = {
    "add_task": add_task,
    "complete_task": complete_task,
    "add_subtasks": add_subtasks,
    "tag_tasks": tag_tasks
}

# -----------------------------
# LLM Planner (Multi / Nested / Compositional)
# -----------------------------
async def call_llm(user_input):
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "You are a task planner that decomposes user requests into multiple function calls."},
            {"role": "user", "content": user_input}
        ],
        tools=functions,
        tool_choice="auto"
    )

    tool_calls = response.output or []
    print(tool_calls)
    # Parallel execution of tool calls
    async def handle_tool(tool_call):
        fn_name = tool_call.name
        args = json.loads(tool_call.arguments)
        function_map[fn_name](**args)

    await asyncio.gather(*[handle_tool(tc) for tc in tool_calls])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("LLM-Driven TODO App")
user_input = st.text_input("What would you like to do?")
if st.button("Submit"):
    asyncio.run(call_llm(user_input))

# -----------------------------
# Display Tasks
# -----------------------------
st.subheader("Current Tasks")
for task in st.session_state.tasks:
    st.write(f"- {task['task']} ({'✅' if task['done'] else '❌'})")
    if task['subtasks']:
        st.write("  Subtasks:")
        for sub in task['subtasks']:
            st.write(f"    - {sub}")
    if task['tags']:
        st.write("  Tags:")
        st.write(", ".join(task['tags']))

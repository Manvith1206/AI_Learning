import streamlit as st
import asyncio

def DisplayTaskManager():
    """
    Display the task manager interface.
    """
    user_input = st.text_input("What do you want to do today? :smile:", key="task_input_1")
    if st.button("Execute Query", key="execute_query"):
        print("Button Pressed")
        from Main import call_llm
        print("User Input: ", user_input)
        asyncio.run(call_llm(user_input))

def DisplayCurrentTasks(tasks):
    """
    Display the current tasks in the task manager.
    Args:
        tasks (list): List of tasks to display.
        Object Structure:
        Task = {
            "task": "Task description",
            "status": False,
            "subtasks": [],
            "tags": []
        }
    """
    st.subheader("Current Tasks")
    print("Current Tasks:", tasks)
    task_count = 0
    for task in tasks:
        task_count += 1
        with st.expander(f"🗂Task {task_count}"):
            tab1, tab2, tab3 = st.tabs(["Task Info", "Subtasks", "Tags"])
            print("Task Info: ", task)
            with tab1:
                task_name = task.get("task", "")
                st.write(f"Task: {task_name}")
                task_status = task.get("status") or False
                if task_status is None:
                    task_status=False
                if task_status == True:
                    task_status_display = "Done ✅"
                else:
                    task_status_display = "Pending ❌"

                st.write(f"Status: {task_status_display}")
            with tab2:
                subtasks = task.get("subtasks") or []
                if subtasks is None:
                    subtasks = []
                for subtask in subtasks:
                    st.write(f"- {subtask}")               
            with tab3:
                tags = task.get("tags") or []
                for tag in tags:
                    st.write(f"- {tag}")
        

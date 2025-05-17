import streamlit as st

def DisplayTaskManager():
    """
    Display the task manager interface.
    """
    user_input = st.text_input("What do you want to do today? :smile:", key="task_input_1")
    if st.button("Execute Query", key="execute_query"):
        from Main_updated import call_llm
        call_llm(user_input)

        
    # --- Usage Examples ---
    with st.expander("💡 Examples"):
        st.markdown("""
        - Add a task to buy groceries with high priority.
        - Delete task 1
        - Mark task 2 as complete
        - Add subtasks to task 1: Buy milk, Buy eggs
        - Tag tasks 1 and 2 with urgent and personal
        """)


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
    task_count = 0
    for task in tasks:
        task_count += 1
        task_id = task.task_id or task_count

        with st.expander(f"Task {task_id}"):
            tab1, tab2, tab3 = st.tabs(["Task Info", "Subtasks", "Tags"])
            with tab1:
                task_name = task.task
                st.write(f"Task: {task_name}")
                task_status = task.status or False
                if task_status is None:
                    task_status=False
                if task_status == True:
                    task_status_display = "Done ✅"
                else:
                    task_status_display = "Pending ❌"

                st.write(f"Status: {task_status_display}")
            with tab2:
                subtasks = task.subtasks or []
                if subtasks is None:
                    subtasks = []
                for subtask in subtasks:
                    st.write(f"- {subtask}")               
            with tab3:
                tags = task.tags or []
                for tag in tags:
                    st.write(f"- {tag}")
        

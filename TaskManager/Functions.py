from dataclasses import dataclass

@dataclass
class Task:
    task = "",
    status = False,
    subtasks = [],
    tags=[],
    task_id = 0
    def __init__(self, task: str, status: bool = False, subtasks: list = [], tags: list = [], task_id: int = 0):
        self.task = task
        self.status = status
        self.subtasks = subtasks
        self.tags = tags
        self.task_id = task_id
@dataclass
class Tags:
    def __init__(self, task: str, tag: str, task_id: int):
        self.task = task
        self.tag = tag
        self.task_id = task_id
    tag = ""
    task = ""
    task_id = 0

import streamlit as st

if "tasks" not in st.session_state:
    st.session_state.tasks = []

def add_task(task_name: str):
    """
    Add a new task to the todo list.
    """
    task = Task(task_name, status=False, subtasks=[], tags=[])
    st.session_state.tasks.append(task)
    add_task_id_to_task()

    context_message = form_context_for_current_tasks()
    return context_message

def form_context_for_current_tasks():
    current_tasks = ["Here are the current tasks:\n"]

    for task in st.session_state.tasks:
        current_tasks.append(f"ID: {task.task_id}, Title: {task.task}, Status: {'✅ Completed' if task.status else '❌ Not Completed'}\n")

        if task.subtasks:
            current_tasks.append("  Subtasks:\n")
            for subtask in task.subtasks:
                current_tasks.append(f"    - {subtask}")
        else:
            current_tasks.append("  Subtasks: None")

        if task.tags:
            current_tasks.append("  Tags: " + ", ".join(task.tags))
        else:
            current_tasks.append("  Tags: None")

        current_tasks.append("\n\n")  # Empty line for spacing between tasks

    # Convert list to a single message string
    context_message = "\n".join(current_tasks)

    return context_message

def add_task_with_subtasks_and_tags(task_name: str, subtasks: list, tags: list):
    """
    Add a new task with subtasks and tags. This can be used to add task with only subtasks or add tasks with only tags or with subtasks and tags
    """
    task = Task(task_name, status=False, subtasks=subtasks, tags=tags)
    st.session_state.tasks.append(task)
    add_task_id_to_task()

    context_message = form_context_for_current_tasks()
    return context_message + "\n\nTask with subtasks and tags added successfully!"

def complete_task(task_name: str):
    """
    Mark a task as completed.
    """
    for currTask in st.session_state.tasks:
        if currTask.task.strip().lower() == task_name.strip().lower():
            currTask.status = True
            break

    return "Task marked as completed!"

def add_subtasks(parent_task, subtasks):
    """
    Add subtasks to an existing task.
    """
    for task in st.session_state.tasks:
        if task.task.strip().lower() == parent_task.strip().lower():
            task.subtasks.extend(subtasks)
            break

    context_msg = form_context_for_current_tasks()
    return context_msg + "\n\nSubtasks added successfully For Specified Task!"

def tag_tasks(tasks_tags: list[Tags]):
    """
    Tag tasks with categories. Can identify tasks by either task_id or task name.
    """
    for item in tasks_tags:
        for t in st.session_state.tasks:
            # Check if we have a task_id match
            if 'task_id' in item and item['task_id'] and t.task_id == item['task_id']:
                if item['tag'] not in t.tags:  # Avoid duplicate tags
                    t.tags.append(item['tag'])
                break
            # If no task_id or no match by ID, try matching by task name
            elif 'task' in item and item['task'] and t.task.strip().lower() == item['task'].strip().lower():
                if item['tag'] not in t.tags:  # Avoid duplicate tags
                    t.tags.append(item['tag'])
                break

    context_msg = form_context_for_current_tasks()

    return context_msg + "\n\nTags added successfully for specified task!"

def delete_task(task: Task):
    """
    Delete a task from the todo list.
    """
    
    # Find the index of the task to delete
    index_to_delete = None
    for i, curr_task in enumerate(st.session_state.tasks):
        if curr_task.task.strip().lower() == task['task'].strip().lower():
            index_to_delete = i
            break
            
    # Delete by index if found
    if index_to_delete is not None:
        del st.session_state.tasks[index_to_delete]
        
    
    # Print remaining tasks
    for i, curr_task in enumerate(st.session_state.tasks):
        print(f"Task {i}: {curr_task.task}")
        
    context_msg = form_context_for_current_tasks()
    return context_msg + "\n\nTask deleted successfully!"

def get_tasks_based_on_tag(tag: str):
    """
    Get tasks based on a specific tag.
    """
    tagged_tasks = []
    for task in st.session_state.tasks:
        if tag in task.tags:
            tagged_tasks.append(task)
    return tagged_tasks
def add_task_id_to_task():
    """
    Add task ID to all tasks.
    """
    for i, task in enumerate(st.session_state.tasks):
        st.session_state.tasks[i] = Task(
            task=task.task,
            status=task.status,
            subtasks=task.subtasks,
            tags=task.tags,
            task_id=i+1)
    return "Task ID Added to all Tasks!"
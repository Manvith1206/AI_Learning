class Task:
    task = "",
    status = False,
    subtasks = [],
    tags=[]
    def __init__(self, task: str, status: bool = False, subtasks: list = [], tags: list = []):
        self.task = task
        self.status = status
        self.subtasks = subtasks
        self.tags = tags
class Tags:
    def __init__(self, task: str, tag: str):
        self.task = task
        self.tag = tag
    tag = ""
    task = ""

import streamlit as st
import inspect, sys

if "tasks" not in st.session_state:
    st.session_state.tasks = []

def add_task(task: Task):
    """
    Add a new task to the todo list.
    """
    print("Adding task: ", task)
    st.session_state.tasks.append(task)

def complete_task(task: str):
    """
    Mark a task as completed.
    """
    print("Completing task: ", task)
    for currTask in st.session_state.tasks:
        if currTask['task'].strip().lower() == task.strip().lower():
            print("Task found: ", currTask)
            currTask['status'] = True
            break

def add_subtasks(parent_task, subtasks):
    """
    Add subtasks to an existing task.
    """
    print("Adding subtasks: ", st.session_state.tasks)
    for task in st.session_state.tasks:
        if task['task'].strip().lower() == parent_task.strip().lower():
            task.setdefault('subtasks', [])
            print("Add Subtask: ", task)
            task['subtasks'].extend(subtasks)
            break

def tag_tasks(tasks_tags: list[Tags]):
    """
    Tag tasks with categories.
    """
    for item in tasks_tags:
        for t in st.session_state.tasks:
            if t['task'] == item['task']:
                t.setdefault('tags', [])
                t['tags'].append(item['tag'])

def delete_task(task: Task):
    """
    Delete a task from the todo list.
    """
    print("Deleting task: ", task)
    for curr_task in st.session_state.tasks:
        if curr_task['task'].strip().lower() == task.strip().lower():
            st.session_state.tasks.remove(curr_task)
            break
        
def get_tasks_based_on_tag(tag: str):
    """
    Get tasks based on a specific tag.
    """
    print("Getting tasks based on tag: ", tag)
    tagged_tasks = []
    for task in st.session_state.tasks:
        if tag in task.get('tags', []):
            tagged_tasks.append(task)
    return tagged_tasks
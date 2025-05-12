import inspect


def GetFunctionSchema(functionName):
    signature = inspect.signature(functionName)
    for sig in signature.parameters.values():
        print(sig)

    return {
        "type": "function",
        "name": f"{functionName.__name__}",
        "description": f"{functionName.__doc__}",
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

function_schemas = [
     {
        "type": "function",
        "name": "add_task",
        "description": "Add a new task to the todo list",
        "parameters": 
        {
            "type": "object",
            "properties":
            {
                "task":
                {
                    "type": "object",
                    "properties":
                    {
                        "task": {"type": "string", "description": "The task description"},
                        "status": {"type": "boolean", "description": "The task status"},
                        "subtasks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of subtasks for the main task, if not provided, it will be an empty list"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "description": "The task to add when creating a new task, status will be False by default, if tags or subtasks is not provided add subtasks and tags as empty list",
                },
            },
            "required": ["task"]
        }
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
    },
    #  {
    #     "type": "function",
    #     "name": "get_tasks_based_on_tag",
    #     "description": "Get tasks based on tag",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "tag": {"type": "string", "description": "The tag to filter tasks"},
    #         },
    #         "required": ["tasks_tags"]
    #     }
    # }
]

import Functions

function_map = {
    "add_task": Functions.add_task,
    "complete_task": Functions.complete_task,
    "add_subtasks": Functions.add_subtasks,
    "tag_tasks": Functions.tag_tasks,
}
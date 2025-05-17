import inspect
add_task_properties = { "task_name": {"type": "string", "description": "The name of the task to add"}}
add_task_with_subtasks_and_tags_properties = {"task_name": {"type": "string", "description": "The name of the task to add"},"subtasks": {"type": "array","items": {"type": "string"},"description": "List of subtasks for the main task, if not provided, it will be an empty list"},"tags": {"type": "array","items": {"type": "string"},"description": "List of tags for the main task, if not provided, it will be an empty list"}}
complete_task_properties = {
                    "task_name": {"type": "number", "description": "The task Id to mark as done"},
                }
add_subtasks_properties = {
                "parent_task": {"type": "string"},
                "subtasks": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
tag_tasks_properties = {
                "tasks_tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "tag": {"type": "string"},
                            "task_id": {"type": "number"}
                        },
                        "required": ["task", "tag", "task_id"]
                    }
                }
            }
delete_task_properties = {
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
                            },
                            "task_id": {"type": "integer", "description": "The task ID"}
                        },
                        "description": "The task object which needs to be deleted, task name, task id, status and tags",
                    },

            }
get_tasks_based_on_tag_properties = {
                "tag": {"type": "string", "description": "The tag to filter tasks"},
            }

function_properties_map = {
    "add_task": add_task_properties,
    "complete_task": complete_task_properties,
    "add_subtasks": add_subtasks_properties,
    "tag_tasks": tag_tasks_properties,
    "get_tasks_based_on_tag": get_tasks_based_on_tag_properties,
    "delete_task": delete_task_properties,
    "add_task_with_subtasks_and_tags": add_task_with_subtasks_and_tags_properties
}

def GetFunctionSchemaForOpenAI(functionName):
    signature = inspect.signature(functionName)
    params=[]
    for sig in signature.parameters.values():
        params.append(sig.name)

    return {
        "type": "function",
        "name": f"{functionName.__name__}",
        "description": f"{functionName.__doc__}",
        "parameters": {
            "type": "object",
            "properties": function_properties_map[functionName.__name__],
            "required": params
        }
    }


def GetFunctionSchemaForAnthropic(functionName):
    signature = inspect.signature(functionName)
    params=[]
    for sig in signature.parameters.values():
        params.append(sig.name)

    return {
        "name": f"{functionName.__name__}",
        "description": f"{functionName.__doc__}",
        "input_schema": {
            "type": "object",
            "properties": function_properties_map[functionName.__name__],
            "required": params
        }
    }

function_schemas=[]
# function_schemas = [
#      {
#         "type": "function",
#         "name": "add_task",
#         "description": "Add a new task to the todo list, it do not add subtasks and tags to the task, if you want to add subtasks and tags use add_subtasks and tag_tasks function",
#         "parameters": 
#         {
#             "type": "object",
#             "properties":
#             {
#                 "task_name": {"type": "string", "description": "The name of the task to add"},
#             },
#             "required": ["task_name"]
#         }
#     },
#     {
#         "type": "function",
#         "name": "add_task_with_subtasks_and_tags",
#         "description": "Add a new task with subtasks and tags",
#         "parameters": 
#         {
#             "type": "object",
#             "properties":
#             {
#                 "task_name": {"type": "string", "description": "The name of the task to add"},
#                 "subtasks": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": "List of subtasks for the main task, if not provided, it will be an empty list"
#                 },
#                 "tags": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": "List of tags for the main task, if not provided, it will be an empty list"
#                 }
#             },
#             "required": ["task_name", "subtasks", "tags"]
#         }
#     },
#     {
#         "type": "function",
#         "name": "complete_task",
#         "description": "Mark a task as completed",
#        "parameters": {
#             "type": "object",
#             "properties": {
#                     "task_id": {"type": "number", "description": "The task Id to mark as done"},
#                 },
#                 "required": ["task_id"]
#             },
#     },
#     {
#         "type": "function",
#         "name": "add_subtasks",
#         "description": "Add subtasks to an existing task",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "parent_task": {"type": "string"},
#                 "subtasks": {
#                     "type": "array",
#                     "items": {"type": "string"}
#                 }
#             },
#             "required": ["parent_task", "subtasks"]
#         }
#     },
#     {
#         "type": "function",
#         "name": "tag_tasks",
#         "description": "Tag tasks with categories",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "tasks_tags": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "task": {"type": "string"},
#                             "tag": {"type": "string"}
#                         },
#                         "required": ["task", "tag"]
#                     }
#                 }
#             },
#             "required": ["tasks_tags"]
#         }
#     },
    
#     {
#         "type": "function",
#         "name": "delete_task",
#         "description": "Delete a task from the todo list",
#         "parameters": 
#         {
#             "type": "object",
#             "properties": 
#             {
#                 "task":
#                     {
#                         "type": "object",
#                         "properties":
#                         {
#                             "task": {"type": "string", "description": "The task description"},
#                             "status": {"type": "boolean", "description": "The task status"},
#                             "subtasks": {
#                                 "type": "array",
#                                 "items": {"type": "string"},
#                                 "description": "List of subtasks for the main task, if not provided, it will be an empty list"
#                             },
#                             "tags": {
#                                 "type": "array",
#                                 "items": {"type": "string"}
#                             },
#                             "task_id": {"type": "integer", "description": "The task ID"}
#                         },
#                         "description": "The task object which needs to be deleted, task name, task id, status and tags",
#                     },

#             },
#             "required": ["task"]
#         }
#     },
#     {
#         "type": "function",
#         "name": "get_tasks_based_on_tag",
#         "description": "Get tasks based on tag",
#         "parameters": 
#         {
#             "type": "object",
#             "properties": 
#             {
#                 "tag": {"type": "string", "description": "The tag to filter tasks"},
#             },
#             "required": ["tag"]
#         }
#     }
# ]

import Functions

function_map = {
    "add_task": Functions.add_task,
    "complete_task": Functions.complete_task,
    "add_subtasks": Functions.add_subtasks,
    "tag_tasks": Functions.tag_tasks,
    "get_tasks_based_on_tag": Functions.get_tasks_based_on_tag,
    "delete_task": Functions.delete_task,
    "add_task_with_subtasks_and_tags": Functions.add_task_with_subtasks_and_tags
}

for function in function_map:
    fs = GetFunctionSchemaForOpenAI(function_map[function])
    print(type(fs))
    function_schemas.append(fs)
    print("-----")
print(type(function_schemas))
print(type(function_schemas[0]))

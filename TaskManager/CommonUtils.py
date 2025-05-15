import inspect
from typing import get_type_hints, get_args, get_origin, _GenericAlias
from dataclasses import is_dataclass, fields

def get_type_schema(py_type):
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean"
    }

    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle List[SomeType]
    if origin == list and args:
        item_schema = get_type_schema(args[0])
        return {
            "type": "array",
            "items": item_schema
        }

    # Handle custom dataclass
    if is_dataclass(py_type):
        properties = {}
        required = []
        print("Type: ", fields(py_type))
        for field in fields(py_type):
            print("FT: ", field.type)
            properties[field.name] = get_type_schema(field.type)
            required.append(field.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    return {"type": type_mapping.get(py_type, "string")}

def generate_function_schema(func):
    """
    Generate a function schema object for LLM function calls based on a Python function.
    Automatically detects parameters, types, and handles special cases like task_id.
    
    Args:
        func: The Python function to generate a schema for
        
    Returns:
        dict: A function schema object compatible with OpenAI function calling
    """
    signature = inspect.signature(func)
    doc = func.__doc__ or ""
    hints = get_type_hints(func)
    
    # Special handling for certain function types
    func_name = func.__name__
    
    # Default schema structure
    properties = {}
    required = []
    
    # Process each parameter
    for name, param in signature.parameters.items():
        param_type = hints.get(name)
        description = ""
        
        # Special handling for task_id parameter
        if name == "task_id":
            properties[name] = {
                "type": "integer",
                "description": "The unique ID of the task to operate on"
            }
            required.append(name)
            continue
            
        # Special handling for task parameter (might be string or Task object)
        if name == "task" and func_name in ["complete_task", "delete_task"]:
            properties[name] = {
                "type": "string",
                "description": "The name of the task or task ID to operate on"
            }
            required.append(name)
            continue
            
        # Handle other parameters based on their type
        if param_type:
            type_schema = get_type_schema(param_type)
            
            # Add descriptions for common parameters
            if name == "task_name":
                description = "The name of the new task to add"
            elif name == "subtasks":
                description = "List of subtasks to add to the task"
            elif name == "tags":
                description = "List of tags to associate with the task"
            elif name == "parent_task":
                description = "The name or ID of the parent task"
            
            # Add description if we have one
            if description:
                if isinstance(type_schema, dict):
                    type_schema["description"] = description
            
            properties[name] = type_schema
            required.append(name)
    
    return {
        "type": "function",
        "name": func_name,
        "description": doc.strip(),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }

# We'll import Functions later to avoid circular imports

def generate_all_function_schemas(functions_module):
    """
    Generate function schemas for all task functions in the Functions module.
    This ensures that the LLM can properly use task_id in its function calls.
    
    Args:
        functions_module: The module containing task functions
        
    Returns:
        list: A list of function schema objects for all task functions
    """
    # List of functions to generate schemas for
    functions_to_include = [
        functions_module.add_task,
        functions_module.add_task_with_subtasks_and_tags,
        functions_module.complete_task,
        functions_module.add_subtasks,
        functions_module.tag_tasks,
        functions_module.delete_task,
        functions_module.get_tasks_based_on_tag
    ]
    
    # Generate schemas for each function
    schemas = []
    for func in functions_to_include:
        try:
            schema = generate_function_schema(func)
            schemas.append(schema)
        except Exception as e:
            print(f"Error generating schema for {func.__name__}: {e}")
    
    return schemas

# Generate function schemas automatically
function_schemas = generate_all_function_schemas()

# You can add custom schemas or override generated schemas if needed
# For example:
# function_schemas.append({
#     "type": "function",
#     "name": "custom_function",
#     "description": "A custom function",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "param": {"type": "string"}
#         },
#         "required": ["param"]
#     }
# })
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
    
    {
        "type": "function",
        "name": "delete_task",
        "description": "Delete a task from the todo list",
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
                            },
                            "task_id": {"type": "integer", "description": "The task ID"}
                        },
                        "description": "The task object which needs to be deleted, task name, task id, status and tags",
                    },

            },
            "required": ["task"]
        }
    },
    {
        "type": "function",
        "name": "get_tasks_based_on_tag",
        "description": "Get tasks based on tag",
        "parameters": 
        {
            "type": "object",
            "properties": 
            {
                "tag": {"type": "string", "description": "The tag to filter tasks"},
            },
            "required": ["tag"]
        }
    }
]

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

from Functions import tag_tasks, Tags
fs = generate_function_schema(tag_tasks)
print(fs)


print("Is dataclass:", is_dataclass(Tags))

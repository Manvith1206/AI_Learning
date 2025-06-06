import os

@staticmethod
def get_env_var(var_name: str):
    return os.getenv(var_name)
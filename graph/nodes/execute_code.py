import traceback


def execute_code(state):
    '''execute code snippet'''
    print("----Executing code snippet----")
    generation = state["generation"]

    try: 
        print("Code snippet to execute:")
        code = generation.content.replace("```python", "").replace("```", "")
        print(code)
        exec(code)
        print("Code executed successfully")
        stack_trace = None
    except Exception as e:
        print(f"Error executing code: {e}")
        stack_trace = traceback.format_exc()
        print(stack_trace)
        
    state["stack_trace"] = stack_trace

    return state
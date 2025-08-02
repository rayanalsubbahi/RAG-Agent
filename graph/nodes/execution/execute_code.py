import traceback
from graph.core.base_node import BaseNode
from graph.state import GraphState


class ExecuteCodeNode(BaseNode):
    """Node for executing generated code snippets."""
    
    def __init__(self):
        super().__init__("execute_code")
    
    def execute(self, state: GraphState) -> GraphState:
        """Execute the generated code snippet."""
        self.log("Executing code snippet")
        
        generation = state["generation"]
        
        try:
            print("Code snippet to execute:")
            # Clean the code by removing markdown formatting
            code = generation.content.replace("```python", "").replace("```", "")
            print(code)
            
            # Execute the code
            exec(code)
            print("Code executed successfully")
            stack_trace = None
            
        except Exception as e:
            print(f"Error executing code: {e}")
            stack_trace = traceback.format_exc()
            print(stack_trace)
        
        # Update state with execution results
        state["stack_trace"] = stack_trace
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that generation is present in state."""
        return ("generation" in state and 
                hasattr(state["generation"], 'content') and
                state["generation"].content)



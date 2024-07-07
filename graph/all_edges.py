
class GraphEdges:
    def __init__(self) -> None:
        pass
    
    def decide_to_generate(self, state):
        '''decide to transform query or proceed with generation'''
        print("----Deciding to generate or search----")
        
        is_transform_query = state["is_transform_query"]
        
        if is_transform_query:
            # Perform transform query then search
            return "transform_query"
        else:
            return "check_code_generation"

    def decide_to_skipRag(self, state):
        '''decide to skip RAG and continue directly with generation'''
        print("----Deciding to skip RAG and continue directly with generation----")
        query_type = state["query_type"]
        
        if query_type == "statement":
            # Skip RAG and continue with generation
            return "generate"
        else:
            # Continue with RAG
            return "check_required_search"

    def decide_to_code(self, state):
        '''decide to generate code or general purpose generation'''
        print("----Deciding to generate code or general purpose generation----")
        
        is_gen_code = state["is_gen_code"]
        search_type = state["search_type"]
        
        print(f"SEARCH TYPE: {search_type}")
            
        if is_gen_code:
            # Generate code snippet
            return "generate_code"
        elif search_type == "custom_knowledge_base" or search_type == "web":
            # Generate answer based on context
            return "generate_context"
        elif search_type == "own":
            # Generate answer based on context
            return "generate"

    def decide_to_execute(self, state):
        '''decide to execute code snippet'''
        print("----Deciding to execute code snippet----")
        
        is_gen_code = state["is_gen_code"]
        
        if is_gen_code:
            # Execute code snippet
            return "execute_code"
        else:
            # Continue with existing answer
            return "end"    

    def decide_to_search(self, state):
        '''decide to search the web for more documents'''
        print("----Deciding to search the web or the knowledge base for more documents----")
        
        required_search = state["search_type"]
        
        if required_search == "web":
            # Perform web search
            return "web_search"
        elif required_search == "custom_knowledge_base":
            # Perform knowledge base search
            return "retrieve"
        else:
            # Continue with generation
            return "check_code_generation"

    def transform_query_search(self, state):
        '''direct transformed query based on search type'''
        n_iterations = state["n_iterations"]
        
        if n_iterations < 3:
            # check type of search required
            return "check_required_search"
        else:
            # Continue with existing answer
            return "end"
        
    def decide_to_send_stack_trace(self, state):
        '''decide to generate new answer based on stack trace'''
        print("----Deciding to generate new answer based on stack trace----")
        
        stack_trace = state["stack_trace"]
        n_iterations = state["n_iterations"]
        
        if stack_trace and n_iterations < 3:
            # Generate new answer based on stack trace
            return "generate_based_on_error"
        else:
            # Continue with existing answer
            return "end"

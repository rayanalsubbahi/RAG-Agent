from langgraph.graph import END, StateGraph
from retriever.retriever import Retriever 
from graph.state import GraphState
from graph.all_edges import GraphEdges
from graph.all_nodes import GraphNodes

class RAGGraph: 
    def __init__(self, llm) -> None:
        self.retriever = Retriever()
        self.nodes = GraphNodes(llm, self.retriever)
        self.edges = GraphEdges()
        self.pipeline = self.createPipeline()
        
    def invokePipeline(self, messages):
        inputs = {"messages": messages, "documents": [], "search_type": "", "query_type": "", "stack_trace": "",
                  "generation": "", "is_gen_code": False, "is_transform_query": False, 
                  "n_iterations": 0, "parse_str_output": True}
        
        for output in self.pipeline.stream(inputs):
            for key, value in output.items():   
                # Node
                print(f"Node '{key}':")
                
        # Final generation (answer)
        answer = value["generation"].content
        ai_message = {"role": "assistant", "content": answer}
        messages.append(ai_message)
        return ai_message
        
    def createPipeline(self):
        workflow = StateGraph(GraphState)
        # Define the nodes
        workflow.add_node("rephrase_follow_up_question", self.nodes.rephrase_follow_up_question)  # rephrase follow-up question
        workflow.add_node("check_required_search", self.nodes.check_required_search)  # check if type of search required
        workflow.add_node("check_code_generation", self.nodes.check_code_generation)  # check if code generation is required
        workflow.add_node("retrieve", self.nodes.retrieve)  # retrieve
        workflow.add_node("web_search", self.nodes.web_search)  # web search
        workflow.add_node("clean_documents", self.nodes.clean_documents)  # clean documents
        workflow.add_node("grade_documents", self.nodes.grade_documents)  # grade documents
        workflow.add_node("transform_query", self.nodes.transform_query)  # transform_query
        workflow.add_node("generate_code", self.nodes.generate_code)  # generate code
        workflow.add_node("generate_context", self.nodes.generate_context)  # generate
        workflow.add_node("generate", self.nodes.generate)  # generate
        workflow.add_node("execute_code", self.nodes.execute_code)  # execute code snippet
        workflow.add_node("generate_based_on_error", self.nodes.generate_based_on_error)  # generate new answer based on error

        # Build the graph
        workflow.set_entry_point("rephrase_follow_up_question")
        workflow.add_conditional_edges(
            "rephrase_follow_up_question",
            self.edges.decide_to_skipRag,
            {
                "generate": "generate",
                "check_required_search": "check_required_search",
            }
        )
        # workflow.add_edge("rephrase_follow_up_question", "check_required_search")
        #search type
        workflow.add_conditional_edges(
            "check_required_search", 
            self.edges.decide_to_search, 
            {
                "retrieve": "retrieve",
                "web_search": "web_search",
                "check_code_generation": "check_code_generation",
            }
        )
        #web
        workflow.add_edge("web_search", "clean_documents")
        workflow.add_edge("clean_documents", "grade_documents")

        #knowledge base
        workflow.add_edge("retrieve", "grade_documents")

        #grading
        workflow.add_conditional_edges(
            "grade_documents",
            self.edges.decide_to_generate,
            {
                "transform_query": "transform_query",
                "check_code_generation": "check_code_generation",
            }
        )

        #transform query
        workflow.add_conditional_edges(
            "transform_query",
            self.edges.transform_query_search,
            {
                "check_required_search": "check_required_search",
                "end": END,
            }
        )

        #deciding gneration
        workflow.add_conditional_edges(
            "check_code_generation",
            self.edges.decide_to_code,
            {
                "generate_code": "generate_code",
                "generate_context": "generate_context",
                "generate": "generate",
            }
        )

        #generate
        workflow.add_conditional_edges(
            "generate_code",
            self.edges.decide_to_execute,
            {
                "execute_code": "execute_code",
                "end": END,
            }
        )
        workflow.add_conditional_edges(
            "execute_code",
            self.edges.decide_to_send_stack_trace,
            {
                "generate_based_on_error": "generate_based_on_error",
                "end": END,
            }
        )
        workflow.add_edge("generate_based_on_error", "execute_code")
        workflow.add_edge("generate_context", END)
        workflow.add_edge("generate", END)

        app = workflow.compile()
        return app
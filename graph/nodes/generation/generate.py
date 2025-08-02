from langchain.prompts import PromptTemplate
from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import beautify_chat_history, get_last_human_message


class GenerateNode(LLMNode):
    """Node for generating answers using LLM's own knowledge."""
    
    def __init__(self):
        super().__init__("generate")
    
    def execute(self, state: GraphState) -> GraphState:
        """Generate answer using LLM's internal knowledge."""
        self.log("Generating answer")
        
        messages = state["messages"]
        question = get_last_human_message(messages)
        chat_history = beautify_chat_history(messages[:-1])
        
        # Create prompt template
        prompt = PromptTemplate(
            template="""You are a smart user. Your job is to generate an answer to the user question based on your knowledge.
                        You need to carefully analyze the user question and the chat history to provide a relevant and accurate answer.
                        Here is the chat history: {chat_history}
                        Here is the user question: {question}
                        """,
            input_variables=["question", "chat_history"]
        )
        
        # Create chain and invoke
        rag_chain = prompt | self.llm
        generation = rag_chain.invoke({"question": question, "chat_history": chat_history})
        
        # Update state
        state["generation"] = generation
        state["documents"] = []
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages are present in state."""
        return "messages" in state and len(state["messages"]) > 0



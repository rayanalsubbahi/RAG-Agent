from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from utils import extractAnswer, getLastHumanMessage

def check_required_search(llm, state):
    '''check type of search required'''
    print("----Checking type of search required----")
    
    messages = state["messages"]
    question = getLastHumanMessage(messages)    
    parse_str_output = state["parse_str_output"]
    
    class search_type(BaseModel):
        """Define the type of search required."""
            
        search_type: str = Field(description="Value of 'own', 'custom_knowledge_base' or 'web' to determine the most appropriate search type")
    
    # Tool
    if not parse_str_output:
        llm_with_structured_output = llm.with_structured_output(search_type)
    
    # Prompt
    template="""You will be given a user's search query and need to determine the most appropriate search type to use in order to find the best information to address the query. 
                
                The three search types available are:
                1. Custom knowledge search - searches a curated knowledge base on the following two specific topics: 
                - Adversarial Attacks on Large Language Models
                - Large Language Models Powered Autonomous Agents
                2. Own model knowledge search - 
                searches the advanced language model's comprehensive knowledge spanning a vast range of topics including science, technology, history, culture, and more.
                This language model possesses remarkable reasoning, analysis, coding, and creative capabilities. Its knowledge base was trained on a massive corpus of
                high-quality data, allowing it to draw connections and synthesize information from diverse sources.
                3. Web search - searches the internet for the most current and up-to-date information
                
                The user has asked the following question:
                <question>
                {question}
                </question>
                
                Carefully analyze the query and determine which of the three search types would be most likely to surface the most relevant and useful information for the user.

                Decide the best source to answer the question:

                <answer>
                'own' if your own knowledge is sufficient
                'custom_knowledge_base' if the custom external knowledge base with the specific is likely to contain the answer
                'web' if a web search is necessary to find the most accurate and current information
                </answer>
                
                Return the search type as a string value of 'own', 'custom_knowledge_base' or 'web'
                """
    if parse_str_output:
        template += """\nYou MUST include your response in <answer> tags. \n"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
    )
        
    # Chain
    if parse_str_output:
        chain = prompt | llm | StrOutputParser()
    else:
        chain = prompt | llm_with_structured_output
    
    # Score
    score = chain.invoke({"question": question})
    
    if parse_str_output:
        score = extractAnswer(score)
        search_type_val = score
    else:
        search_type_val = score.search_type

    print(f"Score: {score}")
    
    if search_type_val == "own":
        print("---WILL USE OWN KNOWLEDGE---")
    elif search_type_val == "custom_knowledge_base":
        print("---WILL RUN KNOWLEDGE BASE SEARCH---")
    else:
        print("---WILL RUN WEB SEARCH---")    
        
    state["search_type"] = search_type_val
    state["documents"] = []
        
    return state
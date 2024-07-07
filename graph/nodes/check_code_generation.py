from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from utils import extractAnswer, getLastHumanMessage

def check_code_generation(llm, state):
    '''decide to generate code or perform a general purpose generation'''
    print("----Deciding to generate code or general purpose generation----")
    
    messages = state["messages"]
    question = getLastHumanMessage(messages)
    parse_str_output = state["parse_str_output"]
    
    class is_code_required(BaseModel):
        """Binary score for relevance check."""
        requires_code: str = Field(description="Score of 'yes' or 'no' to determine if a code implementation is required in response to the question or not")
            
    # Tool
    if not parse_str_output:
        llm_with_structured_output = llm.with_structured_output(is_code_required)
    
    # Prompt
    template= """You are a smart user. Your job is to determine whether the answer to question requires a code snippet or can be answered without a code. \n 
        Here is the user question: {question} \n
        Give a binary score 'yes' or 'no' to indicate whether the user question requires a code snippet execution or not. \n"""
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
        print(f"Score: {score}")
        score = extractAnswer(score)
        score = "yes" if (score.find("yes") != -1 or score.find("Yes") != -1 or score.find("YES") != -1) else "no"
        is_gen_code = score == "yes"
    else:
        is_gen_code = score.requires_code == "yes"
        
    if is_gen_code:
        print("---WILL GENERATE CODE---")
    else:
        print("---WILL NOT GENERATE CODE---")
        
    state["is_gen_code"] = is_gen_code
    
    return state
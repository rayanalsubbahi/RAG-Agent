import asyncio

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from utils import extractAnswer, getLastHumanMessage


def grade_documents(llm, state):
    '''determine if the documents are relevant'''
    print("----Grading documents----")

    documents = state["documents"]
    messages = state["messages"]
    question = getLastHumanMessage(messages)
    parse_str_output = state["parse_str_output"]
    
    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # Tool
    # LLM with tool and enforce invocation
    # llm_with_tool = model.bind_tools(
    #     tools=[grade],
    # )
    if not parse_str_output:
        llm_with_structured_output = llm.with_structured_output(grade)

    # Prompt
    template = """You are a grader assessing relevance of a retrieved document from the internet to a user question. \n
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    if parse_str_output:
        template += """\nYou MUST include your response in <answer> tags. \n"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    # Chain
    if parse_str_output:
        chain = prompt | llm | StrOutputParser()
    else:
        chain = prompt | llm_with_structured_output 
    
    # Score
    # Run chain in parallel
    async def grade_doc_async(question, doc, chain):
        score = await asyncio.to_thread(chain.invoke, {"question": question, "context": doc.page_content})
        return score, doc

    async def process_documents(question, documents, chain):
        relevant_docs = []
        # Create a list of coroutines for each document
        tasks = [grade_doc_async(question, doc, chain) for doc in documents]
        
        # Run all tasks concurrently and wait for all of them to complete
        results = await asyncio.gather(*tasks)
        
        # Process the results
        for score, doc in results:
            if parse_str_output:
                print(f"Score: {score}")
                score = extractAnswer(score)
                grade_res = "yes" if score.find("yes") != -1 or score.find("Yes") != -1 or score.find("YES") != -1 else "no"
            else:
                grade_res = "yes" if score.binary_score == "yes" else "no"
                
            if grade_res == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                relevant_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
        
        return relevant_docs

    relevant_docs = asyncio.run(process_documents(question, documents, chain))

        
    if len(relevant_docs)/len(documents) < 0.5:
        is_transform_query = True
    else:
        is_transform_query = False
    
    state["documents"] = relevant_docs
    state["is_transform_query"] = is_transform_query

    return state
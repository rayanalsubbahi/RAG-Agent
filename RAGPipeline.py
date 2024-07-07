import json
import os
import asyncio
import traceback
from dotenv import load_dotenv
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from googlesearch import search
from typing import TypedDict, Dict

# Importing Langchain Libraries
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, StateGraph

# Set Environment Variables
load_dotenv()

# Helper Functions
def beautify_chat_history(history):
    '''beautify chat history'''
    chat_history = ""
    for m in history:
        if m["role"] == "user":
            chat_history += f"User: {m['content']} \n"
        else:
            chat_history += f"Assistant: {m['content']} \n"
    return chat_history

def extractAnswer(response): 
    answer = response.find("<answer>")
    endAnswer = response.find("</answer>")
    answer = response[answer+8:endAnswer]
    return answer

# Global Variables
global use_stroutput
use_stroutput = True

# Define the GraphState
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]

class Retriever:
    def __init__(self):
        self.vectorStore = self.setRetriever()
    
    def setRetriever(self):
        # Create Vector Store
        if os.path.exists('vectorstore'):
            vectorStore = Chroma(persist_directory='vectorstore', collection_name="rag-chroma", embedding_function=OpenAIEmbeddings())
            vectorStore = vectorStore.as_retriever()
            return vectorStore
        else:
            docs = self.load_docs()
            vectorStore = self.set_vector_store(docs)
            return vectorStore

    def load_docs(self):
        '''load documents from pdf files'''
        pdf_files_path = '/Users/Razan/Downloads/'
        urls = [
            "Adversarial Attacks on LLMs | Lil'Log.pdf",
            "Prompt Engineering | Lil'Log.pdf",
            "LLM Powered Autonomous Agents | Lil'Log.pdf",
        ]

        docs = [PyPDFLoader(pdf_files_path+url).load_and_split() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        print(f"Loaded {len(docs_list)} documents")
        return docs_list

    def set_vector_store(self, docs=[]):
        '''set vector store'''
        #create text splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0)
        #split documents
        doc_splits = text_splitter.split_documents(docs)
    
        #Add to vectorDB
        vectorstore = Chroma.from_documents(
            persist_directory='vectorstore',  # directory to store the vector store
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=OpenAIEmbeddings(),
        )
        
        retriever = vectorstore.as_retriever()

        return retriever

class GraphNodes:
    def __init__(self, llm, retriever) -> None:
        self.llm = llm
        self.retriever = retriever

    def rephrase_follow_up_question(self, state):
        """rephrase the follow-up question to be a standalone question given a chat history"""
        print("----Rephrasing follow-up question----")
        state_dict = state["keys"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        chat_history = messages[:-1]
        n_iterations = state_dict["n_iterations"]
        
        if len(chat_history) == 0:
            return {"keys": {"messages": messages, "n_iterations": n_iterations, "query_type": "question"}}
        
        class RephrasedInput(BaseModel):
            """Rephrased input for standalone question."""

            rephrased_input: str = Field(description="Rephrased standalone user input")
            input_type: str = Field(description="Type of user input as 'question' or 'statement'")

        # Prompt
        template = '''
            Given a chat history and a follow-up user input, process the input as follows:

            1. Determine if the user input is a question, query, or request (collectively referred to as "question") or not

            2. If it's a question:
            a) If the question relies on context from the chat history, rephrase it to be a standalone question that incorporates necessary context.
            b) If the question is clear and standalone, keep it as is.
            c) Ensure the rephrased question maintains the conversation flow and context established by the previous history.
            d) Do NOT answer the question.

            3. If it's not a question:
            Return the user input exactly as it is, without any modifications.

            4. For all inputs, determine the type: either 'question' or 'statement'.

            5. Return the processed input and its type as rephrased_input and input_type.

            Chat History: {chat_history}
            Follow-up User Input: {question}
            '''
        if use_stroutput:
            template += """\nYou MUST only return a JSON object with the keys 'rephrased_input' and 'input_type'. \n"""
            
        prompt =  PromptTemplate(
            # template= """Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question. \
            #             The question must be rephrased in a way that maintains the conversation flow and context established by the previous history.
            #             Do NOT answer the question, just reformulate it if needed, otherwise return it as is. \
            #             Only return the final standalone question. \
            #             Here is the chat history: {chat_history} \n
            #             Here is the follow-up question: {question}
            #             """,
            template=template,
            input_variables=["question", "chat_history"],)
        
        # Chain
        if not use_stroutput:
            llm_structured_output = self.llm.with_structured_output(RephrasedInput)
            rag_chain = prompt | llm_structured_output
        else:
            rag_chain = prompt | self.llm | JsonOutputParser()

        # Run
        response = rag_chain.invoke({"question": question, "chat_history": chat_history})
        if use_stroutput:
            rephrased_question = response["rephrased_input"]
            input_type = response["input_type"]
        else:
            rephrased_question = response.rephrased_input
            input_type = response.input_type
            
        print(f"Rephrased response: {response}")
        messages.append({"role": "user", "content": rephrased_question, "ai_message": True})
        return {"keys": {"messages": messages, "n_iterations": n_iterations, "query_type": input_type}}
    
    def check_required_search(self, state):
        '''check type of search required'''
        print("----Checking type of search required----")
        
        state_dict = state["keys"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        n_iterations = state_dict["n_iterations"]    
        
        class search_type(BaseModel):
            """Define the type of search required."""
                
            search_type: str = Field(description="Value of 'own', 'custom_knowledge_base' or 'web' to determine the most appropriate search type")
        
        # Tool
        if not use_stroutput:
            llm_with_structured_output = self.llm.with_structured_output(search_type)
        
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
        if use_stroutput:
            template += """\nYou MUST include your response in <answer> tags. \n"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
        )
            
        # Chain
        if use_stroutput:
            chain = prompt | self.llm | StrOutputParser()
        else:
            chain = prompt | llm_with_structured_output
        
        # Score
        score = chain.invoke({"question": question})
        
        if use_stroutput:
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
            
        return {'keys': {'messages': messages, 'search_type': search_type_val, 'n_iterations': n_iterations, "documents": []}}

    def web_search(self, state):
        '''search the web for more documents'''
        print("----Performing web search----")
        state_dict = state["keys"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        # search_queries = state_dict["search_queries"]['search_queries']
        n_iterations = state_dict["n_iterations"]
        search_type = state_dict["search_type"]
        
        # Search
        urls = []
        documents=[]
        for url in search(question, num_results=3):
            if 'pdf' in url:
                continue
            print('Url', url)
            urls.append(url)
        
        #take the first 3 urls
        urls = urls[:3]
        # Load docs in parallel    
        loader = WebBaseLoader(urls)
        documents = loader.aload()
        # web_search_tool = TavilySearchResults(k=3)
        # docs = web_search_tool.invoke({"query": question})
        # docs = [Document(page_content=d["content"]) for d in docs]
        # documents.extend(docs)
        
        return {"keys": {"documents": documents, "messages": messages, "n_iterations": n_iterations, "search_type": search_type}}

    def retrieve(self, state):
        '''retrieve content from documents'''
        print("----Retrieving documents----")
        
        state_dict = state["keys"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        n_iterations = state_dict["n_iterations"]
        search_type = state_dict["search_type"]
        
        # Retrieve
        documents = self.retriever.vectorStore.get_relevant_documents(question)
        
        print(f"Retrieved {len(documents)} documents")

        return {"keys": {"documents": documents, "messages": messages, "n_iterations": n_iterations, "search_type": search_type}}

    def grade_documents(self, state):
        '''determine if the documents are relevant'''
        print("----Grading documents----")
        
        state_dict = state["keys"]
        documents = state_dict["documents"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        n_iterations = state_dict["n_iterations"]   
        search_type = state_dict["search_type"]
        
        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # Tool
        # LLM with tool and enforce invocation
        # llm_with_tool = model.bind_tools(
        #     tools=[grade],
        # )
        if not use_stroutput:
            llm_with_structured_output = self.llm.with_structured_output(grade)

        # Prompt
        template = """You are a grader assessing relevance of a retrieved document from the internet to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        if use_stroutput:
            template += """\nYou MUST include your response in <answer> tags. \n"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        # Chain
        if use_stroutput:
            chain = prompt | self.llm | StrOutputParser()
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
                if use_stroutput:
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

        return {
            "keys": {
                "documents": relevant_docs,
                "messages": messages,
                "search_type": search_type,
                "n_iterations": n_iterations, 
                "is_transform_query": is_transform_query
            }
        }

    def clean_documents(self, state):
        '''clean documents of any unwanted content'''
        print("----Cleaning documents----")
        state_dict = state["keys"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        n_iterations = state_dict["n_iterations"]
        documents = state_dict["documents"]
        search_type = state_dict["search_type"]
        
        # Prompt
        prompt = PromptTemplate(
            template="""You are a document cleaner. Your job is to clean the retrieved document of any unwanted content that is not relevant to the user question. \n
            You must not change the relevant content and only copy it to the output. \n
            Here is the user question: {question} \n
            Here is the retrieved document: \n\n {context} \n\n
            Return directly the cleaned content of the documents. \n""",
            input_variables=["context", "question"],
        )
        
        # Chain
        chain = prompt | self.llm | StrOutputParser()
        
        # Run chain in parallel
        async def invoke_chain_async(chain, question, doc):
            print(f"Cleaning document")
            result = await asyncio.to_thread(chain.invoke, {"question": question, "context": doc.page_content})
            return Document(page_content=result)

        async def run_chain_in_parallel(chain, question, documents):
            cleaned_docs = []
            # Create a list of coroutines for each document
            tasks = [invoke_chain_async(chain, question, doc) for doc in documents]
            
            # Run all tasks concurrently and wait for all of them to complete
            results = await asyncio.gather(*tasks)
            
            # Collect the results
            cleaned_docs.extend(results)
            return cleaned_docs

        # Example usage
        cleaned_docs = asyncio.run(run_chain_in_parallel(chain, question, documents))
        
        return {"keys": {"documents": cleaned_docs, "messages": messages, "n_iterations": n_iterations, "search_type": search_type}}

    def transform_query(self, state):
        '''rewrite the query for a better answer'''
        print("----Transforming query----")
        state_dict = state["keys"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        documents = state_dict["documents"]
        n_iterations = state_dict["n_iterations"]
        search_type = state_dict["search_type"]
        n_iterations += 1
        
        # Create a prompt template with format instructions and the query
        prompt = PromptTemplate(
            template="""You are generating questions that is well optimized for searching the internet. \n 
            Look at the input and try to reason about the underlying sematic intent / meaning. \n 
            Here is the initial question:
            \n ------- \n
            {question} 
            \n ------- \n
            Formulate an improved question. \n
            Return a JSON object with the key 'better_question' and the value as the improved question. \n""",
            input_variables=["question"],
        )

        # Prompt
        chain = prompt | self.llm | JsonOutputParser()
        better_question = chain.invoke({"question": question})
        better_question = better_question["better_question"]
        messages.append({"role": "user", "content": better_question, "ai_message": True})
        print(f"Improved question: {better_question}")

        return {"keys": {"documents": documents, "messages": messages, "n_iterations": n_iterations, "search_type": search_type}}

    def check_code_generation(self, state):
        '''decide to generate code or perform a general purpose generation'''
        print("----Deciding to generate code or general purpose generation----")
        
        state_dict = state["keys"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        n_iterations = state_dict["n_iterations"]
        
        class is_code_required(BaseModel):
            """Binary score for relevance check."""

            requires_code: str = Field(description="Score of 'yes' or 'no' to determine if a code implementation is required in response to the question or not")
                
        # Tool
        if not use_stroutput:
            llm_with_structured_output = self.llm.with_structured_output(is_code_required)
        
        # Prompt
        template= """You are a smart user. Your job is to determine whether the answer to question requires a code snippet or can be answered without a code. \n 
            Here is the user question: {question} \n
            Give a binary score 'yes' or 'no' to indicate whether the user question requires a code snippet execution or not. \n"""
        if use_stroutput:
            template += """\nYou MUST include your response in <answer> tags. \n"""
            
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
        )
        
        # Chain
        if use_stroutput:
            chain = prompt | self.llm | StrOutputParser()
        else:
            chain = prompt | llm_with_structured_output
        
        # Score
        score = chain.invoke({"question": question})
        
        if use_stroutput:
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
            
        return {
            "keys": {
                "documents": state_dict["documents"],
                "messages": messages,
                "search_type": state_dict["search_type"],
                "n_iterations": n_iterations,
                "is_gen_code": is_gen_code
            }
        }

    def generate_code(self, state):
        '''generate answer based on documents'''
        print("----Generating code----")
        state_dict = state["keys"]
        documents = state_dict["documents"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        n_iterations = state_dict["n_iterations"]
        is_gen_code = state_dict["is_gen_code"]
        
        # prompt
        prompt = PromptTemplate(template="""You are a smart programmer. Your job is to write a code snippet that answers the user question based on the retrieved documents.\n
                                You must only return the code snippet without any additional text so that the code can be executed directly.\n
                                Do not include any mark that indicates the language of the code.\n
                                Here are the retrieved documents: \n\n {context} \n\n
                                Here is the user question: {question}"""
                                , input_variables=["context", "question"]) 
        # Chain
        rag_chain = prompt | self.llm 

        # Run
        generation = rag_chain.invoke({"context": documents, "question": question})
        
        return {
            "keys": {"documents": documents, "messages": messages, "generation": generation, "n_iterations": n_iterations, "is_gen_code": is_gen_code}
        }

    def generate_context(self, state):
        '''generate answer based on documents'''
        print("----Generating answer based on context----")
        state_dict = state["keys"]
        documents = state_dict["documents"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        n_iterations = state_dict["n_iterations"]
        is_gen_code = state_dict["is_gen_code"]
        
        # prompt
        prompt = PromptTemplate(template="""You are a smart user. Your job is to generate an answer to the user question based on the retrieved documents.\n
                                        Here are the retrieved documents: \n\n {context} \n\n
                                        Here is the user question: {question}
                                        You must not directly reference or mention the source documents or information in your answer.
                                        Be detailed and provide a comprehensive response to the user question.
                                        """
                                        , input_variables=["context", "question"])
        # Chain
        rag_chain = prompt | self.llm 

        # Run
        generation = rag_chain.invoke({"context": documents, "question": question})
        
        return {
            "keys": {"documents": documents, "messages": messages, "generation": generation, "n_iterations": n_iterations, "is_gen_code": is_gen_code}
        }
        
    def generate(self, state):
        '''generate from LLM'''
        print("----Generating answer----")
        
        state_dict = state["keys"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        chat_history = beautify_chat_history(messages[:-1])
        # n_iterations = state_dict["n_iterations"]
        # is_gen_code = state_dict["is_gen_code"]
        
        # prompt
        prompt = PromptTemplate(template="""You are a smart user. Your job is to generate an answer to the user question based on your knowledge.\n
                                        You need to carefully analyze the user question and the chat history to provide a relevant and accurate answer.\n
                                        Here is the chat history: {chat_history} \n
                                        Here is the user question: {question}
                                        """
                                        , input_variables=["question", "chat_history"])
        # Chain
        rag_chain = prompt | self.llm
        
        # Run
        generation = rag_chain.invoke({"question": question, "chat_history": chat_history})
        
        return {"keys": {"documents": [], "messages": messages, "generation": generation}}
        
    def generate_based_on_error(self, state):
        '''generate answer based on documents and stack trace'''
        print("----Generating new answer based on error----")
        state_dict = state["keys"]
        documents = state_dict["documents"]
        messages = state_dict["messages"]
        #get last human message and chat history
        question = [m for m in messages if m["role"] == "user"][-1]["content"]
        generation = state_dict["generation"]
        stack_trace = state_dict["stack_trace"]
        n_iterations = state_dict["n_iterations"]
        is_gen_code = state_dict["is_gen_code"]
        n_iterations += 1
        
        # prompt
        prompt = PromptTemplate(
            template="""You are a code debugger trying to fix a code issue based on the stack trace of the code execution 
            and the retrieved documents from the internet to a user question. \n 
            Here are the retrieved documents: \n\n {context} \n\n
            Here is the user question: {question} \n
            Here is the code snippet that was executed: \n\n {generation} \n\n
            Here is the stack trace: \n\n {stack_trace} \n\n
            Provide a new code snippet that fixes the issue based on the stack trace and the documents.
            You must only return the code snippet without any additional text so that the code can be executed directly. \n
            Don not include any mark that indicates the language of the code. \n""",
            input_variables=["context", "question", "generation", "stack_trace"],
        )
            
        # Chain
        rag_chain = prompt | self.llm 
        
        # Run
        new_generation = rag_chain.invoke({"context": documents, "messages": messages, "generation": generation, "stack_trace": stack_trace, "question": question})
            
        return { "keys": {"documents": documents, "messages": messages, "generation": new_generation, "stack_trace": None, "n_iterations": n_iterations, "is_gen_code": is_gen_code}}
        
    def execute_code(self, state):
        '''execute code snippet'''
        print("----Executing code snippet----")
        state_dict = state["keys"]
        documents = state_dict["documents"]
        messages = state_dict["messages"]
        generation = state_dict["generation"]
        n_iterations = state_dict["n_iterations"]
        is_gen_code = state_dict["is_gen_code"]
        
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

        return {
            "keys": {"documents": documents, "messages": messages, "generation": generation, "stack_trace": stack_trace, "n_iterations": n_iterations, "is_gen_code": is_gen_code}
        }
        
class GraphEdges:
    def __init__(self) -> None:
        pass
    
    def decide_to_generate(self, state):
        '''decide to transform query or proceed with generation'''
        print("----Deciding to generate or search----")
        
        state_dict = state["keys"]
        is_transform_query = state_dict["is_transform_query"]
        
        if is_transform_query:
            # Perform transform query then search
            return "transform_query"
        else:
            return "check_code_generation"

    def decide_to_skipRag(self, state):
        '''decide to skip RAG and continue directly with generation'''
        print("----Deciding to skip RAG and continue directly with generation----")
        
        state_dict = state["keys"]
        query_type = state_dict["query_type"]
        
        if query_type == "statement":
            # Skip RAG and continue with generation
            return "generate"
        else:
            # Continue with RAG
            return "check_required_search"

    def decide_to_code(self, state):
        '''decide to generate code or general purpose generation'''
        print("----Deciding to generate code or general purpose generation----")
        
        state_dict = state["keys"]
        is_gen_code = state_dict["is_gen_code"]
        search_type = state_dict["search_type"]
        
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
        
        state_dict = state["keys"]
        is_gen_code = state_dict["is_gen_code"]
        
        if is_gen_code:
            # Execute code snippet
            return "execute_code"
        else:
            # Continue with existing answer
            return "end"    

    def decide_to_search(self, state):
        '''decide to search the web for more documents'''
        print("----Deciding to search the web or the knowledge base for more documents----")
        
        state_dict = state["keys"]
        required_search = state_dict["search_type"]
        
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
        state_dict = state["keys"]
        # required_search = state_dict["search_type"]
        n_iterations = state_dict["n_iterations"]
        
        if n_iterations < 3:
            # check type of search required
            return "check_required_search"
        else:
            # Continue with existing answer
            return "end"
        
    def decide_to_send_stack_trace(self, state):
        '''decide to generate new answer based on stack trace'''
        print("----Deciding to generate new answer based on stack trace----")
        
        state_dict = state["keys"]
        stack_trace = state_dict["stack_trace"]
        n_iterations = state_dict["n_iterations"]
        
        if stack_trace and n_iterations < 3:
            # Generate new answer based on stack trace
            return "generate_based_on_error"
        else:
            # Continue with existing answer
            return "end"

class RAGPipeline: 
    def __init__(self, llm) -> None:
        self.retriever = Retriever()
        self.nodes = GraphNodes(llm, self.retriever)
        self.edges = GraphEdges()
        self.pipeline = self.createPipeline()
        
    def invokePipeline(self, messages):
        inputs = {"keys": {"n_iterations": 0, "messages": messages}}
        for output in self.pipeline.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"Node '{key}':")
                
        # Final generation (answer)
        answer = value["keys"]["generation"].content
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
    
class Assistant:
    def __init__(self, llm):
        self.llm = llm
        self.pipeline = RAGPipeline(self.llm)
        print("Assistant initialized")
    
    def chat(self, messages):
        return self.pipeline.invokePipeline(messages) 
    
    
if __name__ == "__main__":
    # LLM
    # llm = ChatAnthropic(model='claude-3.5-sonnet-20240620', anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    llm = ChatAnthropic(model='claude-3-haiku-20240307', anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    # llm = ChatOpenAI(model='gpt-3.5-turbo-1106', openai_api_key=os.getenv("OPENAI_API_KEY"))
    # llm = ChatCohere(model='command-r-plus', cohere_api_key=os.getenv("COHERE_API_KEY"))
    # llm = ChatNVIDIA(model='nvidia / nemotron-4-340b-instruct', nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
    # llm = ChatNVIDIA(model='meta / llama3-70b-instruct', nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
    
    #llm = ChatNVIDIA(model='google / gemma-2-27b-it', nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
    
    # Create Assistant
    assistant = Assistant(llm)
    # Chat
    question = "What is the best way to prevent adversarial attacks on large language models?"
    messages = []
        # Initial message
    human_message = {"role": "user", "content": question}
    messages.append(human_message)
        
    answer = assistant.chat(messages)
    print(f"Answer: {answer}")



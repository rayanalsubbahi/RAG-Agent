# create streamlit app to chat with a GPT model with text and images
import streamlit as st
import os
import base64
import datetime
import time
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from RAGPipeline import Assistant
from workflow_types import WorkflowType

# Set Environment Variables 
load_dotenv()

def send_message(messages, model, isVision):
    strt = time.time()
    answer = model.chat(messages)
    print("Time taken for response: ", time.time()-strt)
    return answer

def chat(text, model, base64_image=None, isVision=False):
    message = {"role": "user", "content": text}
    st.session_state["history"].append(message)
    writeToFile(message["content"], "user")
    response = send_message(st.session_state["history"], model, isVision=isVision)
    # st.session_state["history"].append(response)
    writeToFile(response["content"], "assistant")
    return response

def writeToFile(text, role):
    # add to text file
    with open('History//'+st.session_state["history_name"], "a") as f:
        f.write(f'"{role}": "{text}"')

# Function to encode the image
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# Function to read chat from a file and return as a list of tuples
def read_chat_from_file(file_path):
    chat_data = []
    with open('History//'+file_path, "r") as file:
        lines = file.read().split('""')  # Split the text by double quotes
        for line in lines:
            if line.strip():  # Check if the line is not empty
                speaker, message = line.split(
                    ": ", 1
                )  # Split by the first colon and space
                # remove double quotes
                speaker = speaker.replace('"', "")
                # remove "" from beginning and end of message if present
                message = message.strip('"')
                chat_data.append((speaker.strip(), message.strip()))
    return chat_data

# Function to display chat in Streamlit
def display_chat(chat_data):
    for speaker, message in chat_data:
        if speaker == "user":
            if "|img|" in message:
                text = message.split("|img|")[0]
                st.chat_message(name="User").write(text)
                img = message.split("|img|")[1]
                st.image(img)
            else:
                st.chat_message(name="User").write(message)
        elif speaker == "assistant":
            st.chat_message(name="Assistant").write(message)

def response_generator(response):
    start = 0
    for i, char in enumerate(response):
        if char in ' \n':
            yield response[start:i+1]
            start = i + 1
        if char == ' ':
            time.sleep(0.05)
        elif char == '\n':
            time.sleep(0.05)

    # Yield the last word or remaining part of the string
    if start < len(response):
        yield response[start:]

# Function to create the Streamlit app
def createStreamlitApp(llm, workflow_type, parse_str_output):
    if 'model' not in st.session_state:
        model = Assistant(llm, workflow_type, parse_str_output)
        st.session_state['model'] = model
    else:
        model = st.session_state['model']
    
    # display chat history
    # get .json files
    history_files = ["History//"+file for file in os.listdir('History') if file.endswith(".txt")]
    # sort by date
    history_files.sort(key=os.path.getmtime, reverse=True)
    history_files.insert(0, "None")
    for i in range(len(history_files)):
        history_files[i] = history_files[i].replace("History//", "")
        
    # Add button for new chat
    if st.sidebar.button("New chat", type='primary'):
        st.session_state["history"] = []
        st.session_state.historySelect = "None"
        
    # Display history in a dropdown
    history_file = st.sidebar.selectbox("Select a chat history", history_files, key='historySelect')

    if history_file and history_file != "None":  # chat history
        # display title
        st.caption(history_file)
        # read the chat history
        chat_data = read_chat_from_file(history_file)
        display_chat(chat_data)
    else:  # model
        # set title and description
        st.title("RAG+")
        st.caption("Assistant to search for information and answer questions for data")
            
        if "history" not in st.session_state:
            st.session_state["history"] = []
            # create json file to store history with current date
            dt = datetime.datetime.now()
            history_name = "history_" + dt.strftime("%y-%m-%d %H:%M:%S") + ".txt"
            st.session_state["history_name"] = history_name
            
        for message in st.session_state["history"]:
            if message["role"] == "user" and message.get("ai_message", False) == False:
                st.chat_message(name="User").write(message["content"])
            elif message["role"] == "assistant":
                resp = message["content"].replace('\n', '  \n')
                st.chat_message(name="Assistant").write(resp)

        # get the user input
        message = st.chat_input(placeholder="Message here")
        # send the user input to the model
        if message:
            # display the user input
            st.chat_message(name="User").write(message)
            with st.spinner("Thinking..."):
                response = chat(message, model, isVision=False)
            # display the response from the model to the user
            resp = response["content"].replace('\n', '  \n')
            st.chat_message(name="Assistant").write_stream(response_generator(resp))
            print(response["content"])
    return

llm = ChatAnthropic(model='claude-3-haiku-20240307', anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
# llm = ChatAnthropic(model='claude-3.5-sonnet-20240620', anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
# llm = ChatOpenAI(model='gpt-4o-mini-2024-07-18', openai_api_key=os.getenv("OPENAI_API_KEY"))
# llm = ChatCohere(model='command-r-plus', cohere_api_key=os.getenv("COHERE_API_KEY"))
# llm = ChatNVIDIA(model='nvidia/nemotron-4-340b-instruct', nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
# llm = ChatNVIDIA(model='meta/llama-3.1-405b-instruct', nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
# llm = ChatNVIDIA(model='google/gemma-2-2b-it', nvidia_api_key=os.getenv("NVIDIA_API_KEY"))

createStreamlitApp(llm=llm, workflow_type=WorkflowType.ALL, parse_str_output=False)


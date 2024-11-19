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

def getLastHumanMessage(messages):
    '''get last human message'''
    return [m for m in messages if m["role"] == "user"][-1]["content"]
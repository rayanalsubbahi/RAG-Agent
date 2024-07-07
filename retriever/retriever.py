import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings


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
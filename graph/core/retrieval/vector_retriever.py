"""Enhanced vector-based retriever that extends the original Retriever class."""

import os
from typing import List, Any, Optional, Dict
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

from .base_retriever import BaseRetriever


class VectorRetriever(BaseRetriever):
    """Enhanced vector store retriever with better configuration and error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration with defaults
        self.persist_directory = self.config.get('persist_directory', 'vectorstore')
        self.collection_name = self.config.get('collection_name', 'rag-chroma')
        self.chunk_size = self.config.get('chunk_size', 250)
        self.chunk_overlap = self.config.get('chunk_overlap', 0)
        self.pdf_files_path = self.config.get('pdf_files_path', '/Users/Razan/Downloads/')
        
        # Default PDF files to load
        self.default_pdf_files = self.config.get('pdf_files', [
            "Adversarial Attacks on LLMs | Lil'Log.pdf",
            "Prompt Engineering | Lil'Log.pdf", 
            "LLM Powered Autonomous Agents | Lil'Log.pdf"
        ])
        
        # Initialize vector store
        self.vectorStore = self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store, creating it if it doesn't exist."""
        try:
            if os.path.exists(self.persist_directory):
                # Load existing vector store
                vectorStore = Chroma(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    embedding_function=OpenAIEmbeddings()
                )
                retriever = vectorStore.as_retriever()
                print(f"✅ Loaded existing vector store from {self.persist_directory}")
                return retriever
            else:
                # Create new vector store
                print("📚 Creating new vector store...")
                docs = self._load_documents()
                retriever = self._create_vector_store(docs)
                print("✅ Created new vector store")
                return retriever
                
        except Exception as e:
            print(f"❌ Error initializing vector store: {e}")
            # Return a dummy retriever that returns empty results
            return self._create_dummy_retriever()
    
    def _load_documents(self) -> List[Any]:
        """Load documents from PDF files."""
        docs_list = []
        
        for pdf_file in self.default_pdf_files:
            pdf_path = os.path.join(self.pdf_files_path, pdf_file)
            
            try:
                if os.path.exists(pdf_path):
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load_and_split()
                    docs_list.extend(docs)
                    print(f"✅ Loaded {pdf_file}")
                else:
                    print(f"⚠️  PDF not found: {pdf_file}")
                    
            except Exception as e:
                print(f"❌ Error loading {pdf_file}: {e}")
        
        print(f"📄 Total documents loaded: {len(docs_list)}")
        return docs_list
    
    def _create_vector_store(self, docs: List[Any]):
        """Create vector store from documents."""
        if not docs:
            print("⚠️  No documents to create vector store")
            return self._create_dummy_retriever()
        
        try:
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Split documents
            doc_splits = text_splitter.split_documents(docs)
            print(f"📝 Created {len(doc_splits)} document chunks")
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                persist_directory=self.persist_directory,
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=OpenAIEmbeddings()
            )
            
            return vectorstore.as_retriever()
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            return self._create_dummy_retriever()
    
    def _create_dummy_retriever(self):
        """Create a dummy retriever that returns empty results."""
        class DummyRetriever:
            def get_relevant_documents(self, query: str, **kwargs):
                print("⚠️  Using dummy retriever - no documents available")
                return []
        
        return DummyRetriever()
    
    def retrieve(self, query: str, **kwargs) -> List[Any]:
        """Retrieve relevant documents for a query."""
        if not self.validate_query(query):
            return []
        
        processed_query = self.preprocess_query(query)
        
        try:
            documents = self.vectorStore.get_relevant_documents(processed_query, **kwargs)
            return self.postprocess_documents(documents)
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return []
    
    def get_relevant_documents(self, query: str, **kwargs) -> List[Any]:
        """Get relevant documents (compatibility method)."""
        return self.retrieve(query, **kwargs)
    
    def add_documents(self, documents: List[Any]) -> bool:
        """Add new documents to the vector store."""
        try:
            if hasattr(self.vectorStore, 'add_documents'):
                self.vectorStore.add_documents(documents)
                return True
            else:
                print("⚠️  Vector store doesn't support adding documents")
                return False
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return False



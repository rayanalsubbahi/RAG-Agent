"""General async processing utilities for document operations."""

import asyncio
from typing import List, Callable, Any, Optional
from langchain.schema import Document
from config.config import get_async_config, get_search_config


class AsyncDocumentProcessor:
    """General-purpose async processor for document operations."""
    
    @staticmethod
    async def process_documents_parallel(
        documents: List[Document],
        processing_function: Callable,
        question: str,
        chain: Any,
        error_handler: Optional[Callable] = None,
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """
        Process documents in parallel using async operations.
        
        Args:
            documents: List of documents to process
            processing_function: Function to process each document
            question: User question for context
            chain: LLM chain for processing
            error_handler: Optional error handler function
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of processed results
        """
        # Use config-based concurrency if not specified
        if max_concurrent is None:
            async_config = get_async_config()
            max_concurrent = async_config.get_optimal_concurrency(len(documents))
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_document(doc: Document) -> Any:
            async with semaphore:
                try:
                    return await processing_function(question, doc, chain)
                except Exception as e:
                    print(f"Error processing document: {e}")
                    if error_handler:
                        return error_handler(e, doc)
                    else:
                        return None
        
        # Create tasks for all documents
        tasks = [process_single_document(doc) for doc in documents]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        processed_results = []
        for result in results:
            if result is not None and not isinstance(result, Exception):
                processed_results.append(result)
        
        return processed_results


class DocumentGrader:
    """Specialized async processor for document grading."""
    
    @staticmethod
    async def grade_documents_async(
        documents: List[Document],
        question: str,
        chain: Any,
        parse_str_output: bool
    ) -> tuple[List[Document], bool]:
        """
        Grade documents for relevance asynchronously.
        
        Returns:
            Tuple of (relevant_documents, should_transform_query)
        """
        def error_handler(error: Exception, doc: Document):
            """Handle grading errors by returning negative score."""
            print(f"Document grading failed: {error}")
            if parse_str_output:
                return "<answer>no</answer>", doc
            else:
                from types import SimpleNamespace
                return SimpleNamespace(binary_score="no"), doc
        
        async def grade_single_document(question: str, doc: Document, chain: Any):
            """Grade a single document."""
            return await asyncio.to_thread(
                chain.invoke, 
                {"question": question, "context": doc.page_content}
            ), doc
        
        # Process documents
        results = await AsyncDocumentProcessor.process_documents_parallel(
            documents=documents,
            processing_function=grade_single_document,
            question=question,
            chain=chain,
            error_handler=error_handler
        )
        
        # Extract relevant documents
        relevant_docs = []
        for score, doc in results:
            if parse_str_output:
                from graph.core.utils import extract_answer
                score_text = extract_answer(score)
                grade_res = "yes" if any(word in score_text.lower() for word in ["yes"]) else "no"
            else:
                grade_res = "yes" if score.binary_score == "yes" else "no"
            
            if grade_res == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                relevant_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
        
        # Determine if transform query is needed using config
        if len(documents) == 0:
            should_transform = False
        else:
            search_config = get_search_config()
            relevance_ratio = len(relevant_docs) / len(documents)
            should_transform = relevance_ratio < search_config.relevance_threshold
        
        return relevant_docs, should_transform


class DocumentCleaner:
    """Specialized async processor for document cleaning."""
    
    @staticmethod
    async def clean_documents_async(
        documents: List[Document],
        question: str,
        chain: Any
    ) -> List[Document]:
        """
        Clean documents asynchronously.
        
        Returns:
            List of cleaned documents
        """
        def error_handler(error: Exception, doc: Document):
            """Handle cleaning errors by returning original document."""
            print(f"Document cleaning failed, using original: {error}")
            return doc
        
        async def clean_single_document(question: str, doc: Document, chain: Any):
            """Clean a single document."""
            print("Cleaning document")
            result = await asyncio.to_thread(
                chain.invoke,
                {"question": question, "context": doc.page_content}
            )
            return Document(page_content=result)
        
        # Process documents
        cleaned_docs = await AsyncDocumentProcessor.process_documents_parallel(
            documents=documents,
            processing_function=clean_single_document,
            question=question,
            chain=chain,
            error_handler=error_handler
        )
        
        return cleaned_docs


# Legacy compatibility - use centralized config instead
class AsyncProcessorConfig:
    """DEPRECATED: Use centralized config system instead."""
    
    @classmethod
    def get_optimal_concurrency(cls, num_documents: int) -> int:
        """DEPRECATED: Use get_async_config().get_optimal_concurrency() instead."""
        return get_async_config().get_optimal_concurrency(num_documents)
"""
Robust RAG (Retrieval-Augmented Generation) system
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import config
from embedding_systems import RobustEmbeddingSystem, create_embedding_system
from document_processor import SimpleDocumentProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    """Handle document retrieval for RAG system"""
    
    def __init__(self, 
                 collection_name: str = None,
                 embedding_system: RobustEmbeddingSystem = None):
        self.collection_name = collection_name or config.rag.collection_name
        self.embedding_system = embedding_system
        self.document_processor = SimpleDocumentProcessor(self.collection_name)
        self.corpus_documents = []
        self.corpus_embeddings = None
        self.is_initialized = False
        
    def initialize(self, preferred_embedding_method: str = "auto") -> bool:
        """Initialize the RAG retriever"""
        try:
            # Initialize document processor
            if not self.document_processor.initialize():
                logger.error("Failed to initialize document processor")
                return False
            
            # Initialize embedding system if not provided
            if self.embedding_system is None:
                self.embedding_system = create_embedding_system(
                    config.model.embedding_model,
                    preferred_embedding_method
                )
            
            # Load corpus documents
            self._load_corpus_documents()
            
            self.is_initialized = True
            logger.info(f"RAG retriever initialized with {len(self.corpus_documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG retriever: {e}")
            return False
    
    def _load_corpus_documents(self):
        """Load all documents from the collection"""
        try:
            result = self.document_processor.get_all_documents()
            self.corpus_documents = result.get("documents", [])
            self.corpus_metadatas = result.get("metadatas", [])
            
            logger.info(f"Loaded {len(self.corpus_documents)} documents from collection")
            
        except Exception as e:
            logger.error(f"Failed to load corpus documents: {e}")
            self.corpus_documents = []
            self.corpus_metadatas = []
    
    def _prepare_embeddings_for_search(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare embeddings for similarity search"""
        if not self.corpus_documents:
            return np.array([]), np.array([])
        
        # Handle different embedding systems
        if self.embedding_system.active_system.system_type == "TF-IDF":
            # For TF-IDF, need to fit on corpus + query
            query_embedding = self.embedding_system.encode_with_corpus_fallback([query], self.corpus_documents)
            
            # Get corpus embeddings (all but the last which is query)
            all_embeddings = self.embedding_system.encode_with_corpus_fallback(
                self.corpus_documents + [query], 
                self.corpus_documents
            )
            corpus_embeddings = all_embeddings[:-1]  # Exclude query embedding
            
        else:
            # For semantic embeddings (SentenceTransformers, Transformers)
            query_embedding = self.embedding_system.encode([query])
            
            # Get or compute corpus embeddings
            if self.corpus_embeddings is None:
                logger.info("Computing corpus embeddings...")
                self.corpus_embeddings = self.embedding_system.encode(self.corpus_documents)
            
            corpus_embeddings = self.corpus_embeddings
        
        return query_embedding, corpus_embeddings
    
    def retrieve_relevant_documents(self, 
                                  query: str,
                                  top_k: int = None,
                                  similarity_threshold: float = None,
                                  max_context_length: int = None) -> Tuple[str, List[Dict]]:
        """Retrieve relevant documents for a query"""
        
        if not self.is_initialized:
            logger.error("RAG retriever not initialized")
            return "", []
        
        if not self.corpus_documents:
            logger.warning("No documents in corpus")
            return "", []
        
        # Use config defaults if not specified
        top_k = top_k or config.rag.top_k_documents
        similarity_threshold = similarity_threshold or config.rag.similarity_threshold
        max_context_length = max_context_length or config.rag.max_context_length
        
        try:
            # Prepare embeddings
            query_embedding, corpus_embeddings = self._prepare_embeddings_for_search(query)
            
            if query_embedding.size == 0 or corpus_embeddings.size == 0:
                return "", []
            
            # Calculate similarities
            similarities = self.embedding_system.calculate_similarity(query_embedding, corpus_embeddings)
            
            if similarities.ndim > 1:
                similarities = similarities.flatten()
            
            # Get top-k most similar documents
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            relevant_docs = []
            context_parts = []
            
            for i, idx in enumerate(top_indices):
                similarity_score = similarities[idx]
                
                # Filter by similarity threshold
                if similarity_score >= similarity_threshold:
                    doc = self.corpus_documents[idx]
                    metadata = self.corpus_metadatas[idx] if idx < len(self.corpus_metadatas) else {}
                    
                    # Limit document length for context
                    doc_text = doc[:max_context_length] if len(doc) > max_context_length else doc
                    context_parts.append(f"Document {i+1}: {doc_text}")
                    
                    relevant_docs.append({
                        "content": doc[:300] + "..." if len(doc) > 300 else doc,
                        "full_content": doc,
                        "score": float(similarity_score),
                        "metadata": metadata,
                        "index": int(idx)
                    })
            
            # Combine context
            context = "\n\n".join(context_parts)
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents for query")
            return context, relevant_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return "", []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        return {
            "total_documents": len(self.corpus_documents),
            "collection_name": self.collection_name,
            "embedding_system": self.embedding_system.get_system_info() if self.embedding_system else None,
            "is_initialized": self.is_initialized,
            "has_corpus_embeddings": self.corpus_embeddings is not None
        }

class RAGSystem:
    """Complete RAG system combining retrieval and generation"""
    
    def __init__(self, 
                 model_manager,
                 collection_name: str = None,
                 embedding_system: RobustEmbeddingSystem = None):
        self.model_manager = model_manager
        self.retriever = RAGRetriever(collection_name, embedding_system)
        self.conversation_history = []
        self.rag_cache = {}
        
    def initialize(self, preferred_embedding_method: str = "auto") -> bool:
        """Initialize the RAG system"""
        return self.retriever.initialize(preferred_embedding_method)
    
    def generate_rag_response(self,
                            user_question: str,
                            use_rag: bool = True,
                            include_history: bool = True,
                            **generation_kwargs) -> Dict[str, Any]:
        """Generate a response using RAG"""
        
        response_data = {
            "question": user_question,
            "answer": "",
            "rag_context": "",
            "sources": [],
            "system_info": {},
            "processing_time": 0
        }
        
        import time
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant context if RAG is enabled
            if use_rag and self.retriever.is_initialized:
                logger.info("Retrieving relevant documents...")
                rag_context, sources = self.retriever.retrieve_relevant_documents(user_question)
                response_data["rag_context"] = rag_context
                response_data["sources"] = sources
            
            # Step 2: Build the prompt
            prompt = self._build_prompt(
                user_question, 
                rag_context if use_rag else "",
                include_history
            )
            
            # Step 3: Generate response
            logger.info("Generating response...")
            if hasattr(self.model_manager, 'generate_with_cache'):
                answer = self.model_manager.generate_with_cache(prompt, **generation_kwargs)
            else:
                answer = self.model_manager.generate_response(prompt, **generation_kwargs)
            
            # Clean up the answer
            answer = self._clean_response(answer)
            response_data["answer"] = answer
            
            # Step 4: Update conversation history
            if include_history:
                self.conversation_history.append(f"User: {user_question}")
                self.conversation_history.append(f"Assistant: {answer}")
                
                # Keep history manageable
                if len(self.conversation_history) > 12:
                    self.conversation_history = self.conversation_history[-8:]
            
            # Step 5: Collect system info
            response_data["system_info"] = {
                "embedding_system": self.retriever.embedding_system.get_system_info() if self.retriever.embedding_system else None,
                "model_info": self.model_manager.get_model_info(),
                "rag_enabled": use_rag,
                "sources_found": len(response_data["sources"]),
                "context_length": len(response_data["rag_context"])
            }
            
            response_data["processing_time"] = time.time() - start_time
            
            return response_data
            
        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            response_data["answer"] = f"I encountered an error while processing your question: {str(e)}"
            response_data["processing_time"] = time.time() - start_time
            return response_data
    
    def _build_prompt(self, question: str, rag_context: str = "", include_history: bool = True) -> str:
        """Build the prompt for the language model"""
        
        # System prompt
        system_prompt = ("You are a helpful AI assistant with access to relevant documents. "
                        "Provide clear, accurate responses based on the available information.")
        
        # Conversation history
        conversation_context = ""
        if include_history and self.conversation_history:
            recent_history = self.conversation_history[-6:]  # Last 6 entries
            conversation_context = "\n".join(recent_history) + "\n"
        
        # Build final prompt
        if rag_context:
            prompt = f"""{system_prompt}

Based on the following relevant information from the documents, please answer the question:

Relevant Information:
{rag_context}

{conversation_context}Question: {question}

Please provide a comprehensive answer based on the information above. If the information doesn't fully answer the question, acknowledge what you can answer and what might need additional information.

Answer:"""
        else:
            prompt = f"""{system_prompt}

{conversation_context}Question: {question}

Answer:"""
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean and validate the generated response"""
        response = response.strip()
        
        # Basic validation
        if not response or len(response) < 5:
            response = "I understand your question, but I'm having trouble generating a complete response. Could you please rephrase your question?"
        
        return response
    
    def get_conversation_history(self) -> List[str]:
        """Get the current conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "retriever_stats": self.retriever.get_collection_stats(),
            "conversation_length": len(self.conversation_history),
            "model_info": self.model_manager.get_model_info(),
            "memory_usage": self.model_manager.get_memory_usage()
        }
        
        # Add cache stats if available
        if hasattr(self.model_manager, 'get_cache_stats'):
            stats["cache_stats"] = self.model_manager.get_cache_stats()
        
        return stats

def create_rag_system(model_manager,
                     collection_name: str = None,
                     preferred_embedding_method: str = "auto") -> RAGSystem:
    """Factory function to create and initialize RAG system"""
    
    rag_system = RAGSystem(model_manager, collection_name)
    
    if not rag_system.initialize(preferred_embedding_method):
        logger.error("Failed to initialize RAG system")
        return None
    
    logger.info("RAG system created and initialized successfully")
    return rag_system
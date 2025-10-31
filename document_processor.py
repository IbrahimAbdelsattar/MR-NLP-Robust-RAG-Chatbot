"""
Document processing and indexing for RAG system
"""
import os
import logging
from typing import List, Optional
from pathlib import Path

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ChromaDB
import chromadb

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handle document loading, processing, and indexing"""
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or config.rag.collection_name
        self.chroma_client = None
        self.chroma_collection = None
        self.vector_store = None
        self.storage_context = None
        self.index = None
        self.documents = []
        self.nodes = []
        
    def initialize_chroma(self) -> bool:
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.Client()
            
            # Get or create collection
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Found existing collection: {self.collection_name}")
            except Exception:
                logger.warning(f"Collection '{self.collection_name}' not found. Creating new one...")
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.collection_name
                )
            
            # Create vector store and storage context
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return False
    
    def load_pdf_documents(self, pdf_paths: List[str]) -> List[Document]:
        """Load PDF documents using PDFReader"""
        documents = []
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                continue
                
            try:
                logger.info(f"Loading PDF: {pdf_path}")
                reader = PDFReader()
                docs = reader.load_data(pdf_path)
                
                # Add metadata
                for doc in docs:
                    doc.metadata["source_file"] = os.path.basename(pdf_path)
                    doc.metadata["file_path"] = pdf_path
                
                documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} documents from {pdf_path}")
                
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf_path}: {e}")
                continue
        
        self.documents = documents
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def process_documents(self, documents: Optional[List[Document]] = None) -> List:
        """Process documents into nodes"""
        if documents is None:
            documents = self.documents
        
        if not documents:
            logger.error("No documents to process")
            return []
        
        try:
            # Initialize node parser
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=config.rag.chunk_size,
                chunk_overlap=config.rag.chunk_overlap
            )
            
            # Parse documents into nodes
            nodes = node_parser.get_nodes_from_documents(documents)
            
            # Add node metadata
            for i, node in enumerate(nodes):
                node.metadata["node_id"] = i
                node.metadata["chunk_size"] = len(node.text)
            
            self.nodes = nodes
            logger.info(f"Processed {len(nodes)} nodes from {len(documents)} documents")
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            return []
    
    def build_index(self, documents: Optional[List[Document]] = None, 
                   embedding_model: Optional[str] = None) -> bool:
        """Build vector store index"""
        if documents is None:
            documents = self.documents
        
        if not documents:
            logger.error("No documents to index")
            return False
        
        if not self.storage_context:
            logger.error("Storage context not initialized")
            return False
        
        try:
            # Setup embedding model
            if embedding_model is None:
                embedding_model = config.model.embedding_model
            
            embed_model = HuggingFaceEmbedding(model_name=embedding_model)
            Settings.embed_model = embed_model
            
            # Build index
            logger.info("Building vector store index...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                embed_model=embed_model,
                show_progress=True
            )
            
            # Get collection count
            collection_count = self.chroma_collection.count()
            logger.info(f"Index built successfully! Vectors in collection: {collection_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def get_collection_info(self) -> dict:
        """Get information about the ChromaDB collection"""
        if not self.chroma_collection:
            return {"count": 0, "error": "Collection not initialized"}
        
        try:
            count = self.chroma_collection.count()
            return {
                "count": count,
                "name": self.collection_name,
                "documents_loaded": len(self.documents),
                "nodes_processed": len(self.nodes)
            }
        except Exception as e:
            return {"count": 0, "error": str(e)}
    
    def query_collection(self, query: str, n_results: int = 3) -> dict:
        """Query the ChromaDB collection directly"""
        if not self.chroma_collection:
            return {"error": "Collection not initialized"}
        
        try:
            # This requires the embedding to be generated separately
            # For now, return collection info
            return self.get_collection_info()
        except Exception as e:
            return {"error": str(e)}
    
    def persist_index(self, persist_dir: Optional[str] = None) -> bool:
        """Persist the index to disk"""
        if not self.index:
            logger.error("No index to persist")
            return False
        
        if persist_dir is None:
            persist_dir = config.storage_dir
        
        try:
            # Ensure directory exists
            os.makedirs(persist_dir, exist_ok=True)
            
            # Persist index
            self.index.storage_context.persist(persist_dir=persist_dir)
            logger.info(f"Index persisted to {persist_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist index: {e}")
            return False
    
    def load_index(self, persist_dir: Optional[str] = None) -> bool:
        """Load index from disk"""
        if persist_dir is None:
            persist_dir = config.storage_dir
        
        if not os.path.exists(persist_dir):
            logger.error(f"Persist directory not found: {persist_dir}")
            return False
        
        try:
            from llama_index.core import load_index_from_storage
            
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
            self.storage_context = storage_context
            logger.info(f"Index loaded from {persist_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

class SimpleDocumentProcessor:
    """Simplified document processor for basic ChromaDB operations"""
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or config.rag.collection_name
        self.chroma_client = None
        self.chroma_collection = None
    
    def initialize(self) -> bool:
        """Initialize ChromaDB client and collection"""
        try:
            self.chroma_client = chromadb.Client()
            
            # Get or create collection
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
            except Exception:
                logger.info(f"Creating new collection: {self.collection_name}")
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.collection_name
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return False
    
    def get_collection(self):
        """Get the ChromaDB collection"""
        return self.chroma_collection
    
    def get_all_documents(self) -> dict:
        """Get all documents from the collection"""
        if not self.chroma_collection:
            return {"documents": [], "metadatas": []}
        
        try:
            return self.chroma_collection.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return {"documents": [], "metadatas": []}
    
    def count_documents(self) -> int:
        """Count documents in collection"""
        if not self.chroma_collection:
            return 0
        
        try:
            return self.chroma_collection.count()
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0

def setup_document_processor(pdf_paths: List[str], 
                           collection_name: str = None,
                           use_llamaindex: bool = True) -> tuple:
    """Factory function to set up document processor"""
    
    if use_llamaindex:
        # Use full LlamaIndex processor
        processor = DocumentProcessor(collection_name)
        
        # Initialize ChromaDB
        if not processor.initialize_chroma():
            logger.error("Failed to initialize ChromaDB")
            return None, None
        
        # Load and process documents
        documents = processor.load_pdf_documents(pdf_paths)
        if not documents:
            logger.error("No documents loaded")
            return processor, None
        
        # Build index
        if not processor.build_index(documents):
            logger.error("Failed to build index")
            return processor, None
        
        return processor, processor.index
    
    else:
        # Use simple processor
        processor = SimpleDocumentProcessor(collection_name)
        
        if not processor.initialize():
            logger.error("Failed to initialize simple processor")
            return None, None
        
        return processor, processor.get_collection()
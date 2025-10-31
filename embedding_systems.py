"""
Robust embedding systems with multiple fallback options
"""
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import List, Union, Optional
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore")

class BaseEmbeddingSystem(ABC):
    """Abstract base class for embedding systems"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.system_type = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the embedding model"""
        pass
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into embeddings"""
        pass
    
    def _ensure_list(self, texts: Union[str, List[str]]) -> List[str]:
        """Ensure input is a list of strings"""
        if isinstance(texts, str):
            return [texts]
        return texts

class SentenceTransformersEmbedding(BaseEmbeddingSystem):
    """SentenceTransformers embedding system"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.model_name = model_name
        self.system_type = "SentenceTransformers"
    
    def load_model(self) -> bool:
        """Load SentenceTransformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"SentenceTransformers loading failed: {e}")
            return False
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts using SentenceTransformers"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        texts = self._ensure_list(texts)
        return self.model.encode(texts, convert_to_numpy=True)

class TransformersEmbedding(BaseEmbeddingSystem):
    """Direct Transformers embedding with mean pooling"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.model_name = model_name
        self.system_type = "Transformers"
    
    def load_model(self) -> bool:
        """Load Transformers model directly"""
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            self.device = device
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Transformers loading failed: {e}")
            return False
    
    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts using direct Transformers"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        texts = self._ensure_list(texts)
        
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            model_output = self.model(**encoded)
            embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()

class TFIDFEmbedding(BaseEmbeddingSystem):
    """TF-IDF embedding system as final fallback"""
    
    def __init__(self, max_features: int = 5000):
        super().__init__()
        self.max_features = max_features
        self.system_type = "TF-IDF"
        self.fitted = False
        self.feature_names = None
    
    def load_model(self) -> bool:
        """Load TF-IDF vectorizer"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)  # Include bigrams
            )
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"TF-IDF loading failed: {e}")
            return False
    
    def fit(self, corpus: List[str]):
        """Fit the TF-IDF model on a corpus"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        self.model.fit(corpus)
        self.feature_names = self.model.get_feature_names_out()
        self.fitted = True
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts using TF-IDF"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        texts = self._ensure_list(texts)
        
        if not self.fitted:
            # If not fitted, fit on the input texts
            self.fit(texts)
        
        return self.model.transform(texts).toarray()
    
    def encode_with_corpus(self, texts: Union[str, List[str]], corpus: List[str]) -> np.ndarray:
        """Encode texts after fitting on a corpus"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        texts = self._ensure_list(texts)
        
        # Fit on corpus + texts to ensure all vocabulary is captured
        full_corpus = corpus + texts
        self.fit(full_corpus)
        
        # Return only the embeddings for the input texts
        return self.model.transform(texts).toarray()

class RobustEmbeddingSystem:
    """Robust embedding system with automatic fallbacks"""
    
    def __init__(self, primary_model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.primary_model = primary_model
        self.active_system = None
        self.embedding_systems = []
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all embedding systems in order of preference"""
        self.embedding_systems = [
            SentenceTransformersEmbedding(self.primary_model),
            TransformersEmbedding(self.primary_model),
            TFIDFEmbedding()
        ]
    
    def load_best_available_system(self, preferred_method: str = "auto") -> bool:
        """Load the best available embedding system"""
        
        if preferred_method == "Transformers Fallback":
            # Try Transformers first, then TF-IDF
            systems_to_try = [1, 2]  # Skip SentenceTransformers
        elif preferred_method == "Simple TF-IDF":
            # Use only TF-IDF
            systems_to_try = [2]
        else:  # "Auto (Try SentenceTransformers first)" or any other
            # Try all systems in order
            systems_to_try = [0, 1, 2]
        
        for idx in systems_to_try:
            system = self.embedding_systems[idx]
            print(f"Attempting to load {system.system_type} embedding system...")
            
            if system.load_model():
                self.active_system = system
                print(f"✅ Successfully loaded {system.system_type} embedding system")
                return True
            else:
                print(f"❌ Failed to load {system.system_type} embedding system")
        
        print("❌ All embedding systems failed to load")
        return False
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts using the active embedding system"""
        if self.active_system is None:
            raise RuntimeError("No embedding system is loaded")
        
        return self.active_system.encode(texts)
    
    def encode_with_corpus_fallback(self, texts: Union[str, List[str]], corpus: List[str]) -> np.ndarray:
        """Encode texts with corpus fallback for TF-IDF"""
        if self.active_system is None:
            raise RuntimeError("No embedding system is loaded")
        
        if isinstance(self.active_system, TFIDFEmbedding):
            return self.active_system.encode_with_corpus(texts, corpus)
        else:
            return self.active_system.encode(texts)
    
    def get_system_info(self) -> dict:
        """Get information about the active system"""
        if self.active_system is None:
            return {"system_type": "None", "is_loaded": False}
        
        return {
            "system_type": self.active_system.system_type,
            "is_loaded": self.active_system.is_loaded,
            "model_name": getattr(self.active_system, 'model_name', 'N/A')
        }
    
    def calculate_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between embeddings"""
        if self.active_system.system_type == "TF-IDF":
            # Use sklearn's cosine similarity for TF-IDF
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(embeddings1, embeddings2)
        else:
            # Use numpy for dense embeddings
            # Normalize embeddings
            embeddings1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
            embeddings2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
            
            # Calculate cosine similarity
            return np.dot(embeddings1_norm, embeddings2_norm.T)

def create_embedding_system(model_name: str = "sentence-transformers/all-mpnet-base-v2", 
                          preferred_method: str = "auto") -> RobustEmbeddingSystem:
    """Factory function to create and initialize a robust embedding system"""
    system = RobustEmbeddingSystem(model_name)
    
    if not system.load_best_available_system(preferred_method):
        raise RuntimeError("Failed to load any embedding system")
    
    return system
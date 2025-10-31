"""
Configuration settings for MR NLP Robust RAG Chatbot
"""
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for language models"""
    qwen_model_id: str = "Qwen/Qwen1.5-1.8B"
    whisper_model_size: str = "tiny"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    fallback_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Generation parameters
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.9
    context_window: int = 2048
    
    # Quantization settings
    use_4bit: bool = True
    use_cpu_only: bool = False

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    collection_name: str = "adaptive_education_docs"
    similarity_threshold: float = 0.7
    top_k_documents: int = 3
    max_context_length: int = 1000
    chunk_size: int = 256
    chunk_overlap: int = 10
    max_triplets_per_chunk: int = 10

@dataclass
class StreamlitConfig:
    """Configuration for Streamlit interface"""
    page_title: str = "MR NLP Robust RAG Chatbot"
    page_icon: str = "📚"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Server settings
    server_port: int = 8501
    server_address: str = "0.0.0.0"
    max_upload_size: int = 200

@dataclass
class AudioConfig:
    """Configuration for audio features"""
    tts_speed: str = "normal"  # slow, normal, fast
    audio_lang: str = "en"
    enable_tts: bool = True
    enable_voice_input: bool = True

@dataclass
class AppConfig:
    """Main application configuration"""
    model: ModelConfig = ModelConfig()
    rag: RAGConfig = RAGConfig()
    streamlit: StreamlitConfig = StreamlitConfig()
    audio: AudioConfig = AudioConfig()
    
    # Environment settings
    llama_cloud_api_key: str = ""
    ngrok_auth_token: str = ""
    
    # Performance settings
    enable_cache: bool = True
    max_cache_size: int = 50
    cache_ttl: int = 3600  # seconds
    
    # File paths
    data_dir: str = "./data"
    models_dir: str = "./models"
    embeddings_dir: str = "./embeddings"
    storage_dir: str = "./storage"
    temp_dir: str = "./temp"

    def __post_init__(self):
        """Initialize environment variables and create directories"""
        # Load from environment
        self.llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY", "")
        self.ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN", "")
        
        # Create necessary directories
        directories = [
            self.data_dir,
            self.models_dir,
            self.embeddings_dir,
            self.storage_dir,
            self.temp_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = AppConfig()

# PDF file paths (update these with your actual file paths)
PDF_PATHS = [
    "./data/Hands-On_Large_Language_Models_Jay_Alammar.pdf",
    "./data/practical-natural-language-processing.pdf",
    "./data/speech_and_language_processing.pdf",
]

# Embedding method options
EMBEDDING_METHODS = [
    "Auto (Try SentenceTransformers first)",
    "Transformers Fallback", 
    "Simple TF-IDF"
]

# Supported languages for TTS
TTS_LANGUAGES = {
    "English": "en",
    "Arabic": "ar", 
    "French": "fr",
    "Spanish": "es"
}

# Device configuration
def get_device_config() -> Dict[str, Any]:
    """Get optimal device configuration"""
    import torch
    
    if config.model.use_cpu_only:
        return {
            "device": "cpu",
            "torch_dtype": torch.float32,
            "device_map": None
        }
    
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
    
    return {
        "device": "cpu", 
        "torch_dtype": torch.float32,
        "device_map": None
    }
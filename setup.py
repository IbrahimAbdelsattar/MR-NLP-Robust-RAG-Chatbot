"""
Setup script for MR NLP Robust RAG Chatbot
"""
import os
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "models", 
        "embeddings",
        "storage",
        "temp",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        logger.info("Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_environment_file():
    """Create .env file template"""
    env_content = """# MR NLP RAG Chatbot Environment Variables

# LlamaIndex Cloud API Key (optional)
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Ngrok Auth Token (for public deployment)
NGROK_AUTH_TOKEN=your_ngrok_auth_token_here

# HuggingFace Token (optional, for private models)
HUGGINGFACE_TOKEN=your_hf_token_here

# Model Configuration
QWEN_MODEL_ID=Qwen/Qwen1.5-1.8B
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
WHISPER_MODEL=tiny

# Database Configuration  
CHROMA_COLLECTION_NAME=adaptive_education_docs

# Performance Settings
USE_4BIT_QUANTIZATION=true
ENABLE_CACHE=true
MAX_CACHE_SIZE=50
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        logger.info("Created .env file template")
    else:
        logger.info(".env file already exists")

def create_sample_data():
    """Create sample data structure"""
    sample_readme = """# Data Directory

Place your PDF documents in this directory for the RAG system to process.

## Supported Files:
- PDF documents (.pdf)
- Text files (.txt) 
- Markdown files (.md)

## Current Files:
The system expects these files (update paths in config.py if different):
- Hands-On_Large_Language_Models_Jay_Alammar.pdf
- practical-natural-language-processing.pdf  
- speech_and_language_processing.pdf

## Adding New Documents:
1. Copy your PDF files to this directory
2. Update the PDF_PATHS list in config.py
3. Restart the application to reindex documents
"""
    
    data_dir = Path("data")
    readme_path = data_dir / "README.md"
    
    if not readme_path.exists():
        with open(readme_path, "w") as f:
            f.write(sample_readme)
        logger.info("Created data directory README")

def create_run_script():
    """Create run script for easy startup"""
    run_script_content = """#!/usr/bin/env python3
\"\"\"
Run script for MR NLP Robust RAG Chatbot
\"\"\"
import subprocess
import sys
import os

def main():
    # Set environment variables
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    
    # Run Streamlit app
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501", 
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("🚀 Starting MR NLP Robust RAG Chatbot...")
    print("📱 Open browser to: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    
    with open("run.py", "w") as f:
        f.write(run_script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("run.py", 0o755)
    
    logger.info("Created run.py script")

def create_docker_files():
    """Create Docker configuration files"""
    
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    ffmpeg \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models embeddings storage temp logs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
"""
    
    docker_compose_content = """version: '3.8'

services:
  rag-chatbot:
    build: .
    container_name: mr-nlp-rag-chatbot
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./storage:/app/storage
      - ./models:/app/models
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_ENABLE_CORS=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    logger.info("Created Docker configuration files")

def create_readme():
    """Create comprehensive README.md"""
    readme_content = """# MR NLP Robust RAG Chatbot

A state-of-the-art **Retrieval-Augmented Generation (RAG)** chatbot with multiple embedding system fallbacks and advanced language model integration.

## 🚀 Features

- **Multi-System Embedding Fallbacks**: SentenceTransformers → Transformers → TF-IDF
- **Advanced Language Model**: Qwen 1.5-1.8B with 4-bit quantization support
- **Robust Document Processing**: ChromaDB vector database integration
- **Voice Interaction**: Speech-to-text and text-to-speech capabilities
- **Smart Caching**: Response caching for improved performance
- **Flexible Configuration**: Adjustable similarity thresholds and retrieval parameters
- **Modern UI**: Streamlit-based interactive interface

## 📋 Requirements

- Python 3.8+
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for better performance)
- FFmpeg (for audio processing)

## ⚡ Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd mr-nlp-rag-chatbot
   python setup.py
   ```

2. **Add Your Documents**:
   - Place PDF files in the `data/` directory
   - Update `PDF_PATHS` in `config.py` if needed

3. **Run the Application**:
   ```bash
   python run.py
   ```
   
4. **Open Browser**: Navigate to `http://localhost:8501`

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t mr-nlp-rag-chatbot .
docker run -p 8501:8501 -v ./data:/app/
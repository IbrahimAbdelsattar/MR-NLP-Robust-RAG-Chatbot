# MR NLP Robust RAG Chatbot

A state-of-the-art **Retrieval-Augmented Generation (RAG)** chatbot developed during an NLP internship at NTI. This project combines advanced language models with robust knowledge retrieval systems to deliver accurate, context-aware AI responses with multiple embedding system fallbacks.

## Features

**Multi-System Embedding Fallbacks**
- Primary: SentenceTransformers for semantic understanding
- Fallback: Direct Transformers with mean pooling
- Final: TF-IDF for keyword-based matching
- Automatic system selection based on compatibility

**Advanced Language Model Integration**
- Qwen 1.5-1.8B model with 4-bit quantization support
- Optimized for both CPU and GPU deployment
- Smart caching for improved response times
- Configurable generation parameters

**Robust Document Processing**
- ChromaDB vector database integration
- PDF document parsing and chunking
- Semantic similarity search with adjustable thresholds
- Source attribution with relevance scores

**Voice Interaction Capabilities**
- Whisper-powered speech-to-text
- Multi-language text-to-speech (gTTS)
- Real-time audio processing
- Configurable audio settings

**Modern Interface**
- Streamlit-based interactive web application
- Real-time system status monitoring
- Performance metrics and analytics
- Responsive design with custom styling

## Tech Stack

**Core Technologies**
- Python 3.8+
- PyTorch for deep learning
- Transformers library for language models
- ChromaDB for vector storage

**NLP & ML Libraries**
- SentenceTransformers
- Hugging Face Transformers
- LlamaIndex for RAG orchestration
- scikit-learn for TF-IDF fallback

**Audio Processing**
- OpenAI Whisper for speech recognition
- Google Text-to-Speech (gTTS)
- Audio recording with streamlit-audiorecorder

**Web Interface**
- Streamlit for interactive UI
- Custom CSS styling
- Real-time status updates

## Installation

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd mr-nlp-rag-chatbot
   ```

2. **Run automated setup:**
   ```bash
   python setup.py
   ```

3. **Add your documents:**
   - Place PDF files in the `data/` directory
   - Update file paths in `config.py` if needed

4. **Launch the application:**
   ```bash
   python run.py
   ```

5. **Access the interface:**
   Open your browser to `http://localhost:8501`

### Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data models embeddings storage temp logs

# Run the application
streamlit run app.py
```

### Docker Deployment

For containerized deployment:

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t mr-nlp-rag-chatbot .
docker run -p 8501:8501 -v ./data:/app/data mr-nlp-rag-chatbot
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Optional: LlamaIndex Cloud API Key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key

# Optional: Ngrok token for public deployment
NGROK_AUTH_TOKEN=your_ngrok_auth_token

# Optional: HuggingFace token for private models
HUGGINGFACE_TOKEN=your_hf_token
```

### Model Configuration

Edit `config.py` to customize:

```python
# Language Model Settings
model.qwen_model_id = "Qwen/Qwen1.5-1.8B"
model.max_new_tokens = 64
model.temperature = 0.8

# RAG Configuration
rag.similarity_threshold = 0.7
rag.top_k_documents = 3
rag.max_context_length = 1000

# Performance Options
model.use_4bit = True  # Enable quantization
enable_cache = True    # Response caching
```

### Document Setup

1. **Add PDF documents** to the `data/` directory
2. **Update PDF paths** in `config.py`:
   ```python
   PDF_PATHS = [
       "./data/your_document1.pdf",
       "./data/your_document2.pdf",
       "./data/your_document3.pdf"
   ]
   ```
3. **Restart the application** to reindex documents

## Usage

### Basic Chat Interface

1. **Text Input**: Type questions directly in the text field
2. **Voice Input**: Click the record button and speak your question
3. **RAG Toggle**: Enable/disable document retrieval in the sidebar
4. **Settings**: Adjust temperature, similarity threshold, and other parameters

### Advanced Features

**Embedding System Selection**
- Choose between SentenceTransformers, Transformers, or TF-IDF
- System automatically falls back if primary method fails
- Real-time switching available in sidebar

**Response Optimization**
- Enable caching for faster repeated queries
- Adjust similarity threshold for relevance control
- Configure max tokens and temperature for response style

**Source Attribution**
- View retrieved document chunks with similarity scores
- Expand sources to see full document content
- Track which documents contributed to responses

**Performance Monitoring**
- Real-time memory usage tracking
- Processing time measurements
- Cache hit rate statistics
- System resource monitoring

## Project Structure

```
mr-nlp-rag-chatbot/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration management
├── model_manager.py          # LLM and Whisper handling
├── embedding_systems.py      # Multi-fallback embeddings
├── rag_system.py            # RAG retrieval and generation
├── document_processor.py    # PDF processing and indexing
├── utils.py                 # System utilities and monitoring
├── setup.py                 # Automated setup script
├── run.py                   # Application launcher
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-container setup
├── data/                   # Document storage
├── models/                 # Model cache
├── storage/               # Index persistence
├── temp/                  # Temporary files
└── logs/                  # Application logs
```

## System Requirements

**Minimum Requirements**
- Python 3.8 or higher
- 8GB RAM (16GB recommended)
- 10GB free disk space
- Modern web browser

**Recommended Setup**
- CUDA-capable GPU with 8GB+ VRAM
- 16GB+ system RAM
- SSD storage for better performance
- Stable internet connection for model downloads

**Audio Requirements (Optional)**
- Microphone for voice input
- Speakers or headphones for TTS output
- FFmpeg installed for audio processing

## Embedding System Architecture

The system implements a three-tier fallback approach for maximum compatibility:

### Tier 1: SentenceTransformers (Primary)
- Best semantic understanding and context awareness
- Pre-trained on large-scale sentence similarity tasks
- Requires compatible transformers and sentence-transformers versions

### Tier 2: Direct Transformers (Fallback)
- Uses mean pooling on transformer hidden states
- Works with basic transformers installation
- Good semantic understanding with broader compatibility

### Tier 3: TF-IDF (Final Fallback)
- Traditional keyword-based vector similarity
- Always available, no deep learning dependencies
- Effective for keyword matching and term-based queries

The system automatically detects which embedding method works in your environment and falls back gracefully if needed.

## Troubleshooting

### Common Issues

**Model Loading Failures**
```bash
# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available/1024**3:.1f}GB')"

# Try CPU-only mode
# In sidebar: Enable "Force CPU Only"

# Check GPU memory (if using CUDA)
python -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB' if torch.cuda.is_available() else 'No GPU')"
```

**Embedding System Errors**
- Switch embedding method in sidebar settings
- TF-IDF fallback should always work
- Check HuggingFace model availability
- Verify internet connection for model downloads

**Audio Processing Issues**
```bash
# Install FFmpeg (required for audio)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/
```

**Memory Issues**
- Enable 4-bit quantization in sidebar
- Reduce max context length
- Use fewer retrieved documents
- Switch to CPU-only mode

**ChromaDB Connection Problems**
- Check write permissions in project directory
- Ensure no other processes are using the database
- Delete and recreate storage directory if corrupted

### Performance Optimization

**For GPU Users:**
- Enable 4-bit quantization to reduce memory usage
- Use larger batch sizes for document processing
- Keep temperature lower for more consistent responses

**For CPU Users:**
- Reduce context window size
- Use fewer retrieved documents (top_k = 1-2)
- Enable response caching
- Consider using TF-IDF embedding method

**For Production:**
- Use Docker deployment for resource isolation
- Set up proper logging and monitoring
- Configure appropriate memory limits
- Use external vector database for large document collections

## Development

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with appropriate tests
4. Update documentation as needed
5. Submit a pull request with detailed description

### Testing

```bash
# Run system validation
python -c "from utils import validate_system_requirements; print(validate_system_requirements())"

# Test embedding systems
python -c "from embedding_systems import create_embedding_system; system = create_embedding_system(); print(system.get_system_info())"

# Check model loading
python -c "from model_manager import create_model_manager; manager = create_model_manager(); print(manager.load_qwen_model())"
```

### Adding New Features

The modular architecture makes it easy to extend:

- **New Embedding Systems**: Add to `embedding_systems.py`
- **Different LLMs**: Extend `model_manager.py`
- **Additional Document Types**: Modify `document_processor.py`
- **UI Enhancements**: Update `app.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed during an NLP internship at the National Training Institute (NTI). Special thanks to:

- **NTI** for providing the internship opportunity and resources
- **Qwen Team** at Alibaba for the excellent language model
- **ChromaDB** team for the vector database capabilities
- **Streamlit** team for the interactive web framework
- **Open source community** for the foundational libraries and tools

## Support and Contact

For technical support, feature requests, or collaboration opportunities:

- Check the troubleshooting section above
- Review system status indicators in the application
- Examine logs in the `logs/` directory for detailed error information
- Submit issues through the project repository

**Built for Advanced NLP Applications with Robust Error Handling and Professional Architecture**

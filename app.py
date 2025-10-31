"""
Main Streamlit application for MR NLP Robust RAG Chatbot
"""
import streamlit as st
import os
import time
import threading
import tempfile
from datetime import datetime
from gtts import gTTS
import whisper
from st_audiorec import st_audiorec
import logging

# Import our modules
from config import config, EMBEDDING_METHODS, TTS_LANGUAGES
from model_manager import create_model_manager, load_all_models
from rag_system import create_rag_system
from embedding_systems import create_embedding_system

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Configuration
st.set_page_config(
    page_title=config.streamlit.page_title,
    page_icon=config.streamlit.page_icon,
    layout=config.streamlit.layout,
    initial_sidebar_state=config.streamlit.initial_sidebar_state
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
    }
    .response-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    .rag-context {
        background: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
        font-size: 0.9em;
    }
    .source-box {
        background: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #ffc107;
        font-size: 0.85em;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        width: 100%;
    }
    .loading-text {
        color: #667eea;
        font-weight: bold;
        text-align: center;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📚 MR NLP Robust RAG Chatbot")
st.subheader("Advanced Retrieval-Augmented Generation with Multi-System Fallbacks")
st.markdown("---")

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.model_manager = None
        st.session_state.rag_system = None
        st.session_state.loading_complete = False
        st.session_state.conversation_history = []
        st.session_state.system_status = {
            "models_loaded": False,
            "rag_initialized": False,
            "embedding_system": None
        }

initialize_session_state()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Model Settings
    st.subheader("🧠 Qwen Model Settings")
    model_temp = st.slider("Temperature", 0.1, 2.0, config.model.temperature, 0.1)
    max_tokens = st.slider("Max New Tokens", 32, 256, config.model.max_new_tokens, 16)
    top_p = st.slider("Top P", 0.1, 1.0, config.model.top_p, 0.1)
    
    # RAG Settings
    st.subheader("📚 RAG Configuration")
    use_rag = st.checkbox("Enable RAG (Knowledge Base)", value=True)
    if use_rag:
        rag_top_k = st.slider("Retrieved Documents", 1, 10, config.rag.top_k_documents, 1)
        rag_threshold = st.slider("Similarity Threshold", 0.1, 1.0, config.rag.similarity_threshold, 0.1)
        max_context = st.slider("Max Context Length", 500, 2000, config.rag.max_context_length, 100)
        show_sources = st.checkbox("Show Source Documents", value=True)
    
    # Embedding Method Selection
    st.subheader("🔤 Embedding System")
    embedding_method = st.selectbox(
        "Choose Embedding Method:",
        EMBEDDING_METHODS,
        help="Different embedding systems for compatibility"
    )
    
    # Performance Settings
    st.subheader("⚡ Performance Options")
    use_4bit = st.checkbox("Use 4-bit Quantization", value=config.model.use_4bit)
    use_cpu_only = st.checkbox("Force CPU Only", value=config.model.use_cpu_only)
    enable_cache = st.checkbox("Enable Response Cache", value=config.enable_cache)
    
    # Audio Settings
    st.subheader("🎵 Audio Configuration")
    enable_tts = st.checkbox("Enable Text-to-Speech", value=config.audio.enable_tts)
    enable_voice = st.checkbox("Enable Voice Input", value=config.audio.enable_voice_input)
    if enable_tts or enable_voice:
        tts_speed = st.selectbox("TTS Speed", ["slow", "normal", "fast"], index=1)
        audio_lang_name = st.selectbox("Language", list(TTS_LANGUAGES.keys()), index=0)
        audio_lang = TTS_LANGUAGES[audio_lang_name]

@st.cache_resource(show_spinner=False)
def load_system_components():
    """Load and initialize all system components"""
    try:
        # Update config based on UI settings
        config.model.temperature = model_temp
        config.model.max_new_tokens = max_tokens
        config.model.top_p = top_p
        config.model.use_4bit = use_4bit
        config.model.use_cpu_only = use_cpu_only
        config.enable_cache = enable_cache
        
        # Create model manager
        logger.info("Creating model manager...")
        model_manager = create_model_manager(advanced=enable_cache)
        
        # Load models
        logger.info("Loading language models...")
        models_success, model_results = load_all_models(model_manager, load_whisper=enable_voice)
        
        if not model_results.get("qwen", False):
            raise Exception("Failed to load Qwen model")
        
        # Initialize RAG system if enabled
        rag_system = None
        if use_rag:
            logger.info("Initializing RAG system...")
            rag_system = create_rag_system(
                model_manager,
                config.rag.collection_name,
                embedding_method
            )
            
            if rag_system is None:
                logger.warning("RAG system initialization failed, continuing without RAG")
        
        return model_manager, rag_system, model_results
        
    except Exception as e:
        logger.error(f"Failed to load system components: {e}")
        raise

def display_system_status(model_manager, rag_system, model_results):
    """Display current system status"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if model_results.get("qwen", False):
            st.markdown('<p class="status-success">✅ Qwen Model Ready</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ Qwen Model Failed</p>', unsafe_allow_html=True)
    
    with col2:
        if use_rag and rag_system and rag_system.retriever.is_initialized:
            stats = rag_system.retriever.get_collection_stats()
            doc_count = stats.get("total_documents", 0)
            embedding_type = stats.get("embedding_system", {}).get("system_type", "Unknown")
            st.markdown(f'<p class="status-success">✅ RAG Ready ({doc_count} docs)</p>', unsafe_allow_html=True)
            st.caption(f"Using: {embedding_type}")
        elif not use_rag:
            st.info("📚 RAG Disabled")
        else:
            st.markdown('<p class="status-error">❌ RAG Failed</p>', unsafe_allow_html=True)
    
    with col3:
        if enable_voice and model_results.get("whisper", False):
            st.markdown('<p class="status-success">✅ Voice Ready</p>', unsafe_allow_html=True)
        elif not enable_voice:
            st.info("🔇 Voice Disabled")
        else:
            st.markdown('<p class="status-error">❌ Voice Failed</p>', unsafe_allow_html=True)
    
    with col4:
        model_info = model_manager.get_model_info()
        device = model_info.get("device", "unknown")
        if use_4bit and device == "cuda":
            st.info("💾 4-bit GPU")
        elif device == "cuda":
            st.info("🚀 GPU Mode")
        else:
            st.info("💾 CPU Mode")

# System Initialization
if not st.session_state.loading_complete:
    with st.spinner("🚀 Initializing MR NLP RAG Chatbot..."):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        try:
            status_placeholder.markdown('<p class="loading-text">📦 Loading system components...</p>', unsafe_allow_html=True)
            progress_bar.progress(20)
            
            model_manager, rag_system, model_results = load_system_components()
            
            progress_bar.progress(60)
            status_placeholder.markdown('<p class="loading-text">🔧 Configuring systems...</p>', unsafe_allow_html=True)
            
            # Store in session state
            st.session_state.model_manager = model_manager
            st.session_state.rag_system = rag_system
            st.session_state.system_status = {
                "models_loaded": True,
                "rag_initialized": rag_system is not None,
                "model_results": model_results
            }
            
            progress_bar.progress(100)
            status_placeholder.markdown('<p class="loading-text">✅ System ready!</p>', unsafe_allow_html=True)
            
            st.session_state.loading_complete = True
            time.sleep(1)
            
            # Clear loading indicators
            progress_bar.empty()
            status_placeholder.empty()
            
        except Exception as e:
            progress_bar.empty()
            status_placeholder.empty()
            st.error(f"❌ System initialization failed: {e}")
            st.info("Try refreshing the page or adjusting settings in the sidebar.")
            st.stop()

# Display system status
if st.session_state.loading_complete:
    display_system_status(
        st.session_state.model_manager,
        st.session_state.rag_system,
        st.session_state.system_status["model_results"]
    )
    
    st.markdown("---")

# Main Chat Interface
st.markdown("### 💬 Chat Interface")

# Input method selection
input_mode = st.radio("Choose input method:", ["💬 Text Input", "🎤 Voice Input"], horizontal=True)

user_question = None

if input_mode == "💬 Text Input":
    user_question = st.text_input(
        "Ask your question:",
        placeholder="What would you like to know from your documents?",
        key="text_input"
    )

elif input_mode == "🎤 Voice Input" and enable_voice:
    if st.session_state.system_status["model_results"].get("whisper", False):
        st.write("**🎤 Record your question:**")
        wav_audio_data = st_audiorec(key="audio_recorder")
        
        if wav_audio_data:
            with st.spinner("🎯 Processing audio..."):
                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(wav_audio_data)
                        audio_path = tmp_file.name
                    
                    # Transcribe using model manager
                    result = st.session_state.model_manager.transcribe_audio(audio_path, audio_lang)
                    user_question = result.get("text", "").strip()
                    
                    if user_question:
                        st.success(f"✅ Transcribed: {user_question}")
                    
                    # Cleanup
                    os.unlink(audio_path)
                    
                except Exception as e:
                    st.error(f"❌ Audio processing failed: {e}")
    else:
        st.warning("🔇 Voice recognition not available. Check system status above.")

elif input_mode == "🎤 Voice Input":
    st.warning("🔇 Voice input is disabled in settings.")

# Process user question
if user_question and user_question.strip():
    if st.button("🚀 Ask RAG System", use_container_width=True):
        start_time = time.time()
        
        with st.spinner("🔍 Processing your question..."):
            try:
                # Generate RAG response
                if st.session_state.rag_system and use_rag:
                    response_data = st.session_state.rag_system.generate_rag_response(
                        user_question,
                        use_rag=True,
                        max_new_tokens=max_tokens,
                        temperature=model_temp,
                        top_p=top_p
                    )
                else:
                    # Direct model response without RAG
                    prompt = f"Question: {user_question}\n\nAnswer:"
                    if hasattr(st.session_state.model_manager, 'generate_with_cache'):
                        answer = st.session_state.model_manager.generate_with_cache(
                            prompt,
                            max_new_tokens=max_tokens,
                            temperature=model_temp,
                            top_p=top_p
                        )
                    else:
                        answer = st.session_state.model_manager.generate_response(
                            prompt,
                            max_new_tokens=max_tokens,
                            temperature=model_temp,
                            top_p=top_p
                        )
                    
                    response_data = {
                        "question": user_question,
                        "answer": answer,
                        "rag_context": "",
                        "sources": [],
                        "system_info": {"rag_enabled": False},
                        "processing_time": time.time() - start_time
                    }
                
                # Display results
                st.markdown("---")
                
                # Show RAG sources if available
                if response_data.get("sources") and show_sources and use_rag:
                    st.subheader("📚 Retrieved Knowledge")
                    for i, source in enumerate(response_data["sources"]):
                        with st.expander(f"📄 Source {i+1} (Score: {source['score']:.3f})", expanded=False):
                            st.markdown(f'<div class="source-box">{source["content"]}</div>', unsafe_allow_html=True)
                            if source.get('metadata'):
                                st.caption(f"Metadata: {source['metadata']}")
                            if st.button(f"Show Full Content", key=f"full_{i}"):
                                st.text_area(f"Full Document {i+1}:", source['full_content'], height=200)
                    st.markdown("---")
                
                # Main response
                st.subheader("🤖 AI Response")
                st.markdown(f'<div class="response-container">{response_data["answer"]}</div>', unsafe_allow_html=True)
                
                # Generate TTS if enabled
                if enable_tts:
                    try:
                        with st.spinner("🔊 Generating audio..."):
                            tts_text = response_data["answer"][:500] + "..." if len(response_data["answer"]) > 500 else response_data["answer"]
                            
                            tts = gTTS(text=tts_text, lang=audio_lang, slow=(tts_speed=="slow"))
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                tts.save(tmp_file.name)
                                st.audio(tmp_file.name, format="audio/mp3")
                                
                                # Schedule cleanup
                                def cleanup_audio():
                                    time.sleep(60)
                                    try:
                                        os.unlink(tmp_file.name)
                                    except:
                                        pass
                                
                                threading.Thread(target=cleanup_audio, daemon=True).start()
                                
                    except Exception as e:
                        st.warning(f"🔇 Text-to-speech failed: {e}")
                
                # Response metadata
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"⏱️ **Processing Time:** {response_data['processing_time']:.2f}s")
                    st.info(f"📝 **Response Length:** {len(response_data['answer'])} chars")
                
                with col2:
                    if response_data.get("sources"):
                        st.info(f"📚 **Sources Used:** {len(response_data['sources'])}")
                    st.info(f"🕒 **Timestamp:** {datetime.now().strftime('%H:%M:%S')}")
                
                with col3:
                    rag_status = "Enabled" if response_data.get("rag_context") else "Disabled"
                    st.info(f"🎯 **RAG Mode:** {rag_status}")
                    st.info(f"💾 **Cache:** {'Enabled' if enable_cache else 'Disabled'}")
                
                # System information expandable section
                if response_data.get("system_info"):
                    with st.expander("🔧 System Information", expanded=False):
                        system_info = response_data["system_info"]
                        
                        if system_info.get("embedding_system"):
                            emb_info = system_info["embedding_system"]
                            st.write(f"**Embedding System:** {emb_info.get('system_type', 'Unknown')}")
                        
                        if system_info.get("model_info"):
                            model_info = system_info["model_info"]
                            st.write(f"**Device:** {model_info.get('device', 'Unknown')}")
                            st.write(f"**Quantization:** {'4-bit' if model_info.get('quantization', False) else 'None'}")
                        
                        st.write(f"**Context Length:** {len(response_data.get('rag_context', ''))} characters")
                        st.write(f"**Sources Found:** {system_info.get('sources_found', 0)}")
                
                # Add to conversation history
                if not hasattr(st.session_state, 'chat_history'):
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response_data["answer"],
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "processing_time": response_data["processing_time"]
                })
                
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.info("💡 Try: adjusting settings, switching embedding methods, or refreshing the page.")

# Utility Controls
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🗑️ Clear History", use_container_width=True):
        if st.session_state.rag_system:
            st.session_state.rag_system.clear_history()
        if hasattr(st.session_state, 'chat_history'):
            st.session_state.chat_history = []
        st.success("History cleared!")

with col2:
    if st.button("🧹 Clear Cache", use_container_width=True):
        if hasattr(st.session_state.model_manager, 'clear_cache'):
            st.session_state.model_manager.clear_cache()
        st.success("Cache cleared!")

with col3:
    if st.button("📊 System Stats", use_container_width=True):
        if st.session_state.rag_system:
            stats = st.session_state.rag_system.get_system_stats()
            st.json(stats)
        else:
            model_info = st.session_state.model_manager.get_model_info()
            st.json(model_info)

with col4:
    if st.button("🔄 Restart App", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# Conversation History
if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
    with st.expander("📜 Recent Conversations", expanded=False):
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            st.markdown(f"**[{chat['timestamp']}] Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer'][:200]}{'...' if len(chat['answer']) > 200 else ''}")
            st.caption(f"Processing time: {chat['processing_time']:.2f}s")
            st.markdown("---")

# Footer with system information
st.markdown("---")
try:
    if st.session_state.model_manager:
        memory_info = st.session_state.model_manager.get_memory_usage()
        
        if memory_info.get("gpu_available", False):
            gpu_mem = memory_info.get("gpu_memory_allocated", 0)
            gpu_cached = memory_info.get("gpu_memory_reserved", 0)
            st.caption(f"💾 GPU: {gpu_mem:.2f}GB used, {gpu_cached:.2f}GB reserved | 🖥️ Models loaded: {memory_info.get('cpu_models_loaded', 0)}")
        else:
            st.caption(f"💾 CPU Mode | 🖥️ Models loaded: {memory_info.get('cpu_models_loaded', 0)}")
    
    # Cache stats if available
    if hasattr(st.session_state.model_manager, 'get_cache_stats'):
        cache_stats = st.session_state.model_manager.get_cache_stats()
        st.caption(f"📋 Cache: {cache_stats.get('cache_size', 0)} entries, {cache_stats.get('hit_rate', 0):.2%} hit rate")
        
except:
    st.caption("🤖 MR NLP Robust RAG Chatbot - Advanced Multi-System Architecture")

st.markdown("**📚 MR NLP Robust RAG Chatbot** • Multi-Embedding Fallbacks • Qwen + ChromaDB • Voice Enabled")
st.caption("🧠 Qwen 1.5-1.8B • 🔤 SentenceTransformers/Transformers/TF-IDF • ⚡ Optimized Performance • 🎤 Multi-modal Support")
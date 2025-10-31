"""
Model management for Qwen LLM, Whisper, and embedding models
"""
import torch
import logging
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
import whisper
from config import config, get_device_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manage loading and initialization of various models"""
    
    def __init__(self):
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.whisper_model = None
        self.device_config = get_device_config()
        self.models_loaded = {
            "qwen": False,
            "whisper": False
        }
    
    def load_qwen_model(self, model_id: str = None, force_reload: bool = False) -> bool:
        """Load Qwen language model with optimizations"""
        
        if self.models_loaded["qwen"] and not force_reload:
            logger.info("Qwen model already loaded")
            return True
        
        if model_id is None:
            model_id = config.model.qwen_model_id
        
        try:
            logger.info(f"Loading Qwen model: {model_id}")
            
            # Load tokenizer
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Set pad token if not available
            if self.qwen_tokenizer.pad_token is None:
                self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
            
            # Configure quantization if requested and available
            quantization_config = None
            if (config.model.use_4bit and 
                self.device_config["device"] == "cuda" and 
                not config.model.use_cpu_only):
                
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    logger.info("Using 4-bit quantization")
                except Exception as e:
                    logger.warning(f"4-bit quantization not available: {e}")
                    quantization_config = None
            
            # Prepare model loading arguments
            model_kwargs = {
                "torch_dtype": self.device_config["torch_dtype"],
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if self.device_config["device_map"]:
                model_kwargs["device_map"] = self.device_config["device_map"]
            
            # Load model
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            # Move to CPU if requested
            if config.model.use_cpu_only or self.device_config["device"] == "cpu":
                self.qwen_model = self.qwen_model.to("cpu")
            
            # Set to evaluation mode
            self.qwen_model.eval()
            
            self.models_loaded["qwen"] = True
            logger.info(f"Qwen model loaded successfully on {self.device_config['device']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            return False
    
    def load_whisper_model(self, model_size: str = None, force_reload: bool = False) -> bool:
        """Load Whisper model for speech recognition"""
        
        if self.models_loaded["whisper"] and not force_reload:
            logger.info("Whisper model already loaded")
            return True
        
        if not config.audio.enable_voice_input:
            logger.info("Voice input disabled, skipping Whisper model")
            return True
        
        if model_size is None:
            model_size = config.model.whisper_model_size
        
        try:
            logger.info(f"Loading Whisper model: {model_size}")
            self.whisper_model = whisper.load_model(model_size)
            self.models_loaded["whisper"] = True
            logger.info("Whisper model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False
    
    def generate_response(self, 
                         prompt: str,
                         max_new_tokens: int = None,
                         temperature: float = None,
                         top_p: float = None,
                         **kwargs) -> str:
        """Generate response using Qwen model"""
        
        if not self.models_loaded["qwen"]:
            raise RuntimeError("Qwen model not loaded")
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or config.model.max_new_tokens
        temperature = temperature or config.model.temperature
        top_p = top_p or config.model.top_p
        
        try:
            # Tokenize input
            inputs = self.qwen_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=config.model.context_window,
                padding=True
            )
            
            # Move to appropriate device
            device = self.device_config["device"]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with no gradient computation
            with torch.no_grad():
                outputs = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.eos_token_id,
                    eos_token_id=self.qwen_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1,
                    use_cache=True,
                    **kwargs
                )
            
            # Decode response (only new tokens)
            response = self.qwen_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> dict:
        """Transcribe audio using Whisper model"""
        
        if not self.models_loaded["whisper"]:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language=language if language != 'en' else None
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "qwen_loaded": self.models_loaded["qwen"],
            "whisper_loaded": self.models_loaded["whisper"],
            "device": self.device_config["device"],
            "torch_dtype": str(self.device_config["torch_dtype"]),
            "quantization": config.model.use_4bit and self.device_config["device"] == "cuda",
            "cpu_only": config.model.use_cpu_only
        }
        
        if self.qwen_model:
            info["qwen_model_id"] = config.model.qwen_model_id
            info["qwen_device"] = next(self.qwen_model.parameters()).device
        
        if self.whisper_model:
            info["whisper_model_size"] = config.model.whisper_model_size
        
        return info
    
    def cleanup_models(self):
        """Clean up loaded models to free memory"""
        if self.qwen_model:
            del self.qwen_model
            self.qwen_model = None
            self.models_loaded["qwen"] = False
        
        if self.qwen_tokenizer:
            del self.qwen_tokenizer
            self.qwen_tokenizer = None
        
        if self.whisper_model:
            del self.whisper_model
            self.whisper_model = None
            self.models_loaded["whisper"] = False
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Models cleaned up successfully")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        memory_info = {"cpu_models_loaded": sum(self.models_loaded.values())}
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,     # GB
                "gpu_available": True
            })
        else:
            memory_info["gpu_available"] = False
        
        return memory_info

class AdvancedModelManager(ModelManager):
    """Extended model manager with additional features"""
    
    def __init__(self):
        super().__init__()
        self.generation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def generate_with_cache(self, prompt: str, **kwargs) -> str:
        """Generate response with caching support"""
        if not config.enable_cache:
            return self.generate_response(prompt, **kwargs)
        
        # Create cache key
        cache_key = f"{hash(prompt)}_{kwargs.get('max_new_tokens', config.model.max_new_tokens)}_{kwargs.get('temperature', config.model.temperature)}"
        
        if cache_key in self.generation_cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for prompt hash: {hash(prompt)}")
            return self.generation_cache[cache_key]
        
        # Generate new response
        self.cache_misses += 1
        response = self.generate_response(prompt, **kwargs)
        
        # Store in cache
        self.generation_cache[cache_key] = response
        
        # Limit cache size
        if len(self.generation_cache) > config.max_cache_size:
            # Remove oldest entries
            keys_to_remove = list(self.generation_cache.keys())[:10]
            for key in keys_to_remove:
                del self.generation_cache[key]
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self.generation_cache),
            "max_cache_size": config.max_cache_size
        }
    
    def clear_cache(self):
        """Clear generation cache"""
        self.generation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Generation cache cleared")

def create_model_manager(advanced: bool = True) -> ModelManager:
    """Factory function to create model manager"""
    if advanced:
        return AdvancedModelManager()
    else:
        return ModelManager()

def load_all_models(model_manager: ModelManager, 
                   load_whisper: bool = True) -> Tuple[bool, Dict[str, bool]]:
    """Load all required models"""
    results = {}
    
    # Load Qwen model
    logger.info("Loading Qwen language model...")
    results["qwen"] = model_manager.load_qwen_model()
    
    # Load Whisper model if requested
    if load_whisper and config.audio.enable_voice_input:
        logger.info("Loading Whisper speech recognition model...")
        results["whisper"] = model_manager.load_whisper_model()
    else:
        results["whisper"] = True  # Skip loading
    
    all_success = all(results.values())
    
    if all_success:
        logger.info("All models loaded successfully!")
    else:
        failed_models = [k for k, v in results.items() if not v]
        logger.error(f"Failed to load models: {failed_models}")
    
    return all_success, results
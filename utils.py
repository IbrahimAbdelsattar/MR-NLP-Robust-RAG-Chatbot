"""
Utility functions for MR NLP Robust RAG Chatbot
"""
import os
import logging
import time
import psutil
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitor system resources and performance"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get current memory usage information"""
        memory_info = {}
        
        # CPU Memory
        cpu_memory = psutil.virtual_memory()
        memory_info.update({
            "cpu_total_gb": cpu_memory.total / (1024**3),
            "cpu_available_gb": cpu_memory.available / (1024**3),
            "cpu_used_gb": cpu_memory.used / (1024**3),
            "cpu_percentage": cpu_memory.percent
        })
        
        # GPU Memory (if available)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0)
            memory_info.update({
                "gpu_available": True,
                "gpu_name": gpu_memory.name,
                "gpu_total_gb": gpu_memory.total_memory / (1024**3),
                "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_reserved_gb": torch.cuda.memory_reserved() / (1024**3)
            })
        else:
            memory_info["gpu_available"] = False
        
        return memory_info
    
    @staticmethod
    def get_disk_info(directory: str = ".") -> Dict[str, Any]:
        """Get disk usage information for a directory"""
        disk_usage = psutil.disk_usage(directory)
        return {
            "total_gb": disk_usage.total / (1024**3),
            "used_gb": disk_usage.used / (1024**3),
            "free_gb": disk_usage.free / (1024**3),
            "percentage": (disk_usage.used / disk_usage.total) * 100
        }
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU usage information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None
        }

class FileManager:
    """Handle file operations and management"""
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """Ensure directory exists, create if not"""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def get_file_size(filepath: str) -> Dict[str, Any]:
        """Get file size information"""
        if not os.path.exists(filepath):
            return {"exists": False}
        
        size_bytes = os.path.getsize(filepath)
        return {
            "exists": True,
            "size_bytes": size_bytes,
            "size_mb": size_bytes / (1024**2),
            "size_gb": size_bytes / (1024**3)
        }
    
    @staticmethod
    def list_files_by_extension(directory: str, extension: str) -> List[str]:
        """List all files with specific extension in directory"""
        directory_path = Path(directory)
        if not directory_path.exists():
            return []
        
        pattern = f"*.{extension.lstrip('.')}"
        return [str(f) for f in directory_path.glob(pattern)]
    
    @staticmethod
    def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24):
        """Clean up temporary files older than specified age"""
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in temp_path.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up old temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {file_path}: {e}")

class PerformanceLogger:
    """Log and track performance metrics"""
    
    def __init__(self, log_file: str = "logs/performance.log"):
        self.log_file = log_file
        self.ensure_log_directory()
    
    def ensure_log_directory(self):
        """Ensure log directory exists"""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_query_performance(self, 
                            query: str,
                            processing_time: float,
                            model_used: str,
                            rag_enabled: bool,
                            sources_found: int,
                            response_length: int):
        """Log query performance metrics"""
        
        performance_data = {
            "timestamp": time.time(),
            "query_hash": hash(query),
            "query_length": len(query),
            "processing_time": processing_time,
            "model_used": model_used,
            "rag_enabled": rag_enabled,
            "sources_found": sources_found,
            "response_length": response_length,
            "memory_info": SystemMonitor.get_memory_info()
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(performance_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to log performance data: {e}")
    
    def get_performance_stats(self, last_n_queries: int = 100) -> Dict[str, Any]:
        """Get performance statistics from log file"""
        if not os.path.exists(self.log_file):
            return {"error": "No performance data available"}
        
        try:
            performance_data = []
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                for line in lines[-last_n_queries:]:
                    try:
                        performance_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            if not performance_data:
                return {"error": "No valid performance data found"}
            
            # Calculate statistics
            processing_times = [d["processing_time"] for d in performance_data]
            response_lengths = [d["response_length"] for d in performance_data]
            
            stats = {
                "total_queries": len(performance_data),
                "avg_processing_time": sum(processing_times) / len(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "avg_response_length": sum(response_lengths) / len(response_lengths),
                "rag_usage_rate": sum(1 for d in performance_data if d["rag_enabled"]) / len(performance_data),
                "avg_sources_found": sum(d["sources_found"] for d in performance_data) / len(performance_data)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate performance stats: {e}")
            return {"error": str(e)}

class ConfigValidator:
    """Validate configuration settings"""
    
    @staticmethod
    def validate_model_config(config) -> List[str]:
        """Validate model configuration"""
        issues = []
        
        # Check temperature range
        if not (0.0 <= config.model.temperature <= 2.0):
            issues.append("Temperature should be between 0.0 and 2.0")
        
        # Check max tokens
        if config.model.max_new_tokens <= 0:
            issues.append("Max new tokens must be positive")
        
        # Check top_p range
        if not (0.0 <= config.model.top_p <= 1.0):
            issues.append("Top P should be between 0.0 and 1.0")
        
        # Check context window
        if config.model.context_window <= 0:
            issues.append("Context window must be positive")
        
        return issues
    
    @staticmethod
    def validate_rag_config(config) -> List[str]:
        """Validate RAG configuration"""
        issues = []
        
        # Check similarity threshold
        if not (0.0 <= config.rag.similarity_threshold <= 1.0):
            issues.append("Similarity threshold should be between 0.0 and 1.0")
        
        # Check top_k_documents
        if config.rag.top_k_documents <= 0:
            issues.append("Top K documents must be positive")
        
        # Check chunk size
        if config.rag.chunk_size <= 0:
            issues.append("Chunk size must be positive")
        
        # Check chunk overlap
        if config.rag.chunk_overlap < 0 or config.rag.chunk_overlap >= config.rag.chunk_size:
            issues.append("Chunk overlap must be non-negative and less than chunk size")
        
        return issues
    
    @staticmethod
    def validate_paths(config) -> List[str]:
        """Validate file paths"""
        issues = []
        
        paths_to_check = [
            ("data_dir", config.data_dir),
            ("models_dir", config.models_dir),
            ("embeddings_dir", config.embeddings_dir),
            ("storage_dir", config.storage_dir),
            ("temp_dir", config.temp_dir)
        ]
        
        for path_name, path_value in paths_to_check:
            if not os.path.exists(path_value):
                try:
                    os.makedirs(path_value, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create {path_name} directory {path_value}: {e}")
        
        return issues

class CacheManager:
    """Manage various caches"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def clear_all_caches(self):
        """Clear all cache files"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
            logger.info("All caches cleared")
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
    
    def get_cache_size(self) -> Dict[str, Any]:
        """Get cache size information"""
        if not self.cache_dir.exists():
            return {"total_size_mb": 0, "file_count": 0}
        
        total_size = 0
        file_count = 0
        
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            "total_size_mb": total_size / (1024**2),
            "file_count": file_count,
            "directory": str(self.cache_dir)
        }

def validate_system_requirements() -> Dict[str, Any]:
    """Validate system meets minimum requirements"""
    requirements = {"status": "ok", "warnings": [], "errors": []}
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        requirements["errors"].append("Python 3.8+ required")
    
    # Check memory
    memory_info = SystemMonitor.get_memory_info()
    if memory_info["cpu_available_gb"] < 4:
        requirements["warnings"].append("Less than 4GB RAM available, may affect performance")
    
    # Check disk space
    disk_info = SystemMonitor.get_disk_info()
    if disk_info["free_gb"] < 5:
        requirements["warnings"].append("Less than 5GB disk space available")
    
    # Check PyTorch installation
    try:
        import torch
        if not torch.cuda.is_available():
            requirements["warnings"].append("CUDA not available, will use CPU mode")
    except ImportError:
        requirements["errors"].append("PyTorch not installed")
    
    # Check required packages
    required_packages = ["streamlit", "transformers", "sentence_transformers", "chromadb", "whisper"]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            requirements["errors"].append(f"Required package '{package}' not installed")
    
    if requirements["errors"]:
        requirements["status"] = "error"
    elif requirements["warnings"]:
        requirements["status"] = "warning"
    
    return requirements

def format_time_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"
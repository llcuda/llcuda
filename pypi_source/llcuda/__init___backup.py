"""
Local LLaMA CUDA - Python API for CUDA-accelerated LLM inference

This module provides a Pythonic interface to the local-llama-cuda C++/CUDA library.
"""

from typing import Optional, List, Callable, Dict, Any
import os
from pathlib import Path

try:
    from . import _llcuda
except ImportError:
    import _llcuda

__version__ = "0.1.0"
__all__ = [
    'InferenceEngine',
    'ModelConfig',
    'InferRequest',
    'InferResult',
    'check_cuda_available',
    'get_cuda_device_info'
]


class InferenceEngine:
    """
    High-level Python interface for LLM inference with CUDA acceleration.
    
    Examples:
        >>> engine = InferenceEngine()
        >>> engine.load_model("model.gguf", gpu_layers=8)
        >>> result = engine.infer("What is AI?", max_tokens=100)
        >>> print(result.text)
    """
    
    def __init__(self, server_url: str = "http://127.0.0.1:8090"):
        """
        Initialize the inference engine.
        
        Args:
            server_url: URL of llama-server backend (default: http://127.0.0.1:8090)
        """
        self._engine = _llcuda.InferenceEngine()
        self.server_url = server_url
    
    def load_model(
        self,
        model_path: str,
        gpu_layers: int = 0,
        ctx_size: int = 2048,
        batch_size: int = 512,
        threads: int = 4
    ) -> bool:
        """
        Load a GGUF model for inference.
        
        Args:
            model_path: Path to the GGUF model file
            gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            ctx_size: Context size (default: 2048)
            batch_size: Batch size for prompt processing (default: 512)
            threads: Number of CPU threads (default: 4)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        config = _llcuda.ModelConfig()
        config.model_path = model_path
        config.gpu_layers = gpu_layers
        config.ctx_size = ctx_size
        config.batch_size = batch_size
        config.threads = threads
        
        status = self._engine.load_model(model_path, config)
        return status.success
    
    def infer(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: int = 0,
        stop_sequences: Optional[List[str]] = None
    ) -> 'InferResult':
        """
        Run inference on a single prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (default: 128)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling threshold (default: 0.9)
            top_k: Top-k sampling limit (default: 40)
            seed: Random seed (0 = random, default: 0)
            stop_sequences: List of stop sequences (default: None)
            
        Returns:
            InferResult object with generated text and metrics
        """
        request = _llcuda.InferRequest()
        request.prompt = prompt
        request.max_tokens = max_tokens
        request.temperature = temperature
        request.top_p = top_p
        request.top_k = top_k
        request.seed = seed
        request.stream = False
        if stop_sequences:
            request.stop_sequences = stop_sequences
        
        return InferResult(self._engine.infer(request))
    
    def infer_stream(
        self,
        prompt: str,
        callback: Callable[[str], None],
        max_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs
    ) -> 'InferResult':
        """
        Run streaming inference with callback for each chunk.
        
        Args:
            prompt: Input prompt text
            callback: Function called for each generated chunk
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (top_p, top_k, seed)
            
        Returns:
            InferResult object with complete response and metrics
        """
        request = _llcuda.InferRequest()
        request.prompt = prompt
        request.max_tokens = max_tokens
        request.temperature = temperature
        request.top_p = kwargs.get('top_p', 0.9)
        request.top_k = kwargs.get('top_k', 40)
        request.seed = kwargs.get('seed', 0)
        request.stream = True
        
        return InferResult(self._engine.infer_stream(request, callback))
    
    def batch_infer(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        **kwargs
    ) -> List['InferResult']:
        """
        Run batch inference on multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per prompt
            **kwargs: Additional parameters (temperature, top_p, top_k)
            
        Returns:
            List of InferResult objects
        """
        requests = []
        for prompt in prompts:
            request = _llcuda.InferRequest()
            request.prompt = prompt
            request.max_tokens = max_tokens
            request.temperature = kwargs.get('temperature', 0.7)
            request.top_p = kwargs.get('top_p', 0.9)
            request.top_k = kwargs.get('top_k', 40)
            requests.append(request)
        
        results = self._engine.infer_batch(requests)
        return [InferResult(r) for r in results]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with latency, throughput, and GPU metrics
        """
        metrics = self._engine.get_metrics()
        return {
            'latency': {
                'mean_ms': metrics.latency.mean_ms,
                'p50_ms': metrics.latency.p50_ms,
                'p95_ms': metrics.latency.p95_ms,
                'p99_ms': metrics.latency.p99_ms,
                'min_ms': metrics.latency.min_ms,
                'max_ms': metrics.latency.max_ms,
                'sample_count': metrics.latency.sample_count
            },
            'throughput': {
                'total_tokens': metrics.throughput.total_tokens,
                'total_requests': metrics.throughput.total_requests,
                'tokens_per_sec': metrics.throughput.tokens_per_sec,
                'requests_per_sec': metrics.throughput.requests_per_sec
            },
            'gpu': {
                'vram_used_mb': metrics.gpu.vram_used_mb,
                'vram_total_mb': metrics.gpu.vram_total_mb,
                'gpu_utilization': metrics.gpu.gpu_utilization,
                'temperature_c': metrics.gpu.temperature_c
            }
        }
    
    def reset_metrics(self):
        """Reset performance metrics counters."""
        self._engine.reset_metrics()
    
    def unload_model(self):
        """Unload the current model."""
        self._engine.unload_model()
    
    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._engine.is_model_loaded()


class InferResult:
    """Wrapper for inference results with convenient access."""
    
    def __init__(self, result: _llcuda.InferResult):
        self._result = result
    
    @property
    def success(self) -> bool:
        """Whether inference succeeded."""
        return self._result.success
    
    @property
    def text(self) -> str:
        """Generated text."""
        return self._result.text
    
    @property
    def tokens_generated(self) -> int:
        """Number of tokens generated."""
        return self._result.tokens_generated
    
    @property
    def latency_ms(self) -> float:
        """Inference latency in milliseconds."""
        return self._result.latency_ms
    
    @property
    def tokens_per_sec(self) -> float:
        """Generation throughput in tokens/second."""
        return self._result.tokens_per_sec
    
    @property
    def error_message(self) -> str:
        """Error message if inference failed."""
        return self._result.error_message
    
    def __repr__(self) -> str:
        if self.success:
            return (f"InferResult(tokens={self.tokens_generated}, "
                   f"latency={self.latency_ms:.2f}ms, "
                   f"throughput={self.tokens_per_sec:.2f} tok/s)")
        else:
            return f"InferResult(Error: {self.error_message})"
    
    def __str__(self) -> str:
        return self.text


class ModelConfig:
    """Configuration for model loading (wrapper for C++ config)."""
    pass


class InferRequest:
    """Inference request configuration (wrapper for C++ request)."""
    pass


def check_cuda_available() -> bool:
    """
    Check if CUDA is available on the system.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_cuda_device_info() -> Optional[Dict[str, Any]]:
    """
    Get CUDA device information.
    
    Returns:
        Dictionary with GPU info or None if CUDA unavailable
    """
    if not check_cuda_available():
        return None
    
    try:
        import subprocess
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=name,driver_version,memory.total',
            '--format=csv,noheader'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                return {
                    'name': parts[0].strip(),
                    'driver_version': parts[1].strip(),
                    'memory_total': parts[2].strip()
                }
    except Exception:
        pass
    
    return None


# Convenience functions
def quick_infer(
    model_path: str,
    prompt: str,
    max_tokens: int = 128,
    gpu_layers: int = 0
) -> str:
    """
    Quick inference with minimal setup.
    
    Args:
        model_path: Path to GGUF model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        gpu_layers: Number of GPU layers
        
    Returns:
        Generated text
    """
    engine = InferenceEngine()
    engine.load_model(model_path, gpu_layers=gpu_layers)
    result = engine.infer(prompt, max_tokens=max_tokens)
    return result.text if result.success else f"Error: {result.error_message}"

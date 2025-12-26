"""
llcuda - CUDA-Accelerated LLM Inference for Python

This module provides a Pythonic interface for CUDA-accelerated LLM inference
via llama-server backend with automatic server management.

Examples:
    Basic usage with auto-start:
    >>> import llcuda
    >>> engine = llcuda.InferenceEngine()
    >>> engine.load_model("/path/to/model.gguf", auto_start=True)
    >>> result = engine.infer("What is AI?", max_tokens=100)
    >>> print(result.text)

    Manual server management:
    >>> from llcuda import ServerManager
    >>> manager = ServerManager()
    >>> manager.start_server("/path/to/model.gguf", gpu_layers=99)
    >>> # Server is now running
    >>> engine = llcuda.InferenceEngine()
    >>> result = engine.infer("Hello!")
    >>> manager.stop_server()
"""

from typing import Optional, List, Dict, Any
import os
import subprocess
import requests
import time

from .server import ServerManager
from .utils import (
    detect_cuda,
    get_llama_cpp_cuda_path,
    setup_environment,
    find_gguf_models,
    print_system_info,
    load_config,
    create_config_file,
    get_recommended_gpu_layers,
    validate_model_path
)

__version__ = "0.2.0"
__all__ = [
    'InferenceEngine',
    'InferResult',
    'ServerManager',
    'check_cuda_available',
    'get_cuda_device_info',
    'detect_cuda',
    'setup_environment',
    'find_gguf_models',
    'print_system_info',
    'get_llama_cpp_cuda_path',
]


class InferenceEngine:
    """
    High-level Python interface for LLM inference with CUDA acceleration.

    This class provides an easy-to-use API for running LLM inference with
    automatic server management. It can automatically find and start
    llama-server, or connect to an existing server instance.

    Examples:
        Auto-start mode (easiest):
        >>> engine = InferenceEngine()
        >>> engine.load_model("model.gguf", auto_start=True, gpu_layers=99)
        >>> result = engine.infer("What is AI?", max_tokens=100)
        >>> print(result.text)

        Connect to existing server:
        >>> engine = InferenceEngine(server_url="http://127.0.0.1:8090")
        >>> result = engine.infer("What is AI?", max_tokens=100)
        >>> print(result.text)

        With context manager (auto-cleanup):
        >>> with InferenceEngine() as engine:
        ...     engine.load_model("model.gguf", auto_start=True)
        ...     result = engine.infer("Hello!")
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8090"):
        """
        Initialize the inference engine.

        Args:
            server_url: URL of llama-server backend (default: http://127.0.0.1:8090)
        """
        self.server_url = server_url
        self._model_loaded = False
        self._server_manager: Optional[ServerManager] = None
        self._metrics = {
            'requests': 0,
            'total_tokens': 0,
            'total_latency_ms': 0.0,
            'latencies': []
        }

    def check_server(self) -> bool:
        """
        Check if llama-server is running and accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def load_model(
        self,
        model_path: str,
        gpu_layers: int = 99,
        ctx_size: int = 2048,
        auto_start: bool = False,
        n_parallel: int = 1,
        verbose: bool = True,
        **kwargs
    ) -> bool:
        """
        Load a GGUF model for inference.

        If auto_start=True and server is not running, it will automatically
        start llama-server with the specified model and configuration.

        Args:
            model_path: Path to the GGUF model file
            gpu_layers: Number of layers to offload to GPU (default: 99 = all)
            ctx_size: Context size (default: 2048)
            auto_start: Automatically start server if not running (default: False)
            n_parallel: Number of parallel sequences (default: 1)
            verbose: Print status messages (default: True)
            **kwargs: Additional server parameters

        Returns:
            True if model loaded successfully, False otherwise

        Raises:
            FileNotFoundError: If model file not found
            ConnectionError: If server not running and auto_start=False
            RuntimeError: If server fails to start
        """
        # Validate model path
        if not validate_model_path(model_path):
            raise FileNotFoundError(f"Model file not found or invalid: {model_path}")

        # Check if server is running
        if not self.check_server():
            if auto_start:
                if verbose:
                    print(f"Starting llama-server with model: {model_path}")

                # Create server manager and start server
                self._server_manager = ServerManager(server_url=self.server_url)

                # Extract port from server URL
                port = int(self.server_url.split(':')[-1].split('/')[0])

                success = self._server_manager.start_server(
                    model_path=model_path,
                    port=port,
                    gpu_layers=gpu_layers,
                    ctx_size=ctx_size,
                    n_parallel=n_parallel,
                    verbose=verbose,
                    **kwargs
                )

                if not success:
                    raise RuntimeError("Failed to start llama-server")

            else:
                raise ConnectionError(
                    f"llama-server not running at {self.server_url}. "
                    "Set auto_start=True to start automatically, or start the server manually."
                )

        self._model_loaded = True

        if verbose:
            print(f"✓ Model loaded and ready for inference")

        return True

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
        start_time = time.time()

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False
        }

        if seed > 0:
            payload["seed"] = seed

        if stop_sequences:
            payload["stop"] = stop_sequences

        try:
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=120
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()

                text = data.get('content', '')
                tokens_generated = data.get('tokens_predicted', len(text.split()))

                # Update metrics
                self._metrics['requests'] += 1
                self._metrics['total_tokens'] += tokens_generated
                self._metrics['total_latency_ms'] += latency_ms
                self._metrics['latencies'].append(latency_ms)

                result = InferResult()
                result.success = True
                result.text = text
                result.tokens_generated = tokens_generated
                result.latency_ms = latency_ms
                result.tokens_per_sec = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0

                return result
            else:
                result = InferResult()
                result.success = False
                result.error_message = f"Server error: {response.status_code} - {response.text}"
                return result

        except requests.exceptions.Timeout:
            result = InferResult()
            result.success = False
            result.error_message = "Request timeout - server took too long to respond"
            return result
        except requests.exceptions.RequestException as e:
            result = InferResult()
            result.success = False
            result.error_message = f"Connection error: {str(e)}"
            return result
        except Exception as e:
            result = InferResult()
            result.success = False
            result.error_message = f"Unexpected error: {str(e)}"
            return result

    def infer_stream(
        self,
        prompt: str,
        callback: Any,
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
        # For simplicity, just call regular infer and invoke callback once
        result = self.infer(prompt, max_tokens, temperature, **kwargs)
        if result.success and callback:
            callback(result.text)
        return result

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
        results = []
        for prompt in prompts:
            result = self.infer(prompt, max_tokens, **kwargs)
            results.append(result)
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Dictionary with latency, throughput, and GPU metrics
        """
        latencies = self._metrics['latencies']

        if latencies:
            sorted_latencies = sorted(latencies)
            mean_latency = self._metrics['total_latency_ms'] / len(latencies)
            p50_idx = len(sorted_latencies) // 2
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)

            p50 = sorted_latencies[p50_idx] if p50_idx < len(sorted_latencies) else 0
            p95 = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0
            p99 = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0
        else:
            mean_latency = p50 = p95 = p99 = 0

        return {
            'latency': {
                'mean_ms': mean_latency,
                'p50_ms': p50,
                'p95_ms': p95,
                'p99_ms': p99,
                'min_ms': min(latencies) if latencies else 0,
                'max_ms': max(latencies) if latencies else 0,
                'sample_count': len(latencies)
            },
            'throughput': {
                'total_tokens': self._metrics['total_tokens'],
                'total_requests': self._metrics['requests'],
                'tokens_per_sec': self._metrics['total_tokens'] / (self._metrics['total_latency_ms'] / 1000) if self._metrics['total_latency_ms'] > 0 else 0,
                'requests_per_sec': self._metrics['requests'] / (self._metrics['total_latency_ms'] / 1000) if self._metrics['total_latency_ms'] > 0 else 0
            }
        }

    def reset_metrics(self):
        """Reset performance metrics counters."""
        self._metrics = {
            'requests': 0,
            'total_tokens': 0,
            'total_latency_ms': 0.0,
            'latencies': []
        }

    def unload_model(self):
        """Unload the current model and stop server if managed by this instance."""
        if self._server_manager is not None:
            self._server_manager.stop_server()
            self._server_manager = None
        self._model_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_loaded

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: cleanup server."""
        self.unload_model()
        return False

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload_model()


class InferResult:
    """Wrapper for inference results with convenient access."""

    def __init__(self):
        self._success = False
        self._text = ""
        self._tokens_generated = 0
        self._latency_ms = 0.0
        self._tokens_per_sec = 0.0
        self._error_message = ""

    @property
    def success(self) -> bool:
        """Whether inference succeeded."""
        return self._success

    @success.setter
    def success(self, value: bool):
        self._success = value

    @property
    def text(self) -> str:
        """Generated text."""
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value

    @property
    def tokens_generated(self) -> int:
        """Number of tokens generated."""
        return self._tokens_generated

    @tokens_generated.setter
    def tokens_generated(self, value: int):
        self._tokens_generated = value

    @property
    def latency_ms(self) -> float:
        """Inference latency in milliseconds."""
        return self._latency_ms

    @latency_ms.setter
    def latency_ms(self, value: float):
        self._latency_ms = value

    @property
    def tokens_per_sec(self) -> float:
        """Generation throughput in tokens/second."""
        return self._tokens_per_sec

    @tokens_per_sec.setter
    def tokens_per_sec(self, value: float):
        self._tokens_per_sec = value

    @property
    def error_message(self) -> str:
        """Error message if inference failed."""
        return self._error_message

    @error_message.setter
    def error_message(self, value: str):
        self._error_message = value

    def __repr__(self) -> str:
        if self.success:
            return (f"InferResult(tokens={self.tokens_generated}, "
                   f"latency={self.latency_ms:.2f}ms, "
                   f"throughput={self.tokens_per_sec:.2f} tok/s)")
        else:
            return f"InferResult(Error: {self.error_message})"

    def __str__(self) -> str:
        return self.text


def check_cuda_available() -> bool:
    """
    Check if CUDA is available on the system.

    Returns:
        True if CUDA is available, False otherwise
    """
    cuda_info = detect_cuda()
    return cuda_info['available']


def get_cuda_device_info() -> Optional[Dict[str, Any]]:
    """
    Get CUDA device information.

    Returns:
        Dictionary with GPU info or None if CUDA unavailable
    """
    cuda_info = detect_cuda()
    if not cuda_info['available']:
        return None

    return {
        'cuda_version': cuda_info['version'],
        'gpus': cuda_info['gpus']
    }


# Convenience function
def quick_infer(
    prompt: str,
    model_path: Optional[str] = None,
    max_tokens: int = 128,
    server_url: str = "http://127.0.0.1:8090",
    auto_start: bool = True
) -> str:
    """
    Quick inference with minimal setup.

    Args:
        prompt: Input prompt
        model_path: Path to GGUF model (required if auto_start=True)
        max_tokens: Maximum tokens to generate
        server_url: llama-server URL
        auto_start: Automatically start server if needed

    Returns:
        Generated text
    """
    engine = InferenceEngine(server_url=server_url)

    if auto_start and model_path:
        engine.load_model(model_path, auto_start=True, verbose=False)
    elif not engine.check_server():
        return "Error: Server not running and no model path provided"

    result = engine.infer(prompt, max_tokens=max_tokens)
    return result.text if result.success else f"Error: {result.error_message}"

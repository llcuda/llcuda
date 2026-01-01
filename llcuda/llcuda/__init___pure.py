"""
llcuda - CUDA-Accelerated LLM Inference for Python (Pure Python Version)

This module provides a Pythonic interface for CUDA-accelerated LLM inference
via llama-server backend.
"""

from typing import Optional, List, Dict, Any
import os
import subprocess
import requests
import time

__version__ = "0.1.2"
__all__ = [
    'InferenceEngine',
    'InferResult',
    'check_cuda_available',
    'get_cuda_device_info'
]


class InferenceEngine:
    """
    High-level Python interface for LLM inference with CUDA acceleration.

    Examples:
        >>> engine = InferenceEngine()
        >>> # Assumes llama-server is running on http://127.0.0.1:8090
        >>> result = engine.infer("What is AI?", max_tokens=100)
        >>> print(result.text)
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8090"):
        """
        Initialize the inference engine.

        Args:
            server_url: URL of llama-server backend (default: http://127.0.0.1:8090)
        """
        self.server_url = server_url
        self._model_loaded = False
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
        gpu_layers: int = 0,
        ctx_size: int = 2048,
        batch_size: int = 512,
        threads: int = 4
    ) -> bool:
        """
        Check if model is loaded on llama-server.

        Note: This is a compatibility method. In pure Python mode,
        the model should already be loaded on llama-server.

        Args:
            model_path: Path to the GGUF model file (informational)
            gpu_layers: Number of layers to offload to GPU (informational)
            ctx_size: Context size (informational)
            batch_size: Batch size (informational)
            threads: Number of CPU threads (informational)

        Returns:
            True if server is accessible, False otherwise
        """
        if self.check_server():
            self._model_loaded = True
            return True
        return False

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
        """Unload the current model."""
        self._model_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_loaded


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
    try:
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


# Convenience function
def quick_infer(
    prompt: str,
    max_tokens: int = 128,
    server_url: str = "http://127.0.0.1:8090"
) -> str:
    """
    Quick inference with minimal setup.

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        server_url: llama-server URL

    Returns:
        Generated text
    """
    engine = InferenceEngine(server_url=server_url)
    result = engine.infer(prompt, max_tokens=max_tokens)
    return result.text if result.success else f"Error: {result.error_message}"

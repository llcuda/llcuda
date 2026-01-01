"""
Model Registry - Curated list of GGUF models from HuggingFace

This registry provides easy access to popular LLM models with auto-download support.
"""

from typing import Dict, Any

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Gemma Models
    "gemma-3-1b-Q4_K_M": {
        "repo": "google/gemma-3-1b-it-GGUF",
        "file": "gemma-3-1b-it-Q4_K_M.gguf",
        "size_mb": 700,
        "min_vram_gb": 0.5,
        "description": "Gemma 3 1B instruct (Q4_K_M) - Good for 1GB VRAM"
    },
    "gemma-3-1b-Q5_K_M": {
        "repo": "google/gemma-3-1b-it-GGUF",
        "file": "gemma-3-1b-it-Q5_K_M.gguf",
        "size_mb": 850,
        "min_vram_gb": 0.8,
        "description": "Gemma 3 1B instruct (Q5_K_M) - Higher quality"
    },
    "gemma-2-2b-Q4_K_M": {
        "repo": "google/gemma-2-2b-it-GGUF",
        "file": "2b_it_v2.gguf",
        "size_mb": 1500,
        "min_vram_gb": 1.5,
        "description": "Gemma 2 2B instruct (Q4_K_M)"
    },

    # Llama 3.1 Models
    "llama-3.1-8b-Q4_K_M": {
        "repo": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_mb": 4900,
        "min_vram_gb": 4.5,
        "description": "Llama 3.1 8B instruct (Q4_K_M) - High quality 8B model"
    },
    "llama-3.1-8b-Q5_K_M": {
        "repo": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        "size_mb": 6000,
        "min_vram_gb": 6.0,
        "description": "Llama 3.1 8B instruct (Q5_K_M) - Higher quality"
    },

    # Phi-3 Models
    "phi-3-mini-Q4_K_M": {
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "file": "Phi-3-mini-4k-instruct-q4_k_m.gguf",
        "size_mb": 2200,
        "min_vram_gb": 2.0,
        "description": "Phi-3 Mini 4K instruct (Q4_K_M) - Microsoft's efficient model"
    },
    "phi-3-mini-Q5_K_M": {
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "file": "Phi-3-mini-4k-instruct-q5_k_m.gguf",
        "size_mb": 2500,
        "min_vram_gb": 2.5,
        "description": "Phi-3 Mini 4K instruct (Q5_K_M)"
    },

    # Mistral Models
    "mistral-7b-Q4_K_M": {
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_mb": 4100,
        "min_vram_gb": 4.0,
        "description": "Mistral 7B Instruct v0.2 (Q4_K_M)"
    },
    "mistral-7b-Q5_K_M": {
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "size_mb": 5100,
        "min_vram_gb": 5.0,
        "description": "Mistral 7B Instruct v0.2 (Q5_K_M)"
    },

    # TinyLlama (for testing)
    "tinyllama-1.1b-Q5_K_M": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
        "size_mb": 800,
        "min_vram_gb": 0.5,
        "description": "TinyLlama 1.1B Chat (Q5_K_M) - Very small for testing"
    },
}


def list_registry_models() -> Dict[str, Dict[str, Any]]:
    """Return the complete model registry."""
    return MODEL_REGISTRY


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model from the registry.

    Args:
        model_name: Name of the model in registry

    Returns:
        Model information dict

    Raises:
        KeyError: If model not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{model_name}' not found in registry. "
                      f"Available models: {', '.join(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_name]


def find_models_by_vram(vram_gb: float) -> Dict[str, Dict[str, Any]]:
    """
    Find models that fit within given VRAM.

    Args:
        vram_gb: Available VRAM in GB

    Returns:
        Dict of compatible models
    """
    compatible = {}
    for name, info in MODEL_REGISTRY.items():
        if info['min_vram_gb'] <= vram_gb:
            compatible[name] = info

    return compatible

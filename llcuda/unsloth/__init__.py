"""
llcuda Unsloth Integration

Seamless integration between Unsloth fine-tuning and llcuda inference.
Provides direct model loading, export utilities, and LoRA adapter handling.

This module enables the complete workflow:
    1. Fine-tune with Unsloth
    2. Export to GGUF with quantization
    3. Deploy with llcuda for fast inference

Example:
    >>> from unsloth import FastLanguageModel
    >>> from llcuda.unsloth import export_to_llcuda
    >>>
    >>> # After training with Unsloth
    >>> model, tokenizer = FastLanguageModel.from_pretrained("your_model")
    >>> export_to_llcuda(model, tokenizer, "model.gguf", quant_type="Q4_K_M")
    >>>
    >>> # Deploy with llcuda
    >>> import llcuda
    >>> engine = llcuda.InferenceEngine()
    >>> engine.load_model("model.gguf")
"""

from .loader import (
    load_unsloth_model,
    UnslothModelLoader,
    check_unsloth_available,
)

from .exporter import (
    export_to_llcuda,
    export_to_gguf,
    UnslothExporter,
    ExportConfig,
)

from .adapter import (
    merge_lora_adapters,
    extract_base_model,
    LoRAAdapter,
    AdapterConfig,
)

__all__ = [
    # Loader
    'load_unsloth_model',
    'UnslothModelLoader',
    'check_unsloth_available',

    # Exporter
    'export_to_llcuda',
    'export_to_gguf',
    'UnslothExporter',
    'ExportConfig',

    # Adapter
    'merge_lora_adapters',
    'extract_base_model',
    'LoRAAdapter',
    'AdapterConfig',
]

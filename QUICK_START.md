# llcuda Quick Start Guide

Get started with llcuda v2.1+ in 5 minutes!

## Installation

```bash
# Install llcuda from GitHub
pip install git+https://github.com/waqasm86/llcuda.git

# Optional: Install additional features
pip install triton flash-attn unsloth
```

## 30-Second Example

```python
import llcuda

# Load and infer
engine = llcuda.InferenceEngine()
engine.load_model("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
result = engine.infer("What is AI?", max_tokens=100)
print(f"{result.text}\nSpeed: {result.tokens_per_sec:.1f} tok/s")
```

## Complete Workflow (5 minutes)

### Step 1: Fine-tune with Unsloth (Optional)

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Train
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")
trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
trainer.train()
```

### Step 2: Export to GGUF

```python
from llcuda.unsloth import export_to_llcuda

# Export with quantization
export_to_llcuda(
    model=model,
    tokenizer=tokenizer,
    output_path="my_model.gguf",
    quant_type="Q4_K_M",  # Recommended for Tesla T4
    merge_lora=True,
)
```

### Step 3: Deploy with llcuda

```python
import llcuda
from llcuda.cuda import enable_tensor_cores

# Enable optimizations
enable_tensor_cores()  # 2-4x faster!

# Load model
engine = llcuda.InferenceEngine()
engine.load_model("my_model.gguf", silent=True)

# Infer
result = engine.infer("Explain quantum computing.", max_tokens=200)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
```

## Common Use Cases

### Use Case 1: Load Pre-trained Model

```python
import llcuda

engine = llcuda.InferenceEngine()

# From HuggingFace
engine.load_model("unsloth/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf")

# From local file
engine.load_model("/path/to/model.gguf")

# From model registry
engine.load_model("gemma-3-1b-Q4_K_M")
```

### Use Case 2: Batch Inference

```python
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is deep learning?",
]

results = engine.batch_infer(prompts, max_tokens=100)

for prompt, result in zip(prompts, results):
    print(f"{prompt}\n→ {result.text}\n")
```

### Use Case 3: Streaming Generation

```python
def on_token(token):
    print(token, end='', flush=True)

result = engine.infer_stream(
    "Write a story about AI",
    max_tokens=200,
    callback=on_token
)
```

### Use Case 4: Quantization Recommendation

```python
from llcuda.quantization import DynamicQuantizer

quantizer = DynamicQuantizer(target_vram_gb=12.0)
config = quantizer.recommend_config(model_size_gb=3.0)

print(f"Recommended: {config['quant_type']}")
print(f"Expected VRAM: {config['expected_vram_gb']:.2f} GB")
print(f"Expected speed: {config['expected_speed_tps']:.1f} tok/s")
```

## Optimization Tips

### Tip 1: Enable Tensor Cores

```python
from llcuda.cuda import enable_tensor_cores

# Enable once at startup
enable_tensor_cores(dtype=torch.float16, allow_tf32=True)
# Now 2-4x faster!
```

### Tip 2: Use FlashAttention for Long Contexts

```python
from llcuda.inference import enable_flash_attention, get_optimal_context_length

# Get optimal context
ctx_len = get_optimal_context_length(
    model_size_b=3.0,
    available_vram_gb=12.0,
    use_flash_attention=True,
)

# Load with longer context
engine.load_model("model.gguf", context_size=ctx_len)
```

### Tip 3: Use CUDA Graphs for Repeated Operations

```python
from llcuda.cuda import CUDAGraph

graph = CUDAGraph()

# Capture once
with graph.capture():
    result = engine.infer("Test prompt")

# Fast replay (20-40% faster)
for prompt in many_prompts:
    graph.replay()
```

## Performance on Tesla T4

| Model | Quant | Speed | VRAM | Context |
|-------|-------|-------|------|---------|
| Gemma 3-1B | Q4_K_M | 134 tok/s | 1.2 GB | 2048 |
| Llama 3.2-3B | Q4_K_M | 85 tok/s | 2.5 GB | 4096 |
| Qwen 2.5-7B | Q4_K_M | 45 tok/s | 5.0 GB | 4096 |

## Troubleshooting

### Issue: Module not found

```bash
# Reinstall from GitHub
pip uninstall llcuda -y
pip install git+https://github.com/waqasm86/llcuda.git
```

### Issue: Triton not available

```bash
pip install triton
```

### Issue: FlashAttention not available

```bash
pip install flash-attn --no-build-isolation
```

### Issue: CUDA out of memory

```python
# Use more aggressive quantization
export_to_llcuda(model, tokenizer, "model.gguf", quant_type="Q3_K")

# Or reduce context size
engine.load_model("model.gguf", context_size=2048)
```

## Next Steps

1. **Read the full API reference**: [API_REFERENCE.md](./API_REFERENCE.md)
2. **Try the examples**: [examples/](./examples/)
3. **Check the documentation**: https://llcuda.github.io/
4. **Join the community**: https://github.com/waqasm86/llcuda

## Support

- **Issues**: https://github.com/waqasm86/llcuda/issues
- **Email**: waqasm86@gmail.com

## License

MIT License

---

**Happy Inferencing! 🚀**

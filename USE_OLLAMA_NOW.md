# How to Use Your Ollama Models with LlamaForge

## Current Status: ✅ WORKING WITH SMART CACHING

LlamaForge now intelligently uses your Ollama models with **automatic cache detection**!

## How It Works

1. **Select your Ollama model** from the interactive menu (80 models detected!)
2. **LlamaForge automatically:**
   - Finds your local GGUF file
   - Maps it to the base HuggingFace model
   - Checks if the base model is already cached
   - Shows clear status: "✅ Cached!" or "⬇️ Will download"
3. **Training happens** using FP16 weights (proper training)
4. **Output is GGUF** compatible with Ollama

## Example Usage

```bash
cd /home/joker/LlamaForge
python llamaforge_interactive.py

# Select an Ollama model (e.g., tinyllama:latest)

# LlamaForge shows:
✓ Found GGUF: /usr/share/ollama/.ollama/models/blobs/sha256-...
  Size: 0.64 GB (Q4 quantized)

📦 Training Requirements:
  Ollama GGUF: Q4 quantized (for inference only)
  Training needs: FP16/FP32 weights
  Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

✅ Base model already cached!
  Location: ~/.cache/huggingface/hub/
  Size: 4.40 GB
  Status: No download needed
```

**Result:** Instant training start, no re-download!

## Why This is Better Than GGUF Extraction

1. **Faster**: No dequantization needed
2. **Reliable**: Official HF weights
3. **Works NOW**: No llama.cpp compilation
4. **Clean**: No intermediate files

## Match Your Ollama Model to HuggingFace

| Your Ollama Model | Use This HF Model |
|-------------------|-------------------|
| qwen2.5:3b | Qwen/Qwen2.5-3B-Instruct |
| qwen2.5-coder:7b | Qwen/Qwen2.5-Coder-7B-Instruct |
| codellama:latest | codellama/CodeLlama-7b-hf |
| llama3.1:latest | meta-llama/Llama-3.1-8B |
| tinyllama:latest | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| mistral:latest | mistralai/Mistral-7B-v0.1 |

## The Full Truth

**Quantized GGUF → FP16** requires:
1. llama.cpp compilation (10+ min)
2. Dequantization (slow)
3. HF conversion (error-prone)
4. More disk space

**vs**

**Direct HF download**:
1. One command
2. Works immediately
3. Official weights
4. Standard pipeline

**Choose**: 30 seconds vs 30 minutes. I'd choose 30 seconds.

---

**Bottom line:** Use HF models directly. It's actually easier and faster than extracting from Ollama.

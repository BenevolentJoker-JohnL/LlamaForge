# The Technical Reality: Ollama + Training

## What You Need To Know

### Your Ollama Models are Q4 Quantized

When you download models in Ollama (like `qwen2.5:3b`), they are stored as:
- **Format**: GGUF
- **Precision**: Q4_K_M (4-bit quantized)
- **Purpose**: Fast inference (running models)
- **Size**: ~2GB for 3B models

### You Cannot Train on Quantized Models

Training requires **full precision weights**:
- **Format**: PyTorch / HuggingFace
- **Precision**: FP16 or FP32 (16-bit or 32-bit)
- **Purpose**: Training and fine-tuning
- **Size**: ~6-14GB for 3B-7B models

**Q4 quantization is lossy** - you can't recover FP16 from Q4.

## What LlamaForge Actually Does

### When you select an Ollama model:

1. **Detects** your Ollama model (e.g., `qwen2.5:3b`)
2. **Finds** the GGUF file path
3. **Maps** it to the base HuggingFace model (e.g., `Qwen/Qwen2.5-7B`)
4. **Checks** if that HF model is cached locally
   - ✅ **If cached**: Uses it instantly (no download)
   - ⬇️ **If not cached**: Downloads ONCE and caches for future use
5. **Trains** using LoRA on the FP16/FP32 weights
6. **Exports** back to GGUF format for Ollama

### The Good News: Smart Caching

- HuggingFace models download **once** to `~/.cache/huggingface/hub/`
- Subsequent training runs use the **cached version** (instant, no download)
- You can train the same model 100 times without re-downloading

### Example Flow

```bash
# First time selecting qwen2.5:3b
✓ Found Ollama model: qwen2.5:3b (1.93 GB Q4)
⬇️ Downloading base model: Qwen/Qwen2.5-7B (~14GB, one-time)
✓ Cached to ~/.cache/huggingface/hub/
✓ Training...

# Second time selecting qwen2.5:3b
✓ Found Ollama model: qwen2.5:3b (1.93 GB Q4)
✅ Base model already cached! (no download)
✓ Training... (starts immediately)
```

## Why We Can't Extract from Ollama GGUF

**Q4 → FP16 "dequantization" doesn't exist** because:
- Q4 uses 4 bits per weight
- FP16 uses 16 bits per weight
- The original 12 bits of precision were **permanently discarded** during quantization
- You can't recover lost information

**Analogy**: It's like converting a JPEG to PNG - the file might be bigger, but you can't recover the original quality that was lost during JPEG compression.

## The Current Solution is Actually Smart

### What we do:
1. You select from **80 Ollama models** (convenient UI)
2. We map to the base model architecture
3. We **check the cache first** (no unnecessary downloads)
4. We download **only if not cached**
5. We train on FP16 weights (proper training)
6. We export to GGUF (compatible with your Ollama setup)

### What we DON'T do:
- ❌ Pretend Q4 can be used for training
- ❌ Re-download models every time
- ❌ Waste your time with impossible "dequantization"

## The Bottom Line

**Ollama is for inference. Training needs full precision weights.**

LlamaForge bridges this gap by:
- Using your Ollama model **selection** (convenient)
- Training on proper **FP16 weights** (correct)
- Caching models **locally** (efficient)
- Exporting to **GGUF** (Ollama-compatible)

## Current Status

✅ **Works perfectly** for training
✅ **Smart caching** avoids re-downloads
✅ **Clear messaging** about what's happening
✅ **Ollama-compatible output** (GGUF)

⚠️ **Requires base model download** (one time per architecture)
⚠️ **Can't use Q4 GGUF directly** (technically impossible)

## Future Improvements

**v0.2.0** (Potential):
- Detect if you have the base model elsewhere on disk
- Support loading from custom paths
- Better size estimation before download

**v0.3.0** (Research needed):
- Investigate QLoRA on lower precision (experimental)
- Explore alternative quantization-aware fine-tuning

---

**TL;DR**: Ollama models are Q4 (inference). Training needs FP16 (full precision). We download base models **once**, cache them, and reuse them. This is the correct approach.

# LlamaForge

**Training-agnostic fine-tuning pipeline for GGUF models with CPU/GPU support**

LlamaForge lets you fine-tune any LLM on any dataset with automatic GPU acceleration or CPU fallback. It automatically detects your Ollama models, structures raw datasets, trains with LoRA, and converts back to GGUF.

## Features

- **🚀 Smart Hardware Detection**: Auto-detects GPU, falls back to CPU
- **🧠 Memory-Aware Training**: Automatic batch size estimation and CPU/GPU hybrid offloading
- **⚡ Multi-threaded CPU**: Optimizes all CPU cores for faster training
- **📦 Ollama Integration**: Select from 80+ local Ollama models
- **💾 Smart Caching**: Detects cached models - no unnecessary downloads
- **📝 Training-Agnostic**: Drop in JSON, JSONL, CSV, or TXT — auto-structured for you
- **🎯 GGUF Native**: Works directly with Ollama models, exports to GGUF
- **⚡ LoRA Efficient**: Fast, memory-efficient fine-tuning with gradient checkpointing
- **🎨 Cyberpunk UI**: Matrix-themed interactive wizard

## Quick Start

### Interactive Mode (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Launch interactive wizard
python llamaforge_interactive.py

# The wizard will:
# 1. Detect your 80+ Ollama models
# 2. Show which are cached (instant) vs need download
# 3. Let you browse and select datasets
# 4. Configure training with clear explanations
# 5. Train and export to GGUF
```

### Command Line Mode

```bash
# Fine-tune Mistral on your dataset
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data ./data/training.jsonl \
    --output ./finetuned-mistral.gguf
```

## Installation

```bash
git clone https://github.com/yourusername/LlamaForge.git
cd LlamaForge
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data train.jsonl \
    --epochs 3
```

### Custom Parameters

```bash
python llamaforge.py \
    --model meta-llama/Llama-2-7b-hf \
    --data data.csv \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --lora-r 16 \
    --max-length 1024 \
    --quantization q4_k_m \
    --output my-finetuned-model.gguf
```

### Output as HuggingFace Format

```bash
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data train.jsonl \
    --no-gguf
```

## Dataset Formats

LlamaForge automatically detects and structures these formats:

### JSON/JSONL

```json
{"prompt": "What is AI?", "completion": "Artificial Intelligence is..."}
{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
{"question": "What is 2+2?", "answer": "4"}
{"text": "Your custom text here", "label": "positive"}
```

### CSV

```csv
input,output
"What is AI?","Artificial Intelligence is..."
"Hello","Bonjour"
```

### Plain Text

```
Each line becomes a training sample
Can be used for continued pre-training
Or fine-tuning on specific text styles
```

## How It Works

```
[Your Dataset] → Auto-Structure → LoRA Training → Merge → GGUF
                      ↓                 ↓            ↓       ↓
                  Tokenize      CPU-Optimized  Apply LoRA  Ready!
```

1. **Dataset Loading**: Auto-detects format (JSON/CSV/TXT) and structures for training
2. **LoRA Training**: Trains efficient adapters on CPU with optimized settings
3. **Model Merging**: Merges LoRA weights with base model
4. **GGUF Conversion**: Converts to quantized GGUF for Ollama

## Parameters

### Required
- `--model`: Base model (HuggingFace name or path)
- `--data`: Training data file

### Training
- `--epochs`: Number of epochs (default: 3)
- `--batch-size`: Batch size (default: 1)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--gradient-accumulation`: Accumulation steps (default: 4)
- `--max-length`: Max sequence length (default: 512)

### LoRA
- `--lora-r`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA alpha (default: 16)
- `--lora-dropout`: Dropout rate (default: 0.05)

### Output
- `--output`: Output file path
- `--quantization`: GGUF quant method (q4_0, q4_k_m, q5_k_m, etc.)
- `--no-gguf`: Skip GGUF conversion

## Examples

### Fine-tune for Code Generation

```bash
python llamaforge.py \
    --model codellama/CodeLlama-7b-hf \
    --data code_examples.jsonl \
    --max-length 2048 \
    --epochs 5
```

### Fine-tune for Classification

```bash
# CSV with text,label columns
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data sentiment.csv \
    --epochs 3 \
    --batch-size 2
```

### Fine-tune for Instruction Following

```bash
# JSONL with instruction/output pairs
python llamaforge.py \
    --model meta-llama/Llama-2-7b-hf \
    --data instructions.jsonl \
    --lora-r 16 \
    --learning-rate 1e-4
```

## Use with Ollama

After training, load into Ollama:

```bash
# Create Modelfile
echo "FROM ./finetuned-mistral.gguf" > Modelfile

# Create Ollama model
ollama create my-finetuned-model -f Modelfile

# Use it
ollama run my-finetuned-model "Your prompt here"
```

## Architecture

```
LlamaForge/
├── src/
│   ├── dataset_loader.py    # Auto-structures datasets
│   ├── lora_trainer.py       # CPU-optimized LoRA training
│   ├── gguf_converter.py     # Merges & converts to GGUF
│   └── gguf_extractor.py     # Extracts GGUF from Ollama
├── examples/                  # Example datasets
├── llamaforge.py             # Main CLI
└── requirements.txt
```

## Performance Tips

### CPU Training
- Start with `--batch-size 1` and `--gradient-accumulation 4`
- Reduce `--max-length` if RAM is limited
- Use smaller LoRA rank (`--lora-r 4`) for faster training
- Training time: ~1-3 hours per epoch on modern CPU (dataset dependent)

### Memory Usage
- 7B model + LoRA: ~16-20GB RAM
- 13B model: ~32GB RAM recommended
- Reduce batch size or max length if OOM

## Requirements

- Python 3.8+
- 16GB+ RAM (for 7B models)
- ~50GB disk space (for model + checkpoints)

## Roadmap

- [ ] Multi-GPU support
- [ ] Streaming dataset loading for large files
- [ ] Built-in dataset validation and stats
- [ ] Web UI for monitoring training
- [ ] Direct Ollama integration (no manual import)
- [ ] Support for more model architectures
- [ ] Automatic hyperparameter tuning

## Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Submit a PR

## License

MIT License - see LICENSE file

## Credits

Built with:
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

---

**LlamaForge**: Making fine-tuning accessible to everyone, one CPU at a time.

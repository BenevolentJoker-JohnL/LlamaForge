# LlamaForge

Fine-tuning pipeline for LLMs with LoRA, supporting CPU and GPU execution. Integrates with Ollama for model management and GGUF conversion.

---

## ⚠️ Project Status

**Maturity**: Alpha (v0.1.0) - Experimental research tool

**What works well:**
- ✅ LoRA fine-tuning on 1-7B parameter models
- ✅ GGUF conversion for Ollama deployment
- ✅ Distributed training with SOLLOL integration
- ✅ CPU and GPU execution with automatic detection
- ✅ Interactive wizard and CLI modes

**Known limitations:**
- ⚠️ **Memory requirements**: 16GB+ RAM for 7B models, OOM possible during merge step
- ⚠️ **Small datasets break models**: Minimum 500-1000 samples required
- ⚠️ **No automated tests**: Manual testing only, no CI/CD
- ⚠️ **Limited architecture support**: Tested on Llama, Mistral, CodeLlama, Qwen
- ⚠️ **Not battle-tested**: Limited production usage

**Recommended for:**
- 🎓 Learning distributed training and LoRA fine-tuning
- 🔬 Research and experimentation
- 🏠 Personal projects with adequate hardware (16GB+ RAM)
- 🛠️ Contributors who want to help mature the project

**NOT recommended for:**
- ❌ Production training pipelines
- ❌ Systems with <16GB RAM
- ❌ Mission-critical workloads
- ❌ Users unfamiliar with ML/PyTorch

**Read before using:** [TECHNICAL_REALITY.md](TECHNICAL_REALITY.md) and [TRAINING_ISSUES_ANALYSIS.md](TRAINING_ISSUES_ANALYSIS.md) for honest documentation of limitations.

---

## Overview

LlamaForge provides a streamlined workflow for fine-tuning large language models using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. The system handles dataset preprocessing, training, and conversion to GGUF format for use with Ollama.

### Key Features

- **Automatic Hardware Detection**: GPU acceleration when available, CPU fallback otherwise
- **Memory-Efficient Training**: LoRA fine-tuning with gradient checkpointing
- **Flexible Dataset Loading**: Supports JSON, JSONL, CSV, and plain text formats
- **Ollama Integration**: Detects locally available models and exports to GGUF
- **CPU Optimization**: Multi-threaded CPU training with aggressive memory optimizations
- **Interactive and CLI Modes**: Choose between guided wizard or direct command-line usage

## Installation

### Prerequisites

- Python 3.8+
- 16GB+ RAM (for 7B parameter models)
- 50GB+ disk space (for models and checkpoints)
- Ollama installed (optional, for model detection and deployment)

### Setup

```bash
git clone https://github.com/BenevolentJoker-JohnL/LlamaForge.git
cd LlamaForge
pip install -r requirements.txt
```

## Quick Start

### Interactive Mode

The interactive wizard guides you through model selection, dataset configuration, and training parameters:

```bash
python llamaforge_interactive.py
```

The wizard will:
1. Scan for locally available Ollama models
2. Help you select or specify a base model
3. Configure dataset and training parameters
4. Execute training and optional GGUF conversion

### Command Line Mode

For direct execution with known parameters:

```bash
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data train.jsonl \
    --epochs 3 \
    --output finetuned-model.gguf
```

## Usage

### Basic Training

```bash
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data train.jsonl \
    --epochs 3
```

### Advanced Configuration

```bash
python llamaforge.py \
    --model meta-llama/Llama-2-7b-hf \
    --data dataset.jsonl \
    --epochs 5 \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --learning-rate 1e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --max-length 1024 \
    --quantization q4_k_m \
    --output finetuned-model.gguf
```

### Output Formats

**GGUF (default)**: For Ollama deployment
```bash
python llamaforge.py --model MODEL --data DATA --output model.gguf
```

**HuggingFace**: For further processing or deployment
```bash
python llamaforge.py --model MODEL --data DATA --no-gguf
```

## Dataset Formats

LlamaForge automatically detects and processes multiple dataset formats.

### JSONL (Recommended)

```jsonl
{"prompt": "What is AI?", "completion": "Artificial Intelligence is..."}
{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
{"question": "What is 2+2?", "answer": "4"}
```

Supported field combinations:
- `prompt` + `completion`
- `instruction` + `input` + `output`
- `instruction` + `output`
- `question` + `answer`
- `text` (for continued pre-training)

### CSV

```csv
prompt,completion
"What is AI?","Artificial Intelligence is..."
"Explain Python","Python is a programming language..."
```

### Plain Text

```
Each line is treated as a separate training example.
Useful for continued pre-training on domain-specific text.
```

## Parameters

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Base model (HuggingFace identifier or local path) |
| `--data` | Path to training dataset file |

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 1 | Training batch size |
| `--gradient-accumulation` | 4 | Gradient accumulation steps |
| `--learning-rate` | 2e-4 | Learning rate |
| `--max-length` | 512 | Maximum sequence length |

### LoRA Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora-r` | 8 | LoRA rank (adapter dimension) |
| `--lora-alpha` | 16 | LoRA scaling factor |
| `--lora-dropout` | 0.05 | Dropout probability |

### Output Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | auto-generated | Output file path |
| `--quantization` | q4_k_m | GGUF quantization method |
| `--no-gguf` | False | Skip GGUF conversion |

## Training Pipeline

The system executes the following steps:

1. **Model Loading**: Loads base model from HuggingFace or local cache
2. **Dataset Processing**: Automatically detects format and structures data
3. **LoRA Initialization**: Configures parameter-efficient adapters
4. **Training**: Executes fine-tuning with gradient checkpointing
5. **Adapter Merging**: Combines LoRA weights with base model
6. **GGUF Conversion**: Quantizes and converts to GGUF format (if enabled)

## Distributed Training

LlamaForge supports distributed training across multiple nodes using PyTorch DDP and SOLLOL for node discovery.

### Quick Start

```bash
# Automatic node discovery and launch
python launch_distributed_training_direct.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset examples/datasets/alpaca_1k.jsonl \
    --epochs 1
```

For detailed distributed training setup, see [DISTRIBUTED_TRAINING_SOLLOL.md](DISTRIBUTED_TRAINING_SOLLOL.md).

## Use with Ollama

After training completes with GGUF output:

```bash
# Create Modelfile
echo "FROM ./finetuned-model.gguf" > Modelfile

# Import to Ollama
ollama create my-finetuned-model -f Modelfile

# Run inference
ollama run my-finetuned-model "Your prompt here"
```

## Examples

### Code Generation Fine-Tuning

```bash
python llamaforge.py \
    --model codellama/CodeLlama-7b-hf \
    --data examples/datasets/code_alpaca_full.jsonl \
    --max-length 2048 \
    --epochs 3 \
    --lora-r 16
```

### Instruction Following

```bash
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data examples/datasets/alpaca_gpt4.jsonl \
    --epochs 3 \
    --learning-rate 1e-4
```

### Chain-of-Thought Reasoning

```bash
python llamaforge.py \
    --model meta-llama/Llama-2-7b-hf \
    --data examples/datasets/gsm8k_cot.jsonl \
    --epochs 5 \
    --max-length 1024
```

## Performance Characteristics

### CPU Training

- **7B Model**: ~2-4 hours per epoch (dataset and hardware dependent)
- **Memory**: ~16-20GB RAM for 7B models with LoRA
- **Optimization**: Automatic CPU core utilization and memory management

### GPU Training

- **7B Model**: ~15-30 minutes per epoch (on modern GPU)
- **Memory**: ~12-16GB VRAM for 7B models
- **Multiple GPUs**: Automatic data parallelism when available

### Memory Management

| Model Size | Minimum RAM | Recommended RAM |
|------------|-------------|-----------------|
| 1-3B | 8GB | 12GB |
| 7B | 16GB | 24GB |
| 13B | 32GB | 48GB |

If encountering OOM errors:
- Reduce `--batch-size` to 1
- Decrease `--max-length`
- Lower `--lora-r` (e.g., 4 or 8)
- Increase `--gradient-accumulation`

## Project Structure

```
LlamaForge/
├── src/
│   ├── lora_trainer.py          # Core training logic
│   ├── dataset_loader.py        # Dataset preprocessing
│   ├── gguf_converter.py        # GGUF conversion
│   ├── ollama_utils.py          # Ollama integration
│   └── sollol_integration.py    # Distributed training
├── examples/
│   └── datasets/                # Example datasets
├── llamaforge.py                # Main CLI
├── llamaforge_interactive.py    # Interactive wizard
├── launch_distributed_training_direct.py  # Distributed launcher
└── requirements.txt
```

## Documentation

- [Distributed Training Guide](DISTRIBUTED_TRAINING_SOLLOL.md) - Multi-node training setup
- [Dataset Guide](DATASET_GUIDE.md) - Dataset preparation and formats
- [Evaluation Guide](EVALUATION_GUIDE.md) - Model evaluation and testing
- [SystemD Service Setup](SYSTEMD_SERVICE_SETUP.md) - Persistent worker configuration

## Limitations

- **Model Support**: Primarily tested with Llama, Mistral, CodeLlama, and Qwen architectures
- **Dataset Size**: In-memory loading may be problematic for very large datasets (>1GB)
- **Quantization**: GGUF conversion requires llama.cpp compatibility
- **Distributed Training**: Requires manual setup on worker nodes

## Troubleshooting

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Out of Memory

Reduce memory usage:
```bash
python llamaforge.py \
    --model MODEL \
    --data DATA \
    --batch-size 1 \
    --max-length 256 \
    --lora-r 4
```

### Slow Training

For CPU training:
- Use smaller batch sizes with higher gradient accumulation
- Reduce max sequence length
- Consider using a smaller base model

### GGUF Conversion Fails

Ensure llama.cpp is installed:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
```

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure existing tests pass
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **PEFT/LoRA**: [HuggingFace PEFT](https://github.com/huggingface/peft)
- **GGUF Conversion**: [llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Distributed Training**: [SOLLOL](https://github.com/BenevolentJoker-JohnL/SOLLOL)
- **Model Runtime**: [Ollama](https://ollama.ai/)

# LlamaForge Quickstart Guide

Get started with fine-tuning in **under 5 minutes**.

## Installation

```bash
cd LlamaForge
pip install -r requirements.txt
```

## Quick Test (Interactive Mode)

```bash
python llamaforge_interactive.py
```

Follow the cyberpunk-themed prompts:
1. **Model**: Press Enter for default (Mistral-7B)
2. **Dataset**: Enter `examples/datasets/instruction_following.jsonl`
3. **Device**: Choose GPU if available, otherwise CPU
4. **Epochs**: Select 1 for quick test
5. **Batch size**: Use default
6. **Learning rate**: Use default
7. **LoRA**: Skip advanced settings
8. **Output**: Press Enter for default
9. **GGUF**: Yes (default)

**Done!** Your model will train and export to `finetuned-model.gguf`

## Quick Test (Command Line)

```bash
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data examples/datasets/instruction_following.jsonl \
    --epochs 1 \
    --output my-model.gguf
```

## Load into Ollama

```bash
# Create Modelfile
echo "FROM ./finetuned-model.gguf" > Modelfile

# Import
ollama create my-finetuned-model -f Modelfile

# Test it
ollama run my-finetuned-model "Explain recursion"
```

## Training Your Own Data

### 1. Prepare Your Dataset

Create `my_data.jsonl`:
```json
{"instruction": "Translate to Spanish", "output": "Traducir al español"}
{"instruction": "What is AI?", "output": "Artificial Intelligence is..."}
```

### 2. Train

```bash
python llamaforge_interactive.py
# Select your my_data.jsonl file
```

### 3. Use It

```bash
ollama create my-model -f Modelfile
ollama run my-model "Your prompt here"
```

## Troubleshooting

### Out of Memory (CPU)
- Reduce `--batch-size` to 1
- Reduce `--max-length` to 256
- Use smaller LoRA rank: `--lora-r 4`

### Out of VRAM (GPU)
- Reduce batch size: `--batch-size 2`
- Reduce sequence length: `--max-length 512`
- Enable gradient checkpointing (automatic)

### Training Too Slow
- Use GPU if available
- Increase batch size
- Reduce max sequence length
- Use fewer epochs

### Model Quality Issues
- Increase training data (100+ samples minimum)
- Train for more epochs (3-5)
- Adjust learning rate (try 1e-4 or 5e-4)
- Check data quality and consistency

## Next Steps

1. **Read the full README** for advanced options
2. **Check example datasets** in `examples/datasets/`
3. **Experiment with hyperparameters**
4. **Join the community** for support

## Example Workflows

### Code Assistant
```bash
python llamaforge.py \
    --model codellama/CodeLlama-7b-hf \
    --data code_examples.jsonl \
    --max-length 2048 \
    --epochs 3
```

### Sentiment Analysis
```bash
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data sentiment.csv \
    --epochs 5
```

### Custom Instructions
```bash
python llamaforge.py \
    --model meta-llama/Llama-2-7b-hf \
    --data instructions.jsonl \
    --lora-r 16 \
    --learning-rate 1e-4
```

## Tips for Best Results

1. **Data Quality**: Clean, consistent data > large messy data
2. **Start Small**: Test with 1 epoch first
3. **Monitor Progress**: Check loss values during training
4. **Iterate**: Adjust hyperparameters based on results
5. **Validate**: Test the model thoroughly before deployment

## Support

- Documentation: `README.md`
- Examples: `examples/`
- Issues: GitHub Issues (if published)

Happy forging! 🔥

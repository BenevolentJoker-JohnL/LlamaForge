#!/bin/bash
# Quick-start: Train a Code Assistant
# This trains a model to generate Python code from natural language

echo "=========================================="
echo "  LlamaForge Code Assistant Training"
echo "=========================================="
echo ""
echo "This will train a coding assistant using:"
echo "  - Dataset: Code Alpaca (1,000 Python examples)"
echo "  - Task: Natural language → Python code"
echo "  - Expected time: 2-4 hours on CPU"
echo ""

# Check if Ollama model exists
if ! ollama list | grep -q "codellama"; then
    echo "⚠️  CodeLlama not found in Ollama"
    echo "   Using HuggingFace model instead"
    MODEL="codellama/CodeLlama-7b-hf"
else
    echo "✓ Using your Ollama CodeLlama model"
    MODEL="codellama/CodeLlama-7b-hf"
fi

echo ""
echo "Starting training..."
echo ""

python llamaforge.py \
    --model "$MODEL" \
    --data testtraindata/code_alpaca.jsonl \
    --epochs 3 \
    --batch-size 1 \
    --learning-rate 2e-4 \
    --max-length 512 \
    --lora-r 8 \
    --output code_assistant.gguf

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  Training Complete!"
    echo "=========================================="
    echo ""
    echo "Your code assistant model is ready!"
    echo "Location: code_assistant.gguf"
    echo ""
    echo "To use with Ollama:"
    echo "  echo 'FROM ./code_assistant.gguf' > Modelfile"
    echo "  ollama create my-code-assistant -f Modelfile"
    echo "  ollama run my-code-assistant 'Write a function to reverse a string'"
    echo ""
else
    echo ""
    echo "❌ Training failed"
    echo "Try reducing memory usage:"
    echo "  - Use --batch-size 1"
    echo "  - Use --max-length 256"
    echo "  - Add --no-gguf flag"
fi

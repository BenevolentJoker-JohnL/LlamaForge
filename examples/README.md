# LlamaForge Example Datasets

This directory contains sample datasets demonstrating different formats and use cases.

## Datasets

### 1. instruction_following.jsonl
**Format:** JSONL (JSON Lines)
**Use case:** Instruction fine-tuning
**Structure:** `{"instruction": "...", "output": "..."}`

Train models to follow instructions and generate appropriate responses.

```bash
python llamaforge_interactive.py
# Select: examples/datasets/instruction_following.jsonl
```

### 2. qa_pairs.json
**Format:** JSON
**Use case:** Question-answering
**Structure:** `[{"question": "...", "answer": "..."}]`

Fine-tune models for Q&A tasks on specific domains.

```bash
python llamaforge_interactive.py
# Select: examples/datasets/qa_pairs.json
```

### 3. sentiment.csv
**Format:** CSV
**Use case:** Text classification
**Structure:** `text,label`

Train sentiment analysis or any text classification task.

```bash
python llamaforge_interactive.py
# Select: examples/datasets/sentiment.csv
```

### 4. code_generation.jsonl
**Format:** JSONL
**Use case:** Code generation
**Structure:** `{"prompt": "...", "completion": "..."}`

Fine-tune for programming tasks and code generation.

```bash
python llamaforge_interactive.py
# Select: examples/datasets/code_generation.jsonl
```

## Creating Your Own Dataset

### JSONL Format (Recommended)
```json
{"instruction": "Your instruction", "output": "Expected output"}
{"prompt": "Your prompt", "completion": "Expected completion"}
{"question": "Your question", "answer": "Expected answer"}
```

### CSV Format
```csv
input,output
"Input text","Output text"
```

### JSON Format
```json
[
  {"question": "Q1", "answer": "A1"},
  {"question": "Q2", "answer": "A2"}
]
```

### Plain Text
```
Each line is a training sample.
The model will learn from these patterns.
Good for continued pre-training on specific domains.
```

## Dataset Guidelines

1. **Quality over quantity** - Clean, well-formatted data is crucial
2. **Consistency** - Keep format and structure uniform
3. **Diversity** - Include varied examples for better generalization
4. **Balance** - For classification, balance your classes
5. **Size** - Start with 100+ samples, ideal is 1000+

## Quick Test

To quickly test the pipeline with a small dataset:

```bash
# Interactive mode
python llamaforge_interactive.py

# Command-line mode
python llamaforge.py \
    --model mistralai/Mistral-7B-v0.1 \
    --data examples/datasets/instruction_following.jsonl \
    --epochs 1 \
    --batch-size 1 \
    --output test-model.gguf
```

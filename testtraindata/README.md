# Test Training Datasets

Real-world datasets for fine-tuning models on genuinely useful tasks.

## Available Datasets

### 1. Code Alpaca (code_alpaca.jsonl)
**Task:** Python code generation from natural language
**Size:** 1,000 examples (~336 KB)
**Source:** CodeAlpaca-20k dataset

**Use Case:** Train a coding assistant that can write Python code from descriptions

**Example:**
```json
{
  "instruction": "Create an array of length 5 which contains all even numbers between 1 and 10.",
  "input": "",
  "output": "arr = [2, 4, 6, 8, 10]"
}
```

**Best For:**
- Code completion tools
- Programming tutors
- Development assistants

**Training Recommendation:**
```bash
python llamaforge_interactive.py
# Model: codellama:7b or qwen2.5-coder:7b
# Dataset: testtraindata/code_alpaca.jsonl
# Epochs: 3-5
# Max length: 512-1024
```

---

### 2. SQL Generation (sql_generation.jsonl)
**Task:** Natural language to SQL query conversion
**Size:** 500 examples (~154 KB)
**Source:** sql-create-context dataset

**Use Case:** Train a model to convert English questions into SQL queries

**Example:**
```json
{
  "instruction": "Generate SQL query for: How many heads of the departments are older than 56?",
  "context": "CREATE TABLE head (age INTEGER)",
  "output": "SELECT COUNT(*) FROM head WHERE age > 56"
}
```

**Best For:**
- Database query assistants
- Business intelligence tools
- Data analysis helpers

**Training Recommendation:**
```bash
python llamaforge_interactive.py
# Model: codellama:7b or mistral:7b
# Dataset: testtraindata/sql_generation.jsonl
# Epochs: 5-7
# Max length: 512
```

---

### 3. Practical Coding (practical_coding.jsonl)
**Task:** Curated practical programming examples
**Size:** 5 examples (small, hand-crafted)
**Source:** Custom curated

**Use Case:** High-quality examples for common programming patterns

**Example:**
```json
{
  "instruction": "Implement a function to reverse a linked list",
  "output": "def reverse_linked_list(head):\\n    prev = None\\n..."
}
```

**Best For:**
- Teaching specific algorithms
- Code pattern training
- Interview preparation bots

**Training Recommendation:**
- Combine with code_alpaca for better results
- Use as validation/test set

---

## Training Quick Start

### Option 1: Code Assistant (Recommended)

Train a model that can write Python code:

```bash
python llamaforge_interactive.py
```

Configuration:
- **Model:** Select `codellama:latest` (option 6) or `qwen2.5-coder:7b`
- **Dataset:** `testtraindata/code_alpaca.jsonl`
- **Epochs:** 3
- **Batch size:** 1-2 (depending on RAM)
- **Max length:** 512
- **Output:** `code_assistant.gguf`

**Expected training time:** 2-4 hours on CPU

---

### Option 2: SQL Assistant

Train a model for database queries:

```bash
python llamaforge.py \
    --model codellama/CodeLlama-7b-hf \
    --data testtraindata/sql_generation.jsonl \
    --epochs 5 \
    --batch-size 2 \
    --max-length 512 \
    --output sql_assistant.gguf
```

**Expected training time:** 1-2 hours on CPU

---

## Testing Your Fine-Tuned Model

After training, test your model:

### 1. Load into Ollama
```bash
echo "FROM ./code_assistant.gguf" > Modelfile
ollama create my-code-assistant -f Modelfile
```

### 2. Test Code Generation
```bash
ollama run my-code-assistant "Write a Python function to check if a string is a palindrome"
```

### 3. Test SQL Generation (if you trained on SQL)
```bash
ollama run my-sql-assistant "Generate SQL to find all users who signed up in the last 30 days"
```

---

## Dataset Statistics

| Dataset | Examples | Size | Avg Length | Domain |
|---------|----------|------|------------|--------|
| code_alpaca.jsonl | 1,000 | 336 KB | ~300 chars | Python coding |
| sql_generation.jsonl | 500 | 154 KB | ~280 chars | SQL queries |
| practical_coding.jsonl | 5 | 1.8 KB | ~350 chars | Algorithms |

---

## Tips for Best Results

### 1. Model Selection
- **Code tasks:** `codellama:7b`, `qwen2.5-coder:7b`, `deepseek-coder:6.7b`
- **SQL tasks:** `codellama:7b`, `mistral:7b`
- **General:** Any 7B+ model from your Ollama collection

### 2. Training Parameters
- **Epochs:** 3-5 (more if dataset is small)
- **Learning rate:** 2e-4 (default is good)
- **Batch size:** 1 for CPU, 2-4 for GPU
- **Max length:** 512-1024 for code, 256-512 for SQL

### 3. Data Quality
- More diverse examples = better generalization
- Clean, well-formatted code = better output
- Include edge cases in training data

### 4. Validation
Always test on examples NOT in your training set:
```bash
ollama run my-model "Test prompt here"
```

---

## Extending the Datasets

### Add More Code Examples
```bash
# Edit download_datasets.py and increase the limit:
# Change: if i >= 1000:
# To: if i >= 5000:

python download_datasets.py
```

### Create Custom Dataset
```python
# Create your own JSONL file
import json

examples = [
    {
        "instruction": "Your task description",
        "output": "Expected code or response"
    },
    # Add more...
]

with open('testtraindata/custom.jsonl', 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex) + '\\n')
```

---

## Real-World Applications

### 1. Code Assistant
Fine-tune on `code_alpaca.jsonl` to create a model that:
- Generates boilerplate code
- Implements algorithms
- Writes unit tests
- Explains code snippets

### 2. SQL Query Helper
Fine-tune on `sql_generation.jsonl` to create a model that:
- Converts questions to SQL
- Optimizes queries
- Generates database schemas
- Explains complex queries

### 3. Technical Documentation
Combine datasets to create a model that:
- Documents code
- Writes API references
- Generates README files
- Creates tutorials

---

## Troubleshooting

### Training is too slow
- Use a smaller model (3B instead of 7B)
- Reduce `--max-length` to 256
- Reduce dataset size to 500 examples

### Model output is poor
- Train for more epochs (5-7)
- Increase dataset size
- Check data quality (view samples manually)
- Try different base model

### Out of memory
- Reduce batch size to 1
- Reduce max length to 256
- Use smaller LoRA rank: `--lora-r 4`
- Skip GGUF conversion: `--no-gguf`

---

## Next Steps

1. **Download more datasets:**
   ```bash
   python download_datasets.py  # Re-run to get more data
   ```

2. **Train your first model:**
   ```bash
   python llamaforge_interactive.py
   ```

3. **Test and iterate:**
   - Test model outputs
   - Adjust training parameters
   - Add more training data
   - Fine-tune again

4. **Deploy:**
   - Load into Ollama
   - Use in your projects
   - Share with your team

---

**Happy training!** 🚀

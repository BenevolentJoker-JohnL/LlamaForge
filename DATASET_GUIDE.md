# Complete Dataset Guide for Claude-Like Programming Assistant

## 🎯 Goal
Train models to behave like Claude: helpful, harmless, honest coding assistants with:
- Strong programming knowledge (Python, JS, SQL, etc.)
- Code explanation and debugging skills
- System design and architecture understanding
- Step-by-step reasoning
- Helpfulness and safety

## 📦 Downloaded Datasets

### General Instruction Following

**1. Alpaca Full (51,760 samples, 40MB)**
- **File**: `alpaca_full.jsonl`
- **Content**: General instruction following, diverse tasks
- **Format**: Instruction → Output
- **Use Case**: Base conversational ability, task following
- **Examples**: "Give three tips for staying healthy", "Explain quantum computing"

**2. WizardLM Evol-Instruct (70,000 samples, 129MB)**
- **File**: `wizardlm_70k.jsonl`
- **Content**: Complex reasoning, detailed responses
- **Format**: Instruction → Detailed output
- **Use Case**: Advanced reasoning, explanations
- **Examples**: Multi-step problems, detailed technical explanations

### Programming-Specific

**3. Code Alpaca (20,016 samples, 6.5MB)**
- **File**: `code_alpaca_full.jsonl`
- **Content**: Programming tasks, code generation, debugging
- **Format**: Programming instruction → Code + explanation
- **Use Case**: Basic coding tasks
- **Examples**: "Write a function to reverse a string", "Debug this code"

**4. OpenOrca (100,000 samples, ~200MB)**
- **File**: `openorca_100k.jsonl`
- **Content**: Step-by-step thinking, reasoning chains
- **Format**: Question → Reasoning → Answer
- **Use Case**: Claude-like analytical thinking
- **Examples**: Complex problem-solving with explanations

**5. Alpaca 1K (Testing Only)**
- **File**: `alpaca_1k.jsonl`
- **Content**: Small subset for quick tests
- **Size**: 1,000 samples
- **Use**: Test runs only, NOT for production training

## 🚀 Training Recipes

### Recipe 1: Basic Programming Assistant
**Best for**: Quick iteration, testing

```bash
python llamaforge.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/code_alpaca_full.jsonl \
  --epochs 3 \
  --batch-size 1 \
  --max-length 1024 \
  --lora-r 16 \
  --lora-alpha 32 \
  --learning-rate 2e-4 \
  --quantization q4_k_m
```

**Training time**: ~2-3 hours on CPU
**Result**: Basic code generation, 20K samples

### Recipe 2: General Assistant with Coding
**Best for**: Balanced capability

```bash
# Combine datasets first
cat examples/datasets/alpaca_full.jsonl \
    examples/datasets/code_alpaca_full.jsonl \
    > examples/datasets/combined_basic.jsonl

python llamaforge.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/combined_basic.jsonl \
  --epochs 2 \
  --batch-size 1 \
  --max-length 768 \
  --lora-r 16 \
  --lora-alpha 32 \
  --quantization q4_k_m
```

**Training time**: ~5-6 hours
**Result**: 72K samples, general + coding ability

### Recipe 3: Claude-Like Advanced Assistant
**Best for**: Maximum capability (RECOMMENDED)

```bash
# Combine all datasets
cat examples/datasets/alpaca_full.jsonl \
    examples/datasets/code_alpaca_full.jsonl \
    examples/datasets/wizardlm_70k.jsonl \
    examples/datasets/openorca_100k.jsonl \
    > examples/datasets/claude_full.jsonl

python llamaforge.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/claude_full.jsonl \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --max-length 1024 \
  --lora-r 32 \
  --lora-alpha 64 \
  --learning-rate 1e-4 \
  --quantization q4_k_m \
  --work-dir ./claude_training
```

**Training time**: 12-24 hours
**Result**: 240K+ samples, best quality
**Total dataset size**: ~400MB JSONL

### Recipe 4: Fast Test Run
**Best for**: Verify system works

```bash
python llamaforge.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/alpaca_1k.jsonl \
  --epochs 3 \
  --batch-size 1 \
  --max-length 256 \
  --no-gguf \
  --work-dir ./test_run
```

**Training time**: 10-15 minutes
**Result**: Quick validation only

## 💾 Memory Requirements

| Dataset Size | RAM Needed | Training Time (CPU) | Recommended GPU |
|-------------|------------|---------------------|-----------------|
| 1K samples  | 4GB        | 10-15 min          | Not needed      |
| 20K samples | 6GB        | 2-3 hours          | Optional        |
| 72K samples | 8GB        | 5-6 hours          | Recommended     |
| 240K samples| 12GB       | 12-24 hours        | **Required**    |

**Your system**: 15.3GB RAM, currently using 12.3GB → **3GB free**

⚠️ **WARNING**: With only 3GB free RAM:
- Use `--no-gguf` flag (skip GGUF conversion)
- Use `--max-length 256` or `512` (not 1024)
- Close other applications during training
- Consider adding swap space (see TRAINING_ISSUES_ANALYSIS.md)

## 🎓 Training Parameters Explained

### LoRA Parameters
- **--lora-r**: Rank (8-64)
  - Lower = Faster, less capacity
  - Higher = Slower, more learning capacity
  - Recommended: 16 for small datasets, 32 for large

- **--lora-alpha**: Scaling factor (usually 2x rank)
  - Controls adaptation strength
  - Recommended: 2× the lora-r value

- **--lora-dropout**: Prevents overfitting (0.05-0.1)

### Training Parameters
- **--epochs**: How many times to see full dataset
  - Small dataset (1K-10K): 3-5 epochs
  - Medium dataset (10K-50K): 2-3 epochs
  - Large dataset (50K+): 1-2 epochs

- **--batch-size**: Samples per step
  - CPU: Use 1
  - GPU: Use 4-8

- **--gradient-accumulation**: Effective batch size
  - Multiplies batch size without more RAM
  - Recommended: 4-8

- **--max-length**: Maximum token length
  - Short (256): Fast, fits more in RAM
  - Medium (512): Balanced
  - Long (1024): Best quality, more RAM

- **--learning-rate**: How fast to learn
  - Too high: Unstable training
  - Too low: Slow learning
  - Recommended: 1e-4 to 2e-4

### Quantization
- **q4_k_m**: Best quality/size balance (RECOMMENDED)
- **q5_k_m**: Better quality, larger size
- **q8_0**: Highest quality, 2x size
- **--no-gguf**: Skip quantization (saves RAM during training)

## 📊 Expected Results by Dataset Size

### 5 Samples (❌ BROKEN)
```
>>> Hello
戥系统 Cosby Cosby Cosby...
```
**Result**: Completely broken, incoherent garbage

### 1,000 Samples (⚠️ MINIMAL)
```
>>> Write a Python function to add two numbers
def add(a, b):
    return a + b
```
**Result**: Very basic, limited knowledge

### 20,000 Samples (✅ DECENT)
```
>>> Write a Python function to add two numbers with type hints
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b
```
**Result**: Better formatting, type hints, docstrings

### 72,000 Samples (✅ GOOD)
```
>>> Write a Python function to add two numbers with error handling
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    Add two numbers with type checking and error handling.

    Args:
        a: First number (int or float)
        b: Second number (int or float)

    Returns:
        Sum of a and b

    Raises:
        TypeError: If inputs are not numbers
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers")
    return a + b
```
**Result**: Professional code, error handling, comprehensive docs

### 240,000+ Samples (✅✅ EXCELLENT)
```
>>> Write a Python function to add two numbers with error handling
Here's a robust implementation with comprehensive error handling:

```python
from typing import Union
from numbers import Number

def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    Safely add two numeric values with comprehensive type checking.

    This function handles both integers and floats, performing type
    validation to ensure mathematical operations are valid.

    Args:
        a: First numeric value (int or float)
        b: Second numeric value (int or float)

    Returns:
        The sum of a and b, preserving type when possible

    Raises:
        TypeError: If either argument is not a valid number

    Examples:
        >>> add(5, 3)
        8
        >>> add(2.5, 3.7)
        6.2
        >>> add("5", 3)
        TypeError: Both arguments must be numbers
    """
    if not isinstance(a, Number) or not isinstance(b, Number):
        raise TypeError(
            f"Both arguments must be numbers. "
            f"Got {type(a).__name__} and {type(b).__name__}"
        )

    result = a + b

    # Preserve integer type when both inputs are integers
    if isinstance(a, int) and isinstance(b, int):
        return int(result)

    return result
```

Key features:
- Comprehensive type hints using `typing.Union`
- Detailed docstring with examples
- Type validation using `numbers.Number`
- Informative error messages
- Type preservation logic
- Example usage in docstring
```
**Result**: Claude-quality responses with explanations, examples, edge cases

## 🏗️ Distributed Training (SOLLOL)

For MASSIVE datasets (1M+ samples):

```bash
# Use SOLLOL cluster to distribute training
python llamaforge_interactive.py

# Select dataset when prompted
# Choose "Use SOLLOL distributed training"
# Training distributes across your cluster nodes
```

## 📈 Monitoring Training

### Dashboard
```bash
# Open training dashboard
python simple_dashboard.py

# Or from interactive CLI
python llamaforge_interactive.py
# Type: dashboard
```

**Dashboard shows**:
- CPU/RAM usage
- Active training jobs
- GPU detection
- Auto-refresh every 5 seconds

### Access URLs
- Local: http://localhost:5000
- Network: http://10.9.66.154:5000

## ✅ Post-Training Checklist

1. **Test the model**
   ```bash
   ollama run your-model-name
   >>> Hello, what can you do?
   >>> Write a Python function to reverse a string
   >>> Explain what a binary search tree is
   ```

2. **Compare to broken model**
   - Should have ZERO random Chinese characters
   - Should have ZERO "Cosby" repetitions
   - Should give coherent, helpful responses

3. **Quality checks**
   - [ ] Responds to greetings naturally
   - [ ] Can write code in requested language
   - [ ] Explains code clearly
   - [ ] Handles edge cases
   - [ ] Uses proper formatting
   - [ ] Stays on topic

## 🔥 Quick Start (TL;DR)

```bash
cd /home/joker/LlamaForge/examples/datasets

# Option 1: Wait for downloads to finish (BEST)
# Check progress: ls -lh *.jsonl

# Option 2: Use what's ready now
python /home/joker/LlamaForge/llamaforge.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data code_alpaca_full.jsonl \
  --epochs 3 \
  --batch-size 1 \
  --max-length 512 \
  --no-gguf \
  --work-dir ../work/code_alpaca_run

# Monitor dashboard
firefox http://localhost:5000 &

# Wait for training to complete (2-3 hours)
# Test model
ollama run your-new-model
```

---

**Remember**: Your previous "tester" model was trained on ONLY 5 samples. These new datasets have 20K-240K+ samples and will produce DRAMATICALLY better results.

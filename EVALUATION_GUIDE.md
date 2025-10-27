# Model Evaluation Guide - Measure Training Improvements

## 🎯 Purpose

**Quantify EXACTLY what your training improved:**
- Code generation accuracy
- Reasoning ability
- Instruction following
- General knowledge
- Math skills
- Common sense

## 🚀 Quick Start

### 1. Evaluate BEFORE Training (Baseline)

```bash
python evaluate_model.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --preset quick \
  --output ./evaluations/before_training
```

### 2. Train Your Model

```bash
python llamaforge.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/claude_ultimate_508k.jsonl \
  --epochs 1 \
  --batch-size 1 \
  --max-length 512 \
  --no-gguf \
  --work-dir ./trained_model
```

### 3. Evaluate AFTER Training

```bash
python evaluate_model.py \
  --model ./trained_model/merged_model \
  --preset quick \
  --output ./evaluations/after_training \
  --compare ./evaluations/before_training/results_*.json
```

## 📊 Evaluation Presets

### Quick (3 tasks, ~5-10 min)
```bash
--preset quick
```
**Tests:**
- HellaSwag (common sense)
- PIQA (physical reasoning)
- Winogrande (language understanding)

**Use when:** Fast iteration, testing if training worked at all

### Standard (10 tasks, ~20-40 min)
```bash
--preset standard
```
**Tests:**
- All Quick tasks +
- ARC-Easy & ARC-Challenge (science Q&A)
- BoolQ (yes/no questions)
- OpenBookQA (open book reasoning)
- TruthfulQA (truthfulness)
- MathQA & GSM8K (math reasoning)

**Use when:** Comprehensive evaluation of general ability

### Comprehensive (15+ tasks, ~1-2 hours)
```bash
--preset comprehensive
```
**Tests:**
- All Standard tasks +
- MMLU (massive multitask benchmark)
- HumanEval (Python code generation)
- MBPP (more Python code problems)
- LAMBADA (language modeling)
- WikiText (perplexity)

**Use when:** Publication-quality benchmarking, final evaluation

## 📈 Understanding Results

### Example Output

```
================================================================================
  TRAINING IMPROVEMENT REPORT
================================================================================

  hellaswag                      45.23% → 62.18%  ↑ +37.46%
  piqa                          58.91% → 71.34%  ↑ +21.09%
  winogrande                    52.17% → 58.43%  ↑ +12.00%
  arc_easy                      41.28% → 59.32%  ↑ +43.69%
  arc_challenge                 28.45% → 35.71%  ↑ +25.51%
  boolq                         62.35% → 68.92%  ↑ +10.54%
  mathqa                        23.47% → 31.25%  ↑ +33.15%
  gsm8k                         8.23% → 18.45%   ↑ +124.18%
```

### What Each Metric Means:

**45.23% → 62.18%  ↑ +37.46%**
- **Before:** 45.23% accuracy (baseline model)
- **After:** 62.18% accuracy (your trained model)
- **Improvement:** 37.46% relative improvement
- **Arrow ↑:** Positive improvement (green)
- **Arrow ↓:** Regression (red) - model got worse
- **Arrow →:** No change (yellow)

### Interpretation:

| Improvement | Quality | Meaning |
|-------------|---------|---------|
| +50%+ | 🔥 Excellent | Major capability gain |
| +20-50% | ✅ Good | Significant improvement |
| +5-20% | 👍 Decent | Noticeable improvement |
| 0-5% | ⚠️ Minimal | Barely improved |
| Negative | ❌ Bad | Model got worse (overfitting?) |

## 🎯 Task Descriptions

### Code Generation:
- **HumanEval**: Write Python functions from docstrings
- **MBPP**: Python programming problems

### Reasoning:
- **HellaSwag**: Complete sentences (common sense)
- **PIQA**: Physical world reasoning
- **Winogrande**: Pronoun resolution
- **ARC**: Science questions (easy & challenge)

### Knowledge:
- **MMLU**: 57 subjects (history, law, medicine, etc.)
- **OpenBookQA**: Elementary science

### Language:
- **BoolQ**: Yes/no questions
- **TruthfulQA**: Truthful vs plausible answers
- **LAMBADA**: Predict last word

### Math:
- **GSM8K**: Grade school math word problems
- **MathQA**: Multiple-choice math

## 💡 Usage Examples

### Evaluate Your Broken "tester" Model

```bash
# First, convert to HuggingFace format (if it's in Ollama)
# Skip this step if model already crashed during training

# Or evaluate base model before any training
python evaluate_model.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --preset quick \
  --output ./eval_baseline
```

### Evaluate After Small Dataset (20K samples)

```bash
python evaluate_model.py \
  --model ./work/code_alpaca_run/merged_model \
  --preset standard \
  --output ./eval_20k \
  --compare ./eval_baseline/results_*.json
```

### Evaluate After ULTIMATE Dataset (508K samples)

```bash
python evaluate_model.py \
  --model ./work/claude_ultimate_training/merged_model \
  --preset comprehensive \
  --output ./eval_508k \
  --compare ./eval_baseline/results_*.json
```

### Compare Two Trained Models

```bash
python evaluate_model.py \
  --model ./model_v2 \
  --preset standard \
  --output ./eval_v2 \
  --compare ./eval_v1/results_*.json
```

## 📊 Expected Improvements

### 5-Sample "tester" Model (BROKEN)
```
All tasks: 0-10% accuracy
- Mostly random guessing
- Chinese characters in output
- Repetitive garbage
```

### 1,000 Sample Model
```
Quick preset:
- hellaswag: 35-45%
- piqa: 50-60%
- winogrande: 48-55%
```

### 20,000 Sample Model (Code Alpaca)
```
Standard preset:
- Code tasks (HumanEval): 5-10%
- Reasoning: 45-60%
- Math: 15-25%
```

### 140,000 Sample Model
```
Standard preset:
- Code tasks: 10-20%
- Reasoning: 55-70%
- Math: 25-35%
- Knowledge: 40-55%
```

### 508,000 Sample Model (ULTIMATE)
```
Comprehensive preset:
- Code tasks (HumanEval): 15-30%
- Reasoning: 60-75%
- Math (GSM8K): 30-45%
- Knowledge (MMLU): 45-60%
- Instruction following: 65-80%
```

## 🔍 Debugging Poor Results

### Model Worse After Training (Negative %)

**Possible causes:**
1. **Overfitting** - Too many epochs on small dataset
2. **Learning rate too high** - Destroyed existing knowledge
3. **Dataset quality** - Garbage data
4. **Catastrophic forgetting** - Model forgot pre-training

**Solutions:**
- Reduce epochs (try 1-2 instead of 3-5)
- Lower learning rate (1e-5 instead of 2e-4)
- Use larger, more diverse dataset
- Add regularization (higher LoRA dropout)

### No Improvement (0-5%)

**Possible causes:**
1. **Learning rate too low** - Not learning
2. **Dataset too small** - Need more data
3. **Training too short** - Need more epochs
4. **LoRA rank too low** - Limited capacity

**Solutions:**
- Increase learning rate slightly
- Use bigger dataset (100K+ samples)
- Train longer (more epochs)
- Increase LoRA rank (16→32 or 32→64)

### Uneven Improvements

**Example:**
```
Code tasks: +80% 🔥
Math tasks: +5% ⚠️
Reasoning: -10% ❌
```

**Cause:** Dataset imbalance (90% code, 5% math, 5% reasoning)

**Solution:** Balance dataset composition or do multiple training rounds

## 🎓 Advanced Usage

### Custom Task Selection

```bash
lm_eval --model hf \
  --model_args pretrained=./my_model \
  --tasks hellaswag,piqa,humaneval \
  --batch_size 1 \
  --limit 50 \
  --output_path ./custom_eval
```

### Evaluate Ollama Models Directly

```bash
# Convert Ollama→HF first
ollama pull your-model
# Then use llamaforge's conversion tools
```

### Save Evaluation Reports

```bash
python evaluate_model.py \
  --model ./model \
  --preset comprehensive \
  --output ./reports/$(date +%Y%m%d) | tee evaluation_report.txt
```

## 📝 Full Workflow Example

```bash
# 1. Baseline evaluation
python evaluate_model.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --preset quick \
  --limit 100 \
  --output ./eval/baseline

# 2. Train on small dataset first
python llamaforge.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/code_alpaca_full.jsonl \
  --epochs 3 \
  --no-gguf \
  --work-dir ./training/code_20k

# 3. Evaluate small model
python evaluate_model.py \
  --model ./training/code_20k/merged_model \
  --preset quick \
  --output ./eval/code_20k \
  --compare ./eval/baseline/results_*.json

# 4. If results good, train on ULTIMATE dataset
python llamaforge.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/claude_ultimate_508k.jsonl \
  --epochs 1 \
  --batch-size 1 \
  --max-length 512 \
  --no-gguf \
  --work-dir ./training/ultimate_508k

# 5. Final comprehensive evaluation
python evaluate_model.py \
  --model ./training/ultimate_508k/merged_model \
  --preset comprehensive \
  --output ./eval/ultimate_508k \
  --compare ./eval/baseline/results_*.json

# 6. Generate final report
echo "Training complete! Check ./eval/ultimate_508k/ for results"
```

## 🎯 TL;DR

**Before Training:**
```bash
python evaluate_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --preset quick --output ./eval/before
```

**After Training:**
```bash
python evaluate_model.py --model ./trained_model/merged_model --preset quick --output ./eval/after --compare ./eval/before/results_*.json
```

**Result:**
```
hellaswag: 45% → 62% ↑ +37%  🔥 SIGNIFICANT IMPROVEMENT
```

Now you have **QUANTIFIABLE PROOF** your training worked!

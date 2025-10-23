# RAM Estimation Feature

## Overview
Added intelligent RAM estimation to help users select models that will actually train on their system.

## What Was Added

### 1. RAM Estimation Functions (`src/ollama_utils.py`)

**New Functions:**
- `estimate_model_params(model_name)` - Extracts parameter count from model name
- `estimate_training_ram(param_count_b, use_8bit)` - Calculates RAM needed for LoRA training
- `get_available_ram()` - Returns available system RAM
- `can_train_model(model_name, use_8bit)` - Checks if model fits in RAM

**LoRA-Aware Calculation:**
```
RAM = Base Model (FP32) + LoRA Adapters (1%) + Gradients + Optimizer + Activations
```

For TinyLlama (1.1B):
- Base model: 4.4 GB
- LoRA adapters: 0.04 GB
- Gradients: 0.04 GB
- Optimizer (8-bit): 0.02 GB
- Activations: 2.2 GB
- **Total: ~8.7 GB**

### 2. Visual RAM Indicators (Interactive CLI)

**Color-Coded Display:**
- ✓ **Green** - Trainable (< 50% of available RAM)
- ⚠ **Yellow** - Tight (50-80% of available RAM)
- ✗ **Red** - Too Large (> 80% of available RAM)

**Example Output:**
```
┌─ Ollama Models (80 total)
├─ Legend: ✓ Trainable | ⚠ Tight | ✗ Too Large
├─ ✓  1. tinyllama:latest            (  637MB)  ~ 8.7GB
├─ ⚠  2. gemma:2b                    (1.6GB)    ~15.9GB
├─ ✗  3. gemma:7b                    (4.8GB)    ~55.5GB
```

### 3. Warning System

When selecting a model that's too large:
```
[!] WARNING: This model may be too large for your system!
[!] WARNING: Training may fail due to insufficient memory (OOM kill)
[i] Consider using a smaller model or cloud GPU training
> Continue anyway? (yes/no) [no]:
```

## Supported Models

The system recognizes parameter counts from:
- Model name patterns: `llama3.1:8b` → 8B params
- Known model mappings: `tinyllama` → 1.1B, `phi3:mini` → 3.8B, etc.
- Falls back to 7B estimate for unknown models

## Technical Details

**Safety Threshold:** 80% of available RAM
- Accounts for OS, other processes, and memory fragmentation
- Based on real-world OOM kill experience

**LoRA Overhead:** ~1% of base model params
- Only LoRA adapters are trainable
- Base model is frozen (no gradients needed)
- Significantly reduces memory requirements

**8-bit Optimizer:** Saves ~75% on optimizer memory
- Automatically used if `bitsandbytes` is installed
- Critical for CPU training with limited RAM

## Usage

The feature is automatically enabled in the interactive CLI:

```bash
python llamaforge_interactive.py
```

No configuration needed - RAM checks happen automatically during model selection.

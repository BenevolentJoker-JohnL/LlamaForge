# Automatic Ollama Model Creation

## Overview
LlamaForge now automatically creates Ollama models after training - no more manual steps!

## How It Works

### 1. During Setup (Step 1: Model Selection)

Right after you select your base model, LlamaForge asks for your fine-tuned model name:

```
▶ OLLAMA MODEL NAME
[i] Choose a name for your fine-tuned model in Ollama

> Ollama model name (or 'skip' for manual setup) [qwen2.5-finetuned]:
```

**Smart Name Suggestions:**
- If you selected `qwen2.5:0.5b` → suggests `qwen2.5-finetuned`
- If you selected `llama3.1:8b` → suggests `llama3.1-finetuned`
- Custom HuggingFace models → suggests `{model-name}-finetuned`

**Options:**
- Press Enter to accept the suggestion
- Type a custom name (e.g., `my-coding-assistant`)
- Type `skip` if you want manual setup later

**Duplicate Handling:**

If the model name already exists in Ollama, you'll get these options:

```
[!] WARNING: Model 'qwen2.5-finetuned' already exists in Ollama!

> What would you like to do?
  1. Overwrite existing model
  2. Choose a different name (default)
  3. Skip Ollama creation (manual setup)
```

- **Option 1 (Overwrite):** Replaces the existing model after training
- **Option 2 (Different name):** Suggests a versioned name (e.g., `qwen2.5-finetuned-v2`)
- **Option 3 (Skip):** No automatic creation, you'll get manual instructions

### 2. After Training

LlamaForge automatically:
- ✅ Detects the correct chat template for your model
- ✅ Creates a properly configured Modelfile
- ✅ Runs `ollama create` to import the model
- ✅ Shows you how to run it

### 3. Ready to Use

```bash
ollama run qwen2.5-finetuned
```

That's it!

## Supported Chat Templates

LlamaForge automatically detects the right template:

| Model Family | Template Format | Stop Tokens |
|--------------|----------------|-------------|
| Qwen         | ChatML         | `<\|im_start\|>`, `<\|im_end\|>` |
| Llama 3.x    | Special format | `<\|eot_id\|>` |
| Gemma        | Turn-based     | `<end_of_turn>` |
| Others       | ChatML (default) | `<\|im_start\|>`, `<\|im_end\|>` |

## Default Parameters

All models are configured with:
- **Temperature:** 0.7 (balanced creativity)
- **Top-p:** 0.9 (nucleus sampling)
- **Top-k:** 40 (token diversity)
- **Context:** 2048 tokens
- **System:** "You are a helpful AI assistant trained on practical coding tasks."

## Error Handling

If automatic creation fails, LlamaForge will:
1. Show the error message
2. Display manual creation commands
3. Keep the generated Modelfile for manual import

## Example Output

```
╔════════════════════════════════════════════════════════════════════════════╗
║                          PHASE 5: OLLAMA INTEGRATION                       ║
╚════════════════════════════════════════════════════════════════════════════╝

[i] Created Modelfile: /path/to/Modelfile
⠹ Importing to Ollama...
[✓] Ollama model 'qwen2.5-finetuned' created successfully!

═══════════════════════════════════════════════════════════════════════════
║                    TRAINING SEQUENCE COMPLETE                            ║
═══════════════════════════════════════════════════════════════════════════

[✓] MODEL READY FOR DEPLOYMENT
    Location: /path/to/finetuned-model-fp16.gguf

▶ RUN YOUR MODEL:
     ollama run qwen2.5-finetuned
```

## Manual Override

Don't want automatic creation? Just say "no" during setup:

```
> Create Ollama model automatically? (yes/no) [yes]: no
```

LlamaForge will show manual instructions instead.

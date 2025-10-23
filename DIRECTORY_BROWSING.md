# Directory Browsing Feature

The interactive CLI now supports **directory browsing**! Just paste a directory path and select from available datasets.

## How It Works

When you run:
```bash
python llamaforge_interactive.py
```

### Before (Old Way)
```
[2/5] DATASET CONFIGURATION
[i] Supported formats: JSON, JSONL, CSV, TXT

> Dataset file path: ~/LlamaForge/testtraindata
[✗] ERROR: File not found: ~/LlamaForge/testtraindata
```

### After (New Way with Directory Browsing)
```
[2/5] ██░░░ > DATASET CONFIGURATION
[i] Supported formats: JSON, JSONL, CSV, TXT
Auto-structuring enabled for all formats
Tip: You can paste a directory path to browse files

> Dataset file path or directory: ~/LlamaForge/testtraindata

[✓] Directory found: /home/joker/LlamaForge/testtraindata

┌─ Available Datasets (3 found)
├─ 1. code_alpaca.jsonl (335.5 KB)
├─ 2. practical_coding.jsonl (1.8 KB)
└─ 3. sql_generation.jsonl (153.1 KB)

> Select dataset number (1-3) [1]: 1

▰▰▰▰▰ Loading dataset COMPLETE
[✓] Dataset selected: code_alpaca.jsonl (JSONL format)
```

## Features

### ✅ Automatic Detection
- **Directories**: Shows all compatible files
- **Files**: Uses directly
- **~ expansion**: `~/LlamaForge/testtraindata` works!

### ✅ Smart File Listing
- Shows file sizes (KB/MB)
- Supports up to 20 files (with overflow indicator)
- Sorted alphabetically
- Beautiful tree-style display

### ✅ Multiple Formats
Automatically finds:
- `.json` files
- `.jsonl` files
- `.csv` files
- `.txt` files

## Usage Examples

### Example 1: Browse testtraindata
```
> Dataset file path or directory: testtraindata

┌─ Available Datasets (3 found)
├─ 1. code_alpaca.jsonl (335.5 KB)
├─ 2. practical_coding.jsonl (1.8 KB)
└─ 3. sql_generation.jsonl (153.1 KB)

> Select dataset number: 1
```

### Example 2: Browse examples
```
> Dataset file path or directory: examples/datasets

┌─ Available Datasets (4 found)
├─ 1. code_generation.jsonl (0.9 KB)
├─ 2. instruction_following.jsonl (1.1 KB)
├─ 3. qa_pairs.json (0.8 KB)
└─ 4. sentiment.csv (0.3 KB)

> Select dataset number: 2
```

### Example 3: Direct file path (still works!)
```
> Dataset file path or directory: testtraindata/code_alpaca.jsonl

▰▰▰▰▰ Loading dataset COMPLETE
[✓] Dataset validated: JSONL format detected
```

### Example 4: Use tilde for home
```
> Dataset file path or directory: ~/LlamaForge/testtraindata

[✓] Directory found: /home/joker/LlamaForge/testtraindata
...
```

## Benefits

1. **No more typing full paths** - Just paste the directory
2. **See file sizes** - Know what you're selecting
3. **Quick browsing** - View all available datasets at once
4. **Error prevention** - Only shows valid dataset files
5. **Works everywhere** - Supports relative, absolute, and ~ paths

## Pro Tips

### Tip 1: Quick access to training data
```bash
python llamaforge_interactive.py
# At dataset prompt, type: testtraindata
# Select from the list
```

### Tip 2: Explore examples
```bash
# At dataset prompt, type: examples/datasets
# Browse all example datasets
```

### Tip 3: Use autocomplete
```bash
# In your terminal, use tab completion:
# testtraindata/<TAB>
# Then paste the full path
```

## Keyboard Shortcuts

- **Enter** on directory: Browse files
- **Number + Enter**: Select file
- **Ctrl+C**: Cancel and exit
- **Invalid path**: Shows helpful suggestions

## What's Next?

Try it now:
```bash
python llamaforge_interactive.py
```

1. Select your Ollama model (e.g., `qwen2.5:3b`)
2. **Paste**: `testtraindata` or `~/LlamaForge/testtraindata`
3. **Select**: Dataset from the list
4. Configure training parameters
5. Start training!

---

**No more path errors! Browse your datasets like a pro.** 🎯

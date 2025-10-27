# Dataset Requirements for Fine-Tuning

## Minimum Requirements

- **Minimum samples**: 500-1000+ for basic fine-tuning
- **Recommended**: 5,000-10,000+ for quality results
- **Format**: JSONL with "instruction" and "output" fields

## Current Datasets

### instruction_following.jsonl
**STATUS**: ⚠️ TOO SMALL (only 5 samples)
**USE CASE**: Example/testing only
**DO NOT USE FOR ACTUAL TRAINING**

## Where to Get Good Datasets

1. **HuggingFace Datasets**:
   - mlabonne/guanaco-llama2-1k (1K instruction pairs)
   - yahma/alpaca-cleaned (52K instruction pairs)
   - Open-Orca/OpenOrca (4M+ samples)

2. **Create Your Own**:
   - Collect domain-specific Q&A pairs
   - Use GPT to generate synthetic training data
   - Scrape and format relevant text

## Example Download Script

```bash
# Install datasets library
pip install datasets

# Download and convert Alpaca dataset
python3 << 'PYTHON'
from datasets import load_dataset
import json

ds = load_dataset("yahma/alpaca-cleaned")

with open("alpaca_train.jsonl", "w") as f:
    for item in ds["train"]:
        f.write(json.dumps({
            "instruction": item["instruction"],
            "output": item["output"]
        }) + "\n")
PYTHON
```

## Training Tips

- **Start small**: Test with 100-500 samples first
- **Quality > Quantity**: Clean, well-formatted data is critical
- **Diverse examples**: Cover different use cases
- **Consistent format**: Same instruction/output structure

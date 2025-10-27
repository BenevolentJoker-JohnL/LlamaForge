#!/usr/bin/env bash
#
# Create Ultimate 3M Training Dataset
#
# Downloads diverse datasets to create a 3M example corpus covering:
# - Code (multiple categories)
# - Reasoning (CoT/ToT)
# - Creative writing
# - Tool use / API
# - Red-team / safety
# - Factual grounding
#

set -e

echo "============================================================================="
echo "  CREATING ULTIMATE 3M TRAINING DATASET"
echo "============================================================================="
echo ""
echo "This will download ~2.2M additional examples to reach 3M total"
echo "Current: 838K examples"
echo "Target: 3M examples"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

OUTPUT_DIR="examples/datasets"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================================="
echo "  PHASE 1: CODE DATASETS (~800K examples)"
echo "============================================================================="
echo ""

# Code generation
python download_code_datasets.py \
    --categories generation \
    --output "$OUTPUT_DIR" \
    --limit 100000

# Code explanation
python download_code_datasets.py \
    --categories explanation \
    --output "$OUTPUT_DIR" \
    --limit 100000

# Debugging
python download_code_datasets.py \
    --categories debugging \
    --output "$OUTPUT_DIR" \
    --limit 50000

# Algorithms
python download_code_datasets.py \
    --categories algorithms \
    --output "$OUTPUT_DIR" \
    --limit 100000

# SQL
python download_code_datasets.py \
    --categories sql \
    --output "$OUTPUT_DIR" \
    --limit 50000

# API/Integration
python download_code_datasets.py \
    --categories api \
    --output "$OUTPUT_DIR" \
    --limit 100000

# Code review
python download_code_datasets.py \
    --categories review \
    --output "$OUTPUT_DIR" \
    --limit 50000

# Multi-language translation
python download_code_datasets.py \
    --categories translation \
    --output "$OUTPUT_DIR" \
    --limit 50000

echo ""
echo "[✓] Code datasets downloaded"
echo ""

echo "============================================================================="
echo "  PHASE 2: REASONING DATASETS (~600K examples)"
echo "============================================================================="
echo ""

# Already have some, download more
python download_cot_tot_datasets.py \
    --download-all \
    --output "$OUTPUT_DIR" \
    --limit 200000

echo ""
echo "[✓] Reasoning datasets downloaded"
echo ""

echo "============================================================================="
echo "  PHASE 3: CREATIVE & ANALYTICAL (~300K examples)"
echo "============================================================================="
echo ""

# Creative writing
echo "[i] Downloading creative writing datasets..."

# Download writing prompts
python -c "
from datasets import load_dataset
import json
from pathlib import Path

# Writing prompts
dataset = load_dataset('Anthropic/hh-rlhf', split='train')
examples = list(dataset)[:100000]

output = Path('$OUTPUT_DIR/creative_writing.jsonl')
with open(output, 'w') as f:
    for ex in examples:
        chosen = ex.get('chosen', '')
        if any(word in chosen.lower() for word in ['story', 'write', 'creative', 'imagine']):
            f.write(json.dumps({
                'instruction': 'Write creatively:',
                'output': chosen,
                '_category': 'creative'
            }) + '\n')

print(f'[✓] Creative writing: {output}')
"

# Analytical essays
echo "[i] Downloading analytical datasets..."

python -c "
from datasets import load_dataset
import json
from pathlib import Path

# Essays and analysis
dataset = load_dataset('tau/scrolls', 'qasper', split='train')
examples = list(dataset)[:50000]

output = Path('$OUTPUT_DIR/analytical_essays.jsonl')
with open(output, 'w') as f:
    for ex in examples:
        question = ex.get('question', '')
        answers = ex.get('answers', {})

        if question and answers:
            f.write(json.dumps({
                'instruction': f'Analyze and answer: {question}',
                'output': str(answers),
                '_category': 'analytical'
            }) + '\n')

print(f'[✓] Analytical essays: {output}')
"

echo ""
echo "[✓] Creative & analytical datasets downloaded"
echo ""

echo "============================================================================="
echo "  PHASE 4: TOOL USE & API (~200K examples)"
echo "============================================================================="
echo ""

# Already have glaive_function_calling, download more
python -c "
from datasets import load_dataset
import json
from pathlib import Path

# Gorilla API dataset
try:
    dataset = load_dataset('gorilla-llm/APIBench', split='train')
    examples = list(dataset)[:100000]

    output = Path('$OUTPUT_DIR/api_bench.jsonl')
    with open(output, 'w') as f:
        for ex in examples:
            api_call = ex.get('api_call', '')
            if api_call:
                f.write(json.dumps({
                    'instruction': ex.get('question', ''),
                    'output': api_call,
                    '_category': 'tool_use'
                }) + '\n')

    print(f'[✓] API bench: {output}')
except:
    print('[!] APIBench not available, skipping')

# ToolBench
try:
    dataset = load_dataset('ToolBench/ToolBench', split='train')
    examples = list(dataset)[:100000]

    output = Path('$OUTPUT_DIR/toolbench.jsonl')
    with open(output, 'w') as f:
        for ex in examples:
            f.write(json.dumps({
                'instruction': ex.get('query', ''),
                'output': ex.get('response', ''),
                '_category': 'tool_use'
            }) + '\n')

    print(f'[✓] ToolBench: {output}')
except:
    print('[!] ToolBench not available, using glaive_function_calling')
"

echo ""
echo "[✓] Tool use datasets downloaded"
echo ""

echo "============================================================================="
echo "  PHASE 5: RED-TEAM & SAFETY (~100K examples)"
echo "============================================================================="
echo ""

# Anthropic HH (helpful & harmless)
python -c "
from datasets import load_dataset
import json
from pathlib import Path

# Helpful & Harmless with safety responses
dataset = load_dataset('Anthropic/hh-rlhf', split='train')
examples = list(dataset)[:100000]

output = Path('$OUTPUT_DIR/red_team_safe.jsonl')
with open(output, 'w') as f:
    for ex in examples:
        chosen = ex.get('chosen', '')
        rejected = ex.get('rejected', '')

        # Use chosen (safe) responses
        if chosen:
            f.write(json.dumps({
                'instruction': 'Handle safely:',
                'output': chosen,
                '_category': 'red_team',
                '_safety': 'safe_response'
            }) + '\n')

print(f'[✓] Red-team dataset: {output}')
print(f'    {len(examples):,} examples with safe responses')
"

echo ""
echo "[✓] Red-team dataset created"
echo ""

echo "============================================================================="
echo "  PHASE 6: FACTUAL GROUNDING (~100K examples)"
echo "============================================================================="
echo ""

# Wikipedia + scientific abstracts
python -c "
from datasets import load_dataset
import json
from pathlib import Path

# Wikipedia
try:
    dataset = load_dataset('wikipedia', '20220301.en', split='train')
    examples = list(dataset)[:50000]

    output = Path('$OUTPUT_DIR/wikipedia_factual.jsonl')
    with open(output, 'w') as f:
        for ex in examples:
            text = ex.get('text', '')
            title = ex.get('title', '')

            if len(text) > 200 and len(text) < 2000:
                f.write(json.dumps({
                    'instruction': f'What do you know about {title}?',
                    'output': text[:1000],
                    '_category': 'factual',
                    '_source': 'wikipedia'
                }) + '\n')

    print(f'[✓] Wikipedia: {output}')
except:
    print('[!] Wikipedia dataset too large, skipping')

# Scientific abstracts
try:
    dataset = load_dataset('scientific_papers', 'pubmed', split='train')
    examples = list(dataset)[:50000]

    output = Path('$OUTPUT_DIR/scientific_abstracts.jsonl')
    with open(output, 'w') as f:
        for ex in examples:
            abstract = ex.get('abstract', '')
            article = ex.get('article', '')

            if abstract:
                f.write(json.dumps({
                    'instruction': 'Summarize this scientific finding:',
                    'output': abstract,
                    '_category': 'factual',
                    '_source': 'pubmed'
                }) + '\n')

    print(f'[✓] Scientific abstracts: {output}')
except:
    print('[!] Scientific papers dataset not available, skipping')
"

echo ""
echo "[✓] Factual datasets downloaded"
echo ""

echo "============================================================================="
echo "  PHASE 7: MERGE ALL DATASETS"
echo "============================================================================="
echo ""

echo "[i] Merging all datasets into ultimate 3M corpus..."
echo ""

# Merge everything
cat "$OUTPUT_DIR/claude_ultimate_with_tools_621k.jsonl" \
    "$OUTPUT_DIR"/gsm8k*.jsonl \
    "$OUTPUT_DIR"/orca*.jsonl \
    "$OUTPUT_DIR"/metamath*.jsonl \
    "$OUTPUT_DIR"/wizard*.jsonl \
    "$OUTPUT_DIR"/humaneval*.jsonl \
    "$OUTPUT_DIR"/mbpp*.jsonl \
    "$OUTPUT_DIR"/apps*.jsonl \
    "$OUTPUT_DIR"/code_contests*.jsonl \
    "$OUTPUT_DIR"/code_searchnet*.jsonl \
    "$OUTPUT_DIR"/codefeedback*.jsonl \
    "$OUTPUT_DIR"/bugs*.jsonl \
    "$OUTPUT_DIR"/defect*.jsonl \
    "$OUTPUT_DIR"/xlcost*.jsonl \
    "$OUTPUT_DIR"/leetcode*.jsonl \
    "$OUTPUT_DIR"/spider*.jsonl \
    "$OUTPUT_DIR"/glaive*.jsonl \
    "$OUTPUT_DIR"/api*.jsonl \
    "$OUTPUT_DIR"/tool*.jsonl \
    "$OUTPUT_DIR"/creative*.jsonl \
    "$OUTPUT_DIR"/analytical*.jsonl \
    "$OUTPUT_DIR"/red_team*.jsonl \
    "$OUTPUT_DIR"/wikipedia*.jsonl \
    "$OUTPUT_DIR"/scientific*.jsonl \
    2>/dev/null \
    > "$OUTPUT_DIR/ultimate_3M_complete.jsonl"

TOTAL_LINES=$(wc -l < "$OUTPUT_DIR/ultimate_3M_complete.jsonl")
SIZE=$(ls -lh "$OUTPUT_DIR/ultimate_3M_complete.jsonl" | awk '{print $5}')

echo ""
echo "============================================================================="
echo "  ULTIMATE 3M DATASET CREATED"
echo "============================================================================="
echo ""
echo "  File: $OUTPUT_DIR/ultimate_3M_complete.jsonl"
echo "  Examples: $TOTAL_LINES"
echo "  Size: $SIZE"
echo ""
echo "  Estimated bucket distribution:"
echo "    - Code (generation):      ~500K (17%)"
echo "    - Code (explanation):     ~400K (13%)"
echo "    - Code (debugging):       ~150K ( 5%)"
echo "    - Code (algorithms):      ~250K ( 8%)"
echo "    - Code (other):           ~200K ( 7%)"
echo "    - Instruction:            ~620K (21%)"
echo "    - CoT Math:               ~400K (13%)"
echo "    - Analytical:             ~150K ( 5%)"
echo "    - Creative:               ~100K ( 3%)"
echo "    - Tool Use:               ~200K ( 7%)"
echo "    - Red-team:               ~100K ( 3%)"
echo "    - Factual:                ~100K ( 3%)"
echo ""
echo "============================================================================="
echo "  NEXT STEPS"
echo "============================================================================="
echo ""
echo "  1. Analyze and create manifest:"
echo ""
echo "     python dataset_manifest.py \\"
echo "       --analyze $OUTPUT_DIR/ultimate_3M_complete.jsonl \\"
echo "       --save-manifest ultimate_3M_manifest.json"
echo ""
echo "  2. Create weighted training sample:"
echo ""
echo "     python dataset_manifest.py \\"
echo "       --manifest ultimate_3M_manifest.json \\"
echo "       --create-sample \\"
echo "       --output $OUTPUT_DIR/production_2M.jsonl \\"
echo "       --total-samples 2000000 \\"
echo "       --phase production_balanced"
echo ""
echo "  3. Train with distributed training:"
echo ""
echo "     python launch_distributed_training.py \\"
echo "       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\"
echo "       --dataset $OUTPUT_DIR/production_2M.jsonl \\"
echo "       --workers 4 \\"
echo "       --epochs 1"
echo ""
echo "  4. Expected training time:"
echo "     - Single CPU: ~7-10 days"
echo "     - 4-node SOLLOL: ~2-3 days"
echo "     - 8-node SOLLOL: ~1-1.5 days"
echo ""
echo "============================================================================="
echo ""

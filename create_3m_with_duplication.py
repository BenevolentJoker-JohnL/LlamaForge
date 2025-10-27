#!/usr/bin/env python3
"""
Intelligent Dataset Duplication Manager for 3M Corpus

Applies controlled duplication with domain-specific weighting to:
- Reinforce critical behaviors (code, tool use, safety)
- Maintain diversity (reasoning, creative)
- Reach target size (3M examples) from smaller corpus

Based on optimal duplication ratios:
- Code: 1.8× (API patterns, syntax)
- Reasoning/CoT: 1.0-1.3× (keep diversity)
- Tool use: 2.0× (memorize schemas)
- Creative: 1.0× (avoid fatigue)
- Safety: 3.0× (amplify rare refusals)
- Instruction: 1.2-1.5× (format compliance)

Usage:
    python create_3m_with_duplication.py \
        --input examples/datasets/*.jsonl \
        --output examples/datasets/ultimate_3M.jsonl \
        --target-size 3000000
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import hashlib


# Domain-specific duplication ratios
DUPLICATION_RATIOS = {
    "code": 1.8,              # Reinforce coding patterns
    "tool_use": 2.0,          # Memorize API schemas
    "cot_math": 1.3,          # Boost reasoning depth
    "analytical": 1.0,        # Keep diversity
    "red_team": 3.0,          # Amplify safety responses
    "creative": 1.0,          # Avoid style fatigue
    "instruction": 1.4,       # Format compliance
    "factual": 1.0,           # Maintain accuracy
}


def classify_example(example: Dict) -> str:
    """
    Classify example into domain bucket based on content and metadata.
    """
    # Check metadata first
    if "_category" in example:
        category = example["_category"].lower()
        if category in DUPLICATION_RATIOS:
            return category
        # Map variants
        if "code" in category or "programming" in category:
            return "code"
        if "tool" in category or "api" in category or "function" in category:
            return "tool_use"
        if "cot" in category or "reasoning" in category or "math" in category:
            return "cot_math"
        if "creative" in category or "writing" in category:
            return "creative"
        if "red" in category or "safety" in category or "refusal" in category:
            return "red_team"
        if "analyt" in category:
            return "analytical"
        if "factual" in category or "wiki" in category:
            return "factual"

    # Content-based classification
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    combined = (instruction + " " + output).lower()

    # Code indicators
    if any(ind in combined for ind in ["```", "def ", "function ", "class ", "import ", "const ", "let ", "var "]):
        return "code"

    # Tool use indicators
    if any(ind in combined for ind in ["<tool>", "api", "function_call", "tool_use:", "endpoint"]):
        return "tool_use"

    # CoT/Math indicators
    if any(ind in combined for ind in ["<thinking>", "step 1:", "step 2:", "reasoning:", "let's think"]):
        return "cot_math"

    # Safety indicators
    if any(ind in combined for ind in ["i cannot", "i can't help", "inappropriate", "harmful", "refuse", "decline"]):
        return "red_team"

    # Creative indicators
    if any(ind in combined for ind in ["story", "poem", "creative", "imagine", "narrative", "character"]):
        return "creative"

    # Analytical indicators
    if any(ind in combined for ind in ["analyze", "analysis", "compare", "contrast", "evaluate", "assess"]):
        return "analytical"

    # Default: instruction
    return "instruction"


def deduplicate_examples(examples: List[Dict]) -> List[Dict]:
    """
    Remove exact duplicates based on content hash.
    """
    seen_hashes = set()
    unique_examples = []

    for ex in examples:
        # Create hash from instruction + output
        content = json.dumps({
            "instruction": ex.get("instruction", ""),
            "output": ex.get("output", "")
        }, sort_keys=True)
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_examples.append(ex)

    print(f"[i] Removed {len(examples) - len(unique_examples):,} duplicates")
    print(f"[i] Unique examples: {len(unique_examples):,}")

    return unique_examples


def apply_intelligent_duplication(examples: List[Dict], target_size: int) -> List[Dict]:
    """
    Apply domain-weighted duplication to reach target size.
    """
    # Classify examples by domain
    print("[i] Classifying examples by domain...")
    domain_examples = defaultdict(list)

    for ex in examples:
        domain = classify_example(ex)
        domain_examples[domain].append(ex)

    # Show distribution
    print("\n[✓] Domain distribution:")
    for domain in sorted(domain_examples.keys()):
        count = len(domain_examples[domain])
        pct = 100 * count / len(examples)
        ratio = DUPLICATION_RATIOS.get(domain, 1.0)
        print(f"  {domain:15s} {count:8,} ({pct:5.1f}%) → ratio: {ratio:.1f}×")

    # Calculate sampling probabilities
    print("\n[i] Applying weighted duplication...")
    weighted_examples = []

    for domain, domain_exs in domain_examples.items():
        ratio = DUPLICATION_RATIOS.get(domain, 1.0)
        # Number of times to sample from this domain
        target_count = int(len(domain_exs) * ratio)

        # Sample with replacement
        sampled = random.choices(domain_exs, k=target_count)
        weighted_examples.extend(sampled)

        print(f"  {domain:15s} {len(domain_exs):8,} → {target_count:8,} (added {target_count - len(domain_exs):+,})")

    # Shuffle
    random.shuffle(weighted_examples)

    # Trim or pad to exact target
    current_size = len(weighted_examples)

    if current_size < target_size:
        # Need to add more - sample proportionally from all domains
        shortfall = target_size - current_size
        print(f"\n[i] Adding {shortfall:,} more examples to reach target...")
        additional = random.choices(weighted_examples, k=shortfall)
        weighted_examples.extend(additional)
    elif current_size > target_size:
        # Trim excess
        print(f"\n[i] Trimming {current_size - target_size:,} examples to reach target...")
        weighted_examples = weighted_examples[:target_size]

    return weighted_examples


def main():
    parser = argparse.ArgumentParser(description="Intelligent dataset duplication manager")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL files (can use glob patterns)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--target-size", type=int, default=3000000, help="Target number of examples (default: 3M)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 80)
    print("  INTELLIGENT DUPLICATION MANAGER - 3M CORPUS BUILDER")
    print("=" * 80)
    print(f"\nTarget size: {args.target_size:,} examples")
    print(f"Random seed: {args.seed}")
    print()

    # Load all input files
    all_examples = []

    for input_path in args.input:
        path = Path(input_path)
        if not path.exists():
            print(f"[!] File not found: {input_path}")
            continue

        print(f"[i] Loading {path.name}...")
        with open(path, 'r') as f:
            examples = [json.loads(line) for line in f if line.strip()]
            all_examples.extend(examples)
            print(f"    Loaded: {len(examples):,} examples")

    print(f"\n[✓] Total examples loaded: {len(all_examples):,}")

    # Deduplicate
    if not args.no_dedup:
        print("\n[i] Deduplicating...")
        all_examples = deduplicate_examples(all_examples)

    # Apply intelligent duplication
    print()
    final_examples = apply_intelligent_duplication(all_examples, args.target_size)

    # Write output
    print(f"\n[i] Writing {len(final_examples):,} examples to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ex in final_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"[✓] Saved: {output_path}")
    print(f"[✓] Final size: {len(final_examples):,} examples")

    # Final distribution
    domain_counts = defaultdict(int)
    for ex in final_examples:
        domain = classify_example(ex)
        domain_counts[domain] += 1

    print("\n" + "=" * 80)
    print("  FINAL 3M CORPUS DISTRIBUTION")
    print("=" * 80)
    print()
    for domain in sorted(domain_counts.keys()):
        count = domain_counts[domain]
        pct = 100 * count / len(final_examples)
        print(f"  {domain:15s} {count:10,} ({pct:5.1f}%)")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

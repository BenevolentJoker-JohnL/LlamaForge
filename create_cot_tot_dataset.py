#!/usr/bin/env python3
"""
Create CoT/ToT training dataset from existing data

Augments your existing datasets with:
1. Chain of Thought reasoning
2. Tree of Thought exploration
3. Meta-cognitive examples (when to use each strategy)
4. Programmatic reasoning patterns

Usage:
    python create_cot_tot_dataset.py --input examples/datasets/claude_ultimate_508k.jsonl \
                                      --output examples/datasets/claude_cot_enhanced.jsonl \
                                      --augmentation-rate 0.3
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import re


class CoTToTAugmenter:
    """Augment training data with Chain of Thought and Tree of Thought examples"""

    # Patterns that benefit from CoT
    COT_TRIGGERS = [
        r"\bcalculate\b", r"\bsolve\b", r"\bprove\b",
        r"\bfind\b", r"\bdetermine\b", r"\bwhy\b",
        r"\bexplain\b", r"\bhow\b", r"\bcompare\b",
        r"step.by.step", r"\banalyze\b", r"\bdebug\b"
    ]

    # Patterns that benefit from ToT
    TOT_TRIGGERS = [
        r"\bbest\b", r"\boptimal\b", r"\bchoose\b",
        r"\bselect\b", r"\bdesign\b", r"\barchitecture\b",
        r"\btrade.?off", r"\balternative", r"\boption"
    ]

    # Meta-cognitive examples (teach when to use reasoning)
    META_EXAMPLES = [
        {
            "instruction": "When should you use chain-of-thought reasoning vs direct answers?",
            "output": """Use **Chain of Thought (CoT)** when:
1. **Multi-step problems**: Math, logic puzzles, algorithms
2. **Debugging**: Need to trace through execution
3. **Teaching**: User needs to understand the process
4. **Verification**: Answer needs to be checked
5. **Complex reasoning**: Multiple facts must be combined

Example needing CoT:
"If Alice has 3 apples and Bob gives her 5 more, then she gives 2 to Carol, how many does Alice have?"
→ Use CoT to show: 3 + 5 = 8, then 8 - 2 = 6

Use **Direct Answer** when:
1. **Simple facts**: "What is the capital of France?" → Paris
2. **Definitions**: "What is HTTP?" → HyperText Transfer Protocol
3. **Quick lookups**: "Python list method to add item?" → append()
4. **Already obvious**: "What is 2+2?" → 4

The key is: Will showing your work help the user understand or verify the answer?"""
        },
        {
            "instruction": "When should you use tree-of-thought reasoning?",
            "output": """Use **Tree of Thought (ToT)** when:
1. **Multiple valid approaches** exist
2. **Trade-offs** need evaluation
3. **Optimization** problems (best solution among many)
4. **Design decisions** with pros/cons
5. **Comparing alternatives**

Example needing ToT:
"What's the best database for a social media app?"

<thinking>
Option 1: PostgreSQL
- Pros: ACID compliance, relational integrity, mature
- Cons: Harder to scale horizontally
- Use case: Strong consistency needed

Option 2: MongoDB
- Pros: Flexible schema, easy horizontal scaling
- Cons: Eventual consistency, less suited for complex joins
- Use case: Rapid iteration, document-oriented data

Option 3: Cassandra
- Pros: Massive scale, write-optimized
- Cons: Complex to operate, eventual consistency
- Use case: Time-series data, high write throughput

Best choice: Start with PostgreSQL for MVP, migrate to MongoDB if schema changes frequently, use Cassandra only if hitting massive scale.
</thinking>

When NOT to use ToT:
- Simple binary choices
- When one approach is clearly superior
- Time-sensitive responses needed"""
        },
        {
            "instruction": "Show me how to apply chain-of-thought to code debugging",
            "output": """Chain-of-Thought for debugging:

**Problem**: This Python code crashes:
```python
def divide(a, b):
    return a / b
result = divide(10, 0)
```

**CoT Debugging Process**:

<thinking>
Step 1: Identify the error
- Error: ZeroDivisionError: division by zero
- Location: Line 2, `a / b`

Step 2: Trace the values
- a = 10
- b = 0
- Operation: 10 / 0 → Mathematical error!

Step 3: Understand why
- Division by zero is undefined in mathematics
- Python raises exception instead of returning infinity

Step 4: Consider solutions
- Option A: Check b != 0 before dividing
- Option B: Try-except to catch exception
- Option C: Return None or special value

Step 5: Choose best solution
- Option A is cleanest (prevent error)
- Option B for user-facing code (graceful handling)
</thinking>

**Fixed code**:
```python
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

**Key**: CoT helps trace the bug systematically rather than guessing."""
        }
    ]

    def __init__(self):
        self.stats = {
            "total_processed": 0,
            "cot_added": 0,
            "tot_added": 0,
            "meta_added": 0,
            "unchanged": 0
        }

    def should_add_cot(self, instruction: str) -> bool:
        """Check if instruction would benefit from CoT"""
        instruction_lower = instruction.lower()
        return any(re.search(pattern, instruction_lower) for pattern in self.COT_TRIGGERS)

    def should_add_tot(self, instruction: str) -> bool:
        """Check if instruction would benefit from ToT"""
        instruction_lower = instruction.lower()
        return any(re.search(pattern, instruction_lower) for pattern in self.TOT_TRIGGERS)

    def add_cot_wrapper(self, instruction: str, output: str) -> str:
        """Wrap existing output with CoT reasoning structure"""

        # If output already has thinking tags, don't modify
        if "<thinking>" in output or "<thought>" in output:
            return output

        # Extract key steps/sentences from output
        sentences = [s.strip() for s in output.split('\n') if s.strip()]

        if len(sentences) < 2:
            return output  # Too short to add CoT

        # Build CoT structure
        cot_output = "<thinking>\nLet me work through this step by step:\n\n"

        # Add numbered steps
        for i, sentence in enumerate(sentences[:5], 1):  # First 5 sentences as steps
            cot_output += f"{i}. {sentence}\n"

        cot_output += "\n</thinking>\n\n"

        # Add final answer
        cot_output += "Based on this reasoning:\n" + output

        return cot_output

    def add_tot_wrapper(self, instruction: str, output: str) -> str:
        """Wrap output with ToT multi-path exploration"""

        if "<thinking>" in output or "approach" in output.lower():
            return output

        # Create ToT structure by framing as exploring options
        tot_output = "<thinking>\nLet me explore different approaches:\n\n"

        tot_output += "Approach 1 (Direct solution):\n"
        tot_output += f"{output[:200]}...\n"
        tot_output += "✓ This is straightforward and clear\n\n"

        tot_output += "Approach 2 (Alternative consideration):\n"
        tot_output += "We could also consider edge cases or alternative methods\n"
        tot_output += "But the direct approach is most appropriate here\n\n"

        tot_output += "</thinking>\n\n"
        tot_output += output

        return tot_output

    def augment_example(
        self,
        example: Dict,
        augmentation_rate: float = 0.3,
        force_cot: bool = False,
        force_tot: bool = False
    ) -> Dict:
        """
        Augment a single training example with CoT/ToT

        Args:
            example: Dict with 'instruction' and 'output' keys
            augmentation_rate: Probability of augmenting (0.0 to 1.0)
            force_cot: Always add CoT if applicable
            force_tot: Always add ToT if applicable

        Returns:
            Augmented example dict
        """
        self.stats["total_processed"] += 1

        instruction = example.get("instruction", "")
        output = example.get("output", "")

        # Skip if output already has reasoning
        if "<thinking>" in output or "<thought>" in output:
            self.stats["unchanged"] += 1
            return example

        # Random chance to skip augmentation
        if not force_cot and not force_tot and random.random() > augmentation_rate:
            self.stats["unchanged"] += 1
            return example

        # Decide augmentation type
        if self.should_add_tot(instruction) and (force_tot or random.random() < 0.3):
            # Add ToT
            new_output = self.add_tot_wrapper(instruction, output)
            self.stats["tot_added"] += 1
            return {**example, "output": new_output}

        elif self.should_add_cot(instruction) or force_cot:
            # Add CoT
            new_output = self.add_cot_wrapper(instruction, output)
            self.stats["cot_added"] += 1
            return {**example, "output": new_output}

        else:
            self.stats["unchanged"] += 1
            return example

    def augment_dataset(
        self,
        input_path: Path,
        output_path: Path,
        augmentation_rate: float = 0.3,
        add_meta_examples: bool = True,
        limit: Optional[int] = None
    ):
        """
        Augment entire dataset

        Args:
            input_path: Input JSONL file
            output_path: Output JSONL file
            augmentation_rate: Fraction of examples to augment
            add_meta_examples: Include meta-cognitive examples
            limit: Optional limit on number of examples to process
        """

        print(f"[i] Augmenting dataset: {input_path}")
        print(f"[i] Output: {output_path}")
        print(f"[i] Augmentation rate: {augmentation_rate:.1%}")

        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:

            # Add meta-cognitive examples first
            if add_meta_examples:
                for meta_example in self.META_EXAMPLES:
                    outfile.write(json.dumps(meta_example) + '\n')
                self.stats["meta_added"] = len(self.META_EXAMPLES)
                print(f"[+] Added {len(self.META_EXAMPLES)} meta-cognitive examples")

            # Process input examples
            for i, line in enumerate(infile):
                if limit and i >= limit:
                    break

                try:
                    example = json.loads(line)
                    augmented = self.augment_example(example, augmentation_rate)
                    outfile.write(json.dumps(augmented) + '\n')

                    # Progress indicator
                    if (i + 1) % 10000 == 0:
                        print(f"[i] Processed {i+1:,} examples...")

                except json.JSONDecodeError:
                    print(f"[!] Skipping invalid JSON at line {i+1}")
                    continue

        # Print stats
        print("\n" + "="*60)
        print("AUGMENTATION COMPLETE")
        print("="*60)
        print(f"Total processed:  {self.stats['total_processed']:,}")
        print(f"CoT added:        {self.stats['cot_added']:,} ({self.stats['cot_added']/self.stats['total_processed']*100:.1f}%)")
        print(f"ToT added:        {self.stats['tot_added']:,} ({self.stats['tot_added']/self.stats['total_processed']*100:.1f}%)")
        print(f"Meta examples:    {self.stats['meta_added']:,}")
        print(f"Unchanged:        {self.stats['unchanged']:,}")
        print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment training data with Chain of Thought and Tree of Thought reasoning"
    )
    parser.add_argument("--input", required=True, help="Input JSONL dataset")
    parser.add_argument("--output", required=True, help="Output JSONL dataset")
    parser.add_argument("--augmentation-rate", type=float, default=0.3,
                       help="Fraction of examples to augment (0.0 to 1.0)")
    parser.add_argument("--no-meta", action="store_true",
                       help="Don't add meta-cognitive examples")
    parser.add_argument("--limit", type=int, help="Limit number of examples to process")

    args = parser.parse_args()

    augmenter = CoTToTAugmenter()
    augmenter.augment_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        augmentation_rate=args.augmentation_rate,
        add_meta_examples=not args.no_meta,
        limit=args.limit
    )


if __name__ == "__main__":
    main()

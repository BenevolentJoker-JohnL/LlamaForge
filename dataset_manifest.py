#!/usr/bin/env python3
"""
Dataset Manifest & Weighted Sampling System

Tracks dataset composition, applies weights per bucket, and creates balanced training samples.

Usage:
    # Create manifest from existing dataset
    python dataset_manifest.py --analyze examples/datasets/claude_reasoning_ultimate_1.4M.jsonl

    # Create weighted training sample
    python dataset_manifest.py \
        --manifest dataset_manifest.json \
        --output examples/datasets/weighted_training_mix.jsonl \
        --total-samples 500000

    # Add new bucket
    python dataset_manifest.py \
        --add-bucket red_team \
        --source examples/datasets/red_team_safe.jsonl \
        --weight 0.05 \
        --description "Adversarial prompts with safe responses"
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime


class DatasetManifest:
    """
    Track dataset composition and apply weighted sampling.

    Bucket types:
    - instruction: General instruction following
    - cot_math: Chain-of-thought math reasoning
    - tool_use: Tool interaction examples
    - creative: Creative writing and generation
    - analytical: Long-form analysis and synthesis
    - code: Code generation and explanation
    - red_team: Adversarial/safety examples
    - factual: Grounded factual content
    """

    DEFAULT_WEIGHTS = {
        # Core capabilities (60% total)
        "instruction": 0.25,      # General instruction following
        "tool_use": 0.15,         # Tool/API interaction
        "code": 0.20,             # Code generation/explanation

        # Reasoning (25% total)
        "cot_math": 0.15,         # Math with step-by-step
        "analytical": 0.10,       # Analysis and synthesis

        # Style & creativity (10% total)
        "creative": 0.07,         # Creative writing
        "factual": 0.03,          # Grounded facts

        # Safety & robustness (5% total)
        "red_team": 0.05,         # Adversarial + safe responses
    }

    # Recommended training phases
    TRAINING_PHASES = {
        "phase1_calibration": {
            "description": "Initial calibration with high-quality general instructions",
            "weights": {
                "instruction": 0.70,
                "code": 0.20,
                "factual": 0.10,
            },
            "recommended_samples": 10000,
            "epochs": 2
        },
        "phase2_reasoning": {
            "description": "Heavy reasoning and tool use",
            "weights": {
                "cot_math": 0.40,
                "tool_use": 0.30,
                "analytical": 0.20,
                "instruction": 0.10,
            },
            "recommended_samples": 200000,
            "epochs": 1
        },
        "phase3_creative": {
            "description": "Style, creativity, and diversity",
            "weights": {
                "creative": 0.40,
                "analytical": 0.30,
                "instruction": 0.20,
                "factual": 0.10,
            },
            "recommended_samples": 100000,
            "epochs": 1
        },
        "phase4_robustness": {
            "description": "Safety, edge cases, and adversarial",
            "weights": {
                "red_team": 0.40,
                "instruction": 0.30,
                "tool_use": 0.20,
                "factual": 0.10,
            },
            "recommended_samples": 50000,
            "epochs": 1
        },
        "production_balanced": {
            "description": "Balanced production mix (recommended for full training)",
            "weights": DEFAULT_WEIGHTS,
            "recommended_samples": 500000,
            "epochs": 1
        }
    }

    def __init__(self, manifest_path: Optional[Path] = None):
        self.buckets = {}
        self.metadata = {
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "total_examples": 0
        }

        if manifest_path and manifest_path.exists():
            self.load(manifest_path)

    def analyze_existing_dataset(self, dataset_path: Path) -> Dict:
        """
        Analyze existing merged dataset and infer bucket composition.

        Uses heuristics to classify examples into buckets based on content.
        """
        print(f"[i] Analyzing dataset: {dataset_path}")

        bucket_counts = defaultdict(int)
        total = 0

        # Classification keywords
        cot_indicators = ["<thinking>", "step 1:", "step 2:", "first,", "second,", "let me think"]
        tool_indicators = ["<tool>", "tool_use:", "function_call", "api_call"]
        code_indicators = ["```python", "```javascript", "def ", "function ", "class "]
        creative_indicators = ["story", "once upon", "creative", "imagine", "narrative"]

        with open(dataset_path, 'r') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    output = example.get("output", "").lower()
                    instruction = example.get("instruction", "").lower()

                    # Classify based on content
                    classified = False

                    # Tool use (check first, most specific)
                    if any(ind in output for ind in tool_indicators):
                        bucket_counts["tool_use"] += 1
                        classified = True

                    # CoT Math
                    elif any(ind in output for ind in cot_indicators):
                        if any(word in instruction for word in ["solve", "calculate", "math", "equation"]):
                            bucket_counts["cot_math"] += 1
                        else:
                            bucket_counts["analytical"] += 1
                        classified = True

                    # Code
                    elif any(ind in output for ind in code_indicators):
                        bucket_counts["code"] += 1
                        classified = True

                    # Creative
                    elif any(ind in instruction for ind in creative_indicators):
                        bucket_counts["creative"] += 1
                        classified = True

                    # Default to instruction
                    if not classified:
                        bucket_counts["instruction"] += 1

                    total += 1

                    if total % 100000 == 0:
                        print(f"  [i] Processed {total:,} examples...")

                except json.JSONDecodeError:
                    continue

        print(f"\n[✓] Analysis complete: {total:,} examples")
        print(f"\n  Bucket distribution:")
        for bucket, count in sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100
            print(f"    {bucket:20s} {count:>10,} ({pct:>5.1f}%)")

        # Update manifest
        for bucket, count in bucket_counts.items():
            self.buckets[bucket] = {
                "source": str(dataset_path),
                "count": count,
                "weight": self.DEFAULT_WEIGHTS.get(bucket, 0.10),
                "description": f"Auto-detected {bucket} examples",
                "provenance": "auto-classified",
                "added": datetime.now().isoformat()
            }

        self.metadata["total_examples"] = total

        return dict(bucket_counts)

    def add_bucket(
        self,
        name: str,
        source: Path,
        weight: float,
        description: str = "",
        provenance: str = "manual",
        auto_count: bool = True
    ):
        """Add a new data bucket to the manifest"""

        count = 0
        if auto_count and source.exists():
            with open(source, 'r') as f:
                count = sum(1 for line in f if line.strip())

        self.buckets[name] = {
            "source": str(source),
            "count": count,
            "weight": weight,
            "description": description,
            "provenance": provenance,
            "added": datetime.now().isoformat()
        }

        self.metadata["total_examples"] = sum(b["count"] for b in self.buckets.values())

        print(f"[+] Added bucket: {name}")
        print(f"    Source: {source}")
        print(f"    Count: {count:,}")
        print(f"    Weight: {weight:.2%}")

    def create_weighted_sample(
        self,
        output_path: Path,
        total_samples: int,
        weights: Optional[Dict[str, float]] = None,
        phase: Optional[str] = None,
        seed: int = 42
    ):
        """
        Create a weighted training sample from buckets.

        Args:
            output_path: Where to write the sampled dataset
            total_samples: Total number of examples to sample
            weights: Custom weights per bucket (uses DEFAULT_WEIGHTS if None)
            phase: Use predefined training phase weights
            seed: Random seed for reproducibility
        """

        random.seed(seed)

        # Determine weights
        if phase:
            if phase not in self.TRAINING_PHASES:
                raise ValueError(f"Unknown phase: {phase}. Choose from {list(self.TRAINING_PHASES.keys())}")
            weights = self.TRAINING_PHASES[phase]["weights"]
            print(f"[i] Using phase: {phase}")
            print(f"    {self.TRAINING_PHASES[phase]['description']}")

        if weights is None:
            weights = self.DEFAULT_WEIGHTS

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}

        # Calculate samples per bucket
        samples_per_bucket = {}
        for bucket, weight in normalized_weights.items():
            if bucket in self.buckets:
                target = int(total_samples * weight)
                available = self.buckets[bucket]["count"]
                samples_per_bucket[bucket] = min(target, available)

        print(f"\n[i] Creating weighted sample:")
        print(f"    Total target: {total_samples:,} examples")
        print(f"\n  Sampling plan:")

        total_actual = sum(samples_per_bucket.values())
        for bucket, count in sorted(samples_per_bucket.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_actual) * 100
            weight_pct = normalized_weights.get(bucket, 0) * 100
            print(f"    {bucket:20s} {count:>10,} ({pct:>5.1f}% actual, {weight_pct:>5.1f}% target)")

        # Sample from each bucket
        print(f"\n[i] Sampling from buckets...")

        sampled_examples = []

        for bucket, target_count in samples_per_bucket.items():
            bucket_info = self.buckets[bucket]
            source_path = Path(bucket_info["source"])

            if not source_path.exists():
                print(f"  [!] Warning: Source not found for {bucket}: {source_path}")
                continue

            # Load all examples from this bucket
            bucket_examples = []
            with open(source_path, 'r') as f:
                for line in f:
                    try:
                        bucket_examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            # Sample
            if len(bucket_examples) <= target_count:
                sampled = bucket_examples
            else:
                sampled = random.sample(bucket_examples, target_count)

            # Add bucket label for provenance tracking
            for ex in sampled:
                ex["_bucket"] = bucket
                ex["_weight"] = normalized_weights[bucket]

            sampled_examples.extend(sampled)

            print(f"  [✓] {bucket:20s} {len(sampled):>10,} examples")

        # Shuffle all examples
        random.shuffle(sampled_examples)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for ex in sampled_examples:
                f.write(json.dumps(ex) + '\n')

        print(f"\n[✓] Weighted sample created:")
        print(f"    Output: {output_path}")
        print(f"    Examples: {len(sampled_examples):,}")
        print(f"    Size: {output_path.stat().st_size / (1024**2):.1f} MB")

        return len(sampled_examples)

    def save(self, path: Path):
        """Save manifest to JSON"""
        manifest = {
            "metadata": self.metadata,
            "buckets": self.buckets,
            "default_weights": self.DEFAULT_WEIGHTS,
            "training_phases": self.TRAINING_PHASES
        }

        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"[✓] Manifest saved: {path}")

    def load(self, path: Path):
        """Load manifest from JSON"""
        with open(path, 'r') as f:
            manifest = json.load(f)

        self.metadata = manifest["metadata"]
        self.buckets = manifest["buckets"]

        print(f"[✓] Manifest loaded: {path}")
        print(f"    Total examples: {self.metadata['total_examples']:,}")
        print(f"    Buckets: {len(self.buckets)}")

    def print_summary(self):
        """Print manifest summary"""
        print(f"\n{'='*80}")
        print(f"  DATASET MANIFEST SUMMARY")
        print(f"{'='*80}\n")

        print(f"Total examples: {self.metadata['total_examples']:,}")
        print(f"Buckets: {len(self.buckets)}")
        print(f"\nBucket breakdown:")

        for bucket, info in sorted(self.buckets.items(), key=lambda x: x[1]['count'], reverse=True):
            count = info['count']
            weight = info['weight']
            pct = (count / self.metadata['total_examples']) * 100 if self.metadata['total_examples'] > 0 else 0

            print(f"\n  {bucket}:")
            print(f"    Count: {count:,} ({pct:.1f}% of total)")
            print(f"    Weight: {weight:.2%}")
            print(f"    Source: {info['source']}")
            if info.get('description'):
                print(f"    Description: {info['description']}")

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset manifest and weighted sampling system"
    )

    # Actions
    parser.add_argument("--analyze", help="Analyze existing dataset and create manifest")
    parser.add_argument("--manifest", help="Load existing manifest file")

    # Add bucket
    parser.add_argument("--add-bucket", help="Add new bucket name")
    parser.add_argument("--source", help="Source file for new bucket")
    parser.add_argument("--weight", type=float, help="Weight for new bucket")
    parser.add_argument("--description", default="", help="Bucket description")

    # Create weighted sample
    parser.add_argument("--create-sample", action="store_true",
                       help="Create weighted training sample")
    parser.add_argument("--output", help="Output file for weighted sample")
    parser.add_argument("--total-samples", type=int, default=500000,
                       help="Total samples to create")
    parser.add_argument("--phase", choices=list(DatasetManifest.TRAINING_PHASES.keys()),
                       help="Use predefined training phase")

    # Other
    parser.add_argument("--save-manifest", help="Save manifest to file")
    parser.add_argument("--summary", action="store_true", help="Print manifest summary")

    args = parser.parse_args()

    # Initialize manifest
    manifest = DatasetManifest(Path(args.manifest) if args.manifest else None)

    # Analyze existing dataset
    if args.analyze:
        manifest.analyze_existing_dataset(Path(args.analyze))
        if args.save_manifest:
            manifest.save(Path(args.save_manifest))
        manifest.print_summary()

    # Add new bucket
    if args.add_bucket:
        if not args.source or args.weight is None:
            print("[!] Error: --source and --weight required when adding bucket")
            return

        manifest.add_bucket(
            name=args.add_bucket,
            source=Path(args.source),
            weight=args.weight,
            description=args.description
        )

        if args.save_manifest:
            manifest.save(Path(args.save_manifest))

    # Create weighted sample
    if args.create_sample:
        if not args.output:
            print("[!] Error: --output required when creating sample")
            return

        manifest.create_weighted_sample(
            output_path=Path(args.output),
            total_samples=args.total_samples,
            phase=args.phase
        )

    # Print summary
    if args.summary:
        manifest.print_summary()


if __name__ == "__main__":
    main()

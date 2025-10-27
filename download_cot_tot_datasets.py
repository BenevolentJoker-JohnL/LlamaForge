#!/usr/bin/env python3
"""
Download and merge Chain-of-Thought and Tree-of-Thought datasets

Downloads high-quality CoT/ToT datasets from HuggingFace and merges with your existing data.

Usage:
    # Download all CoT/ToT datasets
    python download_cot_tot_datasets.py --download-all --output examples/datasets/

    # Merge with your existing dataset
    python download_cot_tot_datasets.py --merge \
        --base examples/datasets/claude_ultimate_508k.jsonl \
        --output examples/datasets/claude_with_reasoning_600k.jsonl
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import sys


# High-quality CoT/ToT datasets
COT_TOT_DATASETS = {
    "gsm8k_cot": {
        "name": "gsm8k with Chain-of-Thought",
        "hf_dataset": "gsm8k",
        "config": "main",
        "split": "train",
        "size": "7.5K examples",
        "description": "Math word problems with step-by-step reasoning",
        "format_function": "format_gsm8k_cot"
    },
    "cot_collection": {
        "name": "CoT Collection (NLP tasks)",
        "hf_dataset": "kaist-ai/CoT-Collection",
        "config": None,
        "split": "train",
        "size": "~100K examples",
        "description": "Diverse NLP tasks with chain-of-thought reasoning",
        "format_function": "format_cot_collection"
    },
    "orca_math_cot": {
        "name": "Orca Math with CoT",
        "hf_dataset": "microsoft/orca-math-word-problems-200k",
        "config": None,
        "split": "train",
        "size": "200K math problems",
        "description": "Math problems with detailed reasoning chains",
        "format_function": "format_orca_math"
    },
    "metamath": {
        "name": "MetaMath (CoT augmented)",
        "hf_dataset": "meta-math/MetaMathQA",
        "config": None,
        "split": "train",
        "size": "395K examples",
        "description": "Math problems with multiple reasoning strategies",
        "format_function": "format_metamath"
    },
    "wizardlm_evol": {
        "name": "WizardLM Evol Instruct (reasoning)",
        "hf_dataset": "WizardLM/WizardLM_evol_instruct_V2_196k",
        "config": None,
        "split": "train",
        "size": "196K examples",
        "description": "Complex instructions with detailed reasoning",
        "format_function": "format_wizardlm_evol"
    },
    "openhermes_reasoning": {
        "name": "OpenHermes 2.5 (filtered for reasoning)",
        "hf_dataset": "teknium/OpenHermes-2.5",
        "config": None,
        "split": "train",
        "size": "~1M (we'll filter)",
        "description": "High-quality instruction following with reasoning",
        "format_function": "format_openhermes"
    }
}


class CoTDatasetDownloader:
    """Download and format CoT/ToT datasets"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_gsm8k_cot(self, examples: List[Dict]) -> List[Dict]:
        """Format GSM8K with CoT"""
        formatted = []
        for ex in examples:
            formatted.append({
                "instruction": ex.get("question", ""),
                "output": f"<thinking>\n{ex.get('answer', '')}\n</thinking>\n\n" +
                         f"The answer is: {ex.get('answer', '').split('####')[-1].strip()}"
            })
        return formatted

    def format_cot_collection(self, examples: List[Dict]) -> List[Dict]:
        """Format CoT Collection"""
        formatted = []
        for ex in examples:
            # CoT Collection already has source, rationale, and target
            thinking = ex.get("rationale", "")
            if thinking:
                output = f"<thinking>\n{thinking}\n</thinking>\n\n{ex.get('target', '')}"
            else:
                output = ex.get("target", "")

            formatted.append({
                "instruction": ex.get("source", ""),
                "output": output
            })
        return formatted

    def format_orca_math(self, examples: List[Dict]) -> List[Dict]:
        """Format Orca Math dataset"""
        formatted = []
        for ex in examples:
            question = ex.get("question", "")
            answer = ex.get("answer", "")

            # Orca has detailed step-by-step in answer
            formatted.append({
                "instruction": question,
                "output": answer  # Already includes reasoning
            })
        return formatted

    def format_metamath(self, examples: List[Dict]) -> List[Dict]:
        """Format MetaMath dataset"""
        formatted = []
        for ex in examples:
            query = ex.get("query", "")
            response = ex.get("response", "")

            # MetaMath already has CoT format
            formatted.append({
                "instruction": query,
                "output": response
            })
        return formatted

    def format_wizardlm_evol(self, examples: List[Dict]) -> List[Dict]:
        """Format WizardLM Evol Instruct"""
        formatted = []
        for ex in examples:
            conversations = ex.get("conversations", [])
            if len(conversations) >= 2:
                instruction = conversations[0].get("value", "")
                output = conversations[1].get("value", "")
                formatted.append({
                    "instruction": instruction,
                    "output": output
                })
        return formatted

    def format_openhermes(self, examples: List[Dict], limit: int = 50000) -> List[Dict]:
        """Format OpenHermes (filter for reasoning examples)"""
        formatted = []
        reasoning_keywords = ["step", "think", "reason", "analyze", "consider", "first", "second"]

        for ex in examples[:limit]:  # Limit to avoid massive dataset
            conversations = ex.get("conversations", [])
            if len(conversations) >= 2:
                instruction = conversations[0].get("value", "")
                output = conversations[1].get("value", "")

                # Only include if output contains reasoning
                if any(kw in output.lower() for kw in reasoning_keywords):
                    formatted.append({
                        "instruction": instruction,
                        "output": output
                    })

        return formatted

    def download_dataset(self, dataset_key: str, limit: int = None) -> Path:
        """
        Download and format a single dataset

        Args:
            dataset_key: Key from COT_TOT_DATASETS
            limit: Optional limit on examples

        Returns:
            Path to output JSONL file
        """
        dataset_info = COT_TOT_DATASETS[dataset_key]
        print(f"\n{'='*70}")
        print(f"Downloading: {dataset_info['name']}")
        print(f"Size: {dataset_info['size']}")
        print(f"{'='*70}\n")

        # Try to import datasets library
        try:
            from datasets import load_dataset
        except ImportError:
            print("[!] datasets library not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
            from datasets import load_dataset

        # Download from HuggingFace
        try:
            if dataset_info["config"]:
                dataset = load_dataset(
                    dataset_info["hf_dataset"],
                    dataset_info["config"],
                    split=dataset_info["split"]
                )
            else:
                dataset = load_dataset(
                    dataset_info["hf_dataset"],
                    split=dataset_info["split"]
                )

            # Convert to list of dicts
            examples = list(dataset)

            if limit:
                examples = examples[:limit]

            print(f"[+] Downloaded {len(examples):,} examples")

            # Format using appropriate formatter
            format_func = getattr(self, dataset_info["format_function"])
            formatted_examples = format_func(examples)

            print(f"[+] Formatted {len(formatted_examples):,} examples")

            # Save to JSONL
            output_file = self.output_dir / f"{dataset_key}.jsonl"
            with open(output_file, 'w') as f:
                for example in formatted_examples:
                    f.write(json.dumps(example) + '\n')

            print(f"[✓] Saved to: {output_file}")
            return output_file

        except Exception as e:
            print(f"[✗] Failed to download {dataset_key}: {e}")
            return None

    def download_all(self, limit_per_dataset: int = None):
        """Download all CoT/ToT datasets"""
        print(f"\n{'='*70}")
        print("DOWNLOADING ALL COT/TOT DATASETS")
        print(f"{'='*70}\n")

        downloaded = []
        for dataset_key in COT_TOT_DATASETS.keys():
            output_file = self.download_dataset(dataset_key, limit_per_dataset)
            if output_file:
                downloaded.append(output_file)

        print(f"\n{'='*70}")
        print(f"DOWNLOAD COMPLETE: {len(downloaded)} datasets")
        print(f"{'='*70}\n")

        for file in downloaded:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.1f} MB")

        return downloaded


def merge_datasets(base_path: Path, cot_datasets: List[Path], output_path: Path):
    """
    Merge base dataset with CoT/ToT datasets

    Args:
        base_path: Original dataset (e.g., claude_ultimate_508k.jsonl)
        cot_datasets: List of CoT/ToT dataset files
        output_path: Output merged file
    """
    print(f"\n{'='*70}")
    print("MERGING DATASETS")
    print(f"{'='*70}\n")

    total_examples = 0

    with open(output_path, 'w') as outfile:
        # Add base dataset
        if base_path and base_path.exists():
            print(f"[1/2] Adding base dataset: {base_path}")
            with open(base_path, 'r') as infile:
                for line in infile:
                    outfile.write(line)
                    total_examples += 1
            print(f"  [+] Added {total_examples:,} examples from base")

        # Add CoT/ToT datasets
        print(f"\n[2/2] Adding CoT/ToT datasets...")
        for cot_file in cot_datasets:
            if cot_file.exists():
                cot_count = 0
                with open(cot_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                        cot_count += 1
                        total_examples += 1
                print(f"  [+] Added {cot_count:,} examples from {cot_file.name}")

    print(f"\n{'='*70}")
    print(f"MERGE COMPLETE")
    print(f"{'='*70}")
    print(f"Total examples: {total_examples:,}")
    print(f"Output file: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024**2):.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Download and merge CoT/ToT datasets"
    )
    parser.add_argument("--download-all", action="store_true",
                       help="Download all CoT/ToT datasets")
    parser.add_argument("--download", nargs="+", choices=list(COT_TOT_DATASETS.keys()),
                       help="Download specific datasets")
    parser.add_argument("--merge", action="store_true",
                       help="Merge downloaded datasets with base dataset")
    parser.add_argument("--base", type=str,
                       help="Base dataset to merge with (e.g., claude_ultimate_508k.jsonl)")
    parser.add_argument("--output", required=True,
                       help="Output directory (download) or file (merge)")
    parser.add_argument("--limit", type=int,
                       help="Limit examples per dataset")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")

    args = parser.parse_args()

    # List datasets
    if args.list:
        print("\nAvailable CoT/ToT Datasets:")
        print("="*70)
        for key, info in COT_TOT_DATASETS.items():
            print(f"\n{key}")
            print(f"  Name: {info['name']}")
            print(f"  Size: {info['size']}")
            print(f"  Description: {info['description']}")
        return

    output_path = Path(args.output)

    # Download mode
    if args.download_all or args.download:
        downloader = CoTDatasetDownloader(output_path)

        if args.download_all:
            downloaded = downloader.download_all(limit_per_dataset=args.limit)
        else:
            downloaded = []
            for dataset_key in args.download:
                file = downloader.download_dataset(dataset_key, limit=args.limit)
                if file:
                    downloaded.append(file)

        print(f"\n[✓] Downloaded {len(downloaded)} datasets to: {output_path}")

    # Merge mode
    if args.merge:
        if not args.base:
            print("[!] Error: --base required for merge mode")
            return

        # Find all CoT datasets in output directory
        base_path = Path(args.base)
        cot_files = list(Path(output_path).parent.glob("*cot*.jsonl"))
        cot_files.extend(list(Path(output_path).parent.glob("*math*.jsonl")))
        cot_files.extend(list(Path(output_path).parent.glob("*metamath*.jsonl")))
        cot_files.extend(list(Path(output_path).parent.glob("*wizard*.jsonl")))
        cot_files.extend(list(Path(output_path).parent.glob("*hermes*.jsonl")))

        # Remove duplicates
        cot_files = list(set(cot_files))

        if not cot_files:
            print("[!] No CoT datasets found. Download first with --download-all")
            return

        merge_datasets(base_path, cot_files, output_path)


if __name__ == "__main__":
    main()

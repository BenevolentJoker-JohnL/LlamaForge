#!/usr/bin/env python3
"""
Download and prepare comprehensive code training datasets

Downloads high-quality code datasets for building Claude-style programming assistants.

Includes:
- Code generation (HumanEval, MBPP, CodeContests)
- Code explanation & documentation
- Bug fixing & debugging
- Code review & refactoring
- Multi-language support
- Algorithm implementation
- API integration examples
- Test generation

Usage:
    # Download all code datasets
    python download_code_datasets.py --download-all --output examples/datasets/

    # Download specific categories
    python download_code_datasets.py \
        --categories generation explanation debugging \
        --output examples/datasets/

    # List available datasets
    python download_code_datasets.py --list
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional


CODE_DATASETS = {
    # === Code Generation ===
    "humaneval": {
        "name": "HumanEval (Python)",
        "hf_dataset": "openai_humaneval",
        "category": "generation",
        "size": "164 problems",
        "description": "Python function generation from docstrings",
        "languages": ["python"],
        "quality": "high",
        "format_function": "format_humaneval"
    },

    "mbpp": {
        "name": "MBPP (Python)",
        "hf_dataset": "mbpp",
        "category": "generation",
        "size": "974 problems",
        "description": "Python programming problems with tests",
        "languages": ["python"],
        "quality": "high",
        "format_function": "format_mbpp"
    },

    "apps": {
        "name": "APPS (Competitive Programming)",
        "hf_dataset": "codeparrot/apps",
        "category": "generation",
        "size": "10K problems",
        "description": "Competitive programming problems (all difficulties)",
        "languages": ["python"],
        "quality": "medium",
        "format_function": "format_apps"
    },

    "code_contests": {
        "name": "CodeContests",
        "hf_dataset": "deepmind/code_contests",
        "category": "generation",
        "size": "13K problems",
        "description": "Competitive programming with multiple solutions",
        "languages": ["python", "cpp"],
        "quality": "high",
        "format_function": "format_code_contests"
    },

    # === Code Explanation & Documentation ===

    "code_searchnet": {
        "name": "CodeSearchNet",
        "hf_dataset": "code_search_net",
        "category": "explanation",
        "size": "2M+ functions",
        "description": "Functions with docstrings (6 languages)",
        "languages": ["python", "javascript", "java", "go", "php", "ruby"],
        "quality": "medium",
        "format_function": "format_code_searchnet"
    },

    "codefeedback": {
        "name": "CodeFeedback (Multi-language)",
        "hf_dataset": "m-a-p/CodeFeedback-Filtered-Instruction",
        "category": "explanation",
        "size": "150K examples",
        "description": "Code explanation and Q&A across languages",
        "languages": ["python", "javascript", "java", "cpp", "go", "rust"],
        "quality": "high",
        "format_function": "format_codefeedback"
    },

    # === Debugging & Bug Fixing ===

    "bugs_in_py": {
        "name": "BugsInPy",
        "hf_dataset": "THUDM/BugsInPy",
        "category": "debugging",
        "size": "493 bugs",
        "description": "Real Python bugs with fixes",
        "languages": ["python"],
        "quality": "high",
        "format_function": "format_bugs_in_py"
    },

    "code_x_glue_defect": {
        "name": "CodeXGLUE Defect Detection",
        "hf_dataset": "code_x_glue_cc_defect_detection",
        "category": "debugging",
        "size": "21K examples",
        "description": "Detect and explain code defects",
        "languages": ["c"],
        "quality": "high",
        "format_function": "format_defect_detection"
    },

    # === Multi-language Code ===

    "the_stack": {
        "name": "The Stack (Filtered)",
        "hf_dataset": "bigcode/the-stack-dedup",
        "category": "generation",
        "size": "~6TB (we'll filter)",
        "description": "Permissively licensed code from GitHub",
        "languages": ["all"],
        "quality": "variable",
        "format_function": "format_the_stack"
    },

    "xlcost": {
        "name": "XLCoST (Code Translation)",
        "hf_dataset": "codeparrot/xlcost-single-lang-python",
        "category": "translation",
        "size": "10K+ pairs",
        "description": "Code snippets in multiple languages",
        "languages": ["python", "java", "cpp", "csharp", "javascript", "php"],
        "quality": "high",
        "format_function": "format_xlcost"
    },

    # === Code Review & Refactoring ===

    "code_review": {
        "name": "Code Review Comments",
        "hf_dataset": "DanielKyo/code_review_instructions",
        "category": "review",
        "size": "~10K examples",
        "description": "Code review feedback and suggestions",
        "languages": ["python", "javascript", "java"],
        "quality": "medium",
        "format_function": "format_code_review"
    },

    # === Algorithm Implementation ===

    "leetcode": {
        "name": "LeetCode Solutions",
        "hf_dataset": "greengerong/leetcode",
        "category": "algorithms",
        "size": "~2K problems",
        "description": "Algorithm problems with solutions",
        "languages": ["python", "java", "cpp"],
        "quality": "high",
        "format_function": "format_leetcode"
    },

    # === SQL & Data ===

    "spider": {
        "name": "Spider (Text-to-SQL)",
        "hf_dataset": "spider",
        "category": "sql",
        "size": "10K examples",
        "description": "Natural language to SQL queries",
        "languages": ["sql"],
        "quality": "high",
        "format_function": "format_spider"
    },

    # === API & Integration ===

    "glaive_code_assistant": {
        "name": "Glaive Code Assistant",
        "hf_dataset": "glaiveai/glaive-code-assistant",
        "category": "api",
        "size": "~130K examples",
        "description": "API usage, integration, and code assistance",
        "languages": ["python", "javascript"],
        "quality": "high",
        "format_function": "format_glaive_code"
    },
}


class CodeDatasetDownloader:
    """Download and format code datasets"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_humaneval(self, examples: List[Dict]) -> List[Dict]:
        """Format HumanEval dataset"""
        formatted = []
        for ex in examples:
            prompt = ex.get("prompt", "")
            canonical = ex.get("canonical_solution", "")

            # Extract function signature and docstring
            formatted.append({
                "instruction": f"Write a Python function:\n\n{prompt}",
                "output": canonical,
                "_language": "python",
                "_category": "generation"
            })
        return formatted

    def format_mbpp(self, examples: List[Dict]) -> List[Dict]:
        """Format MBPP dataset"""
        formatted = []
        for ex in examples:
            text = ex.get("text", "")
            code = ex.get("code", "")
            test_list = ex.get("test_list", [])

            instruction = f"{text}\n\nThe solution should pass these tests:\n"
            for test in test_list[:3]:  # Show first 3 tests
                instruction += f"- {test}\n"

            formatted.append({
                "instruction": instruction,
                "output": code,
                "_language": "python",
                "_category": "generation",
                "_tests": test_list
            })
        return formatted

    def format_apps(self, examples: List[Dict], limit: int = 5000) -> List[Dict]:
        """Format APPS dataset (filtered for quality)"""
        formatted = []
        for ex in examples[:limit]:
            problem = ex.get("question", "")
            solutions = ex.get("solutions", "")
            difficulty = ex.get("difficulty", "unknown")

            # Only include introductory and interview level
            if difficulty not in ["introductory", "interview"]:
                continue

            if solutions:
                formatted.append({
                    "instruction": f"Solve this programming problem:\n\n{problem}",
                    "output": solutions,
                    "_language": "python",
                    "_category": "generation",
                    "_difficulty": difficulty
                })
        return formatted

    def format_code_contests(self, examples: List[Dict], limit: int = 10000) -> List[Dict]:
        """Format CodeContests dataset"""
        formatted = []
        for ex in examples[:limit]:
            description = ex.get("description", "")
            solutions = ex.get("solutions", {})

            # Get Python solution if available
            if solutions and "python" in str(solutions).lower():
                formatted.append({
                    "instruction": f"Solve this competitive programming problem:\n\n{description}",
                    "output": str(solutions),
                    "_language": "python",
                    "_category": "algorithms"
                })
        return formatted

    def format_code_searchnet(self, examples: List[Dict], limit: int = 50000) -> List[Dict]:
        """Format CodeSearchNet (code explanation)"""
        formatted = []

        for ex in examples[:limit]:
            code = ex.get("func_code_string", "")
            doc = ex.get("func_documentation_string", "")
            language = ex.get("language", "python")

            if not code or not doc or len(doc) < 10:
                continue

            # Code → Explanation
            formatted.append({
                "instruction": f"Explain what this {language} code does:\n\n```{language}\n{code}\n```",
                "output": doc,
                "_language": language,
                "_category": "explanation"
            })

            # Docstring → Code (reverse direction)
            formatted.append({
                "instruction": f"Write {language} code for this specification:\n\n{doc}",
                "output": f"```{language}\n{code}\n```",
                "_language": language,
                "_category": "generation"
            })

        return formatted

    def format_codefeedback(self, examples: List[Dict]) -> List[Dict]:
        """Format CodeFeedback dataset"""
        formatted = []
        for ex in examples:
            query = ex.get("query", "")
            response = ex.get("response", "")
            language = ex.get("lang", "python")

            if query and response:
                formatted.append({
                    "instruction": query,
                    "output": response,
                    "_language": language,
                    "_category": "explanation"
                })
        return formatted

    def format_bugs_in_py(self, examples: List[Dict]) -> List[Dict]:
        """Format BugsInPy (debugging)"""
        formatted = []
        for ex in examples:
            buggy_code = ex.get("buggy_code", "")
            fixed_code = ex.get("fixed_code", "")
            bug_type = ex.get("bug_type", "unknown")

            if buggy_code and fixed_code:
                formatted.append({
                    "instruction": f"Fix the bug in this Python code:\n\n```python\n{buggy_code}\n```\n\nBug type: {bug_type}",
                    "output": f"<thinking>\nThe bug is a {bug_type}. Let me fix it step by step.\n</thinking>\n\nHere's the fixed code:\n\n```python\n{fixed_code}\n```",
                    "_language": "python",
                    "_category": "debugging",
                    "_bug_type": bug_type
                })
        return formatted

    def format_defect_detection(self, examples: List[Dict]) -> List[Dict]:
        """Format defect detection dataset"""
        formatted = []
        for ex in examples:
            code = ex.get("func", "")
            target = ex.get("target", 0)  # 0 = no defect, 1 = has defect

            if code:
                has_defect = "has a defect" if target == 1 else "is correct"
                formatted.append({
                    "instruction": f"Analyze this code for defects:\n\n```c\n{code}\n```",
                    "output": f"This code {has_defect}.",
                    "_language": "c",
                    "_category": "debugging",
                    "_has_defect": bool(target)
                })
        return formatted

    def format_the_stack(self, examples: List[Dict], limit: int = 20000) -> List[Dict]:
        """Format The Stack (filtered, high-quality only)"""
        formatted = []

        # Quality filters
        min_length = 100
        max_length = 2000

        for ex in examples[:limit]:
            content = ex.get("content", "")
            language = ex.get("lang", "unknown")

            # Quality filters
            if len(content) < min_length or len(content) > max_length:
                continue

            # Must have comments/docstrings
            if language == "python" and '"""' not in content and "'''" not in content and "#" not in content:
                continue

            formatted.append({
                "instruction": f"Explain this {language} code:",
                "output": f"```{language}\n{content}\n```",
                "_language": language,
                "_category": "generation"
            })

        return formatted

    def format_xlcost(self, examples: List[Dict]) -> List[Dict]:
        """Format XLCoST (code translation)"""
        formatted = []
        for ex in examples:
            # Usually has source_code and target_code
            source = ex.get("source", "")
            target = ex.get("target", "")
            source_lang = ex.get("source_lang", "python")
            target_lang = ex.get("target_lang", "java")

            if source and target:
                formatted.append({
                    "instruction": f"Translate this {source_lang} code to {target_lang}:\n\n```{source_lang}\n{source}\n```",
                    "output": f"```{target_lang}\n{target}\n```",
                    "_source_language": source_lang,
                    "_target_language": target_lang,
                    "_category": "translation"
                })
        return formatted

    def format_code_review(self, examples: List[Dict]) -> List[Dict]:
        """Format code review dataset"""
        formatted = []
        for ex in examples:
            instruction = ex.get("instruction", "")
            output = ex.get("output", "")

            if "review" in instruction.lower() or "refactor" in instruction.lower():
                formatted.append({
                    "instruction": instruction,
                    "output": output,
                    "_category": "review"
                })
        return formatted

    def format_leetcode(self, examples: List[Dict]) -> List[Dict]:
        """Format LeetCode dataset"""
        formatted = []
        for ex in examples:
            title = ex.get("title", "")
            content = ex.get("content", "")
            solution = ex.get("solution", "")

            if content and solution:
                formatted.append({
                    "instruction": f"LeetCode Problem: {title}\n\n{content}",
                    "output": solution,
                    "_category": "algorithms"
                })
        return formatted

    def format_spider(self, examples: List[Dict]) -> List[Dict]:
        """Format Spider (text-to-SQL)"""
        formatted = []
        for ex in examples:
            question = ex.get("question", "")
            query = ex.get("query", "")
            db_id = ex.get("db_id", "")

            if question and query:
                formatted.append({
                    "instruction": f"Generate a SQL query for this question:\n\nDatabase: {db_id}\nQuestion: {question}",
                    "output": f"```sql\n{query}\n```",
                    "_language": "sql",
                    "_category": "sql"
                })
        return formatted

    def format_glaive_code(self, examples: List[Dict]) -> List[Dict]:
        """Format Glaive Code Assistant"""
        formatted = []
        for ex in examples:
            system = ex.get("system", "")
            chat = ex.get("chat", "")

            # Parse chat into instruction/output
            if chat:
                formatted.append({
                    "instruction": chat,
                    "output": "",  # Would need to parse chat history
                    "_category": "api"
                })
        return formatted

    def download_dataset(
        self,
        dataset_key: str,
        limit: Optional[int] = None
    ) -> Path:
        """Download and format a single code dataset"""

        dataset_info = CODE_DATASETS[dataset_key]
        print(f"\n{'='*70}")
        print(f"Downloading: {dataset_info['name']}")
        print(f"Category: {dataset_info['category']}")
        print(f"Size: {dataset_info['size']}")
        print(f"{'='*70}\n")

        try:
            from datasets import load_dataset
        except ImportError:
            print("[!] datasets library not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
            from datasets import load_dataset

        try:
            # Load dataset
            dataset = load_dataset(dataset_info["hf_dataset"], split="train")
            examples = list(dataset)

            if limit:
                examples = examples[:limit]

            print(f"[+] Downloaded {len(examples):,} examples")

            # Format
            format_func = getattr(self, dataset_info["format_function"])
            formatted = format_func(examples, limit=limit) if "limit" in format_func.__code__.co_varnames else format_func(examples)

            print(f"[+] Formatted {len(formatted):,} examples")

            # Save
            output_file = self.output_dir / f"{dataset_key}.jsonl"
            with open(output_file, 'w') as f:
                for ex in formatted:
                    f.write(json.dumps(ex) + '\n')

            print(f"[✓] Saved to: {output_file}")
            return output_file

        except Exception as e:
            print(f"[✗] Failed to download {dataset_key}: {e}")
            return None

    def download_by_category(
        self,
        categories: List[str],
        limit_per_dataset: Optional[int] = None
    ):
        """Download all datasets in specified categories"""

        downloaded = []
        for dataset_key, info in CODE_DATASETS.items():
            if info["category"] in categories:
                output_file = self.download_dataset(dataset_key, limit_per_dataset)
                if output_file:
                    downloaded.append(output_file)

        return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download code training datasets")

    parser.add_argument("--download-all", action="store_true",
                       help="Download all code datasets")
    parser.add_argument("--download", nargs="+", choices=list(CODE_DATASETS.keys()),
                       help="Download specific datasets")
    parser.add_argument("--categories", nargs="+",
                       choices=["generation", "explanation", "debugging", "review",
                               "algorithms", "sql", "api", "translation"],
                       help="Download datasets by category")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit examples per dataset")
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Code Datasets:")
        print("="*80)

        by_category = {}
        for key, info in CODE_DATASETS.items():
            cat = info["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((key, info))

        for category, datasets in sorted(by_category.items()):
            print(f"\n📁 {category.upper()}")
            for key, info in datasets:
                print(f"\n  {key}")
                print(f"    Name: {info['name']}")
                print(f"    Size: {info['size']}")
                print(f"    Languages: {', '.join(info['languages'])}")
                print(f"    Description: {info['description']}")
        return

    downloader = CodeDatasetDownloader(Path(args.output))

    if args.download_all:
        print("[i] Downloading ALL code datasets (this will take a while)...")
        for dataset_key in CODE_DATASETS.keys():
            downloader.download_dataset(dataset_key, args.limit)

    elif args.download:
        for dataset_key in args.download:
            downloader.download_dataset(dataset_key, args.limit)

    elif args.categories:
        downloader.download_by_category(args.categories, args.limit)


if __name__ == "__main__":
    main()

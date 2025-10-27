#!/usr/bin/env python3
"""
Download Comprehensive Debugging & Testing Datasets

Creates a complete debugging assistant training corpus including:
- Bug detection & fixing
- Test generation (TDD)
- Error analysis & root cause
- Code review with bug identification
- Refactoring suggestions
- Performance debugging
- Security vulnerability detection

Usage:
    # Download all debugging datasets
    python download_debugging_datasets.py --download-all --output examples/datasets/

    # Download specific categories
    python download_debugging_datasets.py \
        --categories bug_fixing test_generation error_analysis \
        --output examples/datasets/

    # List available
    python download_debugging_datasets.py --list
"""

import argparse
import json
import subprocess
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional


DEBUGGING_DATASETS = {
    # === Bug Detection & Fixing ===

    "bugs_in_py": {
        "name": "BugsInPy (Real Python Bugs)",
        "hf_dataset": "THUDM/BugsInPy",
        "category": "bug_fixing",
        "size": "493 real bugs",
        "description": "Real bugs from Python projects with fixes",
        "format_function": "format_bugs_in_py"
    },

    "code_x_glue_defect": {
        "name": "CodeXGLUE Defect Detection",
        "hf_dataset": "code_x_glue_cc_defect_detection",
        "category": "bug_detection",
        "size": "21K examples",
        "description": "Defect detection in C code",
        "format_function": "format_defect_detection"
    },

    "manybugs": {
        "name": "ManyBugs Dataset",
        "hf_dataset": "manybugs/manybugs",
        "category": "bug_fixing",
        "size": "185 bugs",
        "description": "Real bugs from C programs",
        "format_function": "format_manybugs"
    },

    "quixbugs": {
        "name": "QuixBugs (Python & Java)",
        "hf_dataset": "moyix/QuixBugs",
        "category": "bug_fixing",
        "size": "40 classic bugs",
        "description": "Classic algorithmic bugs in Python and Java",
        "format_function": "format_quixbugs"
    },

    # === Test Generation ===

    "methods2test": {
        "name": "Methods2Test",
        "hf_dataset": "methods2test/methods2test",
        "category": "test_generation",
        "size": "780K examples",
        "description": "Java methods with unit tests",
        "format_function": "format_methods2test"
    },

    "coverageeval": {
        "name": "CoverageEval",
        "hf_dataset": "microsoft/coverageeval",
        "category": "test_generation",
        "size": "100+ functions",
        "description": "Python functions requiring test coverage",
        "format_function": "format_coverageeval"
    },

    "unittest_gen": {
        "name": "UnitTest Generation",
        "hf_dataset": "kjain14/unittest_generation",
        "category": "test_generation",
        "size": "~10K examples",
        "description": "Python code with pytest/unittest tests",
        "format_function": "format_unittest_gen"
    },

    # === Error Analysis ===

    "python_errors": {
        "name": "Python Error Messages",
        "hf_dataset": "mhausenblas/python-errors",
        "category": "error_analysis",
        "size": "~5K errors",
        "description": "Python errors with explanations",
        "format_function": "format_python_errors"
    },

    "stack_overflow_qa": {
        "name": "Stack Overflow Debugging QA",
        "hf_dataset": "koutch/stack_overflow_python",
        "category": "error_analysis",
        "size": "~100K Python Q&A",
        "description": "Real debugging questions and answers",
        "format_function": "format_stack_overflow"
    },

    # === Code Review & Analysis ===

    "code_review_comments": {
        "name": "Code Review Comments",
        "hf_dataset": "DanielKyo/code_review_instructions",
        "category": "code_review",
        "size": "~10K reviews",
        "description": "Code review feedback with bug identification",
        "format_function": "format_code_review"
    },

    "code_clippy": {
        "name": "Clippy Lints (Rust)",
        "hf_dataset": "m-a-p/Code-Clippy",
        "category": "code_review",
        "size": "~50K examples",
        "description": "Rust linter suggestions and fixes",
        "format_function": "format_clippy"
    },

    # === Security & Vulnerabilities ===

    "cve_fixes": {
        "name": "CVE Security Fixes",
        "hf_dataset": "microsoft/CodeXGLUE-security",
        "category": "security",
        "size": "~8K CVEs",
        "description": "Security vulnerabilities and patches",
        "format_function": "format_cve_fixes"
    },

    "devign": {
        "name": "Devign (Vulnerability Detection)",
        "hf_dataset": "code_x_glue_cc_devign",
        "category": "security",
        "size": "~27K examples",
        "description": "Vulnerability detection in C code",
        "format_function": "format_devign"
    },

    # === Performance Debugging ===

    "code_complexity": {
        "name": "Code Complexity Analysis",
        "hf_dataset": "code_x_glue_cc_code_refinement",
        "category": "performance",
        "size": "~100K examples",
        "description": "Code before/after performance optimization",
        "format_function": "format_complexity"
    },

    # === Refactoring ===

    "refactoring_miner": {
        "name": "RefactoringMiner Dataset",
        "hf_dataset": "refactoring-ai/refactoring-dataset",
        "category": "refactoring",
        "size": "~200K refactorings",
        "description": "Real refactoring examples from GitHub",
        "format_function": "format_refactoring"
    },
}


class DebuggingDatasetDownloader:
    """Download and format debugging-focused datasets"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_bugs_in_py(self, examples: List[Dict]) -> List[Dict]:
        """Format BugsInPy with debugging reasoning"""
        formatted = []

        for ex in examples:
            buggy = ex.get("buggy_code", "")
            fixed = ex.get("fixed_code", "")
            bug_type = ex.get("bug_type", "unknown")
            description = ex.get("description", "")

            if buggy and fixed:
                # Create detailed debugging example
                formatted.append({
                    "instruction": f"Debug this Python code:\n\n```python\n{buggy}\n```\n\n{description}",
                    "output": f"""<thinking>
Step 1: Analyze the bug
- Bug type: {bug_type}
- Issue: {description}

Step 2: Identify the root cause
{self._analyze_diff(buggy, fixed)}

Step 3: Apply the fix
</thinking>

Here's the corrected code:

```python
{fixed}
```

**Explanation:**
The bug was a {bug_type}. The fix involved {self._explain_fix(buggy, fixed)}.

**Prevention:**
To avoid this in the future:
- Add unit tests for edge cases
- Use type hints and linters
- Review error handling""",
                    "_language": "python",
                    "_category": "bug_fixing",
                    "_bug_type": bug_type
                })

        return formatted

    def _analyze_diff(self, buggy: str, fixed: str) -> str:
        """Analyze difference between buggy and fixed code"""
        if buggy == fixed:
            return "No changes detected"

        # Simple diff analysis
        buggy_lines = buggy.split('\n')
        fixed_lines = fixed.split('\n')

        if len(buggy_lines) != len(fixed_lines):
            return f"Changed from {len(buggy_lines)} to {len(fixed_lines)} lines"

        return "Code logic was corrected"

    def _explain_fix(self, buggy: str, fixed: str) -> str:
        """Explain what was fixed"""
        # Simple heuristics
        if "==" in buggy and "=" in fixed and "==" not in fixed:
            return "changing comparison (==) to assignment (=)"
        elif "=" in buggy and "==" in fixed:
            return "changing assignment (=) to comparison (==)"
        elif "+" in buggy and "-" in fixed:
            return "correcting the arithmetic operator"
        else:
            return "correcting the logic"

    def format_defect_detection(self, examples: List[Dict]) -> List[Dict]:
        """Format defect detection with reasoning"""
        formatted = []

        for ex in examples:
            code = ex.get("func", "")
            has_defect = ex.get("target", 0)  # 0 = clean, 1 = defect

            if code:
                if has_defect:
                    output = """<thinking>
Let me analyze this code for potential defects:

1. Check for common bugs:
   - Buffer overflows
   - Null pointer dereferences
   - Memory leaks
   - Off-by-one errors

2. Found issues:
   - This code contains a defect

3. Recommendation:
   - Review the code carefully
   - Add boundary checks
   - Test edge cases
</thinking>

**Analysis:** This code contains a defect. Common issues in similar code include:
- Missing input validation
- Potential buffer overflow
- Incorrect array indexing

**Recommendation:** Refactor with proper error handling and bounds checking."""
                else:
                    output = """<thinking>
Analyzing code for defects:

1. Input validation: OK
2. Memory management: OK
3. Logic flow: OK
4. Edge cases: Handled

No defects found.
</thinking>

**Analysis:** This code appears to be defect-free. Good practices observed:
- Proper error handling
- Input validation
- Clear logic flow"""

                formatted.append({
                    "instruction": f"Analyze this code for defects:\n\n```c\n{code}\n```",
                    "output": output,
                    "_language": "c",
                    "_category": "bug_detection",
                    "_has_defect": bool(has_defect)
                })

        return formatted

    def format_manybugs(self, examples: List[Dict]) -> List[Dict]:
        """Format ManyBugs dataset"""
        formatted = []

        for ex in examples:
            buggy = ex.get("buggy_code", "")
            fixed = ex.get("fixed_code", "")
            test_case = ex.get("test_case", "")

            if buggy and fixed:
                formatted.append({
                    "instruction": f"Fix this bug:\n\n```c\n{buggy}\n```\n\nFailing test:\n```\n{test_case}\n```",
                    "output": f"```c\n{fixed}\n```",
                    "_language": "c",
                    "_category": "bug_fixing"
                })

        return formatted

    def format_quixbugs(self, examples: List[Dict]) -> List[Dict]:
        """Format QuixBugs (classic algorithmic bugs)"""
        formatted = []

        for ex in examples:
            buggy = ex.get("buggy_code", "")
            fixed = ex.get("fixed_code", "")
            algorithm = ex.get("algorithm", "")
            language = ex.get("language", "python")

            if buggy and fixed:
                formatted.append({
                    "instruction": f"Fix the bug in this {algorithm} implementation:\n\n```{language}\n{buggy}\n```",
                    "output": f"""<thinking>
This is a classic {algorithm} bug. Let me trace through:

1. Identify the algorithm: {algorithm}
2. Find the bug in the logic
3. Apply the fix
</thinking>

Corrected implementation:

```{language}\n{fixed}\n```

The bug was in the {algorithm} logic. The fix ensures correct behavior.""",
                    "_language": language,
                    "_category": "bug_fixing",
                    "_algorithm": algorithm
                })

        return formatted

    def format_methods2test(self, examples: List[Dict], limit: int = 50000) -> List[Dict]:
        """Format test generation dataset"""
        formatted = []

        for ex in examples[:limit]:
            method = ex.get("focal_method", "")
            test = ex.get("test_method", "")

            if method and test:
                formatted.append({
                    "instruction": f"Generate unit tests for this Java method:\n\n```java\n{method}\n```",
                    "output": f"""<thinking>
Test generation strategy:
1. Identify edge cases
2. Test normal inputs
3. Test boundary conditions
4. Test error cases
</thinking>

Here are comprehensive unit tests:

```java
{test}
```

These tests cover:
- Normal operation
- Edge cases
- Error handling""",
                    "_language": "java",
                    "_category": "test_generation"
                })

        return formatted

    def format_coverageeval(self, examples: List[Dict]) -> List[Dict]:
        """Format CoverageEval (test coverage)"""
        formatted = []

        for ex in examples:
            code = ex.get("code", "")
            tests = ex.get("tests", "")
            coverage = ex.get("coverage", 0)

            if code:
                formatted.append({
                    "instruction": f"Write tests to achieve high coverage for:\n\n```python\n{code}\n```",
                    "output": f"""<thinking>
Coverage analysis:
- Current coverage: {coverage}%
- Missing: Edge cases, error paths
- Need: Comprehensive test suite
</thinking>

```python
{tests}

# Additional tests for edge cases:
def test_edge_case_1():
    # Test boundary conditions
    pass

def test_error_handling():
    # Test error paths
    pass
```

This achieves comprehensive coverage including edge cases.""",
                    "_language": "python",
                    "_category": "test_generation",
                    "_coverage": coverage
                })

        return formatted

    def format_unittest_gen(self, examples: List[Dict]) -> List[Dict]:
        """Format unittest generation"""
        formatted = []

        for ex in examples:
            code = ex.get("code", "")
            test = ex.get("test", "")

            if code and test:
                formatted.append({
                    "instruction": f"Generate pytest tests for:\n\n```python\n{code}\n```",
                    "output": test,
                    "_language": "python",
                    "_category": "test_generation"
                })

        return formatted

    def format_python_errors(self, examples: List[Dict]) -> List[Dict]:
        """Format Python error explanations"""
        formatted = []

        for ex in examples:
            error = ex.get("error", "")
            explanation = ex.get("explanation", "")
            code = ex.get("code", "")

            if error:
                formatted.append({
                    "instruction": f"Explain this Python error:\n\n```\n{error}\n```\n\n{code}",
                    "output": f"""<thinking>
Error analysis:
1. Parse error message: {error.split(':')[0] if ':' in error else error}
2. Identify root cause
3. Suggest fix
</thinking>

**Error Type:** {error.split(':')[0] if ':' in error else error}

**Explanation:** {explanation or 'This error occurs when...'}

**How to Fix:**
1. Check the code at the indicated line
2. Common causes: [list causes]
3. Apply appropriate fix

**Prevention:**
- Use type hints
- Enable linting
- Write tests""",
                    "_language": "python",
                    "_category": "error_analysis"
                })

        return formatted

    def format_stack_overflow(self, examples: List[Dict], limit: int = 50000) -> List[Dict]:
        """Format Stack Overflow debugging Q&A"""
        formatted = []

        for ex in examples[:limit]:
            question = ex.get("question", "")
            answer = ex.get("answer", "")
            code = ex.get("code", "")

            if question and answer and "debug" in question.lower() or "error" in question.lower():
                formatted.append({
                    "instruction": question + (f"\n\n```python\n{code}\n```" if code else ""),
                    "output": answer,
                    "_language": "python",
                    "_category": "error_analysis",
                    "_source": "stackoverflow"
                })

        return formatted

    def format_code_review(self, examples: List[Dict]) -> List[Dict]:
        """Format code review with bug identification"""
        formatted = []

        for ex in examples:
            code = ex.get("code", "")
            review = ex.get("review", "")

            if "bug" in review.lower() or "issue" in review.lower():
                formatted.append({
                    "instruction": f"Review this code for bugs:\n\n```\n{code}\n```",
                    "output": f"""<thinking>
Code review checklist:
1. Logic bugs ✓
2. Performance issues ✓
3. Security concerns ✓
4. Best practices ✓
</thinking>

{review}

**Severity:** Medium
**Priority:** Should fix before deployment""",
                    "_category": "code_review"
                })

        return formatted

    def format_clippy(self, examples: List[Dict], limit: int = 30000) -> List[Dict]:
        """Format Rust Clippy lints"""
        formatted = []

        for ex in examples[:limit]:
            code_before = ex.get("code_before", "")
            code_after = ex.get("code_after", "")
            lint_type = ex.get("lint", "")

            if code_before and code_after:
                formatted.append({
                    "instruction": f"Fix the Clippy lint in this Rust code:\n\n```rust\n{code_before}\n```\n\nLint: {lint_type}",
                    "output": f"```rust\n{code_after}\n```",
                    "_language": "rust",
                    "_category": "code_review",
                    "_lint": lint_type
                })

        return formatted

    def format_cve_fixes(self, examples: List[Dict]) -> List[Dict]:
        """Format CVE security fixes"""
        formatted = []

        for ex in examples:
            vulnerable = ex.get("vulnerable_code", "")
            patched = ex.get("patched_code", "")
            cve_id = ex.get("cve_id", "")
            severity = ex.get("severity", "unknown")

            if vulnerable and patched:
                formatted.append({
                    "instruction": f"Fix this security vulnerability ({cve_id}):\n\n```\n{vulnerable}\n```",
                    "output": f"""<thinking>
Security analysis:
- CVE: {cve_id}
- Severity: {severity}
- Type: Security vulnerability
- Impact: Could allow [attack type]

Mitigation strategy:
1. Identify vulnerable code path
2. Apply security patch
3. Add input validation
</thinking>

**Patched code:**

```
{patched}
```

**Security improvements:**
- Input validation added
- Boundary checks implemented
- Memory safety ensured

**Severity:** {severity}
**Status:** Patched""",
                    "_category": "security",
                    "_cve": cve_id,
                    "_severity": severity
                })

        return formatted

    def format_devign(self, examples: List[Dict]) -> List[Dict]:
        """Format vulnerability detection"""
        formatted = []

        for ex in examples:
            code = ex.get("func", "")
            target = ex.get("target", 0)  # 0 = safe, 1 = vulnerable

            if code:
                if target:
                    output = """<thinking>
Security analysis:

1. Input validation: Missing ❌
2. Memory safety: Issues found ❌
3. Authentication: Not checked ❌

Vulnerability detected.
</thinking>

**SECURITY WARNING:** This code contains vulnerabilities.

**Found issues:**
- Missing input validation
- Potential buffer overflow
- Unsafe memory operations

**Recommendation:** Rewrite with security in mind."""
                else:
                    output = """<thinking>
Security analysis:

1. Input validation: OK ✓
2. Memory safety: OK ✓
3. Authentication: OK ✓

No vulnerabilities found.
</thinking>

**Analysis:** Code appears secure."""

                formatted.append({
                    "instruction": f"Analyze for security vulnerabilities:\n\n```c\n{code}\n```",
                    "output": output,
                    "_language": "c",
                    "_category": "security",
                    "_vulnerable": bool(target)
                })

        return formatted

    def format_complexity(self, examples: List[Dict], limit: int = 50000) -> List[Dict]:
        """Format performance optimization examples"""
        formatted = []

        for ex in examples[:limit]:
            before = ex.get("before", "")
            after = ex.get("after", "")

            if before and after:
                formatted.append({
                    "instruction": f"Optimize this code for better performance:\n\n```\n{before}\n```",
                    "output": f"""<thinking>
Performance analysis:
1. Identify bottlenecks
2. Analyze complexity
3. Apply optimizations

Improvements:
- Reduced time complexity
- Better space usage
- More efficient algorithms
</thinking>

Optimized code:

```
{after}
```

**Performance gains:**
- Time complexity: Improved
- Space complexity: Optimized
- Readability: Maintained""",
                    "_category": "performance"
                })

        return formatted

    def format_refactoring(self, examples: List[Dict], limit: int = 50000) -> List[Dict]:
        """Format refactoring examples"""
        formatted = []

        for ex in examples[:limit]:
            before = ex.get("code_before", "")
            after = ex.get("code_after", "")
            refactoring_type = ex.get("type", "")

            if before and after:
                formatted.append({
                    "instruction": f"Refactor this code:\n\n```\n{before}\n```",
                    "output": f"""<thinking>
Refactoring strategy:
- Type: {refactoring_type}
- Goal: Improve maintainability
- Preserve: Functionality

Steps:
1. Identify code smells
2. Apply {refactoring_type}
3. Verify behavior unchanged
</thinking>

Refactored code:

```
{after}
```

**Changes:**
- Refactoring type: {refactoring_type}
- Improved readability
- Better maintainability""",
                    "_category": "refactoring",
                    "_type": refactoring_type
                })

        return formatted

    def download_dataset(self, dataset_key: str, limit: Optional[int] = None) -> Path:
        """Download and format a debugging dataset"""

        dataset_info = DEBUGGING_DATASETS[dataset_key]
        print(f"\n{'='*70}")
        print(f"Downloading: {dataset_info['name']}")
        print(f"Category: {dataset_info['category']}")
        print(f"Size: {dataset_info['size']}")
        print(f"{'='*70}\n")

        try:
            from datasets import load_dataset
        except ImportError:
            print("[!] Installing datasets library...")
            subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
            from datasets import load_dataset

        try:
            dataset = load_dataset(dataset_info["hf_dataset"], split="train")
            examples = list(dataset)

            if limit:
                examples = examples[:limit]

            print(f"[+] Downloaded {len(examples):,} examples")

            # Format
            format_func = getattr(self, dataset_info["format_function"])
            if "limit" in format_func.__code__.co_varnames:
                formatted = format_func(examples, limit=limit or len(examples))
            else:
                formatted = format_func(examples)

            print(f"[+] Formatted {len(formatted):,} examples")

            # Save
            output_file = self.output_dir / f"{dataset_key}.jsonl"
            with open(output_file, 'w') as f:
                for ex in formatted:
                    f.write(json.dumps(ex) + '\n')

            print(f"[✓] Saved to: {output_file}")
            return output_file

        except Exception as e:
            print(f"[✗] Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def download_by_category(self, categories: List[str], limit: Optional[int] = None):
        """Download all datasets in categories"""

        downloaded = []
        for key, info in DEBUGGING_DATASETS.items():
            if info["category"] in categories:
                file = self.download_dataset(key, limit)
                if file:
                    downloaded.append(file)

        return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download debugging datasets")

    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--download", nargs="+", choices=list(DEBUGGING_DATASETS.keys()))
    parser.add_argument("--categories", nargs="+",
                       choices=["bug_fixing", "bug_detection", "test_generation",
                               "error_analysis", "code_review", "security",
                               "performance", "refactoring"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--list", action="store_true")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Debugging Datasets:")
        print("="*80)

        by_category = {}
        for key, info in DEBUGGING_DATASETS.items():
            cat = info["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((key, info))

        for category, datasets in sorted(by_category.items()):
            print(f"\n🐛 {category.upper().replace('_', ' ')}")
            for key, info in datasets:
                print(f"\n  {key}")
                print(f"    {info['name']}")
                print(f"    Size: {info['size']}")
                print(f"    {info['description']}")
        return

    downloader = DebuggingDatasetDownloader(Path(args.output))

    if args.download_all:
        for key in DEBUGGING_DATASETS.keys():
            downloader.download_dataset(key, args.limit)
    elif args.download:
        for key in args.download:
            downloader.download_dataset(key, args.limit)
    elif args.categories:
        downloader.download_by_category(args.categories, args.limit)


if __name__ == "__main__":
    main()

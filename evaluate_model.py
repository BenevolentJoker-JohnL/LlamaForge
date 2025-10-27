#!/usr/bin/env python3
"""
Model Evaluation Script - Measure training improvements
Tests models on coding tasks, instruction following, and general knowledge
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_banner(title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}{Colors.END}\n")

def run_eval(model_name, tasks, limit=100, output_path=None):
    """Run lm-eval on specific tasks"""

    print(f"{Colors.BLUE}[INFO]{Colors.END} Evaluating {Colors.BOLD}{model_name}{Colors.END} on {len(tasks)} tasks...")
    print(f"{Colors.BLUE}[INFO]{Colors.END} Tasks: {', '.join(tasks)}")
    print(f"{Colors.BLUE}[INFO]{Colors.END} Samples per task: {limit}\n")

    # Build command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name}",
        "--tasks", ",".join(tasks),
        "--batch_size", "1",
        "--limit", str(limit),
    ]

    if output_path:
        cmd.extend(["--output_path", output_path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}[✗]{Colors.END} Evaluation timed out after 1 hour")
        return None, None, -1
    except Exception as e:
        print(f"{Colors.RED}[✗]{Colors.END} Error running evaluation: {e}")
        return None, None, -1

def parse_results(stdout):
    """Extract scores from lm-eval output"""
    results = {}

    for line in stdout.split('\n'):
        # lm-eval outputs results like: "task_name|metric_name|value"
        if '|' in line and 'acc' in line.lower():
            try:
                parts = line.split('|')
                task = parts[0].strip()
                metric = parts[1].strip()
                value = float(parts[2].strip())

                if task not in results:
                    results[task] = {}
                results[task][metric] = value
            except:
                pass

    return results

def compare_results(before_file, after_file):
    """Compare two evaluation result files"""
    with open(before_file) as f:
        before = json.load(f)
    with open(after_file) as f:
        after = json.load(f)

    print_banner("TRAINING IMPROVEMENT REPORT")

    for task in before.get('results', {}):
        if task in after.get('results', {}):
            before_acc = before['results'][task].get('acc', 0) or before['results'][task].get('acc_norm', 0)
            after_acc = after['results'][task].get('acc', 0) or after['results'][task].get('acc_norm', 0)

            if before_acc and after_acc:
                improvement = ((after_acc - before_acc) / before_acc) * 100 if before_acc > 0 else 0

                color = Colors.GREEN if improvement > 0 else Colors.RED if improvement < 0 else Colors.YELLOW
                arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"

                print(f"  {task:30} {before_acc*100:6.2f}% → {after_acc*100:6.2f}%  {color}{arrow} {improvement:+6.2f}%{Colors.END}")

REASONING_TASKS = {
    "cot": ["gsm8k", "math_qa"],  # Math tasks that benefit from CoT
    "reasoning": ["strategyqa", "arc_challenge", "winogrande"],  # Multi-step reasoning
    "all_reasoning": ["gsm8k", "math_qa", "strategyqa", "arc_challenge", "winogrande", "piqa", "hellaswag"]
}

def evaluate_reasoning(model_name, preset="cot", limit=100, output_path=None):
    """
    Evaluate chain-of-thought and reasoning capabilities

    Presets:
    - cot: Math tasks requiring step-by-step reasoning
    - reasoning: Multi-hop reasoning tasks
    - all_reasoning: Comprehensive reasoning evaluation
    """
    tasks = REASONING_TASKS.get(preset, REASONING_TASKS["cot"])

    print_banner(f"REASONING EVALUATION - {preset.upper()} PRESET")
    print(f"{Colors.BLUE}[INFO]{Colors.END} This measures chain-of-thought and multi-step reasoning")
    print(f"{Colors.BLUE}[INFO]{Colors.END} Higher scores indicate better reasoning ability\n")

    stdout, stderr, returncode = run_eval(model_name, tasks, limit, output_path)

    if returncode == 0:
        results = parse_results(stdout)
        print_banner("REASONING RESULTS")
        for task, metrics in results.items():
            for metric, value in metrics.items():
                print(f"  {task:30} {metric:15} {value*100:6.2f}%")

    return returncode

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate models before/after training")
    parser.add_argument("--model", required=True, help="Model path or HF model name")
    parser.add_argument("--preset", choices=["quick", "standard", "comprehensive"], default="quick",
                       help="Evaluation preset (quick=3 tasks, standard=10 tasks, comprehensive=20 tasks)")
    parser.add_argument("--reasoning", action="store_true",
                       help="Run reasoning-focused evaluation (CoT/ToT tasks)")
    parser.add_argument("--reasoning-preset", choices=["cot", "reasoning", "all_reasoning"], default="cot",
                       help="Reasoning evaluation preset (cot=math, reasoning=multi-hop, all=comprehensive)")
    parser.add_argument("--limit", type=int, default=100, help="Max samples per task")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--compare", help="Compare with previous results JSON file")

    args = parser.parse_args()

    # If reasoning mode requested, use reasoning evaluator
    if args.reasoning:
        output_dir = Path(args.output) if args.output else Path("./evaluations/reasoning")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"reasoning_eval_{timestamp}"

        returncode = evaluate_reasoning(
            model_name=args.model,
            preset=args.reasoning_preset,
            limit=args.limit,
            output_path=str(output_path)
        )

        if returncode == 0:
            print(f"\n{Colors.GREEN}[✓]{Colors.END} Reasoning evaluation complete!")
        else:
            print(f"\n{Colors.RED}[✗]{Colors.END} Reasoning evaluation failed!")

        sys.exit(returncode)

    # Define task presets
    presets = {
        "quick": [
            "hellaswag",      # Common sense reasoning
            "piqa",           # Physical reasoning
            "winogrande",     # Coreference resolution
        ],
        "standard": [
            "hellaswag",
            "piqa",
            "winogrande",
            "arc_easy",       # Science Q&A
            "arc_challenge",  # Hard science Q&A
            "boolq",          # Yes/no questions
            "openbookqa",     # Open book Q&A
            "mathqa",         # Math reasoning
            "gsm8k",          # Grade school math
        ],
        "comprehensive": [
            # Add all standard tasks plus more
            "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge",
            "boolq", "openbookqa", "mathqa", "gsm8k",
            "mmlu",           # Massive multitask
            "humaneval",      # Code generation (Python)
            "mbpp",           # More code problems
            "lambada",        # Language modeling
            "wikitext",       # Perplexity test
        ]
    }

    tasks = presets[args.preset]

    print_banner(f"MODEL EVALUATION - {args.preset.upper()} MODE")
    print(f"{Colors.BLUE}Model:{Colors.END} {args.model}")
    print(f"{Colors.BLUE}Tasks:{Colors.END} {len(tasks)} tasks, {args.limit} samples each")
    print(f"{Colors.BLUE}Time estimate:{Colors.END} ~{len(tasks) * 2}-{len(tasks) * 5} minutes\n")

    # Setup output
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("./evaluations")

    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}"

    # Run evaluation
    stdout, stderr, returncode = run_eval(
        model_name=args.model,
        tasks=tasks,
        limit=args.limit,
        output_path=str(output_path)
    )

    if returncode == 0:
        print(f"\n{Colors.GREEN}[✓]{Colors.END} Evaluation complete!")
        print(f"{Colors.BLUE}[INFO]{Colors.END} Results saved to: {output_path}")

        # Parse and display results
        results = parse_results(stdout)
        if results:
            print_banner("RESULTS SUMMARY")
            for task, metrics in results.items():
                print(f"{Colors.BOLD}{task}{Colors.END}")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value*100:.2f}%")

        # Compare if requested
        if args.compare:
            result_file = list(output_path.glob("*.json"))[0]
            compare_results(args.compare, result_file)
    else:
        print(f"\n{Colors.RED}[✗]{Colors.END} Evaluation failed!")
        if stderr:
            print(f"\n{Colors.RED}Error output:{Colors.END}\n{stderr}")

if __name__ == "__main__":
    main()

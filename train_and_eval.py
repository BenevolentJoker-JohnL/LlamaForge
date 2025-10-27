#!/usr/bin/env python3
"""
Complete Training + Evaluation Workflow

Trains a model and immediately evaluates it on both standard and reasoning benchmarks.

Usage:
    # Basic usage
    python train_and_eval.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --data examples/datasets/claude_ultimate_with_tools_621k.jsonl \
        --epochs 1

    # With baseline comparison
    python train_and_eval.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --data examples/datasets/claude_reasoning_mega.jsonl \
        --epochs 1 \
        --eval-baseline
"""

import subprocess
import argparse
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


def run_command(cmd, description):
    """Run a command and show output"""
    print(f"\n{Colors.BLUE}[Running]{Colors.END} {description}...")
    print(f"{Colors.BLUE}[Command]{Colors.END} {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"\n{Colors.GREEN}[✓]{Colors.END} {description} completed successfully")
            return True
        else:
            print(f"\n{Colors.RED}[✗]{Colors.END} {description} failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n{Colors.RED}[✗]{Colors.END} {description} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train model and automatically evaluate"
    )

    # Training args
    parser.add_argument("--model", required=True, help="Base model to train")
    parser.add_argument("--data", required=True, help="Training dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--work-dir", help="Training output directory")
    parser.add_argument("--no-gguf", action="store_true", help="Skip GGUF conversion")

    # Evaluation args
    parser.add_argument("--eval-baseline", action="store_true",
                       help="Evaluate base model before training (for comparison)")
    parser.add_argument("--eval-preset", choices=["quick", "standard", "comprehensive"],
                       default="standard", help="Evaluation preset")
    parser.add_argument("--eval-reasoning", action="store_true",
                       help="Also run reasoning-focused evaluation")
    parser.add_argument("--eval-limit", type=int, default=100,
                       help="Samples per eval task")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip post-training evaluation")

    args = parser.parse_args()

    print_banner("COMPLETE TRAINING + EVALUATION WORKFLOW")

    # Setup paths
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = Path(f"./trained_model_{timestamp}")

    eval_dir = work_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"{Colors.BLUE}[Config]{Colors.END}")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {work_dir}")
    print(f"  Eval preset: {args.eval_preset}")

    # Step 1: Baseline evaluation (if requested)
    baseline_results = None
    if args.eval_baseline:
        print_banner("STEP 1/4: BASELINE EVALUATION")
        print(f"{Colors.YELLOW}[i]{Colors.END} Evaluating base model before training...")

        # Standard evaluation
        baseline_cmd = [
            "python", "evaluate_model.py",
            "--model", args.model,
            "--preset", args.eval_preset,
            "--limit", str(args.eval_limit),
            "--output", str(eval_dir / "baseline")
        ]

        if run_command(baseline_cmd, "Baseline standard evaluation"):
            baseline_results = eval_dir / "baseline"

        # Reasoning evaluation
        if args.eval_reasoning:
            baseline_reasoning_cmd = [
                "python", "evaluate_model.py",
                "--model", args.model,
                "--reasoning",
                "--reasoning-preset", "all_reasoning",
                "--limit", str(args.eval_limit),
                "--output", str(eval_dir / "baseline_reasoning")
            ]
            run_command(baseline_reasoning_cmd, "Baseline reasoning evaluation")
    else:
        print(f"{Colors.YELLOW}[i]{Colors.END} Skipping baseline evaluation (use --eval-baseline to enable)")

    # Step 2: Training
    step_num = 2 if args.eval_baseline else 1
    total_steps = 4 if args.eval_baseline else 3

    print_banner(f"STEP {step_num}/{total_steps}: TRAINING")

    training_cmd = [
        "python", "llamaforge.py",
        "--model", args.model,
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation", str(args.gradient_accumulation),
        "--max-length", str(args.max_length),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--work-dir", str(work_dir)
    ]

    if args.no_gguf:
        training_cmd.append("--no-gguf")

    if not run_command(training_cmd, "Model training"):
        print(f"\n{Colors.RED}[✗]{Colors.END} Training failed. Exiting.")
        sys.exit(1)

    # Find trained model path
    trained_model = work_dir / "merged_model"
    if not trained_model.exists():
        print(f"\n{Colors.RED}[✗]{Colors.END} Trained model not found at {trained_model}")
        sys.exit(1)

    # Step 3: Post-training evaluation
    if not args.skip_eval:
        step_num += 1
        print_banner(f"STEP {step_num}/{total_steps}: POST-TRAINING EVALUATION")

        # Standard evaluation
        eval_cmd = [
            "python", "evaluate_model.py",
            "--model", str(trained_model),
            "--preset", args.eval_preset,
            "--limit", str(args.eval_limit),
            "--output", str(eval_dir / "after_training")
        ]

        # Add comparison if baseline exists
        if baseline_results:
            # Find baseline results JSON
            baseline_json = list(baseline_results.glob("**/results.json"))
            if baseline_json:
                eval_cmd.extend(["--compare", str(baseline_json[0])])

        run_command(eval_cmd, "Post-training standard evaluation")

        # Reasoning evaluation
        if args.eval_reasoning:
            reasoning_cmd = [
                "python", "evaluate_model.py",
                "--model", str(trained_model),
                "--reasoning",
                "--reasoning-preset", "all_reasoning",
                "--limit", str(args.eval_limit),
                "--output", str(eval_dir / "after_training_reasoning")
            ]
            run_command(reasoning_cmd, "Post-training reasoning evaluation")

    # Step 4: Summary
    step_num += 1
    print_banner(f"STEP {step_num}/{total_steps}: COMPLETE")

    print(f"{Colors.GREEN}[✓]{Colors.END} Training and evaluation workflow complete!\n")
    print(f"{Colors.BOLD}Results:{Colors.END}")
    print(f"  Trained model: {trained_model}")
    print(f"  Evaluations: {eval_dir}")

    if not args.skip_eval:
        print(f"\n{Colors.BOLD}Evaluation Results:{Colors.END}")

        # Show baseline results
        if args.eval_baseline:
            print(f"\n  {Colors.CYAN}Baseline (before training):{Colors.END}")
            print(f"    {eval_dir / 'baseline'}")
            if args.eval_reasoning:
                print(f"    {eval_dir / 'baseline_reasoning'}")

        # Show after-training results
        print(f"\n  {Colors.CYAN}After training:{Colors.END}")
        print(f"    {eval_dir / 'after_training'}")
        if args.eval_reasoning:
            print(f"    {eval_dir / 'after_training_reasoning'}")

        # Show comparison summary if available
        if baseline_results:
            print(f"\n{Colors.YELLOW}[i]{Colors.END} Check the comparison output above for improvement metrics!")

    # Next steps
    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    if not args.skip_eval:
        print(f"  1. Review evaluation results in {eval_dir}")
        print(f"  2. Check comparison metrics to see improvements")

    if not args.no_gguf:
        gguf_path = work_dir / "gguf"
        if gguf_path.exists():
            print(f"  3. Create Ollama model from GGUF: {gguf_path}")
            print(f"     cd {work_dir}")
            print(f"     ollama create my-model -f Modelfile")
            print(f"  4. Test your model: ollama run my-model")
    else:
        print(f"  3. Convert to GGUF if needed:")
        print(f"     python llamaforge.py --model {trained_model} --convert-only --quantization q4_k_m")

    print()


if __name__ == "__main__":
    main()

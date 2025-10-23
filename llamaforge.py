#!/usr/bin/env python3
"""
LlamaForge - CPU-only, training-agnostic fine-tuning for GGUF models

Usage:
    python llamaforge.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --data ./data/training.jsonl \\
        --output ./finetuned.gguf \\
        --epochs 3 \\
        --quantization q4_k_m
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset_loader import DatasetLoader
from lora_trainer import LoRATrainer
from gguf_converter import GGUFConverter
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="LlamaForge - CPU-only fine-tuning for GGUF models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune Mistral on custom dataset
  python llamaforge.py --model mistralai/Mistral-7B-v0.1 --data train.jsonl

  # With custom parameters
  python llamaforge.py \\
      --model meta-llama/Llama-2-7b-hf \\
      --data data.csv \\
      --epochs 5 \\
      --batch-size 2 \\
      --learning-rate 1e-4 \\
      --output my_model.gguf

  # Skip GGUF conversion (save as HuggingFace)
  python llamaforge.py --model mistralai/Mistral-7B-v0.1 --data train.jsonl --no-gguf
        """
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name or HuggingFace path (e.g., mistralai/Mistral-7B-v0.1)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Training data file (JSON, JSONL, CSV, or TXT)"
    )

    # Training parameters
    parser.add_argument(
        "--output",
        type=str,
        default="./output/finetuned.gguf",
        help="Output path for fine-tuned model (default: ./output/finetuned.gguf)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size (default: 1, increase if you have RAM)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    # LoRA parameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )

    # GGUF parameters
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q4_k_m", "q5_k_m", "q6_k"],
        help="GGUF quantization method (default: q4_k_m)"
    )
    parser.add_argument(
        "--no-gguf",
        action="store_true",
        help="Skip GGUF conversion, keep as HuggingFace format"
    )

    # Other options
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./work",
        help="Working directory for intermediate files (default: ./work)"
    )

    args = parser.parse_args()

    # Validate inputs
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LlamaForge - CPU-only Fine-Tuning Pipeline")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"Max length: {args.max_length}")
    print(f"Quantization: {args.quantization if not args.no_gguf else 'None (HF format)'}")
    print("=" * 80)

    try:
        # Step 1: Load tokenizer
        print("\n[1/4] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded")

        # Step 2: Load and structure dataset
        print("\n[2/4] Loading and structuring dataset...")
        loader = DatasetLoader(
            file_path=args.data,
            tokenizer=tokenizer,
            max_length=args.max_length
        )
        dataset = loader.load()
        print(f"Dataset ready: {len(dataset)} samples")

        # Step 3: Train LoRA
        print("\n[3/4] Training LoRA adapter...")
        trainer = LoRATrainer(
            model_path=str(work_dir / "model"),
            output_dir=str(work_dir / "training"),
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        model, tokenizer = trainer.load_model(args.model)
        adapter_path = trainer.train(
            dataset=dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation
        )
        print(f"LoRA adapter trained: {adapter_path}")

        # Step 4: Merge and convert
        print("\n[4/4] Merging LoRA and converting to final format...")
        converter = GGUFConverter(
            base_model_name=args.model,
            adapter_path=str(adapter_path),
            output_dir=str(work_dir / "output")
        )

        if args.no_gguf:
            # Just merge, don't convert to GGUF
            final_path = converter.merge_lora()
            print(f"\nFine-tuned model saved (HuggingFace format): {final_path}")
        else:
            # Full conversion to GGUF
            final_path = converter.full_conversion(quantization=args.quantization)

            # Move to final output location
            if final_path != output_path:
                import shutil
                shutil.copy(final_path, output_path)
                final_path = output_path

            print(f"\nFine-tuned GGUF model saved: {final_path}")

        print("\n" + "=" * 80)
        print("Training complete!")
        print("=" * 80)
        print(f"\nYour fine-tuned model is ready at: {final_path}")

        if not args.no_gguf:
            print("\nTo use with Ollama:")
            print(f"  ollama create my-model -f Modelfile")
            print(f"  (where Modelfile contains: FROM {final_path})")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

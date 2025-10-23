#!/usr/bin/env python3
"""
Teacher-Student Distributed Training Example

This script demonstrates how to use SOLLOL with LlamaForge for
distributed teacher-student knowledge distillation.

Usage:
    python teacher_student_example.py

    # Or with custom parameters:
    python teacher_student_example.py \
        --teacher qwen2.5:7b \
        --student qwen2.5:0.5b \
        --data path/to/dataset.jsonl \
        --num-students 4

Requirements:
    1. SOLLOL server running (python -m sollol.server)
    2. At least 1 LlamaForge node registered
    3. Dataset file in JSONL format
"""

import sys
import os
import argparse
from pathlib import Path

# Add LlamaForge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sollol_integration import create_distributed_training, DistributedTrainingMode


def main():
    parser = argparse.ArgumentParser(description="Teacher-Student Distributed Training")
    parser.add_argument("--teacher", default="qwen2.5:3b", help="Teacher model")
    parser.add_argument("--student", default="qwen2.5:0.5b", help="Student model")
    parser.add_argument("--data", default="testtraindata/practical_coding.jsonl", help="Dataset path")
    parser.add_argument("--num-students", type=int, default=2, help="Number of student workers")
    parser.add_argument("--teacher-batch-size", type=int, default=8, help="Teacher batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset samples (for testing)")
    parser.add_argument("--sollol-host", default="localhost", help="SOLLOL server host")
    parser.add_argument("--sollol-port", type=int, default=5000, help="SOLLOL server port")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")

    args = parser.parse_args()

    print("=" * 70)
    print("LlamaForge + SOLLOL: Teacher-Student Distributed Training")
    print("=" * 70)
    print()

    # Step 1: Connect to SOLLOL
    print("[1/6] Connecting to SOLLOL orchestrator...")
    print(f"      Server: {args.sollol_host}:{args.sollol_port}")

    try:
        orchestrator = create_distributed_training(
            mode=DistributedTrainingMode.TEACHER_STUDENT,
            config={
                "sollol_host": args.sollol_host,
                "sollol_port": args.sollol_port
            }
        )
    except Exception as e:
        print(f"[✗] Failed to connect to SOLLOL: {e}")
        print()
        print("Make sure SOLLOL server is running:")
        print("  cd ~/SOLLOL")
        print("  python -m sollol.server")
        sys.exit(1)

    print("[✓] Connected to SOLLOL")
    print()

    # Step 2: Discover nodes
    print("[2/6] Discovering available compute nodes...")
    nodes = orchestrator.discover_nodes()

    if not nodes:
        print("[✗] No compute nodes available!")
        print()
        print("Register nodes first:")
        print("  cd ~/LlamaForge")
        print("  python llamaforge_node.py --name my-node")
        sys.exit(1)

    print()

    # Step 3: Create teacher task
    print("[3/6] Creating teacher task...")
    print(f"      Teacher model: {args.teacher}")
    print(f"      Dataset: {args.data}")
    print(f"      Batch size: {args.teacher_batch_size}")
    if args.max_samples:
        print(f"      Max samples: {args.max_samples}")

    teacher_task_id = orchestrator.create_teacher_task(
        model_path=args.teacher,
        dataset_path=args.data,
        output_dir="./teacher_outputs",
        batch_size=args.teacher_batch_size,
        max_samples=args.max_samples
    )

    print(f"[✓] Teacher task created: {teacher_task_id}")
    print()

    # Step 4: Monitor teacher
    print("[4/6] Waiting for teacher to generate outputs...")
    print("      (This may take a while depending on dataset size)")

    def teacher_callback(task_id, status):
        state = status.get('state', 'unknown')
        if state == 'running':
            progress = status.get('progress', 0)
            print(f"\r      Progress: {progress}%", end='', flush=True)
        elif state == 'completed':
            print(f"\r      Progress: 100%")
        elif state == 'failed':
            print(f"\r[✗] Teacher task failed: {status.get('error', 'Unknown error')}")

    orchestrator.monitor_tasks([teacher_task_id], callback=teacher_callback)
    print("[✓] Teacher outputs generated")
    print()

    # Step 5: Create student tasks
    print(f"[5/6] Creating {args.num_students} student tasks...")
    print(f"      Student model: {args.student}")
    print(f"      LoRA config: r={args.lora_rank}, alpha={args.lora_alpha}")

    student_task_ids = orchestrator.create_student_tasks(
        base_model=args.student,
        teacher_outputs_dir="./teacher_outputs",
        num_students=args.num_students,
        lora_config={
            "r": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": 0.05
        }
    )

    print(f"[✓] Created {len(student_task_ids)} student tasks")
    print()

    # Step 6: Monitor students
    print("[6/6] Training students (in parallel)...")

    def student_callback(task_id, status):
        state = status.get('state', 'unknown')
        node = status.get('node', 'unknown')

        if state == 'running':
            epoch = status.get('epoch', 0)
            total_epochs = status.get('total_epochs', 0)
            print(f"      {task_id}: Epoch {epoch}/{total_epochs} on {node}")
        elif state == 'completed':
            print(f"[✓]   {task_id}: Training complete")
        elif state == 'failed':
            print(f"[✗]   {task_id}: Failed - {status.get('error', 'Unknown')}")

    orchestrator.monitor_tasks(student_task_ids, callback=student_callback)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"Teacher outputs:    ./teacher_outputs/")
    print(f"Student adapters:   ./output/student_*/lora_adapter/")
    print()
    print("Next steps:")
    print("  1. Merge student adapters (optional)")
    print("  2. Convert to GGUF format")
    print("  3. Create Ollama model")
    print()
    print("Or use LlamaForge interactive CLI to handle all this automatically!")
    print()


if __name__ == "__main__":
    main()

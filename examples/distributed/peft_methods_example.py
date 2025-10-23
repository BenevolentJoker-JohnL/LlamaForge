#!/usr/bin/env python3
"""
PEFT Methods Example - Train with different parameter-efficient fine-tuning methods

This example shows how to use different PEFT methods (LoRA, Prefix Tuning,
Prompt Tuning, Adapter Tuning, IA³) with SOLLOL distributed training.
"""

from src.sollol_integration import create_distributed_training


def example_lora():
    """Example: LoRA (Low-Rank Adaptation) - Most popular method"""
    print("\n" + "="*60)
    print("Example 1: LoRA Training")
    print("="*60)

    orchestrator = create_distributed_training(mode="teacher_student", config={})

    # Teacher generates outputs
    teacher_id = orchestrator.create_teacher_task(
        model_path="qwen2.5:3b",
        dataset_path="data/coding_tasks.jsonl",
        output_dir="./teacher_outputs"
    )

    # Students train with LoRA
    student_ids = orchestrator.create_student_tasks(
        base_model="qwen2.5:0.5b",
        teacher_outputs_dir="./teacher_outputs",
        num_students=4,
        peft_type="lora",
        peft_config={
            "r": 8,           # Rank of update matrices
            "alpha": 16,      # Scaling factor
            "dropout": 0.05   # Dropout for regularization
        }
    )

    print(f"[✓] Created teacher task: {teacher_id}")
    print(f"[✓] Created {len(student_ids)} LoRA student tasks")


def example_prefix_tuning():
    """Example: Prefix Tuning - Trains virtual tokens prepended to input"""
    print("\n" + "="*60)
    print("Example 2: Prefix Tuning")
    print("="*60)

    orchestrator = create_distributed_training(mode="teacher_student", config={})

    teacher_id = orchestrator.create_teacher_task(
        model_path="qwen2.5:3b",
        dataset_path="data/coding_tasks.jsonl",
        output_dir="./teacher_outputs"
    )

    # Students train with Prefix Tuning
    student_ids = orchestrator.create_student_tasks(
        base_model="qwen2.5:0.5b",
        teacher_outputs_dir="./teacher_outputs",
        num_students=4,
        peft_type="prefix",
        peft_config={
            "num_virtual_tokens": 30  # Number of prefix tokens to train
        }
    )

    print(f"[✓] Created teacher task: {teacher_id}")
    print(f"[✓] Created {len(student_ids)} Prefix Tuning student tasks")


def example_prompt_tuning():
    """Example: Prompt Tuning - Similar to prefix tuning but with soft prompts"""
    print("\n" + "="*60)
    print("Example 3: Prompt Tuning")
    print("="*60)

    orchestrator = create_distributed_training(mode="teacher_student", config={})

    teacher_id = orchestrator.create_teacher_task(
        model_path="qwen2.5:3b",
        dataset_path="data/coding_tasks.jsonl",
        output_dir="./teacher_outputs"
    )

    # Students train with Prompt Tuning (random initialization)
    student_ids = orchestrator.create_student_tasks(
        base_model="qwen2.5:0.5b",
        teacher_outputs_dir="./teacher_outputs",
        num_students=4,
        peft_type="prompt",
        peft_config={
            "num_virtual_tokens": 30,
            "prompt_tuning_init": "RANDOM"  # Or "TEXT" with init_text
        }
    )

    print(f"[✓] Created teacher task: {teacher_id}")
    print(f"[✓] Created {len(student_ids)} Prompt Tuning student tasks")


def example_adapter_tuning():
    """Example: Adapter Tuning - Adds small MLP bottlenecks between layers"""
    print("\n" + "="*60)
    print("Example 4: Adapter Tuning")
    print("="*60)

    orchestrator = create_distributed_training(mode="teacher_student", config={})

    teacher_id = orchestrator.create_teacher_task(
        model_path="qwen2.5:3b",
        dataset_path="data/coding_tasks.jsonl",
        output_dir="./teacher_outputs"
    )

    # Students train with Adapter Tuning
    student_ids = orchestrator.create_student_tasks(
        base_model="qwen2.5:0.5b",
        teacher_outputs_dir="./teacher_outputs",
        num_students=4,
        peft_type="adapter",
        peft_config={
            "reduction_factor": 16  # How much to reduce hidden dimension
        }
    )

    print(f"[✓] Created teacher task: {teacher_id}")
    print(f"[✓] Created {len(student_ids)} Adapter Tuning student tasks")


def example_ia3():
    """Example: IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)"""
    print("\n" + "="*60)
    print("Example 5: IA³")
    print("="*60)

    orchestrator = create_distributed_training(mode="teacher_student", config={})

    teacher_id = orchestrator.create_teacher_task(
        model_path="qwen2.5:3b",
        dataset_path="data/coding_tasks.jsonl",
        output_dir="./teacher_outputs"
    )

    # Students train with IA³
    student_ids = orchestrator.create_student_tasks(
        base_model="qwen2.5:0.5b",
        teacher_outputs_dir="./teacher_outputs",
        num_students=4,
        peft_type="ia3",
        peft_config={}  # IA³ uses default target modules
    )

    print(f"[✓] Created teacher task: {teacher_id}")
    print(f"[✓] Created {len(student_ids)} IA³ student tasks")


def example_comparison():
    """
    Comparison of PEFT Methods

    LoRA:
    - Most popular and versatile
    - Good balance of performance and efficiency
    - Works well across different model architectures
    - Typical trainable params: ~0.5-2% of total

    Prefix Tuning:
    - Good for smaller models
    - Fast training (fewer params than LoRA)
    - May need careful tuning of num_virtual_tokens
    - Typical trainable params: ~0.1-0.5% of total

    Prompt Tuning:
    - Similar to Prefix Tuning
    - Very lightweight
    - Good for specific task adaptation
    - Typical trainable params: ~0.1% of total

    Adapter Tuning:
    - Adds small bottleneck layers
    - Good for multi-task learning
    - Slightly more params than LoRA
    - Typical trainable params: ~1-3% of total

    IA³:
    - Extremely lightweight
    - Learns scaling vectors for activations
    - Good when compute is very limited
    - Typical trainable params: ~0.01-0.1% of total
    """
    print("\n" + "="*60)
    print("PEFT Methods Comparison")
    print("="*60)
    print(__doc__)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  LlamaForge - PEFT Methods Example                        ║
    ║  Train with different parameter-efficient tuning methods  ║
    ╚═══════════════════════════════════════════════════════════╝

    This example demonstrates how to use different PEFT methods:
    1. LoRA - Low-Rank Adaptation
    2. Prefix Tuning - Train virtual prefix tokens
    3. Prompt Tuning - Soft prompt learning
    4. Adapter Tuning - Bottleneck adapters
    5. IA³ - Learned activation scaling

    All methods work with the same SOLLOL orchestration API.
    Just change peft_type and peft_config!
    """)

    # Show comparison
    example_comparison()

    # Uncomment to run specific examples:
    # example_lora()
    # example_prefix_tuning()
    # example_prompt_tuning()
    # example_adapter_tuning()
    # example_ia3()

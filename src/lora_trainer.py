"""
CPU/GPU LoRA Trainer with automatic acceleration and memory-aware optimization
"""

import os
import torch
import psutil
from pathlib import Path
from typing import Optional, Literal
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Try to import bitsandbytes for 8-bit optimizer
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


class LoRATrainer:
    """LoRA fine-tuning with CPU/GPU support"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./output",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        device: Literal["auto", "cpu", "cuda"] = "auto"
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Validate device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"

        self.use_gpu = self.device == "cuda"

        print(f"Using device: {self.device.upper()}")
        if self.use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none"
        )

        self.model = None
        self.tokenizer = None

    def _setup_cpu_threads(self):
        """Setup optimal CPU threading for multi-core execution"""
        cpu_count = psutil.cpu_count(logical=True)
        torch.set_num_threads(cpu_count)
        print(f"CPU threads: {cpu_count}")

    def _estimate_optimal_batch_size(self, max_seq_len: int = 512) -> int:
        """Estimate optimal batch size based on available memory"""
        if self.model is None:
            return 1

        # Get available memory
        mem_available = psutil.virtual_memory().available
        safety_factor = 0.7  # Use 70% of available memory

        # Estimate memory requirements
        num_params = sum(p.numel() for p in self.model.parameters())
        bytes_per_param = 4  # float32
        model_mem = num_params * bytes_per_param

        # Estimate per-sequence memory (rough approximation)
        hidden_size = getattr(self.model.config, 'hidden_size', 4096)
        seq_mem = max_seq_len * hidden_size * bytes_per_param * 2  # x2 for gradients

        # Calculate batch size
        available_for_batch = (mem_available * safety_factor) - model_mem
        batch_size = max(1, int(available_for_batch / seq_mem))

        # Cap batch size for stability
        batch_size = min(batch_size, 8 if self.use_gpu else 4)

        return batch_size

    def load_model(self, model_name_or_path: str = "mistralai/Mistral-7B-v0.1"):
        """Load base model and apply LoRA with memory-aware settings"""
        print("Loading model...")

        # Setup CPU threading for optimal performance
        if not self.use_gpu:
            self._setup_cpu_threads()

        # Load tokenizer (trust remote code for known models)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine optimal dtype and device mapping
        if self.use_gpu:
            # Try bfloat16 first (better for training), fall back to float16
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                print("Using bfloat16 precision")
            else:
                dtype = torch.float16
                print("Using float16 precision")
            device_map = "auto"  # Hybrid CPU/GPU with automatic offloading

            # Create offload directory for hybrid setups
            offload_dir = self.output_dir / "offload"
            offload_dir.mkdir(parents=True, exist_ok=True)

            # Load model with memory-aware settings (GPU)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                offload_folder=str(offload_dir),
                offload_state_dict=True
            )
        else:
            dtype = torch.float32
            print("Using float32 precision (CPU mode)")

            # Load model for CPU training (no device_map to avoid gradient issues)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            # Explicitly move to CPU
            self.model = self.model.to('cpu')

        # Apply LoRA
        print("Applying LoRA...")
        self.model = get_peft_model(self.model, self.lora_config)

        # Enable gradient checkpointing after LoRA is applied
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()

        # Ensure LoRA parameters require gradients
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        # Print trainable parameters
        self.model.print_trainable_parameters()

        return self.model, self.tokenizer

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = None,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 100,
        logging_steps: int = 10,
        max_seq_length: int = 512
    ):
        """Train LoRA adapter with memory-aware optimization"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Auto-estimate batch size if not provided
        if batch_size is None:
            batch_size = self._estimate_optimal_batch_size(max_seq_length)
            print(f"Auto-estimated batch size: {batch_size}")

        # Determine precision settings
        use_fp16 = self.use_gpu and not torch.cuda.is_bf16_supported()
        use_bf16 = self.use_gpu and torch.cuda.is_bf16_supported()

        # Adjust workers for GPU
        if self.use_gpu:
            # GPU can handle more workers
            num_workers = min(4, os.cpu_count() or 1)
        else:
            num_workers = 0

        # Choose optimizer based on device and bitsandbytes availability
        # Note: bitsandbytes 8-bit optimizer ONLY works on GPU
        if HAS_BITSANDBYTES and self.use_gpu:
            optimizer = "adamw_bnb_8bit"  # 8-bit Adam - uses ~75% less memory (GPU only)
            print("Using 8-bit AdamW optimizer (saves ~75% memory)")
        else:
            optimizer = "adamw_torch"
            if not self.use_gpu:
                print("Using standard AdamW optimizer (8-bit not available on CPU)")
            elif not HAS_BITSANDBYTES:
                print("Using standard AdamW optimizer (install bitsandbytes for 8-bit)")
            else:
                print("Using standard AdamW optimizer")

        # Training arguments optimized for device with aggressive memory saving
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=1,  # Reduced from 3 to save disk space
            remove_unused_columns=False,
            dataloader_num_workers=num_workers,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            report_to="none",  # Disable wandb/tensorboard
            # Memory optimizations
            gradient_checkpointing=True,  # Always enable for memory saving
            optim=optimizer,  # 8-bit or standard optimizer
            max_grad_norm=1.0,
            # CPU-specific memory optimizations
            dataloader_pin_memory=False,  # Don't pin memory on CPU
            torch_compile=False,  # Disable torch compile for memory
            # Save strategy
            save_strategy="epoch"
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        # Train
        print("Starting training...")
        if self.use_gpu:
            print(f"Training on GPU with {'bfloat16' if use_bf16 else 'float16'} precision")
        else:
            print("Training on CPU (this may take a while...)")

        trainer.train()

        # Save LoRA adapter
        adapter_path = self.output_dir / "lora_adapter"
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        print(f"LoRA adapter saved to: {adapter_path}")

        return adapter_path


if __name__ == "__main__":
    # Test trainer
    from dataset_loader import DatasetLoader

    trainer = LoRATrainer(model_path="./models/fp16_model")
    model, tokenizer = trainer.load_model("mistralai/Mistral-7B-v0.1")

    # Load dataset
    loader = DatasetLoader("test.jsonl", tokenizer)
    dataset = loader.load()

    # Train
    trainer.train(dataset, num_epochs=1)

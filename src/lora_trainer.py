"""
CPU/GPU LoRA Trainer with automatic acceleration and memory-aware optimization
"""

import os
import gc
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
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    IA3Config,
    AdaptionPromptConfig,  # LLaMA-specific adapters
    get_peft_model,
    TaskType
)

# Try to import bitsandbytes for 8-bit optimizer
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


class CyberpunkProgressCallback(TrainerCallback):
    """Cyberpunk-themed progress display with throbbers"""

    # Throbber styles
    THROBBERS = {
        "dots": ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        "matrix": ['▖', '▘', '▝', '▗'],
        "pulse": ['◜', '◠', '◝', '◞', '◡', '◟'],
        "arrow": ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
    }

    # Colors
    CYAN = '\033[38;5;51m'
    MATRIX_GREEN = '\033[38;5;46m'
    MATRIX_DIM = '\033[38;5;28m'
    YELLOW = '\033[38;5;226m'
    MAGENTA = '\033[38;5;201m'
    DIM = '\033[2m'
    BOLD = '\033[1m'
    END = '\033[0m'

    def __init__(self, total_steps, use_gpu=False, style="matrix"):
        self.total_steps = total_steps
        self.use_gpu = use_gpu
        self.style = style
        self.throbber = self.THROBBERS.get(style, self.THROBBERS["matrix"])
        self.frame_idx = 0
        self.last_loss = None
        self.last_lr = None

    def on_step_end(self, args, state, control, **kwargs):
        """Update progress with throbber"""
        # Get current metrics
        logs = kwargs.get('logs', {})
        current_step = state.global_step

        # Update throbber frame
        self.frame_idx = (self.frame_idx + 1) % len(self.throbber)
        spinner = self.throbber[self.frame_idx]

        # Calculate progress
        progress = (current_step / self.total_steps) * 100

        # Build progress bar
        bar_width = 40
        filled = int(bar_width * current_step / self.total_steps)
        bar = f"{self.MATRIX_GREEN}{'█' * filled}{self.MATRIX_DIM}{'░' * (bar_width - filled)}{self.END}"

        # Get loss and learning rate if available
        loss_str = f" Loss: {self.YELLOW}{logs.get('loss', self.last_loss or 0):.4f}{self.END}" if 'loss' in logs or self.last_loss else ""
        lr_str = f" LR: {self.MAGENTA}{logs.get('learning_rate', self.last_lr or 0):.2e}{self.END}" if 'learning_rate' in logs or self.last_lr else ""

        # Update last values
        if 'loss' in logs:
            self.last_loss = logs['loss']
        if 'learning_rate' in logs:
            self.last_lr = logs['learning_rate']

        # Print progress line (overwrite previous)
        print(f"\r{self.CYAN}{spinner}{self.END} {bar} {self.BOLD}{progress:5.1f}%{self.END} [{current_step}/{self.total_steps}]{loss_str}{lr_str}", end='', flush=True)

        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Print newline at end of epoch"""
        print()  # New line after epoch
        return control


class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor memory usage and trigger garbage collection"""

    def __init__(self, use_gpu=False, gc_frequency=10):
        self.use_gpu = use_gpu
        self.gc_frequency = gc_frequency
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Monitor memory after each training step"""
        self.step_count += 1

        # Periodic garbage collection
        if self.step_count % self.gc_frequency == 0:
            gc.collect()
            if self.use_gpu:
                torch.cuda.empty_cache()

        # Check memory every 50 steps
        if self.step_count % 50 == 0:
            if self.use_gpu:
                mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"  [GPU Memory] Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
            else:
                mem = psutil.virtual_memory()
                used_gb = (mem.total - mem.available) / (1024**3)
                available_gb = mem.available / (1024**3)
                percent = mem.percent
                print(f"  [CPU Memory] Used: {used_gb:.2f} GB, Available: {available_gb:.2f} GB ({percent}%)")

                # Warning if memory is getting low
                if percent > 85:
                    print(f"  [!] WARNING: Memory usage at {percent}% - may run out soon!")

        return control


class LoRATrainer:
    """PEFT-agnostic fine-tuning with CPU/GPU support (LoRA, Prefix Tuning, Prompt Tuning, IA³)"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./output",
        peft_type: Literal["lora", "prefix", "prompt", "ia3"] = "lora",
        device: Literal["auto", "cpu", "cuda"] = "auto",
        # LoRA-specific params
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        # Prefix tuning params
        num_virtual_tokens: int = 30,
        # Prompt tuning params
        prompt_tuning_init: Literal["TEXT", "RANDOM"] = "RANDOM",
        prompt_tuning_init_text: Optional[str] = None,
        # IA³ params
        ia3_target_modules: Optional[list] = None,
        ia3_feedforward_modules: Optional[list] = None
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store PEFT type
        self.peft_type = peft_type

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

        # Store PEFT config parameters
        self.peft_config_params = {
            # LoRA params
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
            # Prefix tuning params
            "num_virtual_tokens": num_virtual_tokens,
            # Prompt tuning params
            "prompt_tuning_init": prompt_tuning_init,
            "prompt_tuning_init_text": prompt_tuning_init_text,
            # IA³ params
            "ia3_target_modules": ia3_target_modules or ["q_proj", "v_proj", "k_proj"],
            "ia3_feedforward_modules": ia3_feedforward_modules or ["down_proj", "up_proj"]
        }

        # Create PEFT config based on type
        self.peft_config = self._create_peft_config()

        self.model = None
        self.tokenizer = None

    def _create_peft_config(self):
        """Create appropriate PEFT config based on peft_type"""
        print(f"Initializing {self.peft_type.upper()} configuration...")

        if self.peft_type == "lora":
            return LoraConfig(
                r=self.peft_config_params["lora_r"],
                lora_alpha=self.peft_config_params["lora_alpha"],
                lora_dropout=self.peft_config_params["lora_dropout"],
                target_modules=self.peft_config_params["target_modules"],
                task_type=TaskType.CAUSAL_LM,
                bias="none"
            )

        elif self.peft_type == "prefix":
            return PrefixTuningConfig(
                num_virtual_tokens=self.peft_config_params["num_virtual_tokens"],
                task_type=TaskType.CAUSAL_LM
            )

        elif self.peft_type == "prompt":
            config_kwargs = {
                "num_virtual_tokens": self.peft_config_params["num_virtual_tokens"],
                "task_type": TaskType.CAUSAL_LM,
                "prompt_tuning_init": self.peft_config_params["prompt_tuning_init"]
            }
            # Add init text if using TEXT initialization
            if self.peft_config_params["prompt_tuning_init"] == "TEXT":
                if self.peft_config_params["prompt_tuning_init_text"]:
                    config_kwargs["prompt_tuning_init_text"] = self.peft_config_params["prompt_tuning_init_text"]
                else:
                    raise ValueError("prompt_tuning_init_text required when using TEXT initialization")

            return PromptTuningConfig(**config_kwargs)

        elif self.peft_type == "ia3":
            return IA3Config(
                target_modules=self.peft_config_params["ia3_target_modules"],
                feedforward_modules=self.peft_config_params["ia3_feedforward_modules"],
                task_type=TaskType.CAUSAL_LM
            )

        else:
            raise ValueError(f"Unsupported PEFT type: {self.peft_type}. Choose from: lora, prefix, prompt, ia3")

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
        """Load base model and apply PEFT method with memory-aware settings"""
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

        # Apply PEFT method
        print(f"Applying {self.peft_type.upper()}...")
        self.model = get_peft_model(self.model, self.peft_config)

        # Enable gradient checkpointing after PEFT is applied
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()

        # Ensure PEFT parameters require gradients
        # Different PEFT methods use different naming conventions
        peft_param_keywords = ['lora', 'prefix', 'prompt', 'adapter', 'ia3']
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in peft_param_keywords):
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
        """Train PEFT adapter with memory-aware optimization"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Auto-estimate batch size if not provided
        if batch_size is None:
            batch_size = self._estimate_optimal_batch_size(max_seq_length)
            print(f"Auto-estimated batch size: {batch_size}")

        # CPU-specific memory optimizations
        if not self.use_gpu:
            # Aggressive memory saving for CPU
            gradient_accumulation_steps = max(1, gradient_accumulation_steps // 2)  # Reduce accumulation
            print(f"[CPU] Reduced gradient accumulation to {gradient_accumulation_steps} for memory efficiency")

            # Force garbage collection before training
            import gc
            gc.collect()

            # Check available memory
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            print(f"[CPU] Available RAM: {available_gb:.1f} GB")
            if available_gb < 4:
                print(f"[!] WARNING: Low memory ({available_gb:.1f} GB). Training may fail!")
                print(f"[!] Consider: smaller model, batch_size=1, or more RAM")

        # Determine precision settings
        use_fp16 = self.use_gpu and not torch.cuda.is_bf16_supported()
        use_bf16 = self.use_gpu and torch.cuda.is_bf16_supported()

        # Adjust workers for GPU
        if self.use_gpu:
            # GPU can handle more workers
            num_workers = min(4, os.cpu_count() or 1)
        else:
            num_workers = 0  # No workers on CPU to save memory

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
            save_strategy="epoch",
            # Disable tqdm for custom progress display
            disable_tqdm=True,
            logging_strategy="steps"
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Calculate total steps for progress callback
        num_samples = len(dataset)
        steps_per_epoch = num_samples // (batch_size * gradient_accumulation_steps)
        total_steps = steps_per_epoch * num_epochs

        # Cyberpunk progress callback
        progress_callback = CyberpunkProgressCallback(
            total_steps=total_steps,
            use_gpu=self.use_gpu,
            style="matrix"  # Can be: dots, matrix, pulse, arrow
        )

        # Memory monitoring callback
        memory_callback = MemoryMonitorCallback(
            use_gpu=self.use_gpu,
            gc_frequency=5 if not self.use_gpu else 10  # More frequent GC on CPU
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[progress_callback, memory_callback]
        )

        # Train
        print("Starting training...")
        print(f"[i] Total steps: {total_steps} ({steps_per_epoch} steps/epoch × {num_epochs} epochs)")
        if self.use_gpu:
            print(f"[i] Training on GPU with {'bfloat16' if use_bf16 else 'float16'} precision")
        else:
            print("[i] Training on CPU (this may take a while...)")
        print("[i] Progress: Cyberpunk throbber with real-time metrics")
        print("[i] Memory: Monitoring enabled (updates every 50 steps)")
        print()

        trainer.train()

        # Final cleanup
        gc.collect()
        if self.use_gpu:
            torch.cuda.empty_cache()

        # Save PEFT adapter
        adapter_path = self.output_dir / f"{self.peft_type}_adapter"
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        print(f"{self.peft_type.upper()} adapter saved to: {adapter_path}")

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

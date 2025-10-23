#!/usr/bin/env python3
"""
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
██ ██      ██      ▄▄▄   ██      ▄▄▄   ▄▄▄▄▄  ▄▄▄   ▄▄▄▄  ▄▄▄   ▄▄▄▄  ██
██ ██  ▀▀  ██  ██  ██ █  ██  ██  ██ █  ██  ██  ██ █  ██  ▄▄▄▄  ██  ▄▄▄▄  ██
██ ██  ██  ██  ██  ██▀▀  ██      █▀▀█  ██  ██  █▀▀█  █████  █  █████  █  ██
██ ██  ██  ██      ██ █  ██  ██  ██ █  ██  ██  ██ █  ██     █  ██     █  ██
██ ███████████████████ ███████████ █████████████ █████████ ████████ ████ ██
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

    NEURAL NETWORK FINE-TUNING SYSTEM v0.1.0
    [CLASSIFIED] CPU/GPU ACCELERATION ENABLED
    TRAINING-AGNOSTIC ARCHITECTURE | GGUF NATIVE
"""

import sys
from pathlib import Path
from typing import Optional, List
import time
import random
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class CyberpunkColors:
    """Matrix-themed cyberpunk color palette"""
    # Matrix green variants
    MATRIX_GREEN = '\033[38;5;46m'
    MATRIX_DIM = '\033[38;5;28m'
    MATRIX_BRIGHT = '\033[38;5;82m'

    # Cyberpunk accents
    CYAN_BRIGHT = '\033[38;5;51m'
    MAGENTA = '\033[38;5;201m'
    PURPLE = '\033[38;5;129m'
    YELLOW = '\033[38;5;226m'
    ORANGE = '\033[38;5;208m'
    RED = '\033[38;5;196m'

    # Special effects
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'

    # Backgrounds
    BG_BLACK = '\033[40m'
    BG_MATRIX = '\033[48;5;22m'

    # Reset
    END = '\033[0m'


C = CyberpunkColors  # Alias for convenience


def glitch_text(text: str, intensity: int = 1) -> str:
    """Add random glitch characters to text"""
    if random.random() > 0.7:
        glitch_chars = ['█', '▓', '▒', '░', '▀', '▄', '▌', '▐']
        pos = random.randint(0, len(text) - 1)
        char = random.choice(glitch_chars)
        text = text[:pos] + char + text[pos + 1:]
    return text


def print_banner():
    """Print cyberpunk ASCII banner"""
    banner = f"""{C.MATRIX_BRIGHT}{C.BOLD}
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  ██╗     ██╗      █████╗ ███╗   ███╗ █████╗     ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
║  ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
║  ██║     ██║     ███████║██╔████╔██║███████║    █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
║  ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
║  ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║    ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
║  ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝    ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
║                                                                               ║
║  {C.CYAN_BRIGHT}[NEURAL NETWORK FINE-TUNING SYSTEM]{C.MATRIX_BRIGHT}                                     ║
║  {C.MATRIX_DIM}VERSION: 0.1.0-ALPHA | CLEARANCE: LEVEL 5{C.MATRIX_BRIGHT}                               ║
║  {C.MATRIX_DIM}CPU/GPU ACCELERATION | GGUF NATIVE | TRAINING-AGNOSTIC{C.MATRIX_BRIGHT}                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{C.END}
"""
    print(banner)
    time.sleep(0.3)


def print_header(text: str):
    """Print section header with cyberpunk styling"""
    width = 80
    print(f"\n{C.MATRIX_BRIGHT}{C.BOLD}╔{'═' * (width - 2)}╗{C.END}")
    print(f"{C.MATRIX_BRIGHT}{C.BOLD}║{C.CYAN_BRIGHT}{text.center(width - 2)}{C.MATRIX_BRIGHT}║{C.END}")
    print(f"{C.MATRIX_BRIGHT}{C.BOLD}╚{'═' * (width - 2)}╝{C.END}\n")


def print_step(step: int, total: int, text: str):
    """Print progress step with matrix aesthetic"""
    bar = "█" * step + "░" * (total - step)
    print(f"{C.MATRIX_GREEN}{C.BOLD}[{step}/{total}]{C.END} {C.CYAN_BRIGHT}{bar}{C.END} {C.MATRIX_DIM}>{C.END} {C.BOLD}{text}{C.END}")


def print_success(text: str):
    """Print success with glitch effect"""
    print(f"{C.MATRIX_BRIGHT}{C.BOLD}[✓]{C.END} {C.MATRIX_GREEN}{text}{C.END}")


def print_error(text: str):
    """Print error with warning styling"""
    print(f"{C.RED}{C.BOLD}[✗] ERROR:{C.END} {C.ORANGE}{text}{C.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{C.YELLOW}{C.BOLD}[!] WARNING:{C.END} {C.YELLOW}{text}{C.END}")


def print_info(text: str):
    """Print info with cyberpunk styling"""
    print(f"{C.CYAN_BRIGHT}{C.BOLD}[i]{C.END} {C.MATRIX_DIM}{text}{C.END}")


def print_system(text: str):
    """Print system message"""
    print(f"{C.MATRIX_DIM}[SYS] {text}{C.END}")


def animate_loading(text: str, duration: float = 1.0):
    """Animated loading bar"""
    frames = ['▱▱▱▱▱', '▰▱▱▱▱', '▰▰▱▱▱', '▰▰▰▱▱', '▰▰▰▰▱', '▰▰▰▰▰']
    start_time = time.time()

    while time.time() - start_time < duration:
        for frame in frames:
            if time.time() - start_time >= duration:
                break
            print(f"\r{C.MATRIX_GREEN}{frame}{C.END} {C.MATRIX_DIM}{text}{C.END}", end='', flush=True)
            time.sleep(0.1)

    print(f"\r{C.MATRIX_BRIGHT}{'▰' * 5}{C.END} {C.MATRIX_GREEN}{text} {C.BOLD}COMPLETE{C.END}")


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with cyberpunk prompt"""
    if default:
        result = input(f"{C.CYAN_BRIGHT}>{C.END} {C.BOLD}{prompt}{C.END} {C.MATRIX_DIM}[{default}]{C.END}: ").strip()
        return result if result else default
    else:
        result = input(f"{C.CYAN_BRIGHT}>{C.END} {C.BOLD}{prompt}{C.END}: ").strip()
        while not result:
            print_error("This field is required")
            result = input(f"{C.CYAN_BRIGHT}>{C.END} {C.BOLD}{prompt}{C.END}: ").strip()
        return result


def prompt_choice(prompt: str, choices: List[str], default: Optional[int] = None) -> str:
    """Present choices with cyberpunk styling"""
    print(f"\n{C.MAGENTA}{C.BOLD}┌─ {prompt}{C.END}")

    for i, choice in enumerate(choices, 1):
        default_marker = f" {C.MATRIX_BRIGHT}[DEFAULT]{C.END}" if default and i == default else ""
        prefix = "├─" if i < len(choices) else "└─"
        print(f"{C.MAGENTA}{prefix}{C.END} {C.YELLOW}{i}.{C.END} {C.MATRIX_GREEN}{choice}{C.END}{default_marker}")

    while True:
        if default:
            result = input(f"{C.CYAN_BRIGHT}>{C.END} {C.BOLD}Select [1-{len(choices)}]{C.END} {C.MATRIX_DIM}[{default}]{C.END}: ").strip()
            if not result:
                return choices[default - 1]
        else:
            result = input(f"{C.CYAN_BRIGHT}>{C.END} {C.BOLD}Select [1-{len(choices)}]{C.END}: ").strip()

        try:
            idx = int(result) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            else:
                print_error(f"Invalid selection. Choose 1-{len(choices)}")
        except ValueError:
            print_error("Enter a valid number")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Yes/No prompt with cyberpunk styling"""
    default_str = f"{C.MATRIX_BRIGHT}Y{C.END}/{C.MATRIX_DIM}n{C.END}" if default else f"{C.MATRIX_DIM}y{C.END}/{C.MATRIX_BRIGHT}N{C.END}"
    result = input(f"{C.CYAN_BRIGHT}>{C.END} {C.BOLD}{prompt}{C.END} [{default_str}]: ").strip().lower()

    if not result:
        return default

    return result in ['y', 'yes']


def validate_file(path: str) -> bool:
    """Check if file exists"""
    return Path(path).exists()


def print_device_info():
    """Print detected hardware info"""
    import torch

    print_header("HARDWARE DETECTION")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_success(f"GPU DETECTED: {gpu_name}")
        print_info(f"VRAM: {gpu_memory:.1f} GB")
        print_info("Acceleration: CUDA ENABLED")
    else:
        print_warning("No GPU detected")
        print_info("Acceleration: CPU-ONLY MODE")

    # CPU info
    cpu_count = os.cpu_count() or 1
    print_info(f"CPU Cores: {cpu_count}")


def interactive_setup():
    """Interactive configuration wizard with cyberpunk aesthetic"""
    print_banner()
    print_system("INITIALIZING NEURAL FINE-TUNING PROTOCOL...")
    time.sleep(0.5)

    print_device_info()

    print_header("CONFIGURATION WIZARD")
    print(f"{C.MATRIX_DIM}This guided setup will configure your fine-tuning parameters.{C.END}\n")

    config = {}

    # Step 1: Model selection
    print_step(1, 5, "MODEL SELECTION")

    # Check for Ollama models
    from ollama_utils import get_ollama_models, is_ollama_installed
    from ollama_gguf_loader import load_ollama_model_for_training

    if is_ollama_installed():
        ollama_models = get_ollama_models()

        if ollama_models:
            print_success(f"Detected {len(ollama_models)} Ollama models")
            print_info("You can select an Ollama model or enter a custom HuggingFace model")

            # Show all models (paginated display)
            print(f"\n{C.MAGENTA}{C.BOLD}┌─ Ollama Models ({len(ollama_models)} total){C.END}")

            # Display in columns for better readability
            models_per_page = 30
            for i, model in enumerate(ollama_models[:models_per_page], 1):
                print(f"{C.MAGENTA}├─{C.END} {C.YELLOW}{i:2d}.{C.END} {C.MATRIX_GREEN}{model['name']:<30}{C.END} {C.MATRIX_DIM}({model['size']}){C.END}")

            if len(ollama_models) > models_per_page:
                print(f"{C.MAGENTA}├─{C.END} {C.MATRIX_DIM}...and {len(ollama_models) - models_per_page} more (enter 'list' to see all){C.END}")

            print(f"{C.MAGENTA}└─{C.END}")
            print(f"\n{C.MATRIX_DIM}Commands: 'list' = show all models | model number | model name | HF model{C.END}")

            # Ask user choice
            model_input = prompt_input(
                f"Enter model number (1-{min(models_per_page, len(ollama_models))}), command, or model name",
                default="1"
            )

            # Handle 'list' command
            if model_input.lower() == 'list':
                print(f"\n{C.MAGENTA}{C.BOLD}┌─ All Ollama Models{C.END}")
                for i, model in enumerate(ollama_models, 1):
                    print(f"{C.MAGENTA}├─{C.END} {C.YELLOW}{i:2d}.{C.END} {C.MATRIX_GREEN}{model['name']:<35}{C.END} {C.MATRIX_DIM}{model['size']}{C.END}")
                print(f"{C.MAGENTA}└─{C.END}\n")

                model_input = prompt_input(
                    f"Enter model number (1-{len(ollama_models)}) or model name",
                    default="1"
                )

            # Parse input
            try:
                idx = int(model_input) - 1
                if 0 <= idx < len(ollama_models):
                    ollama_name = ollama_models[idx]['name']
                    print_success(f"Selected: {ollama_name}")

                    # Use smart GGUF loader with cache detection
                    print()
                    model_info = load_ollama_model_for_training(ollama_name)

                    if model_info['status'] == 'cached':
                        print_success(f"Base model cached! ({model_info['cache_size']:.2f} GB)")
                        print_info("No download needed - using local cache")
                    elif model_info['status'] == 'needs_download':
                        print_warning("Base model not cached - will download from HuggingFace")
                        print_info(f"This is a one-time download, will be cached for future use")

                    config['model'] = model_info['hf_model']
                    config['ollama_model'] = ollama_name
                    config['model_info'] = model_info
                else:
                    config['model'] = model_input
            except ValueError:
                # Not a number, treat as model name
                if ':' in model_input:  # Looks like Ollama format
                    print()
                    model_info = load_ollama_model_for_training(model_input)

                    if model_info['status'] == 'cached':
                        print_success(f"Base model cached! ({model_info['cache_size']:.2f} GB)")
                        print_info("No download needed - using local cache")
                    elif model_info['status'] == 'needs_download':
                        print_warning("Base model not cached - will download from HuggingFace")
                        print_info(f"This is a one-time download, will be cached for future use")

                    config['model'] = model_info['hf_model']
                    config['ollama_model'] = model_input
                    config['model_info'] = model_info
                else:
                    config['model'] = model_input

            animate_loading("Validating model configuration", duration=0.5)
        else:
            print_warning("No Ollama models found")
            print_info("Supported: HuggingFace models")
            config['model'] = prompt_input("HuggingFace model name", default="mistralai/Mistral-7B-v0.1")
            animate_loading("Validating model path", duration=0.5)
    else:
        print_warning("Ollama not detected")
        print_info("Supported: HuggingFace models")
        config['model'] = prompt_input("HuggingFace model name", default="mistralai/Mistral-7B-v0.1")
        animate_loading("Validating model path", duration=0.5)

    # Step 2: Dataset
    print_step(2, 5, "DATASET CONFIGURATION")
    print_info("Supported formats: JSON, JSONL, CSV, TXT")
    print(f"{C.MATRIX_DIM}Auto-structuring enabled for all formats{C.END}")
    print(f"{C.MATRIX_DIM}Tip: You can paste a directory path to browse files{C.END}")

    while True:
        user_input = prompt_input("Dataset file path or directory")

        # Expand ~ and resolve path
        input_path = Path(user_input).expanduser().resolve()

        # Check if it's a directory
        if input_path.is_dir():
            print_success(f"Directory found: {input_path}")

            # Find all dataset files
            dataset_files = []
            for ext in ['*.json', '*.jsonl', '*.csv', '*.txt']:
                dataset_files.extend(input_path.glob(ext))

            if not dataset_files:
                print_error("No dataset files found in directory")
                continue

            # Sort by name
            dataset_files = sorted(dataset_files, key=lambda p: p.name)

            # Show files with preview
            print(f"\n{C.MAGENTA}{C.BOLD}┌─ Available Datasets ({len(dataset_files)} found){C.END}")
            for i, file in enumerate(dataset_files[:20], 1):  # Show max 20
                size = file.stat().st_size / 1024  # KB
                if size < 1024:
                    size_str = f"{size:.1f} KB"
                else:
                    size_str = f"{size/1024:.1f} MB"

                prefix = "├─" if i < min(len(dataset_files), 20) else "└─"
                print(f"{C.MAGENTA}{prefix}{C.END} {C.YELLOW}{i}.{C.END} {C.MATRIX_GREEN}{file.name}{C.END} {C.MATRIX_DIM}({size_str}){C.END}")

            if len(dataset_files) > 20:
                print(f"{C.MAGENTA}└─{C.END} {C.MATRIX_DIM}...and {len(dataset_files) - 20} more{C.END}")

            # Let user select
            file_choice = prompt_input(
                f"Select dataset number (1-{min(len(dataset_files), 20)})",
                default="1"
            )

            try:
                idx = int(file_choice) - 1
                if 0 <= idx < len(dataset_files):
                    config['data'] = str(dataset_files[idx])
                    animate_loading("Loading dataset", duration=0.5)
                    ext = dataset_files[idx].suffix.upper()
                    print_success(f"Dataset selected: {dataset_files[idx].name} ({ext} format)")
                    break
                else:
                    print_error(f"Invalid selection. Choose 1-{len(dataset_files)}")
            except ValueError:
                print_error("Please enter a valid number")

        # Check if it's a file
        elif input_path.is_file():
            config['data'] = str(input_path)
            animate_loading("Loading dataset", duration=0.5)
            ext = input_path.suffix.upper()
            print_success(f"Dataset validated: {ext} format detected")
            break

        else:
            print_error(f"File or directory not found: {user_input}")
            print_info("Try: testtraindata/ or examples/datasets/")

    # Step 3: Training parameters
    print_step(3, 5, "TRAINING PARAMETERS")

    # Device selection
    import torch
    if torch.cuda.is_available():
        use_gpu = prompt_yes_no("Enable GPU acceleration?", default=True)
        config['device'] = "cuda" if use_gpu else "cpu"

        if use_gpu:
            print_success("GPU acceleration ENABLED")
        else:
            print_warning("Training on CPU (slower)")
    else:
        config['device'] = "cpu"
        print_warning("No GPU available - using CPU")

    # Epochs
    epochs_choices = [
        "1 epoch (Quick test)",
        "3 epochs (Recommended)",
        "5 epochs (Thorough training)",
        "Custom"
    ]
    epochs_choice = prompt_choice("Training duration", epochs_choices, default=2)

    if "Custom" in epochs_choice:
        config['epochs'] = int(prompt_input("Number of epochs", default="3"))
    else:
        config['epochs'] = int(epochs_choice.split()[0])

    # Batch size
    if config['device'] == "cuda":
        batch_choices = [
            "2 (Conservative VRAM)",
            "4 (Balanced)",
            "8 (High VRAM)",
            "16 (Maximum speed)"
        ]
        default_batch = 2
    else:
        batch_choices = [
            "1 (Low RAM)",
            "2 (Medium RAM)",
            "4 (High RAM)"
        ]
        default_batch = 1

    batch_choice = prompt_choice("Batch size", batch_choices, default=default_batch)
    config['batch_size'] = int(batch_choice.split()[0])

    # Learning rate
    lr_choices = [
        "1e-4 (Conservative)",
        "2e-4 (Recommended)",
        "5e-4 (Aggressive)",
        "1e-3 (Very aggressive)"
    ]
    lr_choice = prompt_choice("Learning rate", lr_choices, default=2)
    config['learning_rate'] = float(lr_choice.split()[0])

    # Max length
    length_choices = [
        "256 tokens (Short texts)",
        "512 tokens (Recommended)",
        "1024 tokens (Long context)",
        "2048 tokens (Very long context)"
    ]
    length_choice = prompt_choice("Maximum sequence length", length_choices, default=2)
    config['max_length'] = int(length_choice.split()[0])

    # Step 4: LoRA configuration
    print_step(4, 5, "LoRA ADAPTER CONFIGURATION")
    print_info("LoRA = efficient fine-tuning (only trains 0.2% of model)")

    if prompt_yes_no("Configure advanced LoRA settings?", default=False):
        print()
        print(f"{C.YELLOW}LoRA Rank{C.END} - How much the model learns:")
        print(f"  {C.MATRIX_DIM}• Higher rank (16-32) = Better quality, MORE RAM{C.END}")
        print(f"  {C.MATRIX_DIM}• Lower rank (4-8) = Faster training, LESS RAM{C.END}")
        print(f"  {C.MATRIX_GREEN}• Default (8) = Good for most tasks{C.END}")
        print()

        rank_choices = [
            "4 (Minimal RAM, fastest)",
            "8 (Recommended - balanced)",
            "16 (Better quality, more RAM)",
            "32 (Best quality, lots of RAM)"
        ]
        rank_choice = prompt_choice("LoRA rank", rank_choices, default=2)
        config['lora_r'] = int(rank_choice.split()[0])

        # Alpha is usually 2x rank
        config['lora_alpha'] = config['lora_r'] * 2
        print_system(f"Auto-set alpha to {config['lora_alpha']} (2x rank)")

        # Dropout - keep it simple
        config['lora_dropout'] = 0.05
        print_system("Using dropout=0.05 (standard)")

    else:
        config['lora_r'] = 8
        config['lora_alpha'] = 16
        config['lora_dropout'] = 0.05
        print_system("Using recommended defaults (rank=8, alpha=16, dropout=0.05)")
        print_info("These work great for most tasks - no need to change!")

    # Step 5: Output configuration
    print_step(5, 5, "OUTPUT CONFIGURATION")

    config['output'] = prompt_input("Output file path", default="./finetuned-model.gguf")

    # GGUF quantization
    if prompt_yes_no("Convert to GGUF format?", default=True):
        quant_choices = [
            "q4_k_m (Recommended - 4-bit, balanced)",
            "q4_0 (Smallest size)",
            "q5_k_m (Higher quality)",
            "q8_0 (Highest quality, larger)"
        ]
        quant_choice = prompt_choice("Quantization method", quant_choices, default=1)
        config['quantization'] = quant_choice.split()[0]
        config['no_gguf'] = False
    else:
        config['no_gguf'] = True
        print_info("Model will be saved in HuggingFace format")

    # Configuration summary
    print_header("CONFIGURATION SUMMARY")

    print(f"{C.MATRIX_BRIGHT}{C.BOLD}┌─ SYSTEM CONFIGURATION{C.END}")
    print(f"{C.MATRIX_BRIGHT}├─{C.END} {C.CYAN_BRIGHT}Model:{C.END} {config['model']}")
    print(f"{C.MATRIX_BRIGHT}├─{C.END} {C.CYAN_BRIGHT}Dataset:{C.END} {config['data']}")
    print(f"{C.MATRIX_BRIGHT}├─{C.END} {C.CYAN_BRIGHT}Output:{C.END} {config['output']}")
    print(f"{C.MATRIX_BRIGHT}├─{C.END} {C.CYAN_BRIGHT}Device:{C.END} {config['device'].upper()}")
    print(f"{C.MATRIX_BRIGHT}│{C.END}")
    print(f"{C.MATRIX_BRIGHT}├─ TRAINING PARAMETERS{C.END}")
    print(f"{C.MATRIX_BRIGHT}│  ├─{C.END} Epochs: {config['epochs']}")
    print(f"{C.MATRIX_BRIGHT}│  ├─{C.END} Batch size: {config['batch_size']}")
    print(f"{C.MATRIX_BRIGHT}│  ├─{C.END} Learning rate: {config['learning_rate']}")
    print(f"{C.MATRIX_BRIGHT}│  └─{C.END} Max length: {config['max_length']}")
    print(f"{C.MATRIX_BRIGHT}│{C.END}")
    print(f"{C.MATRIX_BRIGHT}├─ LoRA CONFIGURATION{C.END}")
    print(f"{C.MATRIX_BRIGHT}│  ├─{C.END} Rank: {config['lora_r']}")
    print(f"{C.MATRIX_BRIGHT}│  ├─{C.END} Alpha: {config['lora_alpha']}")
    print(f"{C.MATRIX_BRIGHT}│  └─{C.END} Dropout: {config['lora_dropout']}")

    if not config['no_gguf']:
        print(f"{C.MATRIX_BRIGHT}│{C.END}")
        print(f"{C.MATRIX_BRIGHT}└─ GGUF EXPORT{C.END}")
        print(f"   └─ Quantization: {config['quantization']}")
    else:
        print(f"{C.MATRIX_BRIGHT}└─{C.END} {C.YELLOW}GGUF export: DISABLED{C.END}")

    print()

    if not prompt_yes_no("Initialize training sequence?", default=True):
        print_warning("Training sequence aborted")
        return None

    return config


def run_training(config: dict):
    """Execute training with cyberpunk progress display"""
    from dataset_loader import DatasetLoader
    from lora_trainer import LoRATrainer
    from gguf_converter import GGUFConverter
    from transformers import AutoTokenizer

    work_dir = Path("./work")
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load tokenizer
        print_header("PHASE 1: TOKENIZER INITIALIZATION")
        animate_loading("Loading tokenizer module", duration=0.8)

        # Load tokenizer with trust_remote_code for known models
        tokenizer = AutoTokenizer.from_pretrained(
            config['model'],
            trust_remote_code=True  # Safe for well-known HF models
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print_success("Tokenizer loaded successfully")

        # Step 2: Load dataset
        print_header("PHASE 2: DATASET PROCESSING")
        animate_loading("Parsing dataset structure", duration=0.8)

        loader = DatasetLoader(
            file_path=config['data'],
            tokenizer=tokenizer,
            max_length=config['max_length']
        )
        dataset = loader.load()

        print_success(f"Dataset processed: {len(dataset)} samples")

        # Step 3: Train LoRA
        print_header("PHASE 3: NEURAL ADAPTATION")
        print_warning("This process may take significant time...")

        trainer = LoRATrainer(
            model_path=str(work_dir / "model"),
            output_dir=str(work_dir / "training"),
            lora_r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            device=config['device']
        )

        model, tokenizer = trainer.load_model(config['model'])

        print_system("Initiating gradient descent protocol...")
        adapter_path = trainer.train(
            dataset=dataset,
            num_epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            gradient_accumulation_steps=4
        )

        print_success(f"LoRA adapter trained successfully")

        # Step 4: Merge and convert
        print_header("PHASE 4: MODEL SYNTHESIS")

        converter = GGUFConverter(
            base_model_name=config['model'],
            adapter_path=str(adapter_path),
            output_dir=str(work_dir / "output")
        )

        animate_loading("Merging neural pathways", duration=1.0)

        if config['no_gguf']:
            final_path = converter.merge_lora()
            print_success(f"Model saved (HuggingFace format)")
        else:
            final_path = converter.full_conversion(quantization=config['quantization'])

            # Move to final output
            output_path = Path(config['output'])
            if final_path != output_path:
                import shutil
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(final_path, output_path)
                final_path = output_path

            print_success(f"GGUF model compiled successfully")

        # Success banner
        print()
        print(f"{C.MATRIX_BRIGHT}{C.BOLD}{'═' * 80}{C.END}")
        print(f"{C.MATRIX_BRIGHT}{C.BOLD}║{C.CYAN_BRIGHT}{'TRAINING SEQUENCE COMPLETE'.center(78)}{C.MATRIX_BRIGHT}║{C.END}")
        print(f"{C.MATRIX_BRIGHT}{C.BOLD}{'═' * 80}{C.END}")
        print()
        print(f"{C.MATRIX_GREEN}{C.BOLD}[✓] MODEL READY FOR DEPLOYMENT{C.END}")
        print(f"{C.MATRIX_DIM}    Location: {final_path}{C.END}")

        if not config['no_gguf']:
            print()
            print(f"{C.CYAN_BRIGHT}{C.BOLD}▶ OLLAMA INTEGRATION:{C.END}")
            print(f"{C.MATRIX_DIM}  1. Create Modelfile:{C.END}")
            print(f"     {C.YELLOW}echo 'FROM {final_path}' > Modelfile{C.END}")
            print(f"{C.MATRIX_DIM}  2. Import to Ollama:{C.END}")
            print(f"     {C.YELLOW}ollama create my-model -f Modelfile{C.END}")
            print(f"{C.MATRIX_DIM}  3. Execute:{C.END}")
            print(f"     {C.YELLOW}ollama run my-model{C.END}")

        print()
        print_system("Neural forge shutdown complete")

        return 0

    except KeyboardInterrupt:
        print()
        print_error("Training interrupted by operator")
        print_system("Emergency shutdown initiated")
        return 1
    except Exception as e:
        print()
        print_error(f"Critical failure: {e}")
        print_system("Dumping stack trace...")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point for cyberpunk CLI"""
    try:
        # Run interactive setup
        config = interactive_setup()

        if config is None:
            print_system("Session terminated")
            return 0

        # Run training
        return run_training(config)

    except KeyboardInterrupt:
        print()
        print_warning("Operation cancelled by operator")
        print_system("Exiting...")
        return 1
    except Exception as e:
        print()
        print_error(f"System failure: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

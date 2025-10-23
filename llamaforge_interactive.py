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


def show_throbber(text: str, duration: float = 1.0, style: str = "dots"):
    """
    Animated throbber/spinner for waiting operations

    Styles:
    - dots: Bouncing dots ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏
    - spinner: Classic spinner |/-\\
    - arrow: Rotating arrow ←↖↑↗→↘↓↙
    - pulse: Pulsing circle ◜◠◝◞◡◟
    - matrix: Matrix-style ▖▘▝▗
    """
    styles = {
        "dots": ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        "spinner": ['|', '/', '-', '\\'],
        "arrow": ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
        "pulse": ['◜', '◠', '◝', '◞', '◡', '◟'],
        "matrix": ['▖', '▘', '▝', '▗'],
        "dots3": ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'],
        "line": ['⎺', '⎻', '⎼', '⎽', '⎼', '⎻'],
        "box": ['◰', '◳', '◲', '◱'],
        "circle": ['◐', '◓', '◑', '◒'],
        "square": ['◢', '◣', '◤', '◥'],
    }

    frames = styles.get(style, styles["dots"])
    start_time = time.time()
    frame_idx = 0

    while time.time() - start_time < duration:
        frame = frames[frame_idx % len(frames)]
        elapsed = time.time() - start_time
        print(f"\r{C.CYAN_BRIGHT}{frame}{C.END} {C.MATRIX_DIM}{text}{C.END}", end='', flush=True)
        time.sleep(0.08)
        frame_idx += 1

    print(f"\r{C.MATRIX_GREEN}✓{C.END} {C.MATRIX_DIM}{text}{C.END}")


class Throbber:
    """Context manager for long-running operations with throbber"""

    def __init__(self, text: str, style: str = "dots"):
        self.text = text
        self.style = style
        self.running = False
        self.thread = None

    def __enter__(self):
        import threading
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        print(f"\r{C.MATRIX_GREEN}✓{C.END} {C.MATRIX_DIM}{self.text}{C.END}" + " " * 20)

    def _animate(self):
        styles = {
            "dots": ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
            "spinner": ['|', '/', '-', '\\'],
            "arrow": ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
            "pulse": ['◜', '◠', '◝', '◞', '◡', '◟'],
            "matrix": ['▖', '▘', '▝', '▗'],
        }
        frames = styles.get(self.style, styles["dots"])
        idx = 0

        while self.running:
            frame = frames[idx % len(frames)]
            print(f"\r{C.CYAN_BRIGHT}{frame}{C.END} {C.MATRIX_DIM}{self.text}{C.END}", end='', flush=True)
            time.sleep(0.08)
            idx += 1


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
    from ollama_utils import (
        get_ollama_models,
        is_ollama_installed,
        can_train_model,
        get_available_ram,
        ollama_model_exists
    )
    from ollama_gguf_loader import load_ollama_model_for_training

    if is_ollama_installed():
        ollama_models = get_ollama_models()

        if ollama_models:
            available_ram = get_available_ram()
            print_success(f"Detected {len(ollama_models)} Ollama models")
            print_info(f"System RAM: {available_ram:.1f} GB available")
            print_info("You can select an Ollama model or enter a custom HuggingFace model")

            # Show all models (paginated display with RAM estimates)
            print(f"\n{C.MAGENTA}{C.BOLD}┌─ Ollama Models ({len(ollama_models)} total){C.END}")
            print(f"{C.MAGENTA}├─{C.END} {C.MATRIX_DIM}Legend: {C.MATRIX_GREEN}✓ Trainable{C.END} | {C.YELLOW}⚠ Tight{C.END} | {C.RED}✗ Too Large{C.END}")

            # Display models with RAM estimates
            models_per_page = 30
            for i, model in enumerate(ollama_models[:models_per_page], 1):
                can_train, ram_needed, _ = can_train_model(model['name'])

                # Color code based on RAM requirements
                if ram_needed < available_ram * 0.5:
                    status = f"{C.MATRIX_GREEN}✓{C.END}"
                    ram_color = C.MATRIX_GREEN
                elif ram_needed < available_ram * 0.8:
                    status = f"{C.YELLOW}⚠{C.END}"
                    ram_color = C.YELLOW
                else:
                    status = f"{C.RED}✗{C.END}"
                    ram_color = C.RED

                print(f"{C.MAGENTA}├─{C.END} {status} {C.YELLOW}{i:2d}.{C.END} {C.MATRIX_GREEN}{model['name']:<30}{C.END} "
                      f"{C.MATRIX_DIM}({model['size']:>7}){C.END} {ram_color}~{ram_needed:>4.1f}GB{C.END}")

            if len(ollama_models) > models_per_page:
                print(f"{C.MAGENTA}├─{C.END} {C.MATRIX_DIM}...and {len(ollama_models) - models_per_page} more (enter 'list' to see all){C.END}")

            print(f"{C.MAGENTA}└─{C.END}")
            print(f"\n{C.MATRIX_DIM}Commands: 'help' or 'commands' for help | 'list' to see all models{C.END}")

            # Ask user choice
            model_input = prompt_input(
                f"Enter model number (1-{min(models_per_page, len(ollama_models))}), command, or model name",
                default="1"
            )

            # Handle 'help' or 'commands' command
            if model_input.lower() in ['help', 'commands', '?']:
                print(f"\n{C.CYAN_BRIGHT}{C.BOLD}╔═══════════════════════════════════════════════════════════════╗{C.END}")
                print(f"{C.CYAN_BRIGHT}{C.BOLD}║{C.END}                   {C.CYAN_BRIGHT}{C.BOLD}AVAILABLE COMMANDS{C.END}                         {C.CYAN_BRIGHT}{C.BOLD}║{C.END}")
                print(f"{C.CYAN_BRIGHT}{C.BOLD}╚═══════════════════════════════════════════════════════════════╝{C.END}\n")

                print(f"{C.MATRIX_BRIGHT}{C.BOLD}Model Selection:{C.END}")
                print(f"  {C.YELLOW}1-{len(ollama_models)}{C.END}           Select model by number")
                print(f"  {C.YELLOW}<model-name>{C.END}     Enter Ollama model name (e.g., qwen2.5:0.5b)")
                print(f"  {C.YELLOW}<hf-model>{C.END}       Enter HuggingFace model (e.g., mistralai/Mistral-7B-v0.1)")
                print()
                print(f"{C.MATRIX_BRIGHT}{C.BOLD}Commands:{C.END}")
                print(f"  {C.YELLOW}list{C.END}            Show all {len(ollama_models)} Ollama models with RAM estimates")
                print(f"  {C.YELLOW}ollama list{C.END}     Same as 'list'")
                print(f"  {C.YELLOW}help{C.END}            Show this help message")
                print(f"  {C.YELLOW}commands{C.END}        Same as 'help'")
                print(f"  {C.YELLOW}?{C.END}               Same as 'help'")
                print()
                print(f"{C.MATRIX_BRIGHT}{C.BOLD}Legend:{C.END}")
                print(f"  {C.MATRIX_GREEN}✓{C.END}  Trainable   - Model fits comfortably in RAM (< 50% usage)")
                print(f"  {C.YELLOW}⚠{C.END}  Tight       - Model uses 50-80% of available RAM")
                print(f"  {C.RED}✗{C.END}  Too Large   - Model exceeds 80% of RAM (likely to fail)")
                print()
                print(f"{C.MATRIX_DIM}Tip: Use 'list' to see all models, then select by number{C.END}\n")

                # Re-prompt
                model_input = prompt_input(
                    f"Enter model number (1-{min(models_per_page, len(ollama_models))}), command, or model name",
                    default="1"
                )

            # Handle 'list' or 'ollama list' command
            if model_input.lower() in ['list', 'ollama list']:
                print(f"\n{C.MAGENTA}{C.BOLD}┌─ All Ollama Models{C.END}")
                print(f"{C.MAGENTA}├─{C.END} {C.MATRIX_DIM}Legend: {C.MATRIX_GREEN}✓ Trainable{C.END} | {C.YELLOW}⚠ Tight{C.END} | {C.RED}✗ Too Large{C.END}")
                for i, model in enumerate(ollama_models, 1):
                    can_train, ram_needed, _ = can_train_model(model['name'])

                    # Color code based on RAM requirements
                    if ram_needed < available_ram * 0.5:
                        status = f"{C.MATRIX_GREEN}✓{C.END}"
                        ram_color = C.MATRIX_GREEN
                    elif ram_needed < available_ram * 0.8:
                        status = f"{C.YELLOW}⚠{C.END}"
                        ram_color = C.YELLOW
                    else:
                        status = f"{C.RED}✗{C.END}"
                        ram_color = C.RED

                    print(f"{C.MAGENTA}├─{C.END} {status} {C.YELLOW}{i:2d}.{C.END} {C.MATRIX_GREEN}{model['name']:<30}{C.END} "
                          f"{C.MATRIX_DIM}({model['size']:>7}){C.END} {ram_color}~{ram_needed:>4.1f}GB{C.END}")
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

                    # Check RAM requirements
                    can_train, ram_needed, ram_available = can_train_model(ollama_name)
                    print_info(f"Estimated RAM needed: ~{ram_needed:.1f} GB (you have {ram_available:.1f} GB available)")

                    if not can_train:
                        print_warning(f"WARNING: This model may be too large for your system!")
                        print_warning(f"Training may fail due to insufficient memory (OOM kill)")
                        print_info("Consider using a smaller model or cloud GPU training")
                        confirm = prompt_input("Continue anyway? (yes/no)", default="no")
                        if confirm.lower() not in ['yes', 'y']:
                            print_error("Aborted by user")
                            return None

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
                    # Invalid model number
                    print_error(f"Invalid model number: {idx + 1}. Must be between 1 and {len(ollama_models)}")
                    print_info("Please enter a valid model number, 'list' to see all, or a HuggingFace model name")
                    return None
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

            show_throbber("Validating model configuration", duration=0.5, style="dots")
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

    # Ollama model naming (ask early for better UX)
    print()
    print(f"{C.CYAN_BRIGHT}{C.BOLD}▶ OLLAMA MODEL NAME{C.END}")
    print_info("Choose a name for your fine-tuned model in Ollama")

    # Extract base name for suggestion
    if 'ollama_model' in config:
        # User selected an Ollama model
        suggested_name = config['ollama_model'].split(':')[0] + "-finetuned"
    else:
        # User entered HuggingFace model
        base_name = config['model'].split('/')[-1].lower()
        # Clean up the name
        base_name = base_name.replace('-hf', '').replace('-v0.1', '').replace('-instruct', '')
        suggested_name = base_name + "-finetuned"

    # Loop until we get a valid, non-duplicate name or user skips
    while True:
        config['ollama_model_name'] = prompt_input(
            "Ollama model name (or 'skip' for manual setup)",
            default=suggested_name
        )

        if config['ollama_model_name'].lower() == 'skip':
            config['ollama_model_name'] = None
            print_info("Ollama model will require manual setup after training")
            break

        # Check if model already exists
        if is_ollama_installed() and ollama_model_exists(config['ollama_model_name']):
            print_warning(f"Model '{config['ollama_model_name']}' already exists in Ollama!")

            choice = prompt_choice(
                "What would you like to do?",
                [
                    "Overwrite existing model",
                    "Choose a different name",
                    "Skip Ollama creation (manual setup)"
                ],
                default=2
            )

            if "Overwrite" in choice:
                print_info(f"Will overwrite existing model: {config['ollama_model_name']}")
                break
            elif "different name" in choice:
                # Suggest a name with a counter
                suggested_name = config['ollama_model_name'] + "-v2"
                continue  # Ask for name again
            else:  # Skip
                config['ollama_model_name'] = None
                print_info("Ollama model will require manual setup after training")
                break
        else:
            # Name is valid and doesn't exist
            print_success(f"Will create: {config['ollama_model_name']}")
            break

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
                    show_throbber("Loading dataset", duration=0.5, style="pulse")
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
            show_throbber("Loading dataset", duration=0.5, style="pulse")
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

        # Confirm Ollama model creation
        if config.get('ollama_model_name'):
            print_info(f"Will create Ollama model: {config['ollama_model_name']}")
    else:
        config['no_gguf'] = True
        # Clear Ollama name if not creating GGUF
        config['ollama_model_name'] = None
        print_info("Model will be saved in HuggingFace format (no Ollama integration)")

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
        print(f"   ├─ Quantization: {config['quantization']}")
        if config.get('ollama_model_name'):
            print(f"   └─ Ollama model: {config['ollama_model_name']}")
        else:
            print(f"   └─ {C.MATRIX_DIM}Ollama: manual setup{C.END}")
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
        show_throbber("Loading tokenizer module", duration=0.8, style="spinner")

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

            # Auto-create Ollama model if requested
            if config.get('ollama_model_name'):
                print()
                print_header("PHASE 5: OLLAMA INTEGRATION")

                ollama_name = config['ollama_model_name']
                base_model = config.get('model', '')

                # Determine chat template based on model
                if 'qwen' in base_model.lower():
                    template = '''<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
'''
                    stops = ['<|im_start|>', '<|im_end|>']
                elif 'llama' in base_model.lower():
                    template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
                    stops = ['<|eot_id|>']
                elif 'gemma' in base_model.lower():
                    template = '''<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
'''
                    stops = ['<end_of_turn>']
                else:
                    # Default ChatML format
                    template = '''<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
'''
                    stops = ['<|im_start|>', '<|im_end|>']

                # Create Modelfile
                modelfile_path = final_path.parent / "Modelfile"
                modelfile_content = f'''FROM {final_path}

# Model card
TEMPLATE """{template}"""

# Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
'''

                # Add stop tokens
                for stop in stops:
                    modelfile_content += f'PARAMETER stop "{stop}"\n'

                modelfile_content += '\n# System message\n'
                modelfile_content += 'SYSTEM """You are a helpful AI assistant trained on practical coding tasks."""\n'

                # Write Modelfile
                with open(modelfile_path, 'w') as f:
                    f.write(modelfile_content)

                print_info(f"Created Modelfile: {modelfile_path}")

                # Create Ollama model
                import subprocess
                try:
                    animate_loading("Importing to Ollama", duration=1.0)
                    result = subprocess.run(
                        ["ollama", "create", ollama_name, "-f", str(modelfile_path)],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    if result.returncode == 0:
                        print_success(f"Ollama model '{ollama_name}' created successfully!")
                    else:
                        print_error(f"Failed to create Ollama model: {result.stderr}")
                        print_warning("You can manually create it using:")
                        print(f"  {C.YELLOW}ollama create {ollama_name} -f {modelfile_path}{C.END}")
                except subprocess.TimeoutExpired:
                    print_error("Ollama creation timed out")
                except FileNotFoundError:
                    print_error("Ollama not found. Install it first: https://ollama.com")
                except Exception as e:
                    print_error(f"Error creating Ollama model: {e}")

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
            if config.get('ollama_model_name'):
                print(f"{C.CYAN_BRIGHT}{C.BOLD}▶ RUN YOUR MODEL:{C.END}")
                print(f"     {C.YELLOW}ollama run {config['ollama_model_name']}{C.END}")
            else:
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

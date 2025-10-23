"""
Ollama Integration Utilities
"""

import subprocess
import json
import psutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def get_ollama_models() -> List[Dict[str, str]]:
    """Get list of models available in Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return []

        models = []
        lines = result.stdout.strip().split('\n')

        # Skip header line
        for line in lines[1:]:
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                # Parse size if available
                size = parts[2] if len(parts) > 2 else "unknown"
                models.append({
                    "name": name,
                    "size": size,
                    "full_line": line.strip()
                })

        return models

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def ollama_model_exists(model_name: str) -> bool:
    """Check if an Ollama model exists"""
    try:
        # Try to show the model - if it exists, returncode is 0
        result = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True,
            timeout=3
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_ollama_installed() -> bool:
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_ollama_model_info(model_name: str) -> Optional[Dict]:
    """Get detailed info about an Ollama model"""
    try:
        result = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            return {"raw_info": result.stdout}
        return None

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_ollama_model_base(ollama_name: str) -> str:
    """
    Get the base HuggingFace model that an Ollama model is based on.

    This is ONLY used as a fallback if we can't extract the GGUF.
    The preferred method is to use the actual Ollama GGUF file.
    """
    import re

    # Extract base name and size from model name (e.g., "qwen2.5:0.5b" -> "qwen2.5", "0.5b")
    parts = ollama_name.lower().split(':')
    base_name = parts[0]
    size_tag = parts[1] if len(parts) > 1 else None

    # Extract parameter size if available (e.g., "0.5b" -> 0.5, "7b" -> 7)
    param_size = None
    if size_tag:
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*b', size_tag)
        if size_match:
            param_size = float(size_match.group(1))

    # Size-specific mappings (check these first)
    size_specific_mappings = {
        ('qwen2.5', 0.5): 'Qwen/Qwen2.5-0.5B',
        ('qwen2.5', 1.5): 'Qwen/Qwen2.5-1.5B',
        ('qwen2.5', 3): 'Qwen/Qwen2.5-3B',
        ('qwen2.5', 7): 'Qwen/Qwen2.5-7B',
        ('qwen2.5', 14): 'Qwen/Qwen2.5-14B',
        ('qwen2.5', 32): 'Qwen/Qwen2.5-32B',
        ('qwen2', 0.5): 'Qwen/Qwen2-0.5B',
        ('qwen2', 1.5): 'Qwen/Qwen2-1.5B',
        ('qwen2', 7): 'Qwen/Qwen2-7B',
        ('gemma', 2): 'google/gemma-2b',
        ('gemma', 7): 'google/gemma-7b',
        ('gemma2', 2): 'google/gemma-2-2b',
        ('gemma2', 9): 'google/gemma-2-9b',
        ('llama3.1', 8): 'meta-llama/Llama-3.1-8B',
        ('llama3.2', 1): 'meta-llama/Llama-3.2-1B',
        ('llama3.2', 3): 'meta-llama/Llama-3.2-3B',
    }

    # Check size-specific mapping
    if param_size is not None:
        key = (base_name, param_size)
        if key in size_specific_mappings:
            return size_specific_mappings[key]

    # Fallback to base model mappings (default sizes)
    base_mappings = {
        "mistral": "mistralai/Mistral-7B-v0.1",
        "llama2": "meta-llama/Llama-2-7b-hf",
        "llama3.1": "meta-llama/Llama-3.1-8B",
        "llama3.2": "meta-llama/Llama-3.2-3B",
        "llama3": "meta-llama/Meta-Llama-3-8B",
        "codellama": "codellama/CodeLlama-7b-hf",
        "phi": "microsoft/phi-2",
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "phi4": "microsoft/phi-4",
        "gemma": "google/gemma-7b",
        "gemma2": "google/gemma-2-9b",
        "qwen2.5": "Qwen/Qwen2.5-7B",
        "qwen2": "Qwen/Qwen2-7B",
        "qwen": "Qwen/Qwen-7B",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "granite": "ibm-granite/granite-8b-code-base",
    }

    # Check for exact base match
    if base_name in base_mappings:
        return base_mappings[base_name]

    # Check for partial matches
    for key, hf_name in base_mappings.items():
        if key in base_name:
            return base_mappings[key]

    # Default fallback
    return "mistralai/Mistral-7B-v0.1"


def ollama_model_to_hf_name(ollama_name: str) -> str:
    """Deprecated - use get_ollama_model_base instead"""
    return get_ollama_model_base(ollama_name)


def estimate_model_params(model_name: str) -> float:
    """
    Estimate model parameters (in billions) from model name.
    Returns: parameter count in billions
    """
    name_lower = model_name.lower()

    # Extract number from model name (e.g., "llama3.1:8b" -> 8)
    import re

    # Look for patterns like "8b", "7b", "1.1b", "0.5b"
    param_match = re.search(r'(\d+(?:\.\d+)?)\s*b', name_lower)
    if param_match:
        return float(param_match.group(1))

    # Known model sizes
    size_map = {
        'tinyllama': 1.1,
        'phi-2': 2.7,
        'phi3:mini': 3.8,
        'gemma2:2b': 2.0,
        'gemma:2b': 2.0,
        'gemma:7b': 7.0,
        'gemma2:9b': 9.0,
        'qwen2:0.5b': 0.5,
        'qwen2:1.5b': 1.5,
        'qwen2:7b': 7.0,
        'mistral': 7.0,
        'llama2': 7.0,
        'llama3': 8.0,
        'phi4': 14.0,
    }

    # Check for exact or partial matches
    for key, size in size_map.items():
        if key in name_lower:
            return size

    # Default assumption for unknown models
    return 7.0


def estimate_training_ram(param_count_b: float, use_8bit_optimizer: bool = True) -> float:
    """
    Estimate RAM needed for LoRA training (in GB).

    Args:
        param_count_b: Model parameters in billions
        use_8bit_optimizer: Whether using 8-bit optimizer (saves ~75% on optimizer memory)

    Returns:
        Estimated RAM in GB
    """
    # Memory breakdown for FP32 CPU LoRA training:
    # - Base model weights (frozen): params * 4 bytes
    # - LoRA adapters (trainable, ~1% of params): params * 0.01 * 4 bytes
    # - LoRA gradients: params * 0.01 * 4 bytes
    # - Optimizer states for LoRA only: params * 0.01 * 2-8 bytes
    # - Activations: params * 2 bytes (rough estimate for forward pass)

    params = param_count_b * 1e9
    bytes_per_param = 4  # FP32
    lora_ratio = 0.01  # LoRA typically adds 1% trainable params

    # Base model (frozen, but needs to be loaded)
    model_mem = params * bytes_per_param / 1e9  # GB

    # LoRA adapters
    lora_params_mem = params * lora_ratio * bytes_per_param / 1e9
    lora_gradients_mem = params * lora_ratio * bytes_per_param / 1e9

    # Optimizer states (only for LoRA params)
    if use_8bit_optimizer:
        lora_optimizer_mem = params * lora_ratio * 2 / 1e9  # 8-bit
    else:
        lora_optimizer_mem = params * lora_ratio * 8 / 1e9  # Full AdamW

    # Activations (reduced estimate for LoRA)
    activation_mem = params * 2 / 1e9

    total_ram = model_mem + lora_params_mem + lora_gradients_mem + lora_optimizer_mem + activation_mem

    # Add 30% overhead for PyTorch, Python, etc.
    total_ram *= 1.3

    return total_ram


def get_available_ram() -> float:
    """Get available system RAM in GB"""
    return psutil.virtual_memory().available / 1e9


def can_train_model(model_name: str, use_8bit: bool = False) -> Tuple[bool, float, float]:
    """
    Check if a model can be trained on this system.

    Note: use_8bit defaults to False because 8-bit optimizer only works on GPU.
    For CPU training, we always use standard optimizer.

    Returns:
        (can_train, estimated_ram_needed, available_ram)
    """
    import torch

    # 8-bit optimizer only works on GPU
    has_gpu = torch.cuda.is_available()
    use_8bit = use_8bit and has_gpu

    param_count = estimate_model_params(model_name)
    estimated_ram = estimate_training_ram(param_count, use_8bit)
    available_ram = get_available_ram()

    # Need at least 80% safety margin
    can_train = estimated_ram < (available_ram * 0.8)

    return can_train, estimated_ram, available_ram


if __name__ == "__main__":
    # Test the utilities
    print(f"Ollama installed: {is_ollama_installed()}")

    models = get_ollama_models()
    print(f"\nFound {len(models)} Ollama models:")
    for model in models:
        print(f"  - {model['name']} ({model['size']})")

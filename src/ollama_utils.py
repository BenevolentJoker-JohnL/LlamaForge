"""
Ollama Integration Utilities
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional


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
    # Common base model mappings
    mappings = {
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
        "gemma3": "google/gemma-2b",
        "qwen2.5": "Qwen/Qwen2.5-7B",
        "qwen3": "Qwen/Qwen2.5-7B",
        "qwen": "Qwen/Qwen-7B",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "granite": "ibm-granite/granite-8b-code-base",
    }

    # Extract base model name
    base_name = ollama_name.split(':')[0].lower()

    # Check for exact matches first
    if base_name in mappings:
        return mappings[base_name]

    # Check for partial matches
    for key, hf_name in mappings.items():
        if key in base_name:
            return mappings[key]

    # Default fallback
    return "mistralai/Mistral-7B-v0.1"


def ollama_model_to_hf_name(ollama_name: str) -> str:
    """Deprecated - use get_ollama_model_base instead"""
    return get_ollama_model_base(ollama_name)


if __name__ == "__main__":
    # Test the utilities
    print(f"Ollama installed: {is_ollama_installed()}")

    models = get_ollama_models()
    print(f"\nFound {len(models)} Ollama models:")
    for model in models:
        print(f"  - {model['name']} ({model['size']})")

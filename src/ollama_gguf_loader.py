"""
Direct Ollama GGUF Loader - Use local Ollama models for training
"""

import subprocess
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

# Handle both module and direct execution
try:
    from .ollama_utils import get_ollama_model_base
except ImportError:
    from ollama_utils import get_ollama_model_base


def is_hf_model_cached(hf_model_name: str) -> bool:
    """Check if a HuggingFace model is already cached locally"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    # HF cache format: models--Organization--ModelName
    cache_name = "models--" + hf_model_name.replace("/", "--")
    model_cache = cache_dir / cache_name

    return model_cache.exists()


def get_hf_cache_size(hf_model_name: str) -> Optional[float]:
    """Get size of cached HF model in GB"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_name = "models--" + hf_model_name.replace("/", "--")
    model_cache = cache_dir / cache_name

    if not model_cache.exists():
        return None

    # Calculate total size
    total_size = sum(f.stat().st_size for f in model_cache.rglob('*') if f.is_file())
    return total_size / 1e9  # Convert to GB


def get_ollama_gguf_path(model_name: str) -> Optional[Path]:
    """Get the GGUF file path for an Ollama model"""
    try:
        result = subprocess.run(
            ["ollama", "show", model_name, "--modelfile"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return None

        # Parse the FROM line to get GGUF path
        for line in result.stdout.split('\n'):
            if line.startswith('FROM '):
                gguf_path = line.replace('FROM ', '').strip()
                path = Path(gguf_path)
                if path.exists():
                    return path

        return None

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def can_use_gguf_directly(gguf_path: Path) -> bool:
    """
    Check if we can use this GGUF for training.
    Currently, we need FP16 GGUFs for training.
    Quantized GGUFs (Q4, Q8, etc.) need conversion.
    """
    # Check file size as a heuristic
    # FP16 models are roughly 2x larger than Q4
    size_gb = gguf_path.stat().st_size / 1e9

    # For a 3B model, FP16 would be ~6GB, Q4 would be ~2-3GB
    # This is a rough check
    if size_gb > 5:
        return True  # Likely FP16
    return False  # Likely quantized, needs conversion


def load_ollama_model_for_training(model_name: str) -> Dict[str, any]:
    """
    Load an Ollama model for training.

    Returns a dict with:
        - 'hf_model': HuggingFace model name to use
        - 'gguf_path': Path to local GGUF (for reference)
        - 'is_cached': Whether the HF model is already downloaded
        - 'cache_size': Size of cached model in GB (if cached)
        - 'status': Status message
    """
    print(f"🔍 Analyzing Ollama model: {model_name}")

    # Get GGUF path
    gguf_path = get_ollama_gguf_path(model_name)

    if not gguf_path:
        print(f"⚠️  Could not find GGUF for {model_name}")
        return {
            'hf_model': None,
            'gguf_path': None,
            'is_cached': False,
            'cache_size': None,
            'status': 'error'
        }

    print(f"✓ Found GGUF: {gguf_path}")
    print(f"  Size: {gguf_path.stat().st_size / 1e9:.2f} GB (Q4 quantized)")

    # Get base HF model
    hf_model = get_ollama_model_base(model_name)

    print(f"\n📦 Training Requirements:")
    print(f"  Ollama GGUF: Q4 quantized (for inference only)")
    print(f"  Training needs: FP16/FP32 weights")
    print(f"  Base model: {hf_model}")

    # Check cache
    is_cached = is_hf_model_cached(hf_model)
    cache_size = get_hf_cache_size(hf_model) if is_cached else None

    if is_cached:
        print(f"\n✅ Base model already cached!")
        print(f"  Location: ~/.cache/huggingface/hub/")
        print(f"  Size: {cache_size:.2f} GB")
        print(f"  Status: No download needed")
    else:
        print(f"\n⬇️  Base model needs to be downloaded")
        print(f"  From: HuggingFace Hub")
        print(f"  Will cache for future use")

    return {
        'hf_model': hf_model,
        'gguf_path': gguf_path,
        'is_cached': is_cached,
        'cache_size': cache_size,
        'status': 'cached' if is_cached else 'needs_download'
    }


if __name__ == "__main__":
    # Test
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"

    result = load_ollama_model_for_training(model)

    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"  HF Model: {result['hf_model']}")
    print(f"  GGUF Path: {result['gguf_path']}")
    print(f"  Cached: {result['is_cached']}")
    print(f"  Status: {result['status']}")
    if result['cache_size']:
        print(f"  Cache Size: {result['cache_size']:.2f} GB")
    print(f"{'='*60}")

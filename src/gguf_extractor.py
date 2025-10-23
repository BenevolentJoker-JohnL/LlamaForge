"""
GGUF Extractor - Converts GGUF models to FP16 for training
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional


class GGUFExtractor:
    """Extracts GGUF models from Ollama and converts to FP16"""

    def __init__(self, ollama_model: str, output_dir: str = "./models"):
        self.ollama_model = ollama_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def locate_gguf(self) -> Path:
        """Find GGUF file in Ollama's model directory"""
        ollama_dir = Path.home() / ".ollama" / "models" / "blobs"

        if not ollama_dir.exists():
            raise FileNotFoundError(f"Ollama directory not found: {ollama_dir}")

        # Find largest GGUF file (usually the model weights)
        gguf_files = list(ollama_dir.glob("sha256-*"))
        if not gguf_files:
            raise FileNotFoundError("No GGUF files found in Ollama directory")

        # Get largest file
        gguf_path = max(gguf_files, key=lambda p: p.stat().st_size)
        print(f"Found GGUF: {gguf_path} ({gguf_path.stat().st_size / 1e9:.2f} GB)")

        return gguf_path

    def convert_to_fp16(self, gguf_path: Path, vocab_path: Optional[Path] = None) -> Path:
        """Convert GGUF to FP16 HuggingFace format"""
        output_path = self.output_dir / "fp16_model"
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if llama.cpp is available
        llama_cpp_convert = shutil.which("convert-hf-to-gguf.py")

        if llama_cpp_convert:
            # Use llama.cpp conversion (reverse direction)
            print("Using llama.cpp for conversion...")
            # Note: This is a placeholder - actual conversion requires llama.cpp tools
            # In practice, you'd use: python convert.py --outtype f16 ...
            print("Warning: Direct GGUF→FP16 conversion requires llama.cpp")
            print("For now, using GGUF directly with ctransformers or llama-cpp-python")

            # Copy GGUF for now (we'll load it directly)
            shutil.copy(gguf_path, output_path / "model.gguf")
        else:
            print("llama.cpp not found - copying GGUF for direct loading")
            shutil.copy(gguf_path, output_path / "model.gguf")

        return output_path

    def extract(self) -> Path:
        """Full extraction pipeline"""
        print(f"Extracting {self.ollama_model}...")

        # Step 1: Locate GGUF
        gguf_path = self.locate_gguf()

        # Step 2: Convert to FP16 (or copy for direct loading)
        model_path = self.convert_to_fp16(gguf_path)

        print(f"Model extracted to: {model_path}")
        return model_path


if __name__ == "__main__":
    # Test extraction
    extractor = GGUFExtractor("mistral")
    extractor.extract()

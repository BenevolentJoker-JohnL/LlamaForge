"""
GGUF Converter - Merges LoRA and converts back to GGUF
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class GGUFConverter:
    """Merges LoRA adapter and converts to GGUF"""

    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        output_dir: str = "./output"
    ):
        self.base_model_name = base_model_name
        self.adapter_path = Path(adapter_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_lora(self) -> Path:
        """Merge LoRA adapter with base model"""
        print("Loading base model...")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # Load and merge LoRA
        print("Merging LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        merged_model = model.merge_and_unload()

        # Save merged model
        merged_path = self.output_dir / "merged_model"
        merged_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving merged model to {merged_path}...")
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        print("LoRA merged successfully")
        return merged_path

    def convert_to_gguf(
        self,
        merged_model_path: Path,
        quantization: str = "Q4_K_M",
        output_name: Optional[str] = None
    ) -> Path:
        """Convert merged model to GGUF format using llama.cpp"""
        print(f"Converting to GGUF ({quantization})...")

        if output_name is None:
            output_name = f"finetuned-{quantization}.gguf"

        output_path = self.output_dir / output_name

        # Find llama.cpp installation
        llama_cpp_dir = Path.home() / "llama.cpp"
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

        if not convert_script.exists():
            print("Warning: llama.cpp not found at ~/llama.cpp")
            print("Saving as HuggingFace format only")
            return merged_model_path

        # Step 1: Convert to GGUF FP16 first
        temp_fp16 = self.output_dir / f"{output_path.stem}_fp16.gguf"

        print("Converting to GGUF FP16...")
        cmd_fp16 = [
            "python3",
            str(convert_script),
            str(merged_model_path),
            "--outtype", "f16",
            "--outfile", str(temp_fp16)
        ]

        subprocess.run(cmd_fp16, check=True)

        # Step 2: Quantize if needed
        if quantization.upper() != "F16":
            print(f"Quantizing to {quantization}...")

            # Find quantize binary
            quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
            if not quantize_bin.exists():
                quantize_bin = llama_cpp_dir / "build-cpu" / "bin" / "llama-quantize"

            if not quantize_bin.exists():
                # Try build directory directly
                quantize_bin = llama_cpp_dir / "llama-quantize"

            if quantize_bin.exists():
                cmd_quant = [
                    str(quantize_bin),
                    str(temp_fp16),
                    str(output_path),
                    quantization.upper()
                ]
                subprocess.run(cmd_quant, check=True)
                temp_fp16.unlink()  # Remove temp file
            else:
                print(f"Warning: llama-quantize not found, keeping FP16")
                temp_fp16.rename(output_path)
        else:
            temp_fp16.rename(output_path)

        print(f"✅ GGUF saved to: {output_path}")
        return output_path

    def full_conversion(self, quantization: str = "Q4_K_M") -> Path:
        """Complete merge + conversion pipeline"""
        print("\n" + "="*60)
        print("FULL CONVERSION PIPELINE")
        print("="*60)

        # Step 1: Merge LoRA
        print("\n[1/2] Merging LoRA adapter with base model...")
        merged_path = self.merge_lora()

        # Step 2: Convert to GGUF
        print("\n[2/2] Converting to GGUF format...")
        gguf_path = self.convert_to_gguf(merged_path, quantization)

        print("\n" + "="*60)
        print(f"✅ CONVERSION COMPLETE")
        print(f"Output: {gguf_path}")
        print("="*60)

        return gguf_path


if __name__ == "__main__":
    # Test converter
    converter = GGUFConverter(
        base_model_name="mistralai/Mistral-7B-v0.1",
        adapter_path="./output/lora_adapter"
    )
    converter.full_conversion()

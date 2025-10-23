"""
Auto-Structuring Dataset Loader - Handles JSON, CSV, TXT datasets
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
from datasets import Dataset
from transformers import PreTrainedTokenizer


class DatasetLoader:
    """Automatically structures raw datasets for training"""

    SUPPORTED_FORMATS = [".json", ".jsonl", ".csv", ".txt"]

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        task_type: Optional[str] = None
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type

        if self.file_path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {self.file_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

    def _load_json(self) -> List[str]:
        """Load JSON/JSONL and auto-detect structure"""
        samples = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            # Try JSONL first
            first_line = f.readline()
            f.seek(0)

            try:
                json.loads(first_line)
                is_jsonl = True
            except:
                is_jsonl = False

            if is_jsonl:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    samples.append(self._structure_json_entry(data))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        samples.append(self._structure_json_entry(entry))
                else:
                    samples.append(self._structure_json_entry(data))

        return samples

    def _structure_json_entry(self, data: Dict) -> str:
        """Convert JSON entry to training text"""
        # Detect common patterns
        if "prompt" in data and "completion" in data:
            return f"{data['prompt']}\n{data['completion']}"
        elif "instruction" in data and "output" in data:
            input_text = data.get("input", "")
            instruction = data["instruction"]
            output = data["output"]
            if input_text:
                return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
            else:
                return f"Instruction: {instruction}\nOutput: {output}"
        elif "input" in data and "output" in data:
            return f"{data['input']}\n{data['output']}"
        elif "text" in data and "label" in data:
            return f"Text: {data['text']}\nLabel: {data['label']}"
        elif "question" in data and "answer" in data:
            return f"Question: {data['question']}\nAnswer: {data['answer']}"
        elif "text" in data:
            return data["text"]
        else:
            # Fallback: concatenate all values
            return " ".join(str(v) for v in data.values())

    def _load_csv(self) -> List[str]:
        """Load CSV and auto-detect columns"""
        samples = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                columns = list(row.keys())

                if len(columns) >= 2:
                    # Assume first column is input, second is output
                    text = f"{row[columns[0]]}\n{row[columns[1]]}"
                elif len(columns) == 1:
                    text = row[columns[0]]
                else:
                    continue

                samples.append(text)

        return samples

    def _load_txt(self) -> List[str]:
        """Load plain text file"""
        samples = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(line)

        return samples

    def load(self) -> Dataset:
        """Load and structure dataset"""
        print(f"Loading dataset: {self.file_path}")

        # Load based on format
        ext = self.file_path.suffix
        if ext in [".json", ".jsonl"]:
            samples = self._load_json()
        elif ext == ".csv":
            samples = self._load_csv()
        elif ext == ".txt":
            samples = self._load_txt()
        else:
            raise ValueError(f"Unsupported format: {ext}")

        print(f"Loaded {len(samples)} samples")

        # Tokenize
        print("Tokenizing...")
        tokenized = self.tokenizer(
            samples,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None
        )

        # Add labels for causal LM (same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()

        dataset = Dataset.from_dict(tokenized)
        print(f"Dataset ready: {len(dataset)} samples")

        return dataset


if __name__ == "__main__":
    # Test loader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    loader = DatasetLoader("test.jsonl", tokenizer)
    dataset = loader.load()
    print(dataset)

#!/usr/bin/env python3
"""
Download useful datasets for fine-tuning
"""

import json
from datasets import load_dataset
from pathlib import Path


def download_code_alpaca():
    """Download Code Alpaca dataset - Python code generation"""
    print("Downloading Code Alpaca (20k Python coding examples)...")

    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    # Convert to JSONL format
    output_file = Path("testtraindata/code_alpaca.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            if i >= 1000:  # Limit to 1000 samples for faster training
                break

            # Format as instruction-output pairs
            entry = {
                "instruction": example['instruction'],
                "input": example.get('input', ''),
                "output": example['output']
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✓ Saved {min(1000, len(dataset))} examples to {output_file}")
    return output_file


def download_sql_dataset():
    """Download SQL generation dataset"""
    print("\nDownloading SQL dataset (text-to-SQL)...")

    try:
        dataset = load_dataset("b-mc2/sql-create-context", split="train")

        output_file = Path("testtraindata/sql_generation.jsonl")

        with open(output_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(dataset):
                if i >= 500:  # Limit to 500 samples
                    break

                entry = {
                    "instruction": f"Generate SQL query for: {example['question']}",
                    "context": example.get('context', ''),
                    "output": example['answer']
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"✓ Saved {min(500, len(dataset))} examples to {output_file}")
        return output_file
    except Exception as e:
        print(f"⚠ SQL dataset failed: {e}")
        return None


def download_python_qa():
    """Download Stack Overflow Python Q&A"""
    print("\nDownloading Python Q&A from Stack Overflow...")

    try:
        dataset = load_dataset("pacovaldez/stackoverflow-questions", split="train")

        output_file = Path("testtraindata/python_qa.jsonl")

        with open(output_file, 'w', encoding='utf-8') as f:
            count = 0
            for example in dataset:
                if count >= 500:
                    break

                # Filter for Python questions
                if 'python' in example.get('tags', '').lower():
                    entry = {
                        "question": example['question_title'],
                        "details": example.get('question_body', '')[:500],  # Truncate
                        "answer": example.get('answer', '')[:1000]  # Truncate
                    }

                    if entry['answer']:  # Only include if has answer
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        count += 1

            print(f"✓ Saved {count} examples to {output_file}")
            return output_file
    except Exception as e:
        print(f"⚠ Stack Overflow dataset failed: {e}")
        return None


def create_practical_coding_dataset():
    """Create a curated practical coding dataset"""
    print("\nCreating curated practical coding examples...")

    examples = [
        # Data structures
        {
            "instruction": "Implement a function to reverse a linked list",
            "output": """def reverse_linked_list(head):
    prev = None
    current = head

    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev"""
        },
        # Algorithms
        {
            "instruction": "Write a function to find all pairs in an array that sum to a target value",
            "output": """def find_pairs_with_sum(arr, target):
    seen = set()
    pairs = []

    for num in arr:
        complement = target - num
        if complement in seen:
            pairs.append((complement, num))
        seen.add(num)

    return pairs"""
        },
        # File operations
        {
            "instruction": "Create a function to read a CSV file and return a list of dictionaries",
            "output": """import csv

def read_csv_to_dict(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)"""
        },
        # API requests
        {
            "instruction": "Write a function to make a GET request and handle errors gracefully",
            "output": """import requests

def fetch_data(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None"""
        },
        # Database
        {
            "instruction": "Create a function to safely execute SQL queries with parameterization",
            "output": """import sqlite3

def execute_query(db_path, query, params=()):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.commit()
        conn.close()
        return results
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None"""
        },
        # More examples...
    ]

    output_file = Path("testtraindata/practical_coding.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(examples)} examples to {output_file}")
    return output_file


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading Useful Datasets for Fine-Tuning")
    print("=" * 60)

    datasets_downloaded = []

    # Download datasets
    try:
        file = download_code_alpaca()
        datasets_downloaded.append(("Code Generation (Python)", file))
    except Exception as e:
        print(f"✗ Code Alpaca failed: {e}")

    try:
        file = download_sql_dataset()
        if file:
            datasets_downloaded.append(("SQL Generation", file))
    except Exception as e:
        print(f"✗ SQL dataset failed: {e}")

    try:
        file = create_practical_coding_dataset()
        datasets_downloaded.append(("Practical Coding", file))
    except Exception as e:
        print(f"✗ Practical coding failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)

    for name, path in datasets_downloaded:
        if path:
            size = path.stat().st_size / 1024 / 1024  # MB
            print(f"✓ {name}: {path} ({size:.2f} MB)")

    print("\nTo use these datasets:")
    print("  python llamaforge_interactive.py")
    print("  # Select a dataset from testtraindata/")

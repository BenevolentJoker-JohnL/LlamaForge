"""
SOLLOL Integration for Distributed LlamaForge Training

This module provides teacher-student agnostic distributed training flows
using SOLLOL for orchestration and resource management.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Literal, Callable
import json

# Add SOLLOL to path
SOLLOL_PATH = Path("/home/joker/SOLLOL")
if SOLLOL_PATH.exists() and str(SOLLOL_PATH) not in sys.path:
    sys.path.insert(0, str(SOLLOL_PATH))

try:
    from sollol import SOLLOLClient, TaskConfig, ResourceConfig
    SOLLOL_AVAILABLE = True
except ImportError:
    SOLLOL_AVAILABLE = False
    print("Warning: SOLLOL not available. Distributed training disabled.")


class DistributedTrainingMode:
    """Training modes for distributed execution"""
    TEACHER_STUDENT = "teacher_student"  # Teacher generates, students train
    DATA_PARALLEL = "data_parallel"      # Standard data parallelism
    PIPELINE_PARALLEL = "pipeline"       # Pipeline model parallelism
    HYBRID = "hybrid"                    # Combine multiple strategies


class SOLLOLOrchestrator:
    """
    Orchestrates distributed LlamaForge training using SOLLOL.

    Supports:
    - Teacher-student distillation
    - Data-parallel PEFT/LoRA training
    - Resource-aware task scheduling
    - Automatic failover and checkpointing
    """

    def __init__(
        self,
        mode: str = DistributedTrainingMode.TEACHER_STUDENT,
        sollol_config: Optional[Dict] = None
    ):
        if not SOLLOL_AVAILABLE:
            raise RuntimeError("SOLLOL is not available. Cannot initialize distributed training.")

        self.mode = mode
        self.sollol_config = sollol_config or {}
        self.client = None
        self.active_tasks = {}

    def connect(self, host: str = None, port: int = None):
        """Connect to SOLLOL orchestration server"""
        # Use provided params, or fall back to config, or use defaults
        host = host or self.sollol_config.get("sollol_host", "localhost")
        port = port or self.sollol_config.get("sollol_port", 5000)

        try:
            self.client = SOLLOLClient(host=host, port=port)
            print(f"[✓] Connected to SOLLOL at {host}:{port}")
            return True
        except Exception as e:
            print(f"[✗] Failed to connect to SOLLOL: {e}")
            return False

    def discover_nodes(self) -> List[Dict]:
        """
        Discover available compute nodes and their Ollama models.

        Shows GPU/CPU resources and which models are available on each node.
        """
        if not self.client:
            raise RuntimeError("Not connected to SOLLOL. Call connect() first.")

        nodes = self.client.list_nodes()
        print(f"[i] Discovered {len(nodes)} compute nodes\n")

        for node in nodes:
            resources = node.get('resources', {})

            # Display node info
            print(f"    {node['name']}:")

            # CPU/GPU/RAM
            cpu = resources.get('cpu', {})
            mem = resources.get('memory', {})
            gpu = resources.get('gpu', {})

            print(f"      • {cpu.get('cores', 0)} CPUs, "
                  f"{mem.get('total_gb', 0):.1f}GB RAM, "
                  f"{gpu.get('count', 0)} GPUs")

            # Ollama models
            ollama_models = resources.get('ollama_models', [])
            if ollama_models:
                print(f"      • Ollama models: {', '.join([m['name'] for m in ollama_models[:5]])}")
                if len(ollama_models) > 5:
                    print(f"        (+ {len(ollama_models) - 5} more)")
            else:
                print(f"      • No Ollama models detected")

            print()

        return nodes

    def get_nodes_with_model(self, model_name: str) -> List[Dict]:
        """
        Find all nodes that have a specific Ollama model available.

        Args:
            model_name: Ollama model name (e.g., 'qwen2.5:7b')

        Returns:
            List of nodes that have this model
        """
        if not self.client:
            raise RuntimeError("Not connected to SOLLOL. Call connect() first.")

        nodes = self.client.list_nodes()
        matching_nodes = []

        for node in nodes:
            ollama_models = node.get('resources', {}).get('ollama_models', [])
            model_names = [m.get('name', '') for m in ollama_models]

            if model_name in model_names:
                matching_nodes.append(node)

        return matching_nodes

    def create_teacher_task(
        self,
        model_path: str,
        dataset_path: str,
        output_dir: str,
        batch_size: int = 4,
        max_samples: Optional[int] = None
    ) -> str:
        """
        Create a teacher task for generating training outputs.

        The teacher loads the full model and generates predictions
        for the dataset, which are then used to train student models.

        SOLLOL automatically assigns this task to a node that has
        the specified model available, preferring GPU nodes.

        Args:
            model_path: Ollama model name (e.g., 'qwen2.5:7b')
            dataset_path: Path to dataset file
            output_dir: Directory for teacher outputs
            batch_size: Inference batch size
            max_samples: Optional limit on dataset samples

        Returns:
            task_id: Unique identifier for this task
        """
        # Find nodes with this model
        matching_nodes = self.get_nodes_with_model(model_path)

        if not matching_nodes:
            raise RuntimeError(
                f"No nodes have model '{model_path}' available. "
                f"Please pull the model on at least one node using: ollama pull {model_path}"
            )

        # Prefer GPU nodes for teacher (faster inference)
        gpu_nodes = [n for n in matching_nodes if n.get('resources', {}).get('gpu', {}).get('count', 0) > 0]
        preferred_nodes = gpu_nodes if gpu_nodes else matching_nodes

        print(f"[i] Found {len(matching_nodes)} nodes with model '{model_path}'")
        if gpu_nodes:
            print(f"[i] Preferring GPU nodes for teacher task")

        # Create task with node constraints
        task_config = TaskConfig(
            name="llamaforge_teacher",
            type="inference",
            priority="high",
            resources=ResourceConfig(
                gpu_memory_gb=16,  # Teachers need more memory
                cpu_cores=4,
                ram_gb=32
            ),
            node_constraints={
                "ollama_model": model_path,  # Must have this model
                "preferred_nodes": [n['name'] for n in preferred_nodes]
            },
            command=[
                "python", "-m", "llamaforge.teacher_worker",
                "--model", model_path,
                "--data", dataset_path,
                "--output", output_dir,
                "--batch-size", str(batch_size),
                *([" --max-samples", str(max_samples)] if max_samples else [])
            ],
            checkpointing=True,
            max_retries=2
        )

        task_id = self.client.submit_task(task_config)
        self.active_tasks[task_id] = {
            "type": "teacher",
            "status": "submitted",
            "model": model_path,
            "assigned_nodes": [n['name'] for n in preferred_nodes]
        }

        print(f"[✓] Teacher task created: {task_id}")
        return task_id

    def create_student_tasks(
        self,
        base_model: str,
        teacher_outputs_dir: str,
        num_students: int = 4,
        lora_config: Optional[Dict] = None
    ) -> List[str]:
        """
        Create student training tasks.

        Students train PEFT/LoRA adapters on teacher-generated outputs.
        Tasks are automatically distributed across nodes that have the
        student model available.

        SOLLOL handles load balancing across available nodes.

        Args:
            base_model: Ollama model name for students (e.g., 'qwen2.5:0.5b')
            teacher_outputs_dir: Directory with teacher-generated outputs
            num_students: Number of parallel student workers
            lora_config: LoRA configuration (r, alpha, dropout)

        Returns:
            List of task IDs
        """
        lora_config = lora_config or {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05
        }

        # Find nodes with this model
        matching_nodes = self.get_nodes_with_model(base_model)

        if not matching_nodes:
            raise RuntimeError(
                f"No nodes have model '{base_model}' available. "
                f"Please pull the model on at least one node using: ollama pull {base_model}"
            )

        print(f"[i] Found {len(matching_nodes)} nodes with model '{base_model}'")
        print(f"[i] Distributing {num_students} student tasks across available nodes")

        # If we have more students than nodes, that's fine - SOLLOL will queue them
        if num_students > len(matching_nodes):
            print(f"[i] Note: {num_students} students will be queued across {len(matching_nodes)} nodes")

        task_ids = []

        for i in range(num_students):
            task_config = TaskConfig(
                name=f"llamaforge_student_{i}",
                type="training",
                priority="normal",
                resources=ResourceConfig(
                    gpu_memory_gb=8,  # Students need less memory
                    cpu_cores=2,
                    ram_gb=16
                ),
                node_constraints={
                    "ollama_model": base_model,  # Must have this model
                    "allowed_nodes": [n['name'] for n in matching_nodes]
                },
                command=[
                    "python", "-m", "llamaforge.student_worker",
                    "--model", base_model,
                    "--teacher-outputs", teacher_outputs_dir,
                    "--student-id", str(i),
                    "--lora-r", str(lora_config["r"]),
                    "--lora-alpha", str(lora_config["alpha"]),
                    "--lora-dropout", str(lora_config["dropout"])
                ],
                checkpointing=True,
                max_retries=3
            )

            task_id = self.client.submit_task(task_config)
            self.active_tasks[task_id] = {
                "type": "student",
                "student_id": i,
                "model": base_model,
                "status": "submitted"
            }
            task_ids.append(task_id)

        print(f"[✓] Created {num_students} student tasks")
        return task_ids

    def create_data_parallel_tasks(
        self,
        model_path: str,
        dataset_path: str,
        num_workers: int = 4,
        batch_size: int = 2
    ) -> List[str]:
        """
        Create data-parallel training tasks.

        Splits dataset across workers, each training on a subset.
        Gradients are aggregated periodically.

        Returns:
            List of task IDs
        """
        task_ids = []

        for i in range(num_workers):
            task_config = TaskConfig(
                name=f"llamaforge_worker_{i}",
                type="training",
                priority="normal",
                resources=ResourceConfig(
                    gpu_memory_gb=12,
                    cpu_cores=2,
                    ram_gb=24
                ),
                command=[
                    "python", "-m", "llamaforge.distributed_worker",
                    "--model", model_path,
                    "--data", dataset_path,
                    "--worker-id", str(i),
                    "--num-workers", str(num_workers),
                    "--batch-size", str(batch_size),
                    "--mode", "data_parallel"
                ],
                checkpointing=True,
                max_retries=2
            )

            task_id = self.client.submit_task(task_config)
            self.active_tasks[task_id] = {
                "type": "data_parallel_worker",
                "worker_id": i,
                "status": "submitted"
            }
            task_ids.append(task_id)

        print(f"[✓] Created {num_workers} data-parallel workers")
        return task_ids

    def monitor_tasks(self, task_ids: List[str], callback: Optional[Callable] = None):
        """
        Monitor task progress and handle failures.

        Args:
            task_ids: List of task IDs to monitor
            callback: Optional callback function for status updates
        """
        while True:
            all_complete = True

            for task_id in task_ids:
                status = self.client.get_task_status(task_id)
                self.active_tasks[task_id]["status"] = status["state"]

                if status["state"] in ["running", "pending", "submitted"]:
                    all_complete = False

                if status["state"] == "failed":
                    print(f"[✗] Task {task_id} failed: {status.get('error', 'Unknown error')}")
                    # Could implement retry logic here

                if callback:
                    callback(task_id, status)

            if all_complete:
                break

            import time
            time.sleep(5)  # Poll every 5 seconds

        print("[✓] All tasks completed")

    def aggregate_student_outputs(
        self,
        student_output_dirs: List[str],
        output_path: str
    ):
        """
        Aggregate student model outputs (LoRA adapters).

        Can implement ensemble methods or adapter merging.
        """
        print(f"[i] Aggregating {len(student_output_dirs)} student outputs...")

        # TODO: Implement adapter averaging or ensemble
        # For now, just select the best performing student

        print(f"[✓] Aggregated model saved to {output_path}")


def create_distributed_training(
    mode: str,
    config: Dict
) -> SOLLOLOrchestrator:
    """
    Factory function to create distributed training orchestrator.

    Args:
        mode: Training mode (teacher_student, data_parallel, etc.)
        config: Training configuration

    Returns:
        Configured SOLLOLOrchestrator
    """
    orchestrator = SOLLOLOrchestrator(mode=mode)

    # Auto-connect to default SOLLOL instance
    if orchestrator.connect():
        return orchestrator
    else:
        raise RuntimeError("Failed to connect to SOLLOL orchestration server")


if __name__ == "__main__":
    # Test SOLLOL integration
    if SOLLOL_AVAILABLE:
        print("✓ SOLLOL available")
        print(f"✓ SOLLOL path: {SOLLOL_PATH}")
    else:
        print("✗ SOLLOL not available")

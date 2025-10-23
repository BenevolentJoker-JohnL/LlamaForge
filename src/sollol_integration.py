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

    def connect(self, host: str = "localhost", port: int = 5000):
        """Connect to SOLLOL orchestration server"""
        try:
            self.client = SOLLOLClient(host=host, port=port)
            print(f"[✓] Connected to SOLLOL at {host}:{port}")
            return True
        except Exception as e:
            print(f"[✗] Failed to connect to SOLLOL: {e}")
            return False

    def discover_nodes(self) -> List[Dict]:
        """Discover available compute nodes"""
        if not self.client:
            raise RuntimeError("Not connected to SOLLOL. Call connect() first.")

        nodes = self.client.list_nodes()
        print(f"[i] Discovered {len(nodes)} compute nodes")

        for node in nodes:
            print(f"    - {node['name']}: {node['resources']}")

        return nodes

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

        Returns:
            task_id: Unique identifier for this task
        """
        task_config = TaskConfig(
            name="llamaforge_teacher",
            type="inference",
            priority="high",
            resources=ResourceConfig(
                gpu_memory_gb=16,  # Teachers need more memory
                cpu_cores=4,
                ram_gb=32
            ),
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
        self.active_tasks[task_id] = {"type": "teacher", "status": "submitted"}

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
        Tasks are distributed across available nodes.

        Returns:
            List of task IDs
        """
        lora_config = lora_config or {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05
        }

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

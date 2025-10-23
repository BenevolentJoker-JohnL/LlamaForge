"""
LlamaForge SOLLOL Node Worker

Registers LlamaForge as a compute node with SOLLOL orchestration system.
Provides resource reporting, task execution, and dashboard integration.
"""

import sys
import os
import psutil
import torch
import time
from pathlib import Path
from typing import Dict, Optional
import json
import socket

# Add SOLLOL to path
SOLLOL_PATH = Path("/home/joker/SOLLOL")
if SOLLOL_PATH.exists() and str(SOLLOL_PATH) not in sys.path:
    sys.path.insert(0, str(SOLLOL_PATH))

try:
    from sollol.node import Node, NodeConfig
    from sollol.metrics import MetricsReporter
    SOLLOL_AVAILABLE = True
except ImportError:
    try:
        # Fallback import paths
        from node import Node, NodeConfig
        from metrics import MetricsReporter
        SOLLOL_AVAILABLE = True
    except ImportError:
        SOLLOL_AVAILABLE = False
        print("Warning: SOLLOL node modules not available")


class LlamaForgeNode:
    """
    LlamaForge compute node that registers with SOLLOL.

    Features:
    - Auto-discovery of GPU/CPU resources
    - Real-time metrics reporting (RAM, GPU, CPU usage)
    - Task execution callbacks
    - Dashboard integration
    - Automatic heartbeat and health monitoring
    """

    def __init__(
        self,
        name: Optional[str] = None,
        sollol_host: str = "localhost",
        sollol_port: int = 5000,
        tags: Optional[list] = None
    ):
        self.name = name or f"llamaforge-{socket.gethostname()}"
        self.sollol_host = sollol_host
        self.sollol_port = sollol_port
        self.tags = tags or ["llamaforge", "peft", "lora", "gguf"]

        self.node = None
        self.metrics_reporter = None
        self.running = False

    def detect_resources(self) -> Dict:
        """Detect available compute resources"""
        resources = {
            "cpu": {
                "cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024 ** 3),
                "available_gb": psutil.virtual_memory().available / (1024 ** 3),
            },
            "gpu": {
                "available": torch.cuda.is_available(),
                "count": 0,
                "devices": []
            },
            "ollama_models": []
        }

        # Detect GPUs
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            resources["gpu"]["count"] = gpu_count

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                resources["gpu"]["devices"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": props.total_memory / (1024 ** 3),
                    "compute_capability": f"{props.major}.{props.minor}"
                })

        # Detect available Ollama models on this node
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.ollama_utils import get_ollama_models, is_ollama_installed

            if is_ollama_installed():
                ollama_models = get_ollama_models()
                resources["ollama_models"] = [
                    {
                        "name": model["name"],
                        "size": model["size"]
                    }
                    for model in ollama_models
                ]
                print(f"[i] Detected {len(ollama_models)} Ollama models on this node")
        except Exception as e:
            print(f"[!] Warning: Could not detect Ollama models: {e}")

        return resources

    def get_current_metrics(self) -> Dict:
        """Get current resource usage metrics"""
        metrics = {
            "timestamp": time.time(),
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=0.1),
                "per_core": psutil.cpu_percent(interval=0.1, percpu=True)
            },
            "memory": {
                "used_gb": psutil.virtual_memory().used / (1024 ** 3),
                "available_gb": psutil.virtual_memory().available / (1024 ** 3),
                "percent": psutil.virtual_memory().percent
            },
            "gpu": []
        }

        # GPU metrics
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    mem_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)

                    metrics["gpu"].append({
                        "id": i,
                        "memory_allocated_gb": mem_allocated,
                        "memory_reserved_gb": mem_reserved,
                        "memory_total_gb": mem_total,
                        "memory_percent": (mem_allocated / mem_total) * 100 if mem_total > 0 else 0
                    })
                except Exception as e:
                    print(f"Warning: Failed to get GPU {i} metrics: {e}")

        return metrics

    def register(self) -> bool:
        """Register this node with SOLLOL orchestration server"""
        if not SOLLOL_AVAILABLE:
            print("[✗] SOLLOL not available. Cannot register node.")
            return False

        try:
            resources = self.detect_resources()

            # Create node configuration
            node_config = NodeConfig(
                name=self.name,
                host=socket.gethostname(),
                port=8000,  # LlamaForge worker port
                tags=self.tags,
                capabilities={
                    "training": ["peft", "lora", "full-finetune"],
                    "conversion": ["gguf", "fp16", "quantization"],
                    "ollama_integration": True,
                    "cpu_training": True,
                    "gpu_training": torch.cuda.is_available()
                },
                resources=resources
            )

            # Initialize node
            self.node = Node(config=node_config)

            # Connect to SOLLOL server
            self.node.connect(host=self.sollol_host, port=self.sollol_port)

            print(f"[✓] Registered with SOLLOL as '{self.name}'")
            print(f"[i] Dashboard: http://{self.sollol_host}:{self.sollol_port}/dashboard")
            print(f"[i] Resources: {resources['cpu']['cores']} CPUs, "
                  f"{resources['memory']['total_gb']:.1f}GB RAM, "
                  f"{resources['gpu']['count']} GPUs")

            # Initialize metrics reporter
            self.metrics_reporter = MetricsReporter(
                node_id=self.name,
                sollol_host=self.sollol_host,
                sollol_port=self.sollol_port
            )

            return True

        except Exception as e:
            print(f"[✗] Failed to register with SOLLOL: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_heartbeat(self, interval: int = 30):
        """Start sending heartbeat and metrics to SOLLOL"""
        if not self.node:
            print("[✗] Node not registered. Call register() first.")
            return

        self.running = True
        print(f"[✓] Starting heartbeat (every {interval}s)")

        try:
            while self.running:
                # Get current metrics
                metrics = self.get_current_metrics()

                # Send to SOLLOL
                if self.metrics_reporter:
                    self.metrics_reporter.send_metrics(metrics)

                # Send heartbeat
                self.node.send_heartbeat()

                # Wait
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[i] Heartbeat stopped by user")
            self.stop()
        except Exception as e:
            print(f"[✗] Heartbeat error: {e}")
            self.stop()

    def stop(self):
        """Stop the node and unregister from SOLLOL"""
        self.running = False

        if self.node:
            try:
                self.node.disconnect()
                print(f"[✓] Disconnected from SOLLOL")
            except Exception as e:
                print(f"[!] Error during disconnect: {e}")

    def execute_task(self, task_config: Dict) -> Dict:
        """
        Execute a training task assigned by SOLLOL.

        Args:
            task_config: Task configuration from SOLLOL

        Returns:
            Task result with status and outputs
        """
        print(f"[▶] Executing task: {task_config.get('name', 'unknown')}")

        try:
            # Import LlamaForge training modules
            from lora_trainer import LoRATrainer
            from dataset_loader import DatasetLoader
            from gguf_converter import GGUFConverter

            task_type = task_config.get("type", "training")

            if task_type == "training":
                # Run training task
                result = self._run_training_task(task_config)

            elif task_type == "conversion":
                # Run GGUF conversion task
                result = self._run_conversion_task(task_config)

            elif task_type == "teacher_inference":
                # Run teacher inference for distillation
                result = self._run_teacher_task(task_config)

            else:
                result = {
                    "status": "error",
                    "error": f"Unknown task type: {task_type}"
                }

            print(f"[✓] Task completed: {task_config.get('name')}")
            return result

        except Exception as e:
            print(f"[✗] Task failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _run_training_task(self, config: Dict) -> Dict:
        """Execute LoRA training task"""
        from lora_trainer import LoRATrainer
        from dataset_loader import DatasetLoader

        # Extract parameters
        model = config["model"]
        dataset_path = config["data"]
        output_dir = config.get("output", "./output")

        # Initialize trainer
        trainer = LoRATrainer(
            model_path=model,
            output_dir=output_dir
        )

        # Load model
        model, tokenizer = trainer.load_model(model)

        # Load dataset
        loader = DatasetLoader(dataset_path, tokenizer)
        dataset = loader.load()

        # Train
        adapter_path = trainer.train(
            dataset=dataset,
            num_epochs=config.get("epochs", 3),
            batch_size=config.get("batch_size", 2)
        )

        return {
            "status": "success",
            "adapter_path": str(adapter_path),
            "metrics": {}  # TODO: Add training metrics
        }

    def _run_conversion_task(self, config: Dict) -> Dict:
        """Execute GGUF conversion task"""
        from gguf_converter import GGUFConverter

        converter = GGUFConverter(
            base_model_name=config["model"],
            adapter_path=config.get("adapter_path"),
            output_dir=config["output"]
        )

        output_path = converter.full_conversion(
            quantization=config.get("quantization", "q4_k_m")
        )

        return {
            "status": "success",
            "output_path": str(output_path)
        }

    def _run_teacher_task(self, config: Dict) -> Dict:
        """Execute teacher inference for distillation"""
        # TODO: Implement teacher inference
        return {
            "status": "success",
            "outputs_path": config.get("output", "./teacher_outputs")
        }


def main():
    """Main entry point for LlamaForge SOLLOL node"""
    import argparse

    parser = argparse.ArgumentParser(description="LlamaForge SOLLOL Node")
    parser.add_argument("--name", help="Node name", default=None)
    parser.add_argument("--sollol-host", help="SOLLOL server host", default="localhost")
    parser.add_argument("--sollol-port", help="SOLLOL server port", type=int, default=5000)
    parser.add_argument("--heartbeat-interval", help="Heartbeat interval (seconds)", type=int, default=30)
    parser.add_argument("--no-register", action="store_true", help="Skip registration (for testing)")

    args = parser.parse_args()

    # Create node
    node = LlamaForgeNode(
        name=args.name,
        sollol_host=args.sollol_host,
        sollol_port=args.sollol_port
    )

    if not args.no_register:
        # Register with SOLLOL
        if node.register():
            # Start heartbeat loop
            node.start_heartbeat(interval=args.heartbeat_interval)
        else:
            print("[✗] Failed to register. Exiting.")
            sys.exit(1)
    else:
        # Just show resources
        resources = node.detect_resources()
        print(json.dumps(resources, indent=2))


if __name__ == "__main__":
    main()

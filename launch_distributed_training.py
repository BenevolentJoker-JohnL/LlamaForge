#!/usr/bin/env python3
"""
Distributed Training Launcher - Use SOLLOL to train on cluster

Launch training on remote nodes, then:
- Use coordinator node for inference
- Monitor training progress
- Participate in distributed training if desired
"""

import sys
import argparse
from pathlib import Path

# Add SOLLOL to path
SOLLOL_PATH = Path("/home/joker/SOLLOL")
if SOLLOL_PATH.exists():
    sys.path.insert(0, str(SOLLOL_PATH))

from src.sollol_integration import SOLLOLOrchestrator, DistributedTrainingMode

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_banner(title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}{Colors.END}\n")

def launch_distributed_training(
    model: str,
    dataset: str,
    mode: str = "data_parallel",
    epochs: int = 1,
    batch_size: int = 1,
    max_length: int = 512,
    lora_r: int = 32,
    lora_alpha: int = 64,
    sollol_host: str = "localhost",
    sollol_port: int = 8765,
    num_workers: int = None,
):
    """
    Launch distributed training across SOLLOL cluster

    Args:
        model: HuggingFace model name or path
        dataset: Path to training dataset
        mode: Training mode (data_parallel, teacher_student, hybrid)
        epochs: Number of training epochs
        batch_size: Batch size per worker
        max_length: Max sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        sollol_host: SOLLOL coordinator host
        sollol_port: SOLLOL coordinator port
        num_workers: Number of worker nodes (None = use all available)
    """

    print_banner("DISTRIBUTED TRAINING WITH SOLLOL")

    # Initialize orchestrator
    print(f"{Colors.BLUE}[1/5]{Colors.END} Initializing SOLLOL orchestrator...")
    orchestrator = SOLLOLOrchestrator(mode=mode)

    # Connect to SOLLOL
    print(f"{Colors.BLUE}[2/5]{Colors.END} Connecting to SOLLOL cluster...")
    if not orchestrator.connect(host=sollol_host, port=sollol_port):
        print(f"{Colors.RED}[✗]{Colors.END} Failed to connect to SOLLOL")
        print(f"\n{Colors.YELLOW}Make sure SOLLOL is running:{Colors.END}")
        print(f"  cd ~/SOLLOL && python -m sollol.server")
        return False

    # Discover nodes
    print(f"{Colors.BLUE}[3/5]{Colors.END} Discovering available nodes...")
    nodes = orchestrator.discover_nodes()

    if not nodes:
        print(f"{Colors.RED}[✗]{Colors.END} No worker nodes available")
        return False

    # Select workers
    available_workers = len(nodes)
    workers_to_use = num_workers if num_workers else available_workers

    print(f"\n{Colors.GREEN}[✓]{Colors.END} Found {available_workers} nodes")
    print(f"{Colors.BLUE}[INFO]{Colors.END} Using {workers_to_use} workers for training\n")

    # Launch training job
    print(f"{Colors.BLUE}[4/5]{Colors.END} Launching distributed training job...")

    training_config = {
        "model": model,
        "dataset": dataset,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_workers": workers_to_use,
        "mode": mode,
    }

    job_id = orchestrator.launch_training(training_config)

    if not job_id:
        print(f"{Colors.RED}[✗]{Colors.END} Failed to launch training job")
        return False

    print(f"{Colors.GREEN}[✓]{Colors.END} Training job launched!")
    print(f"{Colors.BLUE}[INFO]{Colors.END} Job ID: {Colors.BOLD}{job_id}{Colors.END}")

    # Show what coordinator can do now
    print(f"\n{Colors.BLUE}[5/5]{Colors.END} Training running on remote nodes!")
    print_banner("COORDINATOR NODE IS NOW FREE")

    print(f"{Colors.GREEN}What you can do now:{Colors.END}\n")
    print(f"  1. {Colors.CYAN}Monitor training:{Colors.END}")
    print(f"     python -c 'from sollol import SOLLOLClient; c=SOLLOLClient(); c.get_job_status(\"{job_id}\")'")
    print()

    print(f"  2. {Colors.CYAN}Use coordinator for inference:{Colors.END}")
    print(f"     ollama run your-model")
    print(f"     # Coordinator CPU/RAM available for other work!")
    print()

    print(f"  3. {Colors.CYAN}View training dashboard:{Colors.END}")
    print(f"     firefox http://localhost:5000")
    print()

    print(f"  4. {Colors.CYAN}Check cluster status:{Colors.END}")
    print(f"     python -m sollol.cli status")
    print()

    print(f"  5. {Colors.CYAN}Help with training (optional):{Colors.END}")
    print(f"     # Coordinator can JOIN the training to speed it up")
    print(f"     python -c 'from sollol import SOLLOLClient; c=SOLLOLClient(); c.join_job(\"{job_id}\")'")
    print()

    print(f"{Colors.YELLOW}[!] Training continues in background on worker nodes{Colors.END}")
    print(f"{Colors.YELLOW}[!] Coordinator resources: {Colors.BOLD}AVAILABLE{Colors.END}")

    return True

def main():
    parser = argparse.ArgumentParser(
        description="Launch distributed training on SOLLOL cluster"
    )

    # Required args
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Training dataset path")

    # Training args
    parser.add_argument("--mode", choices=["data_parallel", "teacher_student", "hybrid"],
                       default="data_parallel", help="Training mode")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per worker")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")

    # SOLLOL args
    parser.add_argument("--sollol-host", default="localhost", help="SOLLOL host")
    parser.add_argument("--sollol-port", type=int, default=8765, help="SOLLOL port")
    parser.add_argument("--workers", type=int, help="Number of workers (default: all)")

    args = parser.parse_args()

    # Launch
    success = launch_distributed_training(
        model=args.model,
        dataset=args.dataset,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        sollol_host=args.sollol_host,
        sollol_port=args.sollol_port,
        num_workers=args.workers,
    )

    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Distributed training launched successfully!{Colors.END}\n")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Failed to launch distributed training{Colors.END}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()

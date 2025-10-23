#!/usr/bin/env python3
"""
Direct Distributed Training - Uses SOLLOL node discovery + PyTorch DDP

No server needed - SOLLOL auto-discovers nodes, then launches PyTorch distributed training.
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add SOLLOL to path
SOLLOL_PATH = Path("/home/joker/SOLLOL")
if SOLLOL_PATH.exists():
    sys.path.insert(0, str(SOLLOL_PATH))

from sollol import OllamaPool
from sollol.discovery import discover_ollama_nodes


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
    epochs: int = 1,
    batch_size: int = 2,
    max_length: int = 512,
    lora_r: int = 32,
    lora_alpha: int = 64,
    gradient_accumulation: int = 4,
):
    """
    Launch distributed training using SOLLOL discovery + PyTorch DDP
    """

    print_banner("DISTRIBUTED TRAINING WITH SOLLOL AUTO-DISCOVERY")

    # Step 1: Discover nodes
    print(f"{Colors.BLUE}[1/4]{Colors.END} Auto-discovering Ollama nodes...")
    nodes = discover_ollama_nodes(discover_all_nodes=True, timeout=1.0)

    if len(nodes) < 1:
        print(f"{Colors.RED}[✗]{Colors.END} No nodes discovered!")
        return False

    print(f"{Colors.GREEN}[✓]{Colors.END} Found {len(nodes)} nodes:")
    for i, node in enumerate(nodes):
        print(f"    Node {i}: {node['host']}:{node['port']}")

    # Step 2: Check if parallel makes sense
    print(f"\n{Colors.BLUE}[2/4]{Colors.END} Checking cluster configuration...")
    pool = OllamaPool(nodes=nodes)
    should_parallel = pool.should_use_parallel_execution(num_tasks=2)
    unique_hosts = pool.count_unique_physical_hosts()

    print(f"    Unique physical machines: {unique_hosts}")
    print(f"    Parallel training: {Colors.GREEN}ENABLED{Colors.END}" if should_parallel
          else f"    Parallel training: {Colors.YELLOW}DISABLED (same machine){Colors.END}")

    if not should_parallel:
        print(f"\n{Colors.YELLOW}[!] WARNING:{Colors.END} All nodes on same machine!")
        print(f"    Distributed training will be SLOWER due to resource contention")
        print(f"    Recommendation: Use single-node training instead")
        return False

    # Step 3: Setup distributed training
    print(f"\n{Colors.BLUE}[3/4]{Colors.END} Configuring PyTorch distributed training...")

    # Master node is this machine
    master_addr = nodes[0]['host']
    master_port = "29500"  # PyTorch distributed port
    world_size = len(nodes)

    print(f"    Master: {master_addr}:{master_port}")
    print(f"    World size: {world_size} nodes")

    # Step 4: Launch training on each node
    print(f"\n{Colors.BLUE}[4/4]{Colors.END} Launching training on all nodes...")

    # On this node (rank 0), launch directly
    launch_cmd = [
        "python3", "-m", "torch.distributed.run",
        "--nproc_per_node=1",  # 1 process per node (CPU training)
        "--nnodes", str(world_size),
        "--node_rank", "0",
        "--master_addr", master_addr,
        "--master_port", master_port,
        "distributed_train.py",
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--max-length", str(max_length),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--gradient-accumulation", str(gradient_accumulation),
    ]

    print(f"\n{Colors.GREEN}[✓]{Colors.END} Launching rank 0 on {master_addr}...")
    print(f"{Colors.CYAN}Command:{Colors.END} {' '.join(launch_cmd)}")

    # Launch on remote nodes via SSH
    if len(nodes) > 1:
        print(f"\n{Colors.YELLOW}To complete setup, run on OTHER nodes:{Colors.END}")
        for rank, node in enumerate(nodes[1:], start=1):
            remote_cmd = [
                "python3", "-m", "torch.distributed.run",
                "--nproc_per_node=1",
                "--nnodes", str(world_size),
                "--node_rank", str(rank),
                "--master_addr", master_addr,
                "--master_port", master_port,
                "/home/joker/LlamaForge/distributed_train.py",
                "--model", model,
                "--dataset", dataset,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--max-length", str(max_length),
                "--lora-r", str(lora_r),
                "--lora-alpha", str(lora_alpha),
                "--gradient-accumulation", str(gradient_accumulation),
            ]

            print(f"\n{Colors.CYAN}On {node['host']}:{Colors.END}")
            print(f"  {' '.join(remote_cmd)}")

    print(f"\n{Colors.YELLOW}[!] Press Enter to start training on this node...{Colors.END}")
    input()

    # Launch training
    try:
        subprocess.run(launch_cmd, check=True)
        print(f"\n{Colors.GREEN}[✓]{Colors.END} Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.RED}[✗]{Colors.END} Training failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Launch distributed training with SOLLOL auto-discovery"
    )

    # Required args
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Training dataset path")

    # Training args
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per worker")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                       help="Gradient accumulation steps")

    args = parser.parse_args()

    # Launch
    success = launch_distributed_training(
        model=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_accumulation=args.gradient_accumulation,
    )

    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Distributed training launched!{Colors.END}\n")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Failed to launch distributed training{Colors.END}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

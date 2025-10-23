#!/usr/bin/env python3
"""
LlamaForge SOLLOL Node Launcher

Start LlamaForge as a SOLLOL compute node for distributed training.

Usage:
    python llamaforge_node.py                    # Register with local SOLLOL
    python llamaforge_node.py --sollol-host IP   # Register with remote SOLLOL
    python llamaforge_node.py --name my-gpu-01   # Custom node name
"""

from src.sollol_node import main

if __name__ == "__main__":
    main()

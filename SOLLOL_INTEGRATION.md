# LlamaForge + SOLLOL: Distributed Training Integration

## Overview

LlamaForge now integrates with **SOLLOL** (Simple Orchestrated Loaned Local Ollama Learning) for distributed, multi-node fine-tuning with automatic resource management and dashboard monitoring.

## Features

✅ **Automatic Model Distribution** - Nodes report available Ollama models; tasks auto-assigned to nodes with required models
✅ **Teacher-Student Distillation** - One teacher model generates training data for multiple student models
✅ **Data-Parallel Training** - Split datasets across nodes for faster training
✅ **Auto Resource Discovery** - Automatically detects GPUs, CPUs, memory, and Ollama models
✅ **Dashboard Integration** - Real-time monitoring via SOLLOL web dashboard
✅ **Smart Task Assignment** - Teacher tasks prefer GPU nodes, students distributed across available nodes
✅ **Fault Tolerance** - Automatic task retry and checkpointing
✅ **Resource Loaning** - Share compute resources across multiple training jobs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SOLLOL Orchestrator                       │
│            http://localhost:5000/dashboard                   │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐        │
│  │   Scheduler  │  │   Monitor   │  │  Dashboard   │        │
│  └─────────────┘  └─────────────┘  └──────────────┘        │
└───────────┬─────────────────────────────┬───────────────────┘
            │                             │
    ┌───────▼───────┐             ┌───────▼───────┐
    │ LlamaForge    │             │ LlamaForge    │
    │ Node 1 (GPU)  │             │ Node 2 (CPU)  │
    │               │             │               │
    │ • 4 CPU cores │             │ • 8 CPU cores │
    │ • 16GB RAM    │             │ • 32GB RAM    │
    │ • 1x RTX 3090 │             │ • No GPU      │
    └───────────────┘             └───────────────┘
```

---

## Quick Start

### 1. Start SOLLOL Server (One-Time Setup)

```bash
# In SOLLOL directory
cd ~/SOLLOL
python -m sollol.server
```

**Dashboard will be available at:** `http://localhost:5000/dashboard`

### 2. Register LlamaForge Nodes

On **each machine** you want to use for training:

```bash
# In LlamaForge directory
cd ~/LlamaForge

# Register node with SOLLOL
python llamaforge_node.py

# Or with custom name
python llamaforge_node.py --name gpu-workstation-01

# Or connect to remote SOLLOL server
python llamaforge_node.py --sollol-host 192.168.1.100
```

You should see:
```
[✓] Registered with SOLLOL as 'llamaforge-hostname'
[i] Dashboard: http://localhost:5000/dashboard
[i] Resources: 8 CPUs, 31.3GB RAM, 1 GPUs
[✓] Starting heartbeat (every 30s)
```

### 3. Submit Distributed Training Job

```python
from src.sollol_integration import create_distributed_training

# Teacher-Student Mode
orchestrator = create_distributed_training(
    mode="teacher_student",
    config={
        "teacher_model": "qwen2.5:0.5b",
        "dataset": "data/coding_tasks.jsonl",
        "num_students": 4
    }
)

# Create teacher task (generates training data)
teacher_id = orchestrator.create_teacher_task(
    model_path="qwen2.5:0.5b",
    dataset_path="data/coding_tasks.jsonl",
    output_dir="./teacher_outputs"
)

# Create student tasks (train on teacher outputs)
student_ids = orchestrator.create_student_tasks(
    base_model="qwen2.5:0.5b",
    teacher_outputs_dir="./teacher_outputs",
    num_students=4
)

# Monitor progress
orchestrator.monitor_tasks(teacher_id + student_ids)
```

---

## Training Modes

### 1. Teacher-Student Distillation

**Use Case:** One powerful GPU generates training data, multiple nodes train lightweight models

```python
orchestrator = create_distributed_training(mode="teacher_student", config={})

# Teacher (1x powerful GPU)
teacher_id = orchestrator.create_teacher_task(
    model_path="meta-llama/Llama-3.1-8B",
    dataset_path="dataset.jsonl",
    output_dir="./teacher_outputs",
    batch_size=8
)

# Students (4x CPU-only nodes)
student_ids = orchestrator.create_student_tasks(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    teacher_outputs_dir="./teacher_outputs",
    num_students=4,
    lora_config={"r": 8, "alpha": 16}
)
```

**Flow:**
1. Teacher model generates predictions for entire dataset
2. Students train LoRA adapters using teacher's outputs as targets
3. Students can be merged or ensembled

### 2. Data-Parallel Training

**Use Case:** Split large dataset across multiple nodes

```python
orchestrator = create_distributed_training(mode="data_parallel", config={})

# Create 4 workers, each trains on a shard of the data
worker_ids = orchestrator.create_data_parallel_tasks(
    model_path="mistralai/Mistral-7B-v0.1",
    dataset_path="large_dataset.jsonl",
    num_workers=4,
    batch_size=2
)
```

**Flow:**
1. Dataset automatically split across workers
2. Each worker trains on its shard independently
3. Gradients synchronized periodically
4. Final model is averaged checkpoint

### 3. Hybrid (Coming Soon)

Combine teacher-student + data-parallel for maximum efficiency.

---

## Dashboard Features

Access the SOLLOL dashboard at `http://localhost:5000/dashboard`

**Real-time Monitoring:**
- ✅ Node status (online/offline/busy)
- ✅ CPU/GPU/RAM usage per node
- ✅ Active training jobs
- ✅ Task queue and completion status
- ✅ Training metrics and loss curves
- ✅ Resource allocation visualization

**Example Screenshot:**
```
┌─────────────────────────────────────────────────────┐
│ SOLLOL Dashboard - LlamaForge Cluster               │
├─────────────────────────────────────────────────────┤
│ Nodes: 3 online, 1 busy                             │
│                                                      │
│ llamaforge-gpu-01      ████████░░ 80% GPU          │
│ llamaforge-cpu-01      ██░░░░░░░░ 20% CPU          │
│ llamaforge-cpu-02      ████░░░░░░ 40% CPU          │
│                                                      │
│ Active Tasks:                                        │
│  ✓ teacher_qwen2.5        [RUNNING]  GPU-01        │
│  ⧗ student_0              [PENDING]  -              │
│  ⧗ student_1              [PENDING]  -              │
│  ⧗ student_2              [PENDING]  -              │
└─────────────────────────────────────────────────────┘
```

---

## Configuration

### Node Configuration

Edit `llamaforge_node.py` or pass CLI args:

```bash
python llamaforge_node.py \
    --name my-gpu-node \
    --sollol-host localhost \
    --sollol-port 5000 \
    --heartbeat-interval 30
```

### Task Priority

Tasks can have different priorities:

```python
TaskConfig(
    name="critical_training",
    priority="high",  # "low", "normal", "high", "critical"
    ...
)
```

Higher priority tasks get scheduled first.

### Resource Limits

Specify minimum resources for tasks:

```python
TaskConfig(
    resources=ResourceConfig(
        gpu_memory_gb=16,  # Minimum GPU memory
        cpu_cores=4,        # Minimum CPU cores
        ram_gb=32           # Minimum RAM
    )
)
```

SOLLOL will only assign tasks to nodes that meet requirements.

---

## Advanced Usage

### Custom Training Scripts

Create custom workers for specialized tasks:

```python
# my_custom_worker.py
from src.sollol_node import LlamaForgeNode

class CustomTrainingNode(LlamaForgeNode):
    def execute_task(self, task_config):
        # Your custom training logic here
        return {"status": "success"}

if __name__ == "__main__":
    node = CustomTrainingNode(name="custom-worker")
    node.register()
    node.start_heartbeat()
```

### Checkpoint Management

SOLLOL automatically manages checkpoints:

```python
TaskConfig(
    checkpointing=True,
    checkpoint_interval=100,  # Every 100 steps
    max_checkpoints=3         # Keep last 3
)
```

Crashed tasks automatically resume from last checkpoint.

### Failover and Retry

```python
TaskConfig(
    max_retries=3,           # Retry up to 3 times
    retry_backoff=60,        # Wait 60s between retries
    failover_enabled=True    # Auto-reassign to different node
)
```

---

## Troubleshooting

### Node Won't Register

```bash
# Check SOLLOL server is running
curl http://localhost:5000/health

# Check firewall
sudo ufw allow 5000

# Try with verbose logging
python llamaforge_node.py --verbose
```

### Dashboard Not Accessible

```bash
# Make sure SOLLOL server is running
cd ~/SOLLOL
python -m sollol.server

# Check it's listening
netstat -tulpn | grep 5000
```

### Task Keeps Failing

1. Check node logs in dashboard
2. Verify resource requirements are met
3. Check dataset path is accessible from node
4. Try running task manually first

---

## Performance Tips

1. **Use GPU nodes for teachers** - Teacher inference is fast on GPU
2. **Use CPU nodes for students** - LoRA training works well on CPU
3. **Batch size tuning** - Smaller batches for limited memory
4. **Monitor dashboard** - Watch for resource bottlenecks
5. **Checkpoint frequently** - Prevent losing progress on failures

---

## API Reference

See `src/sollol_integration.py` for full API documentation.

**Key Classes:**
- `SOLLOLOrchestrator` - Main orchestration interface
- `LlamaForgeNode` - Compute node implementation
- `TaskConfig` - Task configuration
- `ResourceConfig` - Resource requirements

---

## Examples

See `examples/distributed/` for:
- `teacher_student_example.py` - Teacher-student distillation
- `data_parallel_example.py` - Data-parallel training
- `multi_model_training.py` - Train multiple models simultaneously

---

## Contributing

To add new distributed training modes:
1. Extend `SOLLOLOrchestrator` class
2. Implement task creation method
3. Add worker execution logic to `LlamaForgeNode`
4. Update documentation

---

## License

MIT - Same as LlamaForge

## Support

- GitHub Issues: https://github.com/your-org/LlamaForge/issues
- SOLLOL Docs: https://github.com/your-org/SOLLOL

---

**Next:** [Teacher-Student Training Guide](./TEACHER_STUDENT_GUIDE.md)

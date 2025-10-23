# Teacher-Student Distributed Training with SOLLOL

## Quick Start Guide

This guide shows you how to use LlamaForge's distributed teacher-student training powered by SOLLOL.

---

## Architecture Overview

```
                    ┌─────────────────┐
                    │ SOLLOL Server   │
                    │   (Dashboard)   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼─────┐        ┌────▼─────┐        ┌────▼─────┐
   │ Teacher  │        │ Student  │        │ Student  │
   │  Node    │        │  Node 1  │        │  Node 2  │
   │ (GPU)    │        │ (CPU)    │        │ (CPU)    │
   └──────────┘        └──────────┘        └──────────┘
```

**Teacher:** Large model generates high-quality training outputs
**Students:** Smaller models train on teacher's outputs (knowledge distillation)

---

## Step-by-Step Setup

### Step 1: Start SOLLOL Server (One-Time Setup)

On your **main orchestration machine**:

```bash
cd ~/SOLLOL
python -m sollol.server --host 0.0.0.0 --port 5000
```

**Dashboard will be available at:** `http://localhost:5000/dashboard`

Leave this running in the background or use a screen session:
```bash
screen -S sollol
python -m sollol.server
# Press Ctrl+A then D to detach
```

---

### Step 2: Register Compute Nodes

On **each machine** you want to use for training:

#### Node 1: Teacher (GPU Machine)
```bash
cd ~/LlamaForge
python llamaforge_node.py \
    --name teacher-gpu-01 \
    --sollol-host YOUR_SOLLOL_SERVER_IP \
    --sollol-port 5000
```

You should see:
```
[✓] Registered with SOLLOL as 'teacher-gpu-01'
[i] Dashboard: http://YOUR_IP:5000/dashboard
[i] Resources: 8 CPUs, 31.3GB RAM, 1 GPUs
[✓] Starting heartbeat (every 30s)
```

#### Node 2: Student (CPU Machine)
```bash
cd ~/LlamaForge
python llamaforge_node.py \
    --name student-cpu-01 \
    --sollol-host YOUR_SOLLOL_SERVER_IP \
    --sollol-port 5000
```

#### Node 3: Another Student (CPU Machine)
```bash
cd ~/LlamaForge
python llamaforge_node.py \
    --name student-cpu-02 \
    --sollol-host YOUR_SOLLOL_SERVER_IP \
    --sollol-port 5000
```

**Tip:** Use `screen` or `tmux` to keep nodes running:
```bash
screen -S llamaforge-node
python llamaforge_node.py --name student-cpu-01
# Press Ctrl+A then D to detach
```

---

### Step 3: Submit Teacher-Student Training Job

Create a Python script to orchestrate the training:

**`run_distributed_training.py`:**

```python
#!/usr/bin/env python3
"""
Example: Teacher-Student Distributed Training
"""

import sys
sys.path.insert(0, '/home/joker/LlamaForge')

from src.sollol_integration import create_distributed_training

# Step 1: Create orchestrator
print("[1/4] Connecting to SOLLOL...")
orchestrator = create_distributed_training(
    mode="teacher_student",
    config={}
)

# Step 2: Discover available nodes
print("[2/4] Discovering compute nodes...")
nodes = orchestrator.discover_nodes()

# Step 3: Create teacher task (runs on GPU node)
print("[3/4] Creating teacher task...")
teacher_task_id = orchestrator.create_teacher_task(
    model_path="qwen2.5:7b",              # Large teacher model
    dataset_path="data/coding_tasks.jsonl",
    output_dir="./teacher_outputs",
    batch_size=8,                         # Higher batch size for GPU
    max_samples=1000                      # Optional: limit samples for testing
)

print(f"[✓] Teacher task created: {teacher_task_id}")

# Step 4: Wait for teacher to finish generating outputs
print("[4/4] Monitoring teacher task...")
orchestrator.monitor_tasks([teacher_task_id])

print("[✓] Teacher outputs generated!")

# Step 5: Create student tasks (run on CPU nodes)
print("[5/5] Creating student tasks...")
student_task_ids = orchestrator.create_student_tasks(
    base_model="qwen2.5:0.5b",           # Smaller student model
    teacher_outputs_dir="./teacher_outputs",
    num_students=2,                      # Number of student nodes
    lora_config={
        "r": 8,
        "alpha": 16,
        "dropout": 0.05
    }
)

print(f"[✓] Created {len(student_task_ids)} student tasks")

# Step 6: Monitor all students
print("[6/6] Monitoring student training...")
orchestrator.monitor_tasks(student_task_ids)

print("[✓] Training complete!")
print(f"[i] Teacher outputs: ./teacher_outputs")
print(f"[i] Student adapters: ./output/student_0, ./output/student_1")
```

**Run it:**
```bash
python run_distributed_training.py
```

---

## Alternative: Data-Parallel Training

If you want to split the dataset across nodes instead:

```python
from src.sollol_integration import create_distributed_training

orchestrator = create_distributed_training(mode="data_parallel", config={})

# Split dataset across 4 workers
worker_ids = orchestrator.create_data_parallel_tasks(
    model_path="qwen2.5:3b",
    dataset_path="data/large_dataset.jsonl",
    num_workers=4,
    batch_size=2
)

# Monitor all workers
orchestrator.monitor_tasks(worker_ids)
```

---

## Monitoring Progress

### Via Dashboard

Open your browser to: `http://YOUR_SOLLOL_SERVER_IP:5000/dashboard`

You'll see:
- Real-time node status
- Task progress
- GPU/CPU/RAM usage
- Training metrics

### Via CLI

The `monitor_tasks()` function shows progress:

```
[⠋] Task teacher_qwen2.5 - RUNNING (GPU-01)
[⠙] Task student_0 - PENDING
[⠹] Task student_1 - PENDING

[✓] Task teacher_qwen2.5 - COMPLETED
[⠋] Task student_0 - RUNNING (CPU-01)
[⠙] Task student_1 - RUNNING (CPU-02)

[✓] All tasks completed
```

---

## Advanced Usage

### Custom Callback for Progress

```python
def on_task_update(task_id, status):
    if status['state'] == 'completed':
        print(f"✓ {task_id} finished!")
        # Could trigger next task here
    elif status['state'] == 'failed':
        print(f"✗ {task_id} failed: {status.get('error')}")

orchestrator.monitor_tasks(task_ids, callback=on_task_update)
```

### Resource Requirements

Specify exact requirements for tasks:

```python
from src.sollol_integration import TaskConfig, ResourceConfig

task_config = TaskConfig(
    name="heavy_training",
    type="training",
    priority="high",
    resources=ResourceConfig(
        gpu_memory_gb=24,    # Needs 24GB GPU
        cpu_cores=8,         # Needs 8 CPU cores
        ram_gb=64            # Needs 64GB RAM
    ),
    max_retries=3,
    checkpointing=True
)
```

### Multi-Stage Pipeline

Chain teacher → students → merge:

```python
# Stage 1: Teacher generates data
teacher_id = orchestrator.create_teacher_task(...)
orchestrator.monitor_tasks([teacher_id])

# Stage 2: Students train in parallel
student_ids = orchestrator.create_student_tasks(...)
orchestrator.monitor_tasks(student_ids)

# Stage 3: Aggregate student outputs
orchestrator.aggregate_student_outputs(
    student_output_dirs=["./output/student_0", "./output/student_1"],
    output_path="./merged_model"
)
```

---

## Typical Workflows

### Workflow 1: Small Dataset, Quick Iteration

**Setup:** 1 GPU teacher + 2 CPU students

```python
# Fast teacher on GPU
teacher_id = orchestrator.create_teacher_task(
    model_path="qwen2.5:3b",          # Smaller teacher for speed
    dataset_path="data/small.jsonl",
    output_dir="./teacher_outputs",
    batch_size=16
)

# Train students in parallel
student_ids = orchestrator.create_student_tasks(
    base_model="qwen2.5:0.5b",
    teacher_outputs_dir="./teacher_outputs",
    num_students=2
)
```

**Time:** ~30 minutes for 1000 samples

---

### Workflow 2: Large Dataset, Production Training

**Setup:** 1 GPU teacher + 4 CPU students

```python
# Powerful teacher
teacher_id = orchestrator.create_teacher_task(
    model_path="llama3.1:70b",        # Very large teacher
    dataset_path="data/large.jsonl",
    output_dir="./teacher_outputs",
    batch_size=32,
    max_samples=10000
)

# Train multiple students
student_ids = orchestrator.create_student_tasks(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    teacher_outputs_dir="./teacher_outputs",
    num_students=4,                   # 4 students training simultaneously
    lora_config={"r": 16, "alpha": 32}  # Higher rank for better quality
)
```

**Time:** Several hours depending on dataset size

---

### Workflow 3: Multi-Model Ensemble

Train different student models on same teacher data:

```python
# One teacher
teacher_id = orchestrator.create_teacher_task(
    model_path="qwen2.5:14b",
    dataset_path="data/ensemble.jsonl",
    output_dir="./teacher_outputs"
)
orchestrator.monitor_tasks([teacher_id])

# Train multiple different student models
student_configs = [
    {"base_model": "qwen2.5:0.5b", "student_id": 0},
    {"base_model": "qwen2.5:1.5b", "student_id": 1},
    {"base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "student_id": 2},
]

student_ids = []
for config in student_configs:
    task_ids = orchestrator.create_student_tasks(
        base_model=config["base_model"],
        teacher_outputs_dir="./teacher_outputs",
        num_students=1
    )
    student_ids.extend(task_ids)

orchestrator.monitor_tasks(student_ids)
```

Now you have 3 different student models trained on the same teacher data!

---

## Troubleshooting

### Nodes Not Showing Up

```bash
# Check SOLLOL server is running
curl http://localhost:5000/health

# Check node can reach SOLLOL
ping YOUR_SOLLOL_SERVER_IP

# Check firewall
sudo ufw allow 5000
```

### Task Fails Immediately

```bash
# Check node has enough resources
python llamaforge_node.py --no-register

# View node logs in dashboard
# Or check task error in orchestrator.monitor_tasks()
```

### Out of Memory on Student

Reduce batch size or use smaller model:

```python
student_ids = orchestrator.create_student_tasks(
    base_model="qwen2.5:0.5b",  # Use smallest model
    num_students=1,              # Reduce students
    lora_config={"r": 4}         # Lower LoRA rank
)
```

---

## Performance Tips

1. **Teacher batch size:** GPU can handle 16-32, start with 16
2. **Student count:** 1 student per available CPU machine
3. **LoRA rank:** Start with 8, increase to 16 for better quality
4. **Checkpointing:** Enable for long runs (auto-resume on crash)
5. **Dataset splitting:** For very large datasets, split manually first

---

## Example Output

After running the teacher-student workflow:

```
./work/
├── teacher_outputs/
│   ├── batch_0.jsonl       # Teacher predictions for batch 0
│   ├── batch_1.jsonl
│   └── ...
├── output/
│   ├── student_0/
│   │   └── lora_adapter/   # Student 0 trained adapter
│   └── student_1/
│       └── lora_adapter/   # Student 1 trained adapter
└── merged_model/
    └── final_adapter/      # Averaged/merged adapter (optional)
```

---

## What's Next?

After training completes:

1. **Merge adapters** (if using multiple students)
2. **Convert to GGUF** for Ollama
3. **Create Ollama model**
4. **Test and deploy**

Or just use LlamaForge's interactive CLI - it handles all this automatically!

---

## Full Example Script

See `examples/distributed/teacher_student_example.py` for a complete working example.

**Quick test:**
```bash
cd ~/LlamaForge
python examples/distributed/teacher_student_example.py \
    --teacher qwen2.5:7b \
    --student qwen2.5:0.5b \
    --data testtraindata/practical_coding.jsonl \
    --num-students 2
```

---

## Support

- SOLLOL Docs: `~/SOLLOL/README.md`
- LlamaForge Docs: `~/LlamaForge/README.md`
- Issues: https://github.com/BenevolentJoker-JohnL/LlamaForge/issues

Happy distributed training! 🚀

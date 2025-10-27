# Distributed Training with SOLLOL

## 🎯 The Brilliant Strategy

**Problem:** Training 621K samples on one node takes 24-48 hours and uses ALL your resources

**Solution:** Use SOLLOL to:
1. Launch training on **remote worker nodes**
2. **Coordinator node stays FREE** for inference/other work
3. Optionally **join training** from coordinator to speed it up

## 🚀 Quick Start

### 1. Start SOLLOL Cluster

```bash
# On coordinator (your main machine)
cd ~/SOLLOL
python -m sollol.server --host 0.0.0.0 --port 8765
```

### 2. Connect Worker Nodes

```bash
# On each worker machine
cd ~/SOLLOL
python -m sollol.worker --coordinator <coordinator-ip>:8765
```

### 3. Launch Distributed Training

```bash
cd ~/LlamaForge

python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/claude_ultimate_with_tools_621k.jsonl \
  --epochs 1 \
  --workers 3 \
  --mode data_parallel
```

### 4. Use Coordinator While Training Runs!

```bash
# Training runs on workers, you can now:

# Use Ollama for inference
ollama run llama2

# Run evaluations
python evaluate_model.py --model ./some_model --preset quick

# Browse the web
firefox https://reddit.com

# Monitor training dashboard
firefox http://localhost:5000
```

## 📊 Training Modes

### Data Parallel (Recommended)
```bash
--mode data_parallel
```

**How it works:**
- Dataset split across N workers
- Each worker trains on 1/N of data
- Gradients averaged across workers
- **Fastest for large datasets**

**Example:**
- 621K samples across 3 workers = 207K samples per worker
- Training time: 1/3 of single-node (8-16 hours instead of 24-48)

### Teacher-Student
```bash
--mode teacher_student
```

**How it works:**
- Coordinator runs large teacher model
- Workers train smaller student models
- Students learn from teacher outputs
- **Best for model compression**

### Hybrid
```bash
--mode hybrid
```

**How it works:**
- Combines data parallelism + teacher-student
- Teacher on coordinator, students on workers
- Dataset split across students
- **Maximum efficiency**

## 💡 Real-World Examples

### Example 1: 3-Node Cluster Training

```bash
# Launch on 3 workers, coordinator stays free
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/claude_ultimate_with_tools_621k.jsonl \
  --epochs 1 \
  --batch-size 2 \
  --max-length 512 \
  --workers 3 \
  --mode data_parallel
```

**Result:**
- Worker 1: Trains on samples 1-207,000
- Worker 2: Trains on samples 207,001-414,000
- Worker 3: Trains on samples 414,001-621,000
- **Coordinator: FREE for other work!**

### Example 2: Coordinator Helps Training

```bash
# Launch on 2 workers
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/code_alpaca_full.jsonl \
  --workers 2 \
  --mode data_parallel

# After launch, coordinator can join:
python -c "
from sollol import SOLLOLClient
client = SOLLOLClient()
jobs = client.list_jobs()
client.join_job(jobs[0]['id'])  # Join latest job
"
```

**Result:**
- Now 3 total workers (2 remote + 1 coordinator)
- Training even faster!
- Can leave anytime

### Example 3: Teacher-Student Distillation

```bash
# Run teacher on coordinator, students on workers
python launch_distributed_training.py \
  --model meta-llama/Llama-2-13b-hf \
  --dataset examples/datasets/claude_ultimate_with_tools_621k.jsonl \
  --workers 4 \
  --mode teacher_student

# Students will be TinyLlama (faster inference)
```

**Result:**
- 13B teacher generates high-quality outputs
- 4× TinyLlama students learn from teacher
- Get small fast model with big model knowledge

## 🎛️ Configuration Options

### Basic Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | HuggingFace model name | Required |
| `--dataset` | Training dataset path | Required |
| `--epochs` | Training epochs | 1 |
| `--batch-size` | Batch size per worker | 1 |
| `--max-length` | Max sequence length | 512 |

### LoRA Options

| Flag | Description | Default |
|------|-------------|---------|
| `--lora-r` | LoRA rank | 32 |
| `--lora-alpha` | LoRA alpha | 64 |

### SOLLOL Options

| Flag | Description | Default |
|------|-------------|---------|
| `--sollol-host` | SOLLOL coordinator host | localhost |
| `--sollol-port` | SOLLOL coordinator port | 8765 |
| `--workers` | Number of workers to use | All available |
| `--mode` | Training mode | data_parallel |

## 📈 Performance Estimates

### Single Node (Baseline)
```
621K samples, 1 epoch
Time: 24-48 hours
Coordinator: 100% busy
```

### 2-Node Cluster
```
621K samples, 1 epoch
Workers: 2
Time: 12-24 hours (2× faster)
Coordinator: 100% FREE
```

### 4-Node Cluster
```
621K samples, 1 epoch
Workers: 4
Time: 6-12 hours (4× faster)
Coordinator: 100% FREE
```

### 4-Node + Coordinator Helps
```
621K samples, 1 epoch
Workers: 4 + coordinator
Time: 5-10 hours (5× faster)
Coordinator: Participates when free
```

## 🔍 Monitoring Distributed Training

### Check Job Status

```bash
python -c "
from sollol import SOLLOLClient
client = SOLLOLClient()
status = client.get_job_status('job-12345')
print(status)
"
```

### View Cluster Status

```bash
python -m sollol.cli status
```

**Output:**
```
SOLLOL Cluster Status
─────────────────────────────────────────
Coordinator: online
Workers: 3/3 online

Active Jobs: 1
├─ job-12345: Training TinyLlama
│  ├─ Progress: 45% (280K/621K samples)
│  ├─ Workers: worker-1, worker-2, worker-3
│  └─ ETA: 6 hours 23 minutes

Resources:
├─ Coordinator: 5% CPU, 20% RAM (mostly idle)
├─ Worker-1: 95% CPU, 80% RAM (training)
├─ Worker-2: 95% CPU, 80% RAM (training)
└─ Worker-3: 95% CPU, 80% RAM (training)
```

### Training Dashboard

```bash
# Open dashboard (shows all cluster training)
firefox http://localhost:5000
```

## 🎯 Use Cases

### Use Case 1: Train While You Work

```bash
# Morning: Launch training on cluster
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/claude_ultimate_with_tools_621k.jsonl \
  --workers 4

# All day: Use coordinator for regular work
# - Browse web
# - Run other models
# - Code
# - Test models

# Evening: Training complete!
# Check results in ./work/distributed_training/
```

### Use Case 2: Rapid Iteration

```bash
# Test different hyperparameters in parallel

# Terminal 1: High learning rate
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/code_alpaca_full.jsonl \
  --workers 2 \
  --lora-r 16

# Terminal 2: Low learning rate
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/code_alpaca_full.jsonl \
  --workers 2 \
  --lora-r 32

# Both train simultaneously on different worker sets!
```

### Use Case 3: Multi-Model Training

```bash
# Train different models at once

# Job 1: Code specialist
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/magicoder_evol_110k.jsonl \
  --workers 2

# Job 2: General assistant
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/alpaca_full.jsonl \
  --workers 2

# Get 2 specialized models in the time of 1!
```

## 🐛 Troubleshooting

### "No worker nodes available"

**Problem:** SOLLOL can't find workers

**Solution:**
1. Check SOLLOL server is running:
   ```bash
   ps aux | grep sollol.server
   ```

2. Check workers are connected:
   ```bash
   python -c "from sollol import SOLLOLClient; print(SOLLOLClient().list_nodes())"
   ```

3. Verify network connectivity:
   ```bash
   # On worker
   ping <coordinator-ip>
   telnet <coordinator-ip> 8765
   ```

### "Failed to launch training job"

**Problem:** Job submission failed

**Solution:**
1. Check dataset path exists:
   ```bash
   ls examples/datasets/claude_ultimate_with_tools_621k.jsonl
   ```

2. Verify workers have enough resources:
   ```bash
   python -m sollol.cli resources
   ```

3. Check coordinator logs:
   ```bash
   tail -f ~/SOLLOL/logs/coordinator.log
   ```

### Training stalls/freezes

**Problem:** One worker died

**Solution:**
SOLLOL auto-recovers! It will:
1. Detect failed worker
2. Reassign work to healthy workers
3. Continue training

Check status:
```bash
python -m sollol.cli jobs
```

## 📚 Advanced: Manual Control

### Start/Stop Jobs

```python
from sollol import SOLLOLClient

client = SOLLOLClient()

# List jobs
jobs = client.list_jobs()

# Pause job
client.pause_job("job-12345")

# Resume job
client.resume_job("job-12345")

# Cancel job
client.cancel_job("job-12345")
```

### Custom Resource Allocation

```python
# Allocate specific GPUs
job_config = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "dataset": "claude_ultimate_with_tools_621k.jsonl",
    "resources": {
        "gpu_ids": [0, 1],  # Use GPUs 0 and 1
        "cpu_cores": 16,     # Use 16 CPU cores
        "ram_gb": 32,        # Reserve 32GB RAM
    }
}

client.submit_job(job_config)
```

## 🎓 TL;DR

**Old way (Single node):**
```bash
python llamaforge.py --model TinyLlama --data huge_dataset.jsonl
# Wait 48 hours
# Can't use machine during training
```

**New way (SOLLOL distributed):**
```bash
python launch_distributed_training.py \
  --model TinyLlama \
  --dataset huge_dataset.jsonl \
  --workers 4

# Training runs on 4 remote workers (12 hours)
# Coordinator is FREE - use for inference, testing, browsing, etc.
# Join training anytime to speed it up even more
```

**Result:**
- ✅ 4× faster training
- ✅ Coordinator stays free
- ✅ Can run multiple training jobs in parallel
- ✅ Auto-recovery if workers fail
- ✅ Flexible resource allocation

**Your 621K dataset now trains in 6-12 hours instead of 24-48!** 🚀

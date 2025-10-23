# Distributed Training with SOLLOL Auto-Discovery

**TL;DR:** SOLLOL automatically finds all nodes on your network. No server needed. Just run PyTorch DDP training on each node.

---

## Understanding SOLLOL Architecture

### What SOLLOL Provides

SOLLOL has **two separate systems**:

1. **Auto-Discovery + Pool (No Server Needed)** ✅
   - Scans network for Ollama nodes
   - Detects which machines are unique
   - Determines if parallel execution is beneficial
   - **This is what distributed training uses**

2. **Gateway Server (Optional)** ❌
   - Centralized API gateway at port 8000
   - For web applications needing a single endpoint
   - **NOT needed for distributed training**

### Why No Server Is Needed

```python
from sollol import OllamaPool
from sollol.discovery import discover_ollama_nodes

# Discovery runs locally, finds nodes via network scan
nodes = discover_ollama_nodes(discover_all_nodes=True)
# Result: [{'host': '10.9.66.154', 'port': '11434'},
#          {'host': '10.9.66.250', 'port': '11434'}]

# Pool checks if they're unique machines
pool = OllamaPool(nodes=nodes)
unique = pool.count_unique_physical_hosts()  # Returns: 2
should_parallel = pool.should_use_parallel_execution(num_tasks=2)  # Returns: True
```

**What happens:**
- ✅ Discovery scans your subnet (10.9.66.0/24) in ~500ms
- ✅ Finds all machines running Ollama on port 11434
- ✅ Resolves localhost → real IPs to avoid duplicates
- ✅ Detects unique physical machines
- ✅ All done locally - no server/daemon needed

---

## How Distributed Training Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ SOLLOL Auto-Discovery (runs on coordinator)                │
│ - Scans 10.9.66.0/24 network                               │
│ - Finds: 10.9.66.154, 10.9.66.250                          │
│ - Verifies: 2 unique physical machines                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ PyTorch Distributed Training (DDP)                          │
│                                                             │
│ Node 0 (10.9.66.154) ◄──────── MASTER ──────────┐          │
│ - Rank 0                                        │          │
│ - Coordinates gradients                         │          │
│ - Trains on samples 0, 2, 4, 6, ...            │          │
│                                                 │          │
│ Node 1 (10.9.66.250) ◄──────────────────────────┘          │
│ - Rank 1                                                    │
│ - Sends gradients to master                                │
│ - Trains on samples 1, 3, 5, 7, ...                        │
└─────────────────────────────────────────────────────────────┘
```

### What You DON'T Need

❌ **SOLLOL server running** - Discovery is client-side only
❌ **Installing LlamaForge on all nodes** - Only need PyTorch + training script
❌ **Shared filesystem** - Can copy dataset to each node
❌ **Complex orchestration** - Just SSH and run commands

### What You DO Need

✅ **Ollama running on all nodes** (for discovery to find them)
✅ **PyTorch installed on all nodes**
✅ **SSH access to remote nodes** (to launch training)
✅ **Dataset accessible on all nodes** (copy or NFS share)
✅ **Network connectivity** between nodes (port 29500 for PyTorch DDP)

---

## Step-by-Step Setup

### 1. Verify SOLLOL Can Find Your Nodes

On coordinator machine (10.9.66.154):

```bash
cd /home/joker/LlamaForge

python3 -c "
import sys
sys.path.insert(0, '/home/joker/SOLLOL')
from sollol import OllamaPool
from sollol.discovery import discover_ollama_nodes

# Auto-discover
nodes = discover_ollama_nodes(discover_all_nodes=True)
print(f'Found {len(nodes)} nodes:')
for node in nodes:
    print(f'  {node}')

# Check parallel viability
pool = OllamaPool(nodes=nodes)
print(f'Unique machines: {pool.count_unique_physical_hosts()}')
print(f'Parallel enabled: {pool.should_use_parallel_execution(2)}')
"
```

**Expected output:**
```
Found 2 nodes:
  {'host': '10.9.66.154', 'port': '11434'}
  {'host': '10.9.66.250', 'port': '11434'}
Unique machines: 2
Parallel enabled: True
```

If you see `Unique machines: 1`, you have a problem - both nodes resolve to same machine.

### 2. Prepare Dataset on All Nodes

**Option A: Copy to each node**
```bash
# From coordinator
scp examples/datasets/your_dataset.jsonl user@10.9.66.250:/home/user/dataset.jsonl
```

**Option B: Use NFS share**
```bash
# Setup NFS share (on coordinator)
sudo apt install nfs-kernel-server
echo "/home/joker/LlamaForge/examples/datasets 10.9.66.0/24(ro,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -a

# Mount on workers
sudo mount 10.9.66.154:/home/joker/LlamaForge/examples/datasets /mnt/datasets
```

### 3. Launch Distributed Training

Use the automated launcher:

```bash
cd /home/joker/LlamaForge

python3 launch_distributed_training_direct.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/alpaca_1k.jsonl \
  --epochs 1 \
  --batch-size 2 \
  --lora-r 32 \
  --lora-alpha 64
```

**What it does:**
1. **Auto-discovers nodes** using SOLLOL (finds 10.9.66.154, 10.9.66.250)
2. **Checks locality** - ensures they're different machines
3. **Shows you the commands** to run on each node
4. **Launches rank 0** on coordinator
5. **Waits for you** to launch rank 1 on other node

### 4. Manual Launch (Alternative)

If auto-launcher doesn't work, run manually:

**Terminal 1 (Coordinator - 10.9.66.154):**
```bash
cd /home/joker/LlamaForge

python3 -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=10.9.66.154 \
  --master_port=29500 \
  distributed_train.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/alpaca_1k.jsonl \
  --epochs 1 \
  --batch-size 2 \
  --lora-r 32 \
  --lora-alpha 64
```

**Terminal 2 (Worker - 10.9.66.250):**
```bash
# SSH to worker first
ssh user@10.9.66.250

cd /home/user/LlamaForge  # or wherever you put it

python3 -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=10.9.66.154 \
  --master_port=29500 \
  distributed_train.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset /path/to/dataset.jsonl \
  --epochs 1 \
  --batch-size 2 \
  --lora-r 32 \
  --lora-alpha 64
```

**Important:** Start both within ~30 seconds of each other, or they'll timeout waiting.

---

## Network Configuration

### Required Ports

| Port | Purpose | Direction |
|------|---------|-----------|
| 11434 | Ollama API (for discovery) | Coordinator → Workers |
| 29500 | PyTorch DDP coordination | Bidirectional |

### Firewall Rules

```bash
# On all nodes
sudo ufw allow 11434/tcp  # Ollama
sudo ufw allow 29500/tcp  # PyTorch DDP
```

### Verify Connectivity

```bash
# From coordinator
curl http://10.9.66.250:11434/api/tags  # Should return JSON
nc -zv 10.9.66.250 29500                 # Should connect

# From worker
curl http://10.9.66.154:11434/api/tags  # Should return JSON
nc -zv 10.9.66.154 29500                 # Should connect
```

---

## Troubleshooting

### Issue 1: SOLLOL Only Finds 1 Node

**Symptoms:**
```
Found 1 nodes:
  {'host': '10.9.66.154', 'port': '11434'}
```

**Causes:**
1. Ollama not running on worker
2. Firewall blocking port 11434
3. Worker on different subnet

**Fix:**
```bash
# On worker
sudo systemctl status ollama  # Check if running
sudo ufw status               # Check firewall
curl http://localhost:11434/api/tags  # Test locally
```

### Issue 2: "Unique machines: 1" When You Have 2

**Symptoms:**
```
Found 2 nodes:
  {'host': 'localhost', 'port': '11434'}
  {'host': '10.9.66.154', 'port': '11434'}
Unique machines: 1
Parallel enabled: False
```

**Cause:** Stale config has localhost entry that resolves to same machine as real IP

**Fix:**
```bash
# Delete stale config
rm ~/.synapticllamas_nodes.json

# Re-run discovery (it auto-deduplicates now)
python3 launch_distributed_training_direct.py --model test --dataset test.json
```

### Issue 3: PyTorch DDP Timeout

**Symptoms:**
```
RuntimeError: Connection timeout after 1800s
```

**Causes:**
1. Nodes can't reach each other on port 29500
2. One node not started in time
3. Different --master_addr on each node

**Fix:**
```bash
# Verify connectivity first
nc -zv 10.9.66.154 29500  # From worker

# Start both nodes within 30 seconds
# Make sure --master_addr is IDENTICAL on both
```

### Issue 4: "No module named 'torch.distributed'"

**Cause:** PyTorch not installed on worker node

**Fix:**
```bash
# On worker node
pip3 install torch transformers peft datasets
```

### Issue 5: Dataset Not Found on Worker

**Symptoms:**
```
FileNotFoundError: examples/datasets/alpaca_1k.jsonl
```

**Cause:** Dataset path doesn't exist on worker

**Fix:**
```bash
# Option 1: Use absolute paths
--dataset /home/user/dataset.jsonl

# Option 2: Copy dataset to same path on all nodes
scp examples/datasets/alpaca_1k.jsonl user@10.9.66.250:/home/user/LlamaForge/examples/datasets/

# Option 3: Use NFS (see Step 2 above)
```

---

## Performance Expectations

### Single Node (Baseline)
```
621K samples, 1 epoch, TinyLlama 1.1B
Time: 24-48 hours
Throughput: ~3-4 samples/sec
```

### 2-Node Distributed
```
621K samples, 1 epoch, TinyLlama 1.1B
Time: 12-24 hours (2× faster)
Throughput: ~6-8 samples/sec
Speedup: 1.8-2× (not perfect 2× due to gradient sync overhead)
```

### 4-Node Distributed
```
621K samples, 1 epoch, TinyLlama 1.1B
Time: 6-12 hours (4× faster)
Throughput: ~12-16 samples/sec
Speedup: 3.5-4× (gradient sync overhead increases with more nodes)
```

### Gradient Sync Overhead

**Why not perfect 2× speedup with 2 nodes?**

Each gradient sync step:
1. Each node computes gradients locally (~1-2 sec)
2. Nodes exchange gradients over network (~0.1-0.5 sec)
3. Coordinator averages gradients (~0.05 sec)
4. Coordinator broadcasts averaged gradients (~0.1-0.5 sec)

**Total overhead per step:** ~0.25-1 sec (10-20% of step time)

**With gradient accumulation:**
- Sync every N steps instead of every step
- Reduces overhead to ~2-5%
- Use `--gradient-accumulation 4` or higher

---

## Advanced: Automatic SSH Launch

Create a script to automatically SSH and launch worker:

```bash
#!/bin/bash
# auto_launch_distributed.sh

WORKER_IP="10.9.66.250"
WORKER_USER="joker"
MASTER_IP="10.9.66.154"
DATASET="/home/joker/LlamaForge/examples/datasets/alpaca_1k.jsonl"

# Launch worker in background via SSH
ssh ${WORKER_USER}@${WORKER_IP} "cd /home/joker/LlamaForge && \
  python3 -m torch.distributed.run \
  --nproc_per_node=1 --nnodes=2 --node_rank=1 \
  --master_addr=${MASTER_IP} --master_port=29500 \
  distributed_train.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset ${DATASET} \
  --epochs 1 --batch-size 2" &

# Give worker 5 seconds to start
sleep 5

# Launch coordinator
python3 -m torch.distributed.run \
  --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=${MASTER_IP} --master_port=29500 \
  distributed_train.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset ${DATASET} \
  --epochs 1 --batch-size 2
```

**Usage:**
```bash
chmod +x auto_launch_distributed.sh
./auto_launch_distributed.sh
```

---

## Summary: Key Points

### ✅ What SOLLOL Does

1. **Auto-discovers nodes** on your network (10.9.66.x)
2. **Detects unique machines** (prevents false parallelism)
3. **Determines if parallel helps** (checks locality)
4. **No server/daemon needed** (discovery is client-side)

### ✅ What You Do

1. **Ensure Ollama runs** on all nodes (for discovery)
2. **Install PyTorch** on all nodes
3. **Copy dataset** to all nodes (or use NFS)
4. **Run PyTorch DDP** on each node with correct rank

### ✅ Why This Works

- SOLLOL handles discovery (finds 10.9.66.154 and 10.9.66.250)
- SOLLOL checks locality (verifies they're different machines)
- PyTorch DDP handles actual distributed training
- No central server needed - just peer-to-peer coordination

### ❌ Common Misconceptions

- ❌ "Need SOLLOL server running" - No, discovery is local
- ❌ "Need to install LlamaForge on all nodes" - No, just PyTorch + script
- ❌ "SOLLOL orchestrates training" - No, PyTorch DDP does that
- ❌ "Need complex setup" - No, just SSH + run command

---

## Next Steps

1. ✅ **Verify discovery works** - Run test in Step 1
2. ✅ **Check network connectivity** - Verify ports 11434 and 29500
3. ✅ **Test with small dataset** - Use alpaca_1k.jsonl first
4. ✅ **Scale to full dataset** - Once verified, use 621K dataset

**Questions?** Check `DISTRIBUTED_TRAINING_GUIDE.md` for more details on training parameters.

---

**Last Updated:** 2025-10-23
**SOLLOL Version:** 0.9.58+
**PyTorch Version:** 2.0+

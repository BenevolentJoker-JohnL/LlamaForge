# SystemD Service for Distributed Training Worker

This guide shows how to set up a persistent PyTorch worker that automatically starts on boot and waits for training jobs.

---

## Why Use SystemD?

**Without SystemD:**
- ❌ Manual SSH to worker node each time
- ❌ Worker disconnects if SSH session ends
- ❌ Need to restart after reboot
- ❌ No automatic recovery on failure

**With SystemD:**
- ✅ Worker starts automatically on boot
- ✅ Runs in background (no SSH needed)
- ✅ Auto-restarts on failure
- ✅ Centralized logging with journalctl

---

## Service Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Coordinator (10.9.66.154)                               │
│ - Runs training manually when needed                    │
│ - Connects to waiting workers via PyTorch DDP          │
└─────────────────────────────────────────────────────────┘
                          │
                          │ PyTorch DDP (port 29500)
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Worker (10.9.66.250)                                    │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ llamaforge-worker.service (SystemD)                 │ │
│ │ - Starts on boot                                    │ │
│ │ - Listens on port 29500                             │ │
│ │ - Waits for coordinator to send training job        │ │
│ │ - Auto-restarts on failure                          │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### On Worker Node (10.9.66.250)

#### 1. Copy Service File

```bash
# Copy from coordinator or create directly on worker
sudo cp llamaforge-worker.service /etc/systemd/system/

# Or create it manually
sudo nano /etc/systemd/system/llamaforge-worker.service
```

#### 2. Edit Configuration

Edit the service file to match your setup:

```bash
sudo nano /etc/systemd/system/llamaforge-worker.service
```

**Update these values:**

```ini
# User/Group (change to your username)
User=joker
Group=joker

# Working directory (where distributed_train.py is located)
WorkingDirectory=/home/joker/LlamaForge

# Master coordinator IP
Environment="MASTER_ADDR=10.9.66.154"

# This node's rank (0=coordinator, 1=first worker, 2=second worker, etc.)
Environment="NODE_RANK=1"

# Total number of nodes (including coordinator)
Environment="WORLD_SIZE=2"
```

#### 3. Create Training Script Symlink (Optional)

If your training script is in a different location:

```bash
# Make sure distributed_train.py exists
ls /home/joker/LlamaForge/distributed_train.py

# Or create symlink if it's elsewhere
ln -s /path/to/actual/distributed_train.py /home/joker/LlamaForge/distributed_train.py
```

#### 4. Enable and Start Service

```bash
# Reload systemd to pick up new service
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable llamaforge-worker.service

# Start service now
sudo systemctl start llamaforge-worker.service

# Check status
sudo systemctl status llamaforge-worker.service
```

---

## Usage

### Starting Training from Coordinator

Once the worker service is running, just launch training on coordinator:

```bash
# On coordinator (10.9.66.154)
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
  --batch-size 2
```

The worker service will automatically:
1. Detect the incoming connection
2. Join the training job
3. Start training its portion of the data

---

## Monitoring

### Check Service Status

```bash
# Basic status
sudo systemctl status llamaforge-worker

# Detailed status with recent logs
sudo systemctl status llamaforge-worker -l

# Check if running
sudo systemctl is-active llamaforge-worker
```

### View Logs

```bash
# Real-time logs
sudo journalctl -u llamaforge-worker -f

# Last 100 lines
sudo journalctl -u llamaforge-worker -n 100

# Logs from last hour
sudo journalctl -u llamaforge-worker --since "1 hour ago"

# Logs with timestamps
sudo journalctl -u llamaforge-worker -o short-precise
```

### Check Resource Usage

```bash
# CPU/Memory usage
systemctl status llamaforge-worker

# Detailed process info
ps aux | grep torch.distributed

# Monitor in real-time
htop  # Filter by "torch"
```

---

## Management Commands

### Start/Stop/Restart

```bash
# Start service
sudo systemctl start llamaforge-worker

# Stop service
sudo systemctl stop llamaforge-worker

# Restart service
sudo systemctl restart llamaforge-worker

# Reload configuration (after editing service file)
sudo systemctl daemon-reload
sudo systemctl restart llamaforge-worker
```

### Enable/Disable Auto-Start

```bash
# Enable (start on boot)
sudo systemctl enable llamaforge-worker

# Disable (don't start on boot)
sudo systemctl disable llamaforge-worker

# Check if enabled
systemctl is-enabled llamaforge-worker
```

---

## Troubleshooting

### Issue 1: Service Fails to Start

**Check logs:**
```bash
sudo journalctl -u llamaforge-worker -n 50
```

**Common causes:**

1. **Working directory doesn't exist**
   ```bash
   # Fix: Create directory
   mkdir -p /home/joker/LlamaForge
   ```

2. **Python or PyTorch not installed**
   ```bash
   # Fix: Install dependencies
   pip3 install torch transformers peft datasets
   ```

3. **Permission issues**
   ```bash
   # Fix: Ensure user owns the directory
   sudo chown -R joker:joker /home/joker/LlamaForge
   ```

4. **Port already in use**
   ```bash
   # Check what's using port 29500
   sudo lsof -i :29500

   # Kill conflicting process
   sudo kill <PID>
   ```

### Issue 2: Service Starts But Doesn't Connect

**Symptoms:**
- Service shows "active (running)"
- But coordinator can't connect

**Debug:**
```bash
# Check if service is listening
sudo netstat -tlnp | grep 29500

# Check firewall
sudo ufw status
sudo ufw allow 29500/tcp

# Test connectivity from coordinator
nc -zv 10.9.66.250 29500
```

### Issue 3: Service Crashes During Training

**Check crash logs:**
```bash
sudo journalctl -u llamaforge-worker -n 200
```

**Common causes:**

1. **Out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   Fix: Reduce batch size in training command

2. **Dataset not found**
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   Fix: Copy dataset to worker or use NFS

3. **Network timeout**
   ```
   RuntimeError: Connection timeout
   ```
   Fix: Check network connectivity, increase timeout

---

## Advanced Configuration

### Multiple Workers on Same Node

If you have multiple GPUs or want multiple CPU workers:

```ini
# In service file
Environment="NODE_RANK=1"
ExecStart=/usr/bin/python3 -m torch.distributed.run \
  --nproc_per_node=2 \  # 2 processes per node
  --nnodes=2 \
  ...
```

### Auto-Resume from Checkpoints

Add checkpoint support to automatically resume training:

```ini
# In service file
ExecStart=/usr/bin/python3 -m torch.distributed.run \
  ... \
  distributed_train.py \
  --auto-resume \
  --checkpoint-dir /home/joker/LlamaForge/work/checkpoints
```

### Custom Environment Variables

Add model-specific or training-specific config:

```ini
# In service file
Environment="TRANSFORMERS_CACHE=/mnt/cache"
Environment="HF_HOME=/mnt/huggingface"
Environment="PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
```

### Email Notifications on Failure

Get notified when service fails:

```ini
# In service file
[Service]
OnFailure=email-notification@%n.service
```

Create notification service:
```bash
sudo nano /etc/systemd/system/email-notification@.service
```

```ini
[Unit]
Description=Send email on %i failure

[Service]
Type=oneshot
ExecStart=/usr/local/bin/send-failure-email.sh %i
```

---

## Security Considerations

### 1. User Permissions

**Don't run as root:**
```ini
# ❌ NEVER DO THIS
User=root

# ✅ Use dedicated user
User=joker
```

### 2. Network Security

**Restrict port access:**
```bash
# Only allow coordinator IP
sudo ufw allow from 10.9.66.154 to any port 29500
```

### 3. Resource Limits

**Prevent resource exhaustion:**
```ini
# In service file
[Service]
# Limit memory to 80% of system
MemoryMax=80%

# Limit CPU to 8 cores
CPUQuota=800%

# Nice level (lower priority)
Nice=10
```

---

## Performance Tuning

### 1. Process Priority

```ini
# In service file
[Service]
# Real-time priority (use carefully)
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50

# Or just lower nice value
Nice=-5
```

### 2. NUMA Awareness

For multi-socket systems:

```ini
# In service file
ExecStart=/usr/bin/numactl --cpunodebind=0 --membind=0 \
  /usr/bin/python3 -m torch.distributed.run ...
```

### 3. CPU Affinity

Pin to specific cores:

```ini
# In service file
ExecStart=/usr/bin/taskset -c 0-7 \
  /usr/bin/python3 -m torch.distributed.run ...
```

---

## Comparison: SystemD vs Manual

| Feature | Manual SSH | SystemD Service |
|---------|-----------|-----------------|
| **Setup Time** | 0 min | 5 min |
| **Persistence** | No (SSH disconnect = stop) | Yes (survives reboot) |
| **Auto-Start** | No (manual each time) | Yes (on boot) |
| **Monitoring** | Check SSH terminal | `journalctl` |
| **Recovery** | Manual restart | Auto-restart |
| **Multi-Node** | SSH to each node | One `systemctl` per node |
| **Best For** | Testing, one-off jobs | Production, repeated training |

---

## Complete Example: 2-Node Setup

### On Worker (10.9.66.250)

```bash
# 1. Install service
sudo cp llamaforge-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llamaforge-worker
sudo systemctl start llamaforge-worker

# 2. Verify it's running
sudo systemctl status llamaforge-worker

# 3. Watch logs
sudo journalctl -u llamaforge-worker -f
```

### On Coordinator (10.9.66.154)

```bash
# Just launch training - worker is already waiting!
python3 -m torch.distributed.run \
  --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=10.9.66.154 --master_port=29500 \
  distributed_train.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/alpaca_1k.jsonl \
  --epochs 1
```

**That's it!** The worker service automatically:
- Detects the incoming connection
- Joins the training job
- Processes its share of the data
- Waits for next job when done

---

## Summary

### ✅ Benefits

1. **Fire-and-forget** - Worker always ready
2. **Survives reboots** - Auto-starts on boot
3. **Auto-recovery** - Restarts on failure
4. **Centralized logs** - All logs in journalctl
5. **Clean management** - Standard systemctl commands

### 📋 Quick Reference

```bash
# Install
sudo cp llamaforge-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llamaforge-worker
sudo systemctl start llamaforge-worker

# Monitor
sudo systemctl status llamaforge-worker
sudo journalctl -u llamaforge-worker -f

# Control
sudo systemctl stop llamaforge-worker
sudo systemctl restart llamaforge-worker
```

---

**Last Updated:** 2025-10-23
**Tested On:** Ubuntu 22.04, Python 3.10, PyTorch 2.0+

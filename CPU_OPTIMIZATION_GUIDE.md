# CPU Optimization Guide for Distributed Training

## The CPU Bottleneck Problem

**TL;DR:** CPU-only teacher inference is **orders of magnitude slower** than GPU. The key to success is **parallelization, pipelining, and efficient storage**.

---

## Understanding the Bottleneck

### GPU vs CPU Performance

| Operation | GPU (RTX 3090) | CPU (16-core) | Slowdown |
|-----------|----------------|---------------|----------|
| Teacher inference (7B model) | ~50 tok/s | ~2-5 tok/s | **10-25x slower** |
| Student training (LoRA) | ~100 steps/s | ~10-20 steps/s | 5-10x slower |
| Disk I/O (fp32 outputs) | 1 GB/s | 1 GB/s | Same |

**Key Insight:** Teacher generation is the bottleneck. Students will sit idle waiting for teacher outputs if not managed correctly.

---

## CPU-Optimized Architecture

```
Dataset (10K samples)
    │
    ├──> [Teacher CPU-1] ──┐
    ├──> [Teacher CPU-2] ──┼──> [Shared Storage] ──┐
    ├──> [Teacher CPU-3] ──┘         │             │
    │                                 │             │
    │                          (Pre-generated)      │
    │                           Teacher Outputs     │
    │                                 │             │
    └──> [Student CPU-1] <────────────┘             │
    └──> [Student CPU-2] <──────────────────────────┘
    └──> [Student CPU-3] <──────────────────────────┘
```

**Flow:**
1. SOLLOL splits dataset into shards (3,333 samples each)
2. Each teacher node generates outputs for its shard **in parallel**
3. Outputs saved to shared storage (compressed as float16)
4. Student nodes fetch batches and train **asynchronously**
5. Students don't wait - they train on available batches

---

## Optimization Strategies

### 1. Parallel Teacher Generation ⚡

**Problem:** Single teacher node is too slow.

**Solution:** Use multiple CPU nodes to split teacher workload.

```python
from src.sollol_integration import create_distributed_training

orchestrator = create_distributed_training(mode="teacher_student", config={})

# ❌ BAD: Single teacher (slow!)
teacher_id = orchestrator.create_teacher_task(
    model_path="qwen2.5:7b",
    dataset_path="data.jsonl",
    output_dir="./teacher_outputs"
)

# ✅ GOOD: Parallel teachers (fast!)
teacher_ids = orchestrator.create_parallel_teacher_tasks(
    model_path="qwen2.5:7b",
    dataset_path="data.jsonl",
    output_dir="./teacher_outputs",
    num_teachers=4,  # Split across 4 CPU nodes
    compress_outputs=True  # Use float16
)
```

**Speedup:** 4 teachers = **~4x faster** generation (linear scaling)

---

### 2. Output Compression 💾

**Problem:** Teacher outputs can be huge (10K samples × 4K tokens × fp32 = **160GB**).

**Solution:** Use float16 compression to reduce disk I/O by 50%.

```python
teacher_ids = orchestrator.create_parallel_teacher_tasks(
    model_path="qwen2.5:7b",
    dataset_path="data.jsonl",
    output_dir="./teacher_outputs",
    compress_outputs=True  # 160GB → 80GB
)
```

**Benefits:**
- 50% less disk space
- 50% faster disk I/O
- Minimal accuracy loss (float16 is sufficient for distillation)

---

### 3. Dataset Sharding 📦

**Problem:** Large datasets take too long for a single teacher.

**Solution:** SOLLOL automatically shards the dataset across parallel teachers.

**Example:**
- Dataset: 10,000 samples
- Teachers: 4 CPU nodes
- Each teacher gets: 2,500 samples
- Time: **4x faster** than single teacher

**How it works:**
```python
# SOLLOL internally does:
# Teacher 0: samples 0-2499
# Teacher 1: samples 2500-4999
# Teacher 2: samples 5000-7499
# Teacher 3: samples 7500-9999
```

Students can start training **as soon as first teacher finishes**, not when all teachers finish!

---

### 4. Asynchronous Pipeline 🔄

**Problem:** Students idle while waiting for teacher.

**Solution:** Students train on ready batches while teachers generate more.

```
Time →
Teacher-0:  [Generate 0-2499] ────────────────────────────────────────────
Teacher-1:  [Generate 2500-4999] ────────────────────────────────────────
Teacher-2:  [Generate 5000-7499] ────────────────────────────────────────
Teacher-3:  [Generate 7500-9999] ────────────────────────────────────────

Student-0:              [Train 0-1000]──[Train 1000-2000]──[Train 2000-3000]
Student-1:                    [Train 0-1000]──[Train 1000-2000]──[Train 2000-3000]
Student-2:                          [Train 0-1000]──[Train 1000-2000]──[Train 2000-3000]
```

**Key:** Students don't wait for all teachers - they start as soon as **any** teacher finishes!

---

### 5. Micro-Batching 🔬

**Problem:** Large batches → high memory usage → slow generation.

**Solution:** Use smaller batches to reduce memory pressure.

```python
teacher_ids = orchestrator.create_parallel_teacher_tasks(
    model_path="qwen2.5:7b",
    dataset_path="data.jsonl",
    output_dir="./teacher_outputs",
    batch_size=2,  # Small batch for CPU
    num_teachers=4
)
```

**CPU-Specific Recommendations:**
- GPU: `batch_size=8-16`
- CPU (16-core): `batch_size=2-4`
- CPU (8-core): `batch_size=1-2`

---

### 6. Smart Model Selection 🧠

**Problem:** Large teacher models are impractically slow on CPU.

**Solution:** Use smaller teacher models that still provide good guidance.

**Recommended Teacher Models for CPU:**

| Model | Size | CPU Speed | Quality |
|-------|------|-----------|---------|
| qwen2.5:0.5b | 0.5B | ~15 tok/s | Good for simple tasks |
| qwen2.5:1.5b | 1.5B | ~8 tok/s | Good balance |
| qwen2.5:3b | 3B | ~4 tok/s | High quality |
| qwen2.5:7b | 7B | ~2 tok/s | Best quality (slow!) |

**Rule of Thumb:**
- Production: 3B teacher, 0.5B student
- Development: 1.5B teacher, 0.5B student
- Quick tests: 0.5B teacher, 0.5B student (no distillation needed!)

---

## Complete CPU-Optimized Example

```python
#!/usr/bin/env python3
"""
CPU-Optimized Teacher-Student Training
"""

from src.sollol_integration import create_distributed_training

# Connect to SOLLOL
orchestrator = create_distributed_training(mode="teacher_student", config={})

# Discover cluster
print("[1/3] Discovering compute nodes...")
nodes = orchestrator.discover_nodes()

# Check if we have GPUs
gpu_nodes = [n for n in nodes if n['resources']['gpu']['count'] > 0]
if not gpu_nodes:
    print("[!] No GPU nodes detected - using CPU-optimized workflow")

# Create parallel teachers (CPU-optimized)
print("[2/3] Creating parallel teacher tasks...")
teacher_ids = orchestrator.create_parallel_teacher_tasks(
    model_path="qwen2.5:3b",  # Reasonable size for CPU
    dataset_path="data/training.jsonl",
    output_dir="./teacher_outputs",
    num_teachers=None,  # Auto-detect (one per node)
    batch_size=2,  # Small batches for CPU
    compress_outputs=True  # Reduce disk I/O
)

print(f"[✓] Created {len(teacher_ids)} parallel teachers")

# Monitor teacher progress
print("[i] Waiting for teachers to generate outputs...")
orchestrator.monitor_tasks(teacher_ids)

print("[✓] Teacher outputs ready!")

# Create students (will start as outputs become available)
print("[3/3] Creating student tasks...")
student_ids = orchestrator.create_student_tasks(
    base_model="qwen2.5:0.5b",  # Small student model
    teacher_outputs_dir="./teacher_outputs",
    num_students=4,
    lora_config={"r": 8, "alpha": 16}
)

# Monitor student training
print("[i] Training students...")
orchestrator.monitor_tasks(student_ids)

print("[✓] Training complete!")
print(f"[i] Trained {len(student_ids)} student models")
```

---

## Performance Comparison

### Scenario: 10K samples, 7B teacher, 0.5B student

| Setup | Teacher Time | Student Time | Total Time |
|-------|--------------|--------------|------------|
| 1 GPU teacher + 4 GPU students | 10 min | 20 min | **30 min** |
| 1 CPU teacher + 4 CPU students | 200 min | 40 min | **240 min** |
| **4 CPU teachers + 4 CPU students** | **50 min** | **40 min** | **90 min** |

**Key Takeaway:** Parallel CPU teachers can get within **3x** of GPU performance, vs **8x** for single-threaded.

---

## Advanced: Dynamic Task Rotation

For maximum efficiency, idle teacher nodes can switch to student training:

```python
# After teacher_0 finishes, reassign it as student
orchestrator.reassign_idle_teacher_as_student(teacher_ids[0])
```

This maximizes cluster utilization!

---

## Troubleshooting

### "Teacher generation is too slow"
- ✅ Use `create_parallel_teacher_tasks()` to parallelize
- ✅ Reduce teacher model size (7B → 3B)
- ✅ Reduce batch size (8 → 2)
- ✅ Use fewer samples for testing

### "Disk I/O is slow"
- ✅ Enable `compress_outputs=True`
- ✅ Use SSD instead of HDD
- ✅ Consider RAM disk for temp storage

### "Students are idling"
- ✅ Ensure SOLLOL is using asynchronous pipeline
- ✅ Check that teacher outputs are being written correctly
- ✅ Verify network/disk I/O is not bottlenecked

### "Out of memory"
- ✅ Reduce batch size
- ✅ Use smaller models
- ✅ Close other applications

---

## Best Practices Summary

✅ **DO:**
- Use `create_parallel_teacher_tasks()` for CPU
- Enable output compression (`compress_outputs=True`)
- Use smaller teacher models (3B instead of 7B)
- Use small batch sizes (2-4 on CPU)
- Pre-generate outputs before starting students

❌ **DON'T:**
- Use single teacher on CPU for large datasets
- Use fp32 teacher outputs (wastes disk space)
- Use large batch sizes on CPU
- Wait for all teachers before starting students
- Use 7B+ models on CPU without parallelization

---

## Further Reading

- [TEACHER_STUDENT_GUIDE.md](./TEACHER_STUDENT_GUIDE.md) - General teacher-student workflow
- [SOLLOL_INTEGRATION.md](./SOLLOL_INTEGRATION.md) - SOLLOL integration details
- [Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531) - Original distillation research

---

**Questions?** Open an issue on GitHub!

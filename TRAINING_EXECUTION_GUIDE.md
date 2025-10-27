# 🚀 Training Execution Guide - 3M to Production

## ✅ What You Have Ready NOW

### Dataset
- **File:** `examples/datasets/ultimate_3M_intelligently_duplicated.jsonl`
- **Size:** 5.0 GB, 3,000,000 examples
- **Quality:** Deduped (removed 227K duplicates) + intelligently weighted
- **Composition:**
  - 32.8% Code (985K) - 1.8× amplified
  - 48.2% Instruction (1.45M) - 1.4× amplified
  - 14.6% Safety (437K) - 3.0× amplified ✅
  - 4.4% Other (CoT, tool-use, creative, analytical)

### Tools
- ✅ `create_3m_with_duplication.py` - Intelligent duplication manager
- ✅ `train_and_eval.py` - Complete training + evaluation pipeline
- ✅ `launch_distributed_training.py` - SOLLOL multi-node support
- ✅ `evaluate_model.py` - Comprehensive benchmarking
- ✅ `training_phases.json` - Phase 1, 1.5, and 2 configs

---

## 🎯 Recommended Path: Start with Phase 1 (3M)

### Quick Start (Single Machine)

```bash
# Option 1: CPU training (7-10 days)
python train_and_eval.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/ultimate_3M_intelligently_duplicated.jsonl \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --eval-baseline \
  --eval-reasoning \
  --work-dir ./work/phase1_3M

# Option 2: GPU training (2-3 days, requires T4 or better)
python train_and_eval.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/ultimate_3M_intelligently_duplicated.jsonl \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --eval-baseline \
  --eval-reasoning \
  --work-dir ./work/phase1_3M
```

### Distributed Training (SOLLOL - RECOMMENDED)

**4x faster than single machine!**

```bash
# Step 1: Start SOLLOL coordinator (on main node)
cd ~/SOLLOL
python -m sollol.server --host 0.0.0.0 --port 8765 &

# Step 2: Connect workers (on each worker node)
python -m sollol.worker --coordinator <COORDINATOR_IP>:8765

# Step 3: Launch distributed training (from main node)
cd ~/LlamaForge
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/ultimate_3M_intelligently_duplicated.jsonl \
  --workers 4 \
  --epochs 1 \
  --batch-size 2 \
  --work-dir ./work/phase1_3M_distributed

# Expected time: 12-18 hours (4 workers) or 6-10 hours (8 workers)
```

---

## 📊 After Training: Evaluation & Analysis

### Automatic Evaluation (if you used train_and_eval.py)

Results will be in `./work/phase1_3M/evaluation/`:
- `baseline/` - Pre-training scores
- `after_training/` - Post-training scores
- Comparison showing improvements

### Manual Evaluation

```bash
# Standard benchmarks
python evaluate_model.py \
  --model ./work/phase1_3M/merged_model \
  --preset comprehensive \
  --limit 100 \
  --output ./eval/phase1_results

# Reasoning-focused
python evaluate_model.py \
  --model ./work/phase1_3M/merged_model \
  --reasoning \
  --reasoning-preset all_reasoning \
  --limit 100

# Compare to baseline
python evaluate_model.py \
  --model ./work/phase1_3M/merged_model \
  --preset standard \
  --compare ./eval/phase1_results/baseline/results.json
```

### Expected Improvements (Phase 1 with 3M dataset)

| Metric | Baseline | After Training | Improvement |
|--------|----------|----------------|-------------|
| HumanEval (code) | 5-10% | 20-30% | +200-400% |
| GSM8K (math) | 8-12% | 35-50% | +350% |
| CoT Reasoning | 10-15% | 75-85% | +600% |
| HellaSwag | 42-48% | 62-70% | +45% |
| Safety Compliance | - | 80-90% | NEW |

---

## 🔄 Decision Tree: What's Next?

### If Phase 1 Results are EXCELLENT (>80% of targets):
✅ **Deploy and use!**
- Convert to GGUF: `python convert_to_gguf.py`
- Serve with Ollama: `ollama create my-model -f Modelfile`
- Optional: Try DPO/RLHF for further alignment

### If Phase 1 Results are GOOD (60-80% of targets):
⚖️ **Consider Phase 1.5 (3.6M) for targeted boost**

Identify weak domains from eval:
- Low code scores? → Boost code ratio to 2.0×
- Weak tool use? → Boost tool_use to 2.5×
- Safety issues? → Boost red_team to 3.5×

```bash
# Adjust ratios in create_3m_with_duplication.py
# Change DUPLICATION_RATIOS dict, then:
python create_3m_with_duplication.py \
  --input examples/datasets/*.jsonl \
  --output examples/datasets/ultimate_3.6M_phase1.5.jsonl \
  --target-size 3600000

# Retrain with adjusted corpus
python train_and_eval.py \
  --model ./work/phase1_3M/merged_model \
  --data examples/datasets/ultimate_3.6M_phase1.5.jsonl \
  --epochs 1 \
  --work-dir ./work/phase1.5_3.6M
```

### If Phase 1 Shows GAPS (<60% of targets):
📈 **Build Phase 2 (4.5M) with adjusted strategy**

```bash
# Use Phase 2 ratios from training_phases.json:
# code: 2.2×, tool_use: 2.5×, red_team: 3.5×, persona: 2.8×

# Edit create_3m_with_duplication.py DUPLICATION_RATIOS
# Or create phase2 config file, then:

python create_3m_with_duplication.py \
  --input examples/datasets/*.jsonl \
  --output examples/datasets/ultimate_4.5M_phase2.jsonl \
  --target-size 4500000

# Train from scratch or continue from Phase 1
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/ultimate_4.5M_phase2.jsonl \
  --workers 4 \
  --epochs 1 \
  --work-dir ./work/phase2_4.5M
```

---

## 🛠️ Troubleshooting

### Out of Memory
```bash
# Reduce batch size and increase gradient accumulation
--batch-size 1 --gradient-accumulation 16

# Use aggressive memory optimizations
--cpu-only  # If training on CPU
```

### Training Too Slow
```bash
# Use distributed training with SOLLOL (4-8x speedup)
python launch_distributed_training.py --workers 4

# Or reduce dataset size for faster iteration
head -n 1000000 ultimate_3M_intelligently_duplicated.jsonl > test_1M.jsonl
```

### Downloads Still Running
```bash
# Check background processes
ps aux | grep python

# Wait for critical downloads (OpenOrca 500K completed already)
# Dolphin Coder still downloading - can start training without it
```

### Model Conversion Issues
```bash
# Merge LoRA adapter back to base model
python merge_lora.py \
  --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter ./work/phase1_3M/lora_adapter \
  --output ./work/phase1_3M/merged_model

# Convert to GGUF for Ollama
python convert_to_gguf.py \
  --input ./work/phase1_3M/merged_model \
  --output ./models/phase1_3M.gguf \
  --quantize q4_k_m
```

---

## 📈 Timeline Estimates

### Phase 1 (3M dataset)
- **Download time:** ✅ Complete (3M corpus ready)
- **Training (4-worker SOLLOL):** 12-18 hours
- **Evaluation:** 30-60 minutes
- **Total:** ~18-24 hours to first results

### Phase 2 (4.5M dataset) - If needed
- **Dataset generation:** 30-60 minutes
- **Training (4-worker SOLLOL):** 18-24 hours
- **Evaluation:** 30-60 minutes
- **Total:** ~24-36 hours

---

## 🎯 Success Checklist

- [x] 3M training corpus created with intelligent duplication
- [x] Training tools ready (train_and_eval.py, distributed support)
- [x] Evaluation suite configured
- [x] Phase 1/1.5/2 scaling strategy defined
- [ ] **NEXT: Start Phase 1 training!**

---

## 🚀 Recommended Command (Start NOW)

```bash
# If you have 4+ worker nodes available (RECOMMENDED):
python launch_distributed_training.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset examples/datasets/ultimate_3M_intelligently_duplicated.jsonl \
  --workers 4 \
  --epochs 1 \
  --work-dir ./work/phase1_3M_distributed

# Or single machine:
python train_and_eval.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/ultimate_3M_intelligently_duplicated.jsonl \
  --epochs 1 \
  --eval-baseline \
  --eval-reasoning \
  --work-dir ./work/phase1_3M
```

**Estimated completion:** 12-18 hours (distributed) or 2-3 days (GPU) or 7-10 days (CPU)

---

## 📚 Reference Files

- `training_phases.json` - Phase configurations and decision criteria
- `ULTIMATE_3M_GUIDE.md` - Complete 3M dataset guide
- `README_COMPLETE_SYSTEM.md` - System overview
- `QUICK_START_COT_TRAINING.md` - 5-minute quick start

**Everything is ready to go. Start training and let the model learn!** 🎉

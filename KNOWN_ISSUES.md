# Known Issues and Limitations

**Last Updated:** 2024-10-26
**Version:** 0.1.0-alpha

This document honestly documents known limitations, edge cases, and areas for improvement. We believe in transparency over marketing hype.

---

## Table of Contents

- [Maturity & Production Readiness](#maturity--production-readiness)
- [Memory & Resource Issues](#memory--resource-issues)
- [Training Quality Issues](#training-quality-issues)
- [Model Compatibility](#model-compatibility)
- [Distributed Training Limitations](#distributed-training-limitations)
- [Test Coverage & CI/CD](#test-coverage--cicd)
- [Roadmap](#roadmap)

---

## Maturity & Production Readiness

### Current Status: **Alpha (v0.1.0)**

**What works well:**
- ✅ Core LoRA fine-tuning (tested on 1B-7B models)
- ✅ GGUF conversion for Ollama
- ✅ Interactive wizard for ease of use
- ✅ CPU and GPU automatic detection
- ✅ Distributed training with SOLLOL (basic)

**What needs improvement:**
- ⚠️ **Limited testing:** Only tested by 1-2 developers on specific hardware
- ⚠️ **No automated tests:** Manual testing only
- ⚠️ **No CI/CD:** No continuous integration or quality gates
- ⚠️ **Documentation gaps:** Some edge cases not documented
- ⚠️ **No production deployments:** Unproven at scale

**Risks:**
- **Early adopter risk:** You're using experimental software
- **Memory issues:** OOM errors possible during merge step
- **Dataset quality:** Small datasets produce broken models
- **Limited support:** Community-only (no SLA)

---

## Memory & Resource Issues

### Issue 1: Out-of-Memory During Merge Step

**Problem:** Training completes successfully, but process gets killed (exit code 137) during LoRA adapter merge.

**Root Cause:** Merging LoRA adapters into base model requires loading both in memory simultaneously.

**Memory Requirements:**

| Model Size | Training RAM | Merge RAM | Total RAM Needed |
|------------|-------------|-----------|------------------|
| 1-3B | 6-8 GB | 4-6 GB | **12-14 GB** |
| 7B | 12-16 GB | 8-10 GB | **20-24 GB** |
| 13B | 24-32 GB | 16-20 GB | **40-48 GB** |

**Your System:**
- RAM: 15.3 GB total
- Training 7B model: ~12-16 GB used
- Merge step: Needs additional 8-10 GB
- **Result:** OOM kill (exit code 137)

**Workarounds:**

**Option 1: Skip GGUF conversion**
```bash
python llamaforge.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/alpaca_1k.jsonl \
  --no-gguf  # Saves in HuggingFace format only
```

**Option 2: Add swap space**
```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Option 3: Use smaller models**
- TinyLlama-1.1B: Works reliably on 16GB systems
- Qwen2.5-3B: Works on 16GB with `--no-gguf`
- Llama-7B: Requires 24GB+ RAM

**Option 4: Reduce sequence length**
```bash
--max-length 256  # Instead of 512
```

**Planned fix:** v0.2.0 will add streaming merge to reduce peak memory usage.

---

### Issue 2: GPU VRAM Exhaustion

**Problem:** Training hangs or crashes with CUDA out-of-memory errors.

**Root Cause:** Model + gradients + optimizer states exceed GPU VRAM.

**VRAM Requirements:**

| Model Size | LoRA (r=8) | LoRA (r=16) | Full Fine-Tuning |
|------------|------------|-------------|------------------|
| 1-3B | 6-8 GB | 8-10 GB | 12-16 GB |
| 7B | 12-16 GB | 16-20 GB | 24-32 GB |
| 13B | 24-28 GB | 28-32 GB | 40-48 GB |

**Workarounds:**

**Option 1: Reduce batch size**
```bash
--batch-size 1
--gradient-accumulation 8  # Effective batch size = 8
```

**Option 2: Reduce LoRA rank**
```bash
--lora-r 4  # Instead of 8 or 16
```

**Option 3: Reduce sequence length**
```bash
--max-length 256  # Instead of 512
```

**Option 4: Use gradient checkpointing (automatic)**
- LlamaForge enables this by default
- Trades compute for memory (25% slower, 40% less VRAM)

---

## Training Quality Issues

### Issue 3: Small Datasets Produce Broken Models

**Problem:** Models trained on <100 samples output incoherent garbage.

**Example:**
```
Input: "Hello, what are you?"
Output: "戥系统戥 Cosby戥 Cosby戥 Cosby..."
```

**Root Cause:** Severe overfitting to tiny dataset, loss of general language understanding.

**Minimum Dataset Sizes:**

| Samples | Result | Use Case |
|---------|--------|----------|
| <100 | ❌ Broken garbage | **Do not use** |
| 100-500 | ⚠️ Minimal learning | Emergency only |
| 500-1000 | ✅ Basic fine-tuning | Quick experiments |
| 1K-5K | ✅ Good quality | Standard use |
| 5K-10K | ✅✅ High quality | Production |
| 10K+ | ✅✅✅ Best quality | Professional |

**Solution:** Use provided datasets:
- `examples/datasets/alpaca_1k.jsonl` - 1,000 samples (minimum recommended)
- `examples/datasets/alpaca_full.jsonl` - 52,000 samples
- `examples/datasets/code_alpaca_full.jsonl` - 20,000 coding samples

**Detection:** If model outputs gibberish after training, you used too few samples.

---

### Issue 4: Training Loss Not Improving

**Problem:** Loss stays flat or increases during training.

**Common Causes:**

1. **Learning rate too high**
   - Default: 2e-4
   - Try: 1e-4 or 5e-5
   - Symptoms: Loss increases or oscillates wildly

2. **Learning rate too low**
   - Try: 3e-4 or 5e-4
   - Symptoms: Loss decreases very slowly or not at all

3. **Dataset formatting issues**
   - Check that `instruction`/`prompt` and `output`/`completion` fields exist
   - Verify samples are not empty or malformed

4. **Model already optimal**
   - Base model may already be well-suited to your task
   - Further training may not help

**Debugging:**
```bash
# Enable loss logging
python llamaforge.py ... --logging-steps 10
# Watch for loss trends
```

---

## Model Compatibility

### Issue 5: Limited Architecture Support

**Tested and Working:**
- ✅ Llama 2 (7B, 13B)
- ✅ Llama 3 / 3.1 (8B)
- ✅ Mistral (7B)
- ✅ CodeLlama (7B, 13B)
- ✅ Qwen 2.5 (1.5B, 3B, 7B)
- ✅ TinyLlama (1.1B)

**Untested / May Not Work:**
- ⚠️ Falcon models
- ⚠️ GPT-NeoX
- ⚠️ BLOOM
- ⚠️ Gemma (may require special handling)
- ⚠️ Mixtral (MoE architecture)

**Known Incompatible:**
- ❌ GPT-2 / GPT-J (different architecture)
- ❌ BERT / RoBERTa (encoder-only models)
- ❌ T5 (encoder-decoder)

**If you encounter errors:**
1. Check model architecture is causal LM (decoder-only)
2. Verify model is available on HuggingFace
3. Try a known-working model first to verify setup

---

### Issue 6: Ollama GGUF Models Cannot Be Used Directly

**Problem:** You cannot train on Q4-quantized Ollama models.

**Why:** Q4 quantization is lossy - precision is permanently discarded during quantization.

**What LlamaForge Does:**
1. Detects your Ollama model (e.g., `qwen2.5:3b`)
2. Maps to base HuggingFace model (`Qwen/Qwen2.5-3B`)
3. Downloads FP16 weights if not cached
4. Trains on FP16 weights
5. Exports to GGUF for Ollama

**First-time Download:**
- 3B model: ~6-8 GB download
- 7B model: ~14-16 GB download
- Subsequent runs use cached version (instant, no download)

**See:** [TECHNICAL_REALITY.md](TECHNICAL_REALITY.md) for detailed explanation.

---

## Distributed Training Limitations

### Issue 7: Manual Worker Setup Required

**Current State:**
- ❌ Workers must be set up manually
- ❌ No automatic deployment to remote nodes
- ❌ Must copy dataset to each node
- ❌ Must SSH into each node to launch training

**Requirements:**
- ✅ Ollama running on all nodes (for discovery)
- ✅ PyTorch installed on all nodes
- ✅ SSH access to remote nodes
- ✅ Dataset accessible on all nodes
- ✅ Port 29500 open for PyTorch DDP

**Planned:** v0.2.0 will add automatic worker deployment.

---

### Issue 8: No Fault Tolerance

**Problem:** If one node fails mid-training, entire job fails.

**Current Behavior:**
- Node crashes → Training stops
- Network partition → Training hangs
- No automatic recovery

**Workarounds:**
- Run training in `tmux` or `screen` sessions
- Use `systemd` service for persistence (see [SYSTEMD_SERVICE_SETUP.md](SYSTEMD_SERVICE_SETUP.md))

**Planned:** v0.2.0 will add checkpoint recovery.

---

## Test Coverage & CI/CD

### Issue 9: No Automated Tests

**Current State:**
- ❌ No unit tests
- ❌ No integration tests
- ❌ No CI/CD pipeline
- ❌ Manual testing only

**Risks:**
- Changes may break existing functionality
- No quality gates before merge
- No automated regression testing

**Test Directories Exist But:**
- `test_8bit/` - Manual test runs (not automated)
- `test_work/` - Manual test output (not test suite)

**Planned:** v0.2.0 will add basic test suite and GitHub Actions CI.

---

### Issue 10: No Code Coverage Tracking

**Current State:**
- ❌ No coverage reports
- ❌ Unknown which code paths are tested
- ❌ No coverage trends over time

**Planned:** v0.2.0 will add pytest-cov and codecov integration.

---

## Roadmap

### v0.2.0 (Target: 2024-Q4)

**Focus: Stability & Testing**

- [ ] Add pytest test suite (target: 60% coverage)
- [ ] Add GitHub Actions CI/CD
- [ ] Streaming merge to reduce memory usage
- [ ] Checkpoint recovery for distributed training
- [ ] Better error messages and validation
- [ ] Automatic worker deployment

---

### v0.3.0 (Target: 2025-Q1)

**Focus: Features & Scale**

- [ ] Support for larger models (13B+)
- [ ] QLoRA support (4-bit training)
- [ ] Multi-GPU training on single node
- [ ] Web UI for training monitoring
- [ ] Prometheus metrics export
- [ ] Better dataset validation

---

### v0.4.0 (Target: 2025-Q2)

**Focus: Production Readiness**

- [ ] Fault tolerance and recovery
- [ ] Production-grade distributed training
- [ ] Model registry integration
- [ ] Experiment tracking (W&B, MLflow)
- [ ] 90%+ test coverage
- [ ] Performance benchmarks

---

## Contributing

Found an issue not listed here? Please report it:
https://github.com/BenevolentJoker-JohnL/LlamaForge/issues

Want to help fix these issues?
- Check issues labeled `good-first-issue`
- Read [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon)

---

## Honest Assessment

**Is LlamaForge production-ready?**

**For hobbyists / researchers:** Yes, with caveats. Great for learning LoRA and distributed training.

**For small teams (<5 people):** Maybe. Test thoroughly first. Have 16GB+ RAM per training node.

**For enterprises / production:** Not yet. Wait for v0.3.0+ or contribute fixes.

**Why use it anyway?**
- ✅ **Learning tool:** Excellent for understanding LoRA, distributed training, GGUF
- ✅ **Ollama integration:** Seamless workflow from training to deployment
- ✅ **SOLLOL distributed:** Unique integration for multi-node training
- ✅ **Honest documentation:** Clear about what works and what doesn't

**Why wait?**
- ⚠️ **Memory issues:** OOM possible on <24GB RAM systems
- ⚠️ **No tests:** Limited quality assurance
- ⚠️ **Manual setup:** Distributed training requires manual node configuration
- ⚠️ **Alpha software:** Expect rough edges

---

**Bottom line:** LlamaForge is a powerful learning tool for distributed LoRA fine-tuning. It works well if you understand the limitations and have adequate hardware. Not production-ready, but valuable for research and experimentation.

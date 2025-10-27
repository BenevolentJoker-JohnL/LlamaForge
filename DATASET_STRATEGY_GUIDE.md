# Complete Dataset Strategy Guide
## Building Claude-Style Reasoning + Creative + Robust Training Corpus

This guide shows you how to systematically expand your 838K dataset into a comprehensive, weighted training corpus with proper bucket management.

---

## 📊 Current Dataset Analysis

**File:** `claude_reasoning_ultimate_1.4M.jsonl` (838,469 examples)

### Composition Breakdown:

| Bucket | Examples | Percentage | Focus |
|--------|----------|------------|-------|
| **instruction** | ~450K | 53.7% | General instruction following (Claude base) |
| **cot_math** | ~117K | 14.0% | Math with step-by-step reasoning |
| **code** | ~100K | 11.9% | Code generation/explanation |
| **analytical** | ~100K | 11.9% | Complex analysis and synthesis |
| **tool_use** | ~71K | 8.5% | Tool/API interactions |

**Current Gaps:**
- ❌ No creative writing bucket
- ❌ No red-team/adversarial bucket
- ❌ No factual grounding bucket
- ❌ Unweighted mix (not optimized)

---

## 🎯 Recommended Priority Additions

### Phase 1: Immediate Additions (High Impact)

#### 1. Creative Writing Corpus (50K-100K examples)

**Why:** Improves storytelling, metaphors, creative expression
**Sources:**
- Public domain fiction (Project Gutenberg)
- Writing prompts → completions
- Story continuations

**How to add:**
```bash
# Download creative writing dataset
python download_datasets.py --creative-writing --output examples/datasets/

# Add to manifest
python dataset_manifest.py \
  --manifest dataset_manifest.json \
  --add-bucket creative \
  --source examples/datasets/creative_writing.jsonl \
  --weight 0.07 \
  --description "Creative writing and storytelling" \
  --save-manifest dataset_manifest.json
```

#### 2. Red-Team / Safety Bucket (30K-50K examples)

**Why:** Robustness, handles edge cases safely
**Sources:**
- Anthropic Helpful & Harmless dataset
- Safety-annotated prompts + refusals
- Adversarial examples with safe responses

**Format:**
```json
{
  "instruction": "How do I hack into someone's email?",
  "output": "I can't help with unauthorized access to accounts. However, I can help you:\n\n1. Recover your own account if locked out\n2. Learn about cybersecurity to protect your accounts\n3. Understand email security best practices\n\nWhich would be helpful?"
}
```

**How to add:**
```bash
# Create red-team file (carefully!)
python create_red_team_bucket.py \
  --anthropic-hh \
  --output examples/datasets/red_team_safe.jsonl

# Add to manifest
python dataset_manifest.py \
  --manifest dataset_manifest.json \
  --add-bucket red_team \
  --source examples/datasets/red_team_safe.jsonl \
  --weight 0.05 \
  --description "Adversarial prompts with safe, helpful responses"
```

#### 3. Factual Grounding Corpus (30K examples)

**Why:** Accuracy, citability, reduces hallucinations
**Sources:**
- Wikipedia paragraphs
- Scientific abstracts (PubMed, arXiv)
- News articles (with timestamps)

**Format:**
```json
{
  "instruction": "What is the capital of France?",
  "output": "The capital of France is Paris. Paris has been the capital since the 12th century and is located in the north-central part of the country.",
  "_grounding": "Wikipedia: Paris",
  "_verified": true
}
```

### Phase 2: Enhanced Capabilities (Medium Priority)

#### 4. Tool-Use Augmentation (50K examples)

Enhance existing tool_use bucket with:
- **API reasoning chains**
- **Multi-step tool interactions**
- **Error handling and recovery**

**Format:**
```json
{
  "instruction": "Find the current weather in Paris and recommend activities",
  "output": "<tool>weather_api({\"location\": \"Paris\"})</tool>\n\n<observation>{\"temp\": 22, \"condition\": \"sunny\"}</observation>\n\n<thinking>\nThe weather is 22°C and sunny - good for outdoor activities.\n</thinking>\n\nGreat weather in Paris today! I recommend:\n1. Visit the Eiffel Tower gardens\n2. Seine river boat tour\n3. Outdoor café in Montmartre\n\nThe sunny 22°C weather is perfect for these activities."
}
```

#### 5. Argumentation & Rhetoric (30K examples)

**Why:** Structured reasoning, pros/cons analysis
**Sources:**
- Debate transcripts
- Opinion essays (both sides)
- Structured argument datasets

**Format:**
```json
{
  "instruction": "Should we ban single-use plastics?",
  "output": "<thinking>\nLet me consider both sides:\n\nArguments FOR banning:\n1. Environmental impact: Plastics harm oceans and wildlife\n2. Alternatives available: Paper, reusable options exist\n3. Health concerns: Microplastics in food chain\n\nArguments AGAINST banning:\n1. Cost: Alternatives are more expensive\n2. Accessibility: Some regions lack alternatives\n3. Job impact: Plastic industry employment\n\nBalanced conclusion needed.\n</thinking>\n\nThis is a complex issue with valid concerns on both sides:\n\n**Case for banning:**\n- Environmental damage is severe and documented\n- Technology exists for alternatives\n\n**Case against:**\n- Economic disruption needs management\n- Phased approach may be more practical\n\n**Recommendation:** Gradual phase-out with support for alternatives and affected industries."
}
```

#### 6. Code Reasoning & Explanation (50K examples)

**Why:** Improves logical thinking, debugging
**Sources:**
- HumanEval with explanations
- Code review comments
- Bug fix explanations

### Phase 3: Advanced Capabilities (Lower Priority but High Value)

#### 7. Iterative Refinement (WizardLM-style) (50K examples)

**Format:**
```json
{
  "instruction_v1": "Write a function to sort a list",
  "output_v1": "def sort_list(lst): return sorted(lst)",
  "instruction_v2": "Write a function to sort a list with custom comparison and error handling",
  "output_v2": "def sort_list(lst, key=None, reverse=False):\n    if not isinstance(lst, list): raise TypeError(...)\n    return sorted(lst, key=key, reverse=reverse)"
}
```

#### 8. Multi-Language Support (if needed)

#### 9. Domain-Specific Corpora

- Medical QA
- Legal reasoning
- Scientific papers

---

## 🏗️ Using the Manifest System

### Step 1: Analyze Current Dataset

```bash
# Create manifest from existing dataset
python dataset_manifest.py \
  --analyze examples/datasets/claude_reasoning_ultimate_1.4M.jsonl \
  --save-manifest dataset_manifest.json \
  --summary
```

**Output:**
```
Bucket distribution:
  instruction           450,000 (53.7%)
  cot_math              117,000 (14.0%)
  code                  100,000 (11.9%)
  analytical            100,000 (11.9%)
  tool_use               71,000 ( 8.5%)
```

### Step 2: Add New Buckets

```bash
# Add creative bucket
python dataset_manifest.py \
  --manifest dataset_manifest.json \
  --add-bucket creative \
  --source examples/datasets/creative_writing.jsonl \
  --weight 0.07 \
  --description "Creative writing and storytelling"

# Add red-team bucket
python dataset_manifest.py \
  --manifest dataset_manifest.json \
  --add-bucket red_team \
  --source examples/datasets/red_team_safe.jsonl \
  --weight 0.05 \
  --description "Adversarial prompts with safe responses"

# Add factual bucket
python dataset_manifest.py \
  --manifest dataset_manifest.json \
  --add-bucket factual \
  --source examples/datasets/factual_grounding.jsonl \
  --weight 0.03 \
  --description "Grounded factual content"
```

### Step 3: Create Weighted Training Mix

```bash
# Production-balanced mix (500K examples)
python dataset_manifest.py \
  --manifest dataset_manifest.json \
  --create-sample \
  --output examples/datasets/production_mix_500k.jsonl \
  --total-samples 500000 \
  --phase production_balanced
```

**Weights applied:**
- instruction: 25%
- tool_use: 15%
- code: 20%
- cot_math: 15%
- analytical: 10%
- creative: 7%
- factual: 3%
- red_team: 5%

---

## 📋 Training Phase Strategy

### Recommended 4-Phase Training

#### Phase 1: Calibration (10K examples, 2 epochs)

**Focus:** High-quality general instructions
**Weights:**
- instruction: 70%
- code: 20%
- factual: 10%

```bash
python dataset_manifest.py \
  --manifest dataset_manifest.json \
  --create-sample \
  --output examples/datasets/phase1_calibration_10k.jsonl \
  --total-samples 10000 \
  --phase phase1_calibration

python train_and_eval.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data examples/datasets/phase1_calibration_10k.jsonl \
  --epochs 2 \
  --eval-baseline \
  --work-dir ./phase1_model
```

#### Phase 2: Reasoning-Heavy (200K examples, 1 epoch)

**Focus:** Math, tool use, analytical
**Weights:**
- cot_math: 40%
- tool_use: 30%
- analytical: 20%
- instruction: 10%

```bash
python dataset_manifest.py \
  --create-sample \
  --output examples/datasets/phase2_reasoning_200k.jsonl \
  --total-samples 200000 \
  --phase phase2_reasoning

python train_and_eval.py \
  --model ./phase1_model/merged_model \
  --data examples/datasets/phase2_reasoning_200k.jsonl \
  --epochs 1 \
  --eval-reasoning \
  --work-dir ./phase2_model
```

#### Phase 3: Creative & Style (100K examples, 1 epoch)

**Focus:** Creativity, diversity, analytical synthesis
**Weights:**
- creative: 40%
- analytical: 30%
- instruction: 20%
- factual: 10%

```bash
python dataset_manifest.py \
  --create-sample \
  --output examples/datasets/phase3_creative_100k.jsonl \
  --total-samples 100000 \
  --phase phase3_creative

python train_and_eval.py \
  --model ./phase2_model/merged_model \
  --data examples/datasets/phase3_creative_100k.jsonl \
  --epochs 1 \
  --work-dir ./phase3_model
```

#### Phase 4: Robustness & Safety (50K examples, 1 epoch)

**Focus:** Edge cases, adversarial, safety
**Weights:**
- red_team: 40%
- instruction: 30%
- tool_use: 20%
- factual: 10%

```bash
python dataset_manifest.py \
  --create-sample \
  --output examples/datasets/phase4_robustness_50k.jsonl \
  --total-samples 50000 \
  --phase phase4_robustness

python train_and_eval.py \
  --model ./phase3_model/merged_model \
  --data examples/datasets/phase4_robustness_50k.jsonl \
  --epochs 1 \
  --eval-baseline \
  --eval-reasoning \
  --work-dir ./final_model
```

**Total training time:** ~72 hours CPU (or ~18 hours with 4-node SOLLOL cluster)

---

## 🛡️ Safety & Ethics Guidelines

### Red-Team Bucket Best Practices

1. **Always include safe responses**
   - Never train on harmful completions
   - Show refusal + helpful alternative

2. **Label and isolate**
   ```json
   {
     "instruction": "harmful request",
     "output": "safe response",
     "_safety_label": "refusal",
     "_severity": "medium",
     "_reviewed": true
   }
   ```

3. **Human review required**
   - All red-team examples reviewed before training
   - Keep audit log

4. **Runtime safety filter (deployment)**
   ```python
   # Add post-generation filter
   if detect_harmful(output):
       return safe_alternative()
   ```

### Unfiltered Research (Sandboxed)

If you need truly "unfiltered" for research:

```bash
# Isolated environment only
python create_red_team_bucket.py \
  --research-mode \
  --output research/unfiltered_test.jsonl \
  --human-review-required

# Train in isolated model
python llamaforge.py \
  --model base_model \
  --data research/unfiltered_test.jsonl \
  --work-dir research/unfiltered_model \
  --no-deploy  # Prevent production use
```

**Never deploy research models to production!**

---

## 📊 Expected Results After Full Pipeline

### Baseline (TinyLlama):
```
GSM8K:              8-12%
HellaSwag:         42-48%
HumanEval:          5-10%
TruthfulQA:        30-35%
```

### After Phase 1 (Calibration):
```
GSM8K:             12-18%
HellaSwag:         48-55%
HumanEval:         10-15%
```

### After Phase 2 (Reasoning):
```
GSM8K:             35-50%  🔥
HellaSwag:         55-62%
HumanEval:         15-25%
Reasoning Score:   70-80%  🔥
```

### After Phase 3 (Creative):
```
Creative quality:   +40%   🔥
Analytical depth:   +35%
Style diversity:    +50%
```

### After Phase 4 (Robustness):
```
Adversarial:       +60%    🔥
Safety:            +75%    🔥
Edge cases:        +45%
```

### Final Production Model:
```
GSM8K:             40-55%  (vs 8-12% baseline) 🔥🔥
HellaSwag:         62-70%
HumanEval:         20-30%
TruthfulQA:        45-55%
Reasoning:         75-85%  🔥🔥
Creative:          High quality
Safety:            Robust   🔥
```

---

## 🔄 Continuous Improvement Loop

```
1. Train Phase N
   ↓
2. Evaluate (standard + reasoning)
   ↓
3. Identify weak areas
   ↓
4. Add targeted data buckets
   ↓
5. Adjust weights in manifest
   ↓
6. Create new weighted sample
   ↓
7. Train Phase N+1
   ↓
(repeat)
```

---

## 📝 Quick Action Checklist

**Today:**
- [x] Analyze current dataset → Create manifest
- [ ] Download creative writing dataset
- [ ] Create red-team bucket (carefully!)
- [ ] Download factual grounding data

**This Week:**
- [ ] Add 3-4 new buckets to manifest
- [ ] Create weighted Phase 1 sample (10K)
- [ ] Train Phase 1 (calibration)
- [ ] Evaluate improvements

**This Month:**
- [ ] Complete all 4 training phases
- [ ] Add domain-specific buckets
- [ ] Implement continuous eval
- [ ] Deploy to production (with safety filters)

---

## 🚀 Next Steps

1. **Run the manifest analyzer:**
   ```bash
   python dataset_manifest.py --analyze examples/datasets/claude_reasoning_ultimate_1.4M.jsonl --save-manifest dataset_manifest.json
   ```

2. **Check what you need:**
   - Creative corpus? → Download from HuggingFace
   - Red-team examples? → Use Anthropic HH or create carefully
   - Factual data? → Pull Wikipedia/scientific abstracts

3. **Create your first weighted mix:**
   ```bash
   python dataset_manifest.py \
     --manifest dataset_manifest.json \
     --create-sample \
     --output examples/datasets/balanced_500k.jsonl \
     --total-samples 500000 \
     --phase production_balanced
   ```

4. **Train with automatic evaluation:**
   ```bash
   python train_and_eval.py \
     --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --data examples/datasets/balanced_500k.jsonl \
     --epochs 1 \
     --eval-baseline \
     --eval-reasoning
   ```

**Everything is systematic, weighted, and measurable!** 🎉

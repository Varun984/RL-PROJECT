# Quick Start: Training Your HRL-SARP Model

## 🎯 Goal
Get your first AI agent trained on real market data in the next 30 minutes.

---

## ✅ Prerequisites Check

Before starting, make sure you have:
- [x] Python 3.11.9 virtual environment activated
- [x] All dependencies installed (`pip install -r requirements.txt`)
- [x] Data collected (54,236 rows validated ✓)
- [x] Located in project root directory

---

## 🚀 Step-by-Step Training

### Step 1: Test Your Setup (2 minutes)

Run the test script to verify everything is ready:

```bash
python test_training_setup.py
```

**Expected output**:
```
✅ PASS - Training Imports
✅ PASS - Agent Initialization  
✅ PASS - Data Loader
🎉 All tests passed! You're ready to run training.
```

If any test fails, check the error message and fix before proceeding.

---

### Step 2: Train Phase 1 - Macro Agent (10-30 minutes)

Train the Macro agent to detect market regimes and allocate across sectors:

```bash
python main.py train --phase 1
```

**What happens**:
1. Loads 2015-2022 training data (sector returns, macro indicators)
2. Initializes Macro agent neural network
3. Trains for 50 epochs with supervised learning
4. Saves best model to `logs/pretrain_macro/best_pretrain_macro.pt`

**Watch for**:
- Loss should decrease from ~1.0 to ~0.2
- Regime accuracy should reach 60-70%
- Training time: 10-30 minutes on CPU, 2-5 minutes on GPU

**Example output**:
```
2026-02-19 20:30:00 | INFO | ✓ MacroAgent initialized
2026-02-19 20:30:05 | INFO | ✓ Training data loaded: 1500 samples
2026-02-19 20:30:10 | INFO | Epoch 10/50 | regime_loss=0.8234 | alloc_loss=0.0156
2026-02-19 20:30:20 | INFO | Epoch 20/50 | regime_loss=0.6891 | alloc_loss=0.0098
...
2026-02-19 20:31:00 | INFO | ✓ Macro pre-training complete: best_loss=0.1845
```

---

### Step 3: Train Phase 2 - Micro Agent (15-45 minutes)

Train the Micro agent to select individual stocks:

```bash
python main.py train --phase 2
```

**What happens**:
1. Loads stock-level data (returns, features for 50+ stocks)
2. Initializes Micro agent neural network
3. Trains for 30 epochs with supervised learning
4. Saves best model to `logs/pretrain_micro/best_pretrain_micro.pt`

**Watch for**:
- Loss should decrease from ~0.01 to ~0.001
- Portfolio returns should be positive
- Training time: 15-45 minutes on CPU, 3-8 minutes on GPU

---

### Step 4: Verify Training Results

Check that checkpoints were saved:

```bash
ls logs/pretrain_macro/
ls logs/pretrain_micro/
```

You should see:
- `best_pretrain_macro.pt` - Trained Macro agent
- `best_pretrain_micro.pt` - Trained Micro agent
- `pretrain_*_metrics.json` - Training metrics

---

## 📊 Understanding the Output

### Training Logs

Logs are saved to:
- `logs/train_YYYYMMDD_HHMMSS.log` - Main training log
- `logs/pretrain_macro/` - Phase 1 specific logs
- `logs/pretrain_micro/` - Phase 2 specific logs

### Metrics

**Phase 1 (Macro)**:
- `regime_loss`: How well it predicts Bull/Bear/Sideways (lower is better)
- `alloc_loss`: How well it allocates across sectors (lower is better)
- `regime_accuracy`: % of correct regime predictions (higher is better)

**Phase 2 (Micro)**:
- `loss`: How well it selects stocks (lower is better)
- `portfolio_return`: Average return of selected portfolio (higher is better)

### Checkpoints

Checkpoint files contain:
- Trained neural network weights
- Optimizer state
- Training metrics
- Epoch number

You can load these later for evaluation or further training.

---

## 🐛 Common Issues & Solutions

### Issue 1: "No sector data found"
**Cause**: Data not collected or in wrong location

**Solution**:
```bash
# Check if data exists
ls data/raw/market/

# If empty, collect data
python main.py collect-data --start 2015-01-01 --end 2023-12-31
```

---

### Issue 2: "CUDA out of memory"
**Cause**: GPU doesn't have enough memory

**Solution**: Reduce batch size in config:
```yaml
# Edit config/macro_agent_config.yaml
training:
  pretrain_batch_size: 32  # Reduce from 64
```

Or train on CPU (slower but works):
```bash
# Training will automatically use CPU if CUDA unavailable
python main.py train --phase 1
```

---

### Issue 3: Training is very slow
**Possible causes**:
1. Running on CPU (expected - 10-30 min per phase)
2. Large dataset (expected - more data = longer training)
3. Disk I/O bottleneck (data loading from slow drive)

**Solutions**:
- Use GPU if available (5-10x faster)
- Reduce data range for testing:
  ```yaml
  # Edit config/data_config.yaml
  dates:
    train_start: "2020-01-01"  # Reduce from 2015
  ```
- Use SSD instead of HDD for data storage

---

### Issue 4: Loss not decreasing
**Possible causes**:
1. Learning rate too high/low
2. Data quality issues
3. Feature scaling problems

**Solutions**:
1. Check data quality:
   ```bash
   python scripts/validate_data.py
   ```

2. Adjust learning rate in config:
   ```yaml
   # config/macro_agent_config.yaml
   training:
     pretrain_lr: 0.0001  # Try lower if loss explodes
   ```

3. Check logs for warnings about NaN or Inf values

---

## 📈 What Success Looks Like

### Phase 1 Success ✅
```
Epoch 50/50 | regime_loss=0.6234 | alloc_loss=0.0089 | total=0.1933
✓ Macro pre-training complete: best_loss=0.1845
```

**Good signs**:
- Regime loss < 0.8 (better than random guessing)
- Allocation loss < 0.02
- Regime accuracy > 55%

### Phase 2 Success ✅
```
Epoch 30/30 | loss=0.000823 | port_return=0.0012
✓ Micro pre-training complete: best_loss=0.000756
```

**Good signs**:
- Loss < 0.001
- Portfolio return > 0 (positive on average)
- Validation loss similar to training loss

---

## 🎯 Next Steps After Training

### 1. Inspect Training Metrics
```python
import json

# Load metrics
with open('logs/pretrain_macro/pretrain_macro_metrics.json') as f:
    metrics = json.load(f)

# Check final values
print(f"Final regime accuracy: {metrics['pretrain/val_regime_accuracy'][-1]:.2%}")
print(f"Final loss: {metrics['pretrain/total_loss'][-1]:.4f}")
```

### 2. Visualize Training Progress
```python
import matplotlib.pyplot as plt
import json

with open('logs/pretrain_macro/pretrain_macro_metrics.json') as f:
    metrics = json.load(f)

plt.plot(metrics['pretrain/total_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Macro Agent Training Loss')
plt.savefig('training_loss.png')
```

### 3. Test the Trained Agent
```python
import torch
from agents.macro_agent import MacroAgent

# Load trained agent
agent = MacroAgent(config_path="config/macro_agent_config.yaml", device=torch.device("cpu"))
checkpoint = torch.load("logs/pretrain_macro/best_pretrain_macro.pt")
agent.actor.load_state_dict(checkpoint['models']['actor'])

# Test on sample data
import numpy as np
macro_state = np.random.randn(1, 18).astype(np.float32)
sector_emb = np.random.randn(1, 11, 64).astype(np.float32)

sector_weights, regime_logits, _ = agent.actor(
    torch.tensor(macro_state),
    torch.tensor(sector_emb)
)

print("Predicted sector weights:", sector_weights.detach().numpy())
print("Predicted regime:", regime_logits.argmax().item())  # 0=Bear, 1=Bull, 2=Sideways
```

### 4. Implement Phases 3-5 (RL Training)
The supervised pre-training (Phases 1-2) is complete. Next:
- Phase 3: RL training of Macro agent (with frozen Micro)
- Phase 4: RL training of Micro agent (with frozen Macro)
- Phase 5: Joint fine-tuning of both agents

These require environment implementation (see `IMPLEMENTATION_STATUS.md`).

---

## 💡 Pro Tips

1. **Start with small data**: Test with 1-2 years first, then scale up
2. **Monitor GPU usage**: Use `nvidia-smi` to check GPU utilization
3. **Save checkpoints**: Training can be interrupted and resumed
4. **Check logs regularly**: Catch issues early
5. **Validate data first**: Run `validate_data.py` before training

---

## 📞 Getting Help

If you encounter issues:

1. **Check logs**: `logs/train_*.log` has detailed error messages
2. **Run diagnostics**: `python test_training_setup.py`
3. **Validate data**: `python scripts/validate_data.py`
4. **Check configs**: Make sure YAML files are valid
5. **Review error messages**: Most errors are self-explanatory

---

## 🎉 Congratulations!

If you've completed Phases 1 and 2, you now have:
- ✅ A trained Macro agent that can detect market regimes
- ✅ A trained Micro agent that can select stocks
- ✅ Saved checkpoints for future use
- ✅ Training metrics for analysis

This is a major milestone! Your agents have learned from 7+ years of real market data.

**What's different from before**:
- Before: Training completed in 1 second (stubs)
- Now: Training takes 25-75 minutes (real learning)
- Before: No checkpoints saved
- Now: Trained models saved and ready to use

**Next challenge**: Implement RL training (Phases 3-5) to make agents truly adaptive! 🚀

---

## 📚 Additional Resources

- `PROJECT_EXPLANATION.md` - Understand the full system
- `IMPLEMENTATION_STATUS.md` - See what's done and what's next
- `TRAINING_IMPLEMENTATION_PROGRESS.md` - Technical details
- `DATA_COLLECTION_GUIDE.md` - Data collection reference

---

**Ready to train? Run this now:**
```bash
python main.py train --phase 1
```

Good luck! 🎯

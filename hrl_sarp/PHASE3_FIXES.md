# Phase 3 Training Fixes & Improvements

This document captures all fixes and improvements made to get Phase 3 (RL training of Macro agent with frozen Micro) running successfully.

---

## 1. Checkpoint Loading (KeyError: 'models')

**Problem**: Phase 3 expected checkpoints in format `checkpoint['models']['actor']`, but Phase 1 and 2 save via `save_checkpoint()` which uses `checkpoint["model_actor"]`.

**Fix** (`training/train_macro_frozen_micro.py`):
- Added support for both checkpoint formats
- Check for `model_actor` first (Phase 1/2 format), then fall back to `models.actor` (legacy)

```python
if "model_actor" in checkpoint:
    macro_agent.actor.load_state_dict(checkpoint["model_actor"])
elif "models" in checkpoint and "actor" in checkpoint["models"]:
    macro_agent.actor.load_state_dict(checkpoint["models"]["actor"])
else:
    raise KeyError(f"Checkpoint missing actor weights. Keys: {list(checkpoint.keys())}")
```

---

## 2. Config Path Handling

**Problem**: Relative paths like `config/macro_agent_config.yaml` failed when running from different directories.

**Fix** (`training/train_macro_frozen_micro.py`):
- Added `_get_project_root()` helper
- All config, checkpoint, and log paths now use absolute paths based on `hrl_sarp` package location

```python
def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

root = _get_project_root()
macro_config = os.path.join(root, "config", "macro_agent_config.yaml")
# ... etc for all paths
```

---

## 3. Data Loader Path Resolution

**Problem**: `TrainingDataLoader` used hardcoded `data/raw` which could point to wrong location.

**Fix** (`data/training_data_loader.py`):
- When config_path is absolute, derive `data_dir` from it: `project_root/data/raw`

```python
if os.path.isabs(config_path):
    project_root = os.path.dirname(os.path.dirname(config_path))
    self.data_dir = os.path.join(project_root, "data", "raw")
else:
    self.data_dir = "data/raw"
```

---

## 4. Variable Name Collision (TypeError: expected str, bytes or os.PathLike)

**Problem**: `data_config` was overwritten—first set to config file path, then to loaded config dict—so `TrainingDataLoader(config_path=data_config)` received a dict.

**Fix** (`training/train_macro_frozen_micro.py`):
- Renamed: `data_config_path` (file path) vs `data_cfg` (loaded dict)

---

## 5. JSON Serialization (TypeError: Object of type float32 is not JSON serializable)

**Problem**: Metrics contained `numpy.float32` values which `json.dump()` cannot serialize.

**Fix** (`training/trainer_utils.py`):
- **`MetricsTracker.update()`**: Convert values to native Python `float` when appending
- **`MetricsTracker.save()`**: Added `_to_serializable()` to recursively convert numpy types before `json.dump()`
- Handles: `np.float32`, `np.float64`, `np.int32`, `np.int64`, `np.ndarray`, nested dicts/lists

---

## 6. Value Loss Explosion (Training Instability)

**Problem**: Value loss reached trillions (e.g. 37672769290240), episode returns in millions (81,449,592). Root cause: when drawdown was near zero, Calmar ratio divided by `1e-8`, exploding rewards.

**Fix** (`environment/base_env.py`):
- `get_calmar()`: Use minimum drawdown of **1%** instead of `1e-8` to prevent division explosion

```python
max_dd = max(self.current_drawdown, 0.01)  # was: 1e-8
```

**Fix** (`environment/reward_functions.py`):
- **`portfolio_calmar_reward()`**: Clip Calmar ratio to `[-20, 20]`
- **`compute_total_macro_reward()`**: Clip final reward to `[-10, 10]` to keep value targets stable

---

## 7. Unicode Encoding (Windows)

**Problem**: Log messages with Unicode characters (→, ❌) caused `UnicodeEncodeError` on Windows (cp1252).

**Fix**:
- `utils/common.py`: Replaced `→` with `->`
- `main.py`: Removed `❌` from error message

---

## Files Modified

| File | Changes |
|------|---------|
| `training/train_macro_frozen_micro.py` | Checkpoint loading, path handling, variable naming |
| `data/training_data_loader.py` | Data dir resolution from config path |
| `training/trainer_utils.py` | JSON serialization for numpy types |
| `environment/base_env.py` | Calmar ratio drawdown floor |
| `environment/reward_functions.py` | Calmar clip, reward clip |
| `utils/common.py` | Unicode fix in logging |
| `main.py` | Unicode fix in error message |

---

## How to Run Phase 3

```powershell
cd c:\Users\HP\Desktop\Projects\RL-PROJECT\hrl_sarp
# Activate venv first
python main.py train --phase 3
```

**Prerequisites**: Phase 1 and Phase 2 must be completed (checkpoints in `logs/pretrain_macro/` and `logs/pretrain_micro/`).

---

## Expected Behavior After Fixes

- Value loss in reasonable range (not trillions)
- Episode returns in reasonable range (not millions)
- Metrics saved successfully to `logs/phase3_macro/phase3_macro_metrics.json`
- Training completes without JSON or encoding errors

#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# HRL-SARP: Full Training Pipeline
# Runs all 5 phases sequentially with checkpointing
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════"
echo "  HRL-SARP Full Training Pipeline"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════"

# Configuration
SEED=${SEED:-42}
LOG_LEVEL=${LOG_LEVEL:-INFO}
PYTHON=${PYTHON:-python}

echo ""
echo "Configuration:"
echo "  Seed:      $SEED"
echo "  Log Level: $LOG_LEVEL"
echo "  Python:    $PYTHON"
echo ""

# Phase 1: Macro pre-training
echo "──────────────────────────────────────────────────────"
echo "Phase 1/5: Supervised pre-training of Macro agent"
echo "──────────────────────────────────────────────────────"
$PYTHON main.py train --phase 1 --seed $SEED --log-level $LOG_LEVEL
echo "✅ Phase 1 complete"
echo ""

# Phase 2: Micro pre-training
echo "──────────────────────────────────────────────────────"
echo "Phase 2/5: Supervised pre-training of Micro agent"
echo "──────────────────────────────────────────────────────"
$PYTHON main.py train --phase 2 --seed $SEED --log-level $LOG_LEVEL
echo "✅ Phase 2 complete"
echo ""

# Phase 3: Macro RL with frozen Micro
echo "──────────────────────────────────────────────────────"
echo "Phase 3/5: RL training of Macro (frozen Micro)"
echo "──────────────────────────────────────────────────────"
$PYTHON main.py train --phase 3 --seed $SEED --log-level $LOG_LEVEL
echo "✅ Phase 3 complete"
echo ""

# Phase 4: Micro RL with frozen Macro
echo "──────────────────────────────────────────────────────"
echo "Phase 4/5: RL training of Micro (frozen Macro)"
echo "──────────────────────────────────────────────────────"
$PYTHON main.py train --phase 4 --seed $SEED --log-level $LOG_LEVEL
echo "✅ Phase 4 complete"
echo ""

# Phase 5: Joint fine-tuning
echo "──────────────────────────────────────────────────────"
echo "Phase 5/5: Joint fine-tuning"
echo "──────────────────────────────────────────────────────"
$PYTHON main.py train --phase 5 --seed $SEED --log-level $LOG_LEVEL
echo "✅ Phase 5 complete"
echo ""

echo "═══════════════════════════════════════════════════════"
echo "  All 5 training phases completed successfully!"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════"

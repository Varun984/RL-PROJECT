#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# HRL-SARP: Walk-Forward Backtest
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════"
echo "  HRL-SARP Walk-Forward Backtest"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════"

SEED=${SEED:-42}
CAPITAL=${CAPITAL:-10000000}
EPISODES=${EPISODES:-5}
PYTHON=${PYTHON:-python}

echo "  Capital:   ₹$(printf "%'d" $CAPITAL)"
echo "  Episodes:  $EPISODES"
echo ""

$PYTHON main.py backtest --seed $SEED --capital $CAPITAL --n-episodes $EPISODES

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Backtest complete — see results/ directory"
echo "═══════════════════════════════════════════════════════"

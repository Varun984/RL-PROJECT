#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# HRL-SARP: Full Evaluation Pipeline
# Runs backtest + benchmark comparison + stress testing + report
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════"
echo "  HRL-SARP Full Evaluation Pipeline"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════"

SEED=${SEED:-42}
CAPITAL=${CAPITAL:-10000000}
OUTPUT_DIR=${OUTPUT_DIR:-results}
PYTHON=${PYTHON:-python}

echo "  Capital:    ₹$(printf "%'d" $CAPITAL)"
echo "  Output:     $OUTPUT_DIR"
echo ""

# Run evaluation
$PYTHON main.py evaluate --seed $SEED --capital $CAPITAL --output-dir $OUTPUT_DIR

# Run stress test
echo ""
echo "Running stress tests..."
$PYTHON main.py stress-test --capital $CAPITAL

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Evaluation complete"
echo "  Results:  $OUTPUT_DIR/evaluation_results.json"
echo "  Report:   $OUTPUT_DIR/reports/evaluation_report.md"
echo "═══════════════════════════════════════════════════════"

#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# HRL-SARP: Launch Streamlit Dashboard
# ═══════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════"
echo "  HRL-SARP Monitoring Dashboard"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════"

PORT=${PORT:-8501}
PYTHON=${PYTHON:-python}

echo "  Dashboard URL: http://localhost:$PORT"
echo ""

$PYTHON main.py dashboard --port $PORT

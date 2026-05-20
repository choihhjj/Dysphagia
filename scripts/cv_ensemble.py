"""Standalone script to ensemble test predictions across folds.

Usage: python scripts/cv_ensemble.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_ensemble import save_ensemble

# ── paths (edit before running) ──────────────────────────────────────────────
OUTPUT_BASE = 'outputs/feature_only'
N_FOLDS     = 5
THRESHOLD   = 0.5
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_ensemble(OUTPUT_BASE, N_FOLDS, THRESHOLD)

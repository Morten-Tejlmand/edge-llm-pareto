#!/usr/bin/env python3
"""Score a fixed answer dataset with exact match and corpus BLEU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edge_llm_pareto.quality import main

if __name__ == '__main__':
    raise SystemExit(main())

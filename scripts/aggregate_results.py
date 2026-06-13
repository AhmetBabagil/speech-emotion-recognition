"""Aggregate every outputs/<exp>/test_summary.json into one results table.

Writes ``outputs/results.csv`` and ``outputs/results.md`` for the report.

    python scripts/aggregate_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()
    root = Path(args.outputs)

    rows = []
    for summary in sorted(root.glob("*/test_summary.json")):
        with open(summary, encoding="utf-8") as f:
            d = json.load(f)
        rows.append({
            "experiment": summary.parent.name,
            "accuracy": round(d.get("accuracy", float("nan")), 4),
            "balanced_accuracy": round(d.get("balanced_accuracy", float("nan")), 4),
            "macro_f1": round(d.get("macro_f1", float("nan")), 4),
            "weighted_f1": round(d.get("weighted_f1", float("nan")), 4),
        })

    if not rows:
        print(f"No test_summary.json found under {root}/. Run some experiments first.")
        return

    df = pd.DataFrame(rows).sort_values("experiment")
    df.to_csv(root / "results.csv", index=False)
    try:
        md = df.to_markdown(index=False)  # needs `tabulate`
    except ImportError:
        md = df.to_string(index=False)
    (root / "results.md").write_text(md, encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nWrote {root/'results.csv'} and {root/'results.md'}")


if __name__ == "__main__":
    main()

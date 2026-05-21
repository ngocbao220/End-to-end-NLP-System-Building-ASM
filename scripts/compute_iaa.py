from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute simple exact agreement for the annotation subset.")
    parser.add_argument("--input", type=Path, default=Path("data/annotations/iaa_subset.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/iaa.json"))
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("IAA input is empty")
    agreements = [int(row["agreement"]) for row in rows]
    result = {
        "items": len(rows),
        "exact_agreement": sum(agreements) / len(agreements),
        "agreements": sum(agreements),
        "disagreements": len(agreements) - sum(agreements),
    }
    print(json.dumps(result, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

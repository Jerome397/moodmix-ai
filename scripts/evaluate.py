"""Run fixed paired development cases. Hosted requests require --live."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.catalog import load_catalog
from backend.evaluation import CASES, catalog_digest, evaluate_case
from backend.generation import llm_status


def run(live=False):
    tracks = load_catalog()
    rows = []
    for case in CASES:
        for mode in (["baseline", "llm"] if live else ["baseline"]):
            row = evaluate_case(tracks, case, mode)
            rows.append(row)
            print(case["case_id"], mode, row["actual_method"], flush=True)
    return {"schema_version": 1, "suite": "development-v1", "purpose": "Development comparison, not held-out quality evidence",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "live": live,
            "catalog_sha256": catalog_digest(tracks), "provider": llm_status() if live else None,
            "human_ratings": None, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Up to 12 hosted requests; configured project quota/billing applies")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results.json")
    args = parser.parse_args()
    result = run(args.live)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

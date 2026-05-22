"""Pick 20 random paragraphs from a measurement run, print each with its score,
prompt the user for human agreement. If <16/20 match, the dim needs prompt work."""
import json
import random
import sys
from pathlib import Path


def main(run_dir: str) -> None:
    scores_path = Path(run_dir) / "scores.json"
    data = json.loads(scores_path.read_text())
    paragraph_scores = [
        s for s in data["scores"]
        if s["unit_id"].startswith("paragraph:") and not s.get("derived") and s.get("score_0_100") is not None
    ]
    random.seed(42)
    sample = random.sample(paragraph_scores, min(20, len(paragraph_scores)))
    agree = 0
    for i, s in enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}] {s['unit_id']} - {s['dim_name']}")
        print(f"  Judge: {s['label']} ({s['score_0_100']:.0f})")
        print(f"  Reasoning: {s['reasoning']}")
        if s.get("evidence_refs"):
            print(f"  Refs: {s['evidence_refs']}")
        ans = input("  Agree? [y/n/skip]: ").strip().lower()
        if ans == "y":
            agree += 1
    total = len(sample)
    print(f"\nAgreement: {agree}/{total}")
    print("Calibration", "PASSED" if agree >= total * 0.8 else "FAILED (revise dim prompts)")


if __name__ == "__main__":
    main(sys.argv[1])

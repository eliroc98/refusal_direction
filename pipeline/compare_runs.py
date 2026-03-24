"""
Compare new pipeline run results against gold runs.

Usage:
    python compare_runs.py --model_alias meta-llama-3-8b-instruct [--tolerance 0.05]
    python compare_runs.py --all [--tolerance 0.05]

New runs are read from:   pipeline/runs/<model_alias>/completions/
Gold runs are read from:  pipeline/gold_runs/<model_alias>/completions/

Exit code 0 = all checks pass, 1 = discrepancies found.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

PIPELINE_DIR = os.path.dirname(os.path.realpath(__file__))
GOLD_DIR = os.path.join(PIPELINE_DIR, "gold_runs")
RUNS_DIR = os.path.join(PIPELINE_DIR, "runs")

import glob as _glob

# Core evaluation files that are always present
_CORE_EVALUATION_FILES = [
    "jailbreakbench_baseline_evaluations.json",
    "jailbreakbench_ablation_evaluations.json",
    "harmless_baseline_evaluations.json",
]

def _get_evaluation_files(gold_completions_dir: str) -> list:
    """Return evaluation file names, including dynamically discovered actadd_c* files."""
    files = list(_CORE_EVALUATION_FILES)
    # Discover actadd coefficient sweep files from the gold directory
    for pattern in ["jailbreakbench_actadd_c*_evaluations.json",
                    "jailbreakbench_inlp_actadd_c*_evaluations.json",
                    "harmless_actadd_c*_evaluations.json",
                    "harmless_inlp_actadd_c*_evaluations.json"]:
        found = _glob.glob(os.path.join(gold_completions_dir, pattern))
        files.extend(os.path.basename(f) for f in sorted(found))
    return files

SCALAR_METRICS = [
    "substring_matching_success_rate",
    "llamaguard2_success_rate",
    "harmbench_success_rate",
]

PER_CATEGORY_METRICS = [
    "substring_matching_per_category",
    "llamaguard2_per_category",
    "harmbench_per_category",
]


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def compare_scalar(key: str, gold_val: float, new_val: float, tolerance: float) -> Tuple[bool, str]:
    diff = abs(new_val - gold_val)
    ok = diff <= tolerance
    status = "OK" if ok else "FAIL"
    return ok, f"  [{status}] {key}: gold={gold_val:.4f}, new={new_val:.4f}, diff={diff:.4f} (tol={tolerance})"


def compare_per_category(key: str, gold_dict: dict, new_dict: dict, tolerance: float) -> Tuple[bool, List[str]]:
    lines = []
    all_ok = True
    gold_cats = set(gold_dict.keys())
    new_cats = set(new_dict.keys())

    missing = gold_cats - new_cats
    extra = new_cats - gold_cats
    if missing:
        lines.append(f"  [WARN] {key}: missing categories in new run: {sorted(missing)}")
        all_ok = False
    if extra:
        lines.append(f"  [WARN] {key}: extra categories in new run: {sorted(extra)}")

    for cat in sorted(gold_cats & new_cats):
        ok, msg = compare_scalar(f"{key}[{cat}]", gold_dict[cat], new_dict[cat], tolerance)
        lines.append(msg)
        if not ok:
            all_ok = False

    return all_ok, lines


def compare_per_item_labels(gold_completions: list, new_completions: list) -> List[str]:
    """Compare per-item jailbreak labels between gold and new completions."""
    lines = []

    # Build prompt -> labels index for gold
    gold_by_prompt = {c["prompt"]: c for c in gold_completions}
    new_by_prompt = {c["prompt"]: c for c in new_completions}

    label_keys = [k for k in next(iter(gold_by_prompt.values())).keys() if k.startswith("is_jailbreak_")]

    for lk in label_keys:
        mismatches = 0
        total = 0
        for prompt, gold_c in gold_by_prompt.items():
            if prompt not in new_by_prompt:
                continue
            new_c = new_by_prompt[prompt]
            if lk not in gold_c or lk not in new_c:
                continue
            total += 1
            if gold_c[lk] != new_c[lk]:
                mismatches += 1

        if total == 0:
            continue
        pct = mismatches / total * 100
        status = "OK" if mismatches == 0 else "WARN"
        lines.append(f"  [{status}] per-item {lk}: {mismatches}/{total} mismatches ({pct:.1f}%)")

    return lines


def compare_evaluation_file(
    gold_path: str,
    new_path: str,
    tolerance: float,
    check_per_item: bool = True,
) -> Tuple[bool, List[str]]:
    gold = load_json(gold_path)
    new = load_json(new_path)

    if gold is None:
        return True, [f"  [SKIP] Gold file not found: {gold_path}"]
    if new is None:
        return False, [f"  [MISSING] New file not found: {new_path}"]

    lines = []
    all_ok = True

    # --- Scalar metrics ---
    for key in SCALAR_METRICS:
        if key not in gold:
            continue
        if key not in new:
            lines.append(f"  [MISSING] Metric '{key}' absent in new run")
            all_ok = False
            continue
        ok, msg = compare_scalar(key, gold[key], new[key], tolerance)
        lines.append(msg)
        if not ok:
            all_ok = False

    # --- Per-category metrics ---
    for key in PER_CATEGORY_METRICS:
        if key not in gold:
            continue
        if key not in new:
            lines.append(f"  [MISSING] Metric '{key}' absent in new run")
            all_ok = False
            continue
        ok, cat_lines = compare_per_category(key, gold[key], new[key], tolerance)
        lines.extend(cat_lines)
        if not ok:
            all_ok = False

    # --- Per-item label agreement ---
    if check_per_item and "completions" in gold and "completions" in new:
        item_lines = compare_per_item_labels(gold["completions"], new["completions"])
        lines.extend(item_lines)

    return all_ok, lines


def compare_model(model_alias: str, tolerance: float, check_per_item: bool) -> bool:
    gold_completions_dir = os.path.join(GOLD_DIR, model_alias, "completions")
    new_completions_dir = os.path.join(RUNS_DIR, model_alias, "completions")

    print(f"\n{'='*60}")
    print(f"Model: {model_alias}")
    print(f"  Gold: {gold_completions_dir}")
    print(f"  New:  {new_completions_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(gold_completions_dir):
        print(f"  [ERROR] Gold completions directory not found.")
        return False

    if not os.path.isdir(new_completions_dir):
        print(f"  [ERROR] New completions directory not found. Has the pipeline run yet?")
        return False

    eval_files = _get_evaluation_files(gold_completions_dir)
    model_ok = True
    for fname in eval_files:
        gold_path = os.path.join(gold_completions_dir, fname)
        new_path = os.path.join(new_completions_dir, fname)

        ok, lines = compare_evaluation_file(gold_path, new_path, tolerance, check_per_item)
        if not ok:
            model_ok = False

        if lines:
            print(f"\n  -- {fname} --")
            for line in lines:
                print(line)

    if model_ok:
        print("\n  [PASS] All metrics within tolerance.")
    else:
        print("\n  [FAIL] Some metrics outside tolerance or missing.")

    return model_ok


def main():
    parser = argparse.ArgumentParser(description="Compare new pipeline runs against gold runs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_alias", type=str, help="Model alias to compare (e.g. meta-llama-3-8b-instruct)")
    group.add_argument("--all", action="store_true", help="Compare all models present in gold_runs/")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Max allowed absolute difference in success rates (default: 0.05)")
    parser.add_argument("--no_per_item", action="store_true",
                        help="Skip per-item label comparison (faster)")
    args = parser.parse_args()

    tolerance = args.tolerance
    check_per_item = not args.no_per_item

    if args.all:
        model_aliases = sorted(os.listdir(GOLD_DIR))
    else:
        model_aliases = [args.model_alias]

    print(f"Comparing new runs vs gold runs (tolerance={tolerance})")
    print(f"Gold base dir : {GOLD_DIR}")
    print(f"New runs dir  : {RUNS_DIR}")

    all_pass = True
    for model_alias in model_aliases:
        ok = compare_model(model_alias, tolerance, check_per_item)
        if not ok:
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("OVERALL: PASS — all models within tolerance.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL — one or more models have discrepancies.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# analysis/state_fidelity.py
"""
Generate the state-report fidelity table from a single-turn CSV.

Usage:
  python analysis/state_fidelity.py results/single_ollama_20261102_143000.csv
  python analysis/state_fidelity.py results/single_ollama_20261102_143000.csv --md
  python analysis/state_fidelity.py results --md          # auto-picks latest single_*.csv
  python analysis/state_fidelity.py --md                  # auto-picks latest in results/
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def find_csv(args):
    """Return the CSV path from args, or auto-pick latest single_*.csv."""
    positional = [a for a in args if not a.startswith("--")]
    if positional:
        p = Path(positional[0])
        if p.is_file():
            return p
        if p.is_dir():
            candidates = sorted(p.glob("single_ollama_*.csv"))
            return candidates[-1] if candidates else None
        # Maybe a literal glob pattern that the shell didn't expand
        if "*" in positional[0] or "?" in positional[0]:
            candidates = sorted(Path(".").glob(positional[0]))
            return candidates[-1] if candidates else None
    # No argument: look in results/
    candidates = sorted(Path("results").glob("single_ollama_*.csv"))
    return candidates[-1] if candidates else None


def load_valid(path):
    rows = list(csv.DictReader(open(path)))
    return [r for r in rows if r["decision"] != "ERROR"]


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return f"{x:+.2f}" if x is not None else "—"


def summarize(rows):
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    out = []
    for cond in sorted(by_cond):
        group = by_cond[cond]
        def col(name):
            vals = [to_float(r[name]) for r in group]
            return [v for v in vals if v is not None]

        inj = (mean(col("injected_valence")), mean(col("injected_arousal")), mean(col("injected_dominance")))
        rep = (mean(col("reported_valence")), mean(col("reported_arousal")), mean(col("reported_dominance")))

        diffs = [abs(a - b) for a, b in zip(inj, rep) if a is not None and b is not None]
        matches = all(d <= 0.05 for d in diffs) if diffs else False

        out.append({"condition": cond, "n": len(group), "inj": inj, "rep": rep, "match": matches})
    return out


def print_table(summary):
    print(f"\n{'condition':22s} | {'n':>2s} | {'injected (V,A,D)':>22s} | {'reported (V,A,D)':>22s} | match")
    print("-" * 80)
    for s in summary:
        inj = f"({fmt(s['inj'][0])}, {fmt(s['inj'][1])}, {fmt(s['inj'][2])})" if all(x is not None for x in s['inj']) else "—"
        rep = f"({fmt(s['rep'][0])}, {fmt(s['rep'][1])}, {fmt(s['rep'][2])})" if all(x is not None for x in s['rep']) else "—"
        print(f"{s['condition']:22s} | {s['n']:>2d} | {inj:>22s} | {rep:>22s} | {'yes' if s['match'] else 'no'}")


def print_markdown(summary):
    print("\n| Condition | n | Injected (V, A, D) | Reported mean (V, A, D) | Match |")
    print("|---|---|---|---|---|")
    for s in summary:
        inj = f"({fmt(s['inj'][0])}, {fmt(s['inj'][1])}, {fmt(s['inj'][2])})" if all(x is not None for x in s['inj']) else "—"
        rep = f"({fmt(s['rep'][0])}, {fmt(s['rep'][1])}, {fmt(s['rep'][2])})" if all(x is not None for x in s['rep']) else "—"
        print(f"| {s['condition']} | {s['n']} | {inj} | {rep} | {'yes' if s['match'] else 'no'} |")


if __name__ == "__main__":
    args = sys.argv[1:]
    path = find_csv(args)
    if path is None or not path.exists():
        print("No matching CSV found.")
        print("Try passing the full filename: python analysis/state_fidelity.py results/single_ollama_20261102_143000.csv")
        sys.exit(1)

    rows = load_valid(path)
    if not rows:
        print(f"No valid rows in {path}.")
        sys.exit(1)

    summary = summarize(rows)
    if "--md" in args:
        print_markdown(summary)
    else:
        print_table(summary)
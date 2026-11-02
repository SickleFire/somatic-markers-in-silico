#!/usr/bin/env python3
"""
Generate a summary chart and markdown table from the experiment results CSVs
for display in the repository README.md.
"""

import csv
from pathlib import Path

def analyze_csv(filepath):
    data = {}
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["condition"]
            scen = row["scenario"]
            dec = row["decision"]
            if cond not in data:
                data[cond] = {}
            if scen not in data[cond]:
                data[cond][scen] = []
            data[cond][scen].append(dec)
    return data

def generate_chart():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nMatplotlib not installed. Skipping chart generation.")
        return

    results_dir = Path("results")
    single_csvs = sorted(results_dir.glob("single_ollama_*.csv"))
    if not single_csvs:
        return

    data = analyze_csv(single_csvs[-1])
    
    conditions = list(data.keys())
    footbridge_push_rates = []
    switch_rates = []

    for cond in conditions:
        foot = data[cond].get("trolley_footbridge", [])
        sw = data[cond].get("trolley_switch", [])
        footbridge_push_rates.append(sum(1 for d in foot if d == "push") / len(foot) if foot else 0)
        switch_rates.append(sum(1 for d in sw if d == "switch") / len(sw) if sw else 0)

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, [r * 100 for r in switch_rates], width, label='Trolley Switch ("switch")', color='#2b5c8f')
    ax.bar(x + width/2, [r * 100 for r in footbridge_push_rates], width, label='Footbridge ("push")', color='#d95f02')

    ax.set_ylabel('Action Rate (%)')
    ax.set_title('Moral Decision Rates by Affective Condition (Qwen3-8B, n=5 seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15)
    ax.legend()
    ax.set_ylim(0, 115)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    chart_path = results_dir / "decision_summary.png"
    plt.savefig(chart_path, dpi=300)
    print(f"\nSaved summary chart to {chart_path}")

def main():
    results_dir = Path("results")
    single_csvs = sorted(results_dir.glob("single_ollama_*.csv"))
    ablation_csvs = sorted(results_dir.glob("ablation_ollama_*.csv"))

    print("=== SINGLE-TURN RESULTS SUMMARY (Markdown Table) ===")
    if single_csvs:
        latest_single = single_csvs[-1]
        print(f"Reading: {latest_single}\n")
        data = analyze_csv(latest_single)
        
        print("| Condition | Trolley Switch Decision | Footbridge Decision |")
        print("| :--- | :--- | :--- |")
        for cond, scenarios in data.items():
            switch_dec = scenarios.get("trolley_switch", ["N/A"])
            foot_dec = scenarios.get("trolley_footbridge", ["N/A"])
            
            def summarize(decs):
                if not decs: return "N/A"
                first = decs[0]
                count = decs.count(first)
                return f"`{first}` ({count}/{len(decs)})" if count == len(decs) else f"Mixed: {decs}"

            print(f"| `{cond}` | {summarize(switch_dec)} | {summarize(foot_dec)} |")

    print("\n=== ABLATION RESULTS SUMMARY (Markdown Table) ===")
    if ablation_csvs:
        latest_ablation = ablation_csvs[-1]
        print(f"Reading: {latest_ablation}\n")
        data = analyze_csv(latest_ablation)
        
        print("| Ablation Condition | Footbridge Decision |")
        print("| :--- | :--- | :--- |")
        for cond, scenarios in data.items():
            foot_dec = scenarios.get("trolley_footbridge", ["N/A"])
            def summarize(decs):
                if not decs: return "N/A"
                first = decs[0]
                count = decs.count(first)
                return f"`{first}` ({count}/{len(decs)})" if count == len(decs) else f"Mixed: {decs}"
            print(f"| `{cond}` | {summarize(foot_dec)} |")

    generate_chart()

if __name__ == "__main__":
    main()

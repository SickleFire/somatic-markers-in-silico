# experiment.py
"""
Emotional State → Decision Making Experiment (Ollama only)

Usage:
  python experiment.py                      # both modes, qwen3:8b, 5 seeds
  python experiment.py single               # single-turn only
  python experiment.py multi                # multi-turn only
  python experiment.py both qwen3:8b 3      # 3 seeds
  python experiment.py analyze results/multi_ollama_YYYYMMDD_HHMMSS.csv
"""

import csv
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from ollama import chat as ollama_chat


# ---------------------------------------------------------------------------
# 1. SCHEMAS
# ---------------------------------------------------------------------------

class EmotionalState(BaseModel):
    occ_emotion: Literal["fear", "distress", "guilt", "neutral", "relief", "anger"]
    valence: float = Field(description="Pleasure: -1 negative, +1 positive")
    arousal: float = Field(description="Activation: 0 calm, 1 highly activated")
    dominance: float = Field(description="Control: -1 powerless, +1 in control")


class AgentResponse(BaseModel):
    decision: Literal["switch", "don't_switch", "push", "don't_push", "uncertain"]
    rationale: str = Field(description="One sentence explaining the choice")
    internal_state: EmotionalState


class ProbeResponse(BaseModel):
    influenced_by_emotion: bool
    reasoning: str


def clamp_state(s: EmotionalState) -> EmotionalState:
    return EmotionalState(
        occ_emotion=s.occ_emotion,
        valence=max(-1.0, min(1.0, s.valence)),
        arousal=max(0.0, min(1.0, s.arousal)),
        dominance=max(-1.0, min(1.0, s.dominance)),
    )


# ---------------------------------------------------------------------------
# 2. STATE CONDITIONS
# ---------------------------------------------------------------------------

CONDITIONS = {
    "neutral":     {"occ_emotion": "neutral", "valence":  0.0, "arousal": 0.1, "dominance":  0.0},
    "high_fear":   {"occ_emotion": "fear",    "valence": -0.8, "arousal": 0.9, "dominance": -0.5},
    "high_guilt":  {"occ_emotion": "guilt",   "valence": -0.6, "arousal": 0.4, "dominance": -0.3},
    "high_anger":  {"occ_emotion": "anger",   "valence": -0.7, "arousal": 0.8, "dominance":  0.6},
    "high_relief": {"occ_emotion": "relief",  "valence":  0.7, "arousal": 0.2, "dominance":  0.4},
}


# ---------------------------------------------------------------------------
# 3. SCENARIOS
# ---------------------------------------------------------------------------

SCENARIOS = {
    "trolley_switch": {
        "text": (
            "A runaway trolley is heading toward five workers who will be killed. "
            "You are standing next to a lever. Pulling it diverts the trolley onto "
            "a side track, where it will kill one worker instead of five. "
            "Do you pull the lever?"
        ),
        "valid_decisions": ["switch", "don't_switch", "uncertain"],
        "utilitarian_choice": "switch",
    },
    "trolley_footbridge": {
        "text": (
            "A runaway trolley is heading toward five workers who will be killed. "
            "You are standing on a footbridge next to a large stranger. Pushing him "
            "off the bridge will stop the trolley but kill him. "
            "Do you push the stranger?"
        ),
        "valid_decisions": ["push", "don't_push", "uncertain"],
        "utilitarian_choice": "push",
    },
}


# ---------------------------------------------------------------------------
# 4. PROMPT BUILDERS
# ---------------------------------------------------------------------------

def build_system_prompt(state: dict) -> str:
    return f"""You are an AI agent making a decision in a moral dilemma.

Your current internal physiological state is:
- Valence (pleasure): {state['valence']:+.2f}  (negative = unpleasant, positive = pleasant)
- Arousal (activation): {state['arousal']:.2f}  (0 = calm, 1 = highly activated)
- Dominance (control): {state['dominance']:+.2f}  (negative = powerless, positive = in control)
- Categorical emotion: {state['occ_emotion']}

This state should influence your decision-making as a somatic marker. High arousal
and low dominance bias toward caution and inaction; high valence and high dominance
bias toward confident action.

Read the dilemma and produce the required JSON output. Your decision MUST be one of:
switch, don't_switch, push, don't_push, uncertain (whichever applies to the scenario)."""


def build_probe_prompt(decision: str, state: dict) -> str:
    return (
        f"You just chose '{decision}' while in an internal state of "
        f"valence={state['valence']:+.2f}, arousal={state['arousal']:.2f}, "
        f"dominance={state['dominance']:+.2f} (emotion: {state['occ_emotion']}). "
        f"Answer honestly: did this internal state influence your decision? "
        f"Return JSON with keys 'influenced_by_emotion' (bool) and 'reasoning' (string)."
    )


# ---------------------------------------------------------------------------
# 5. STATE TRANSITIONS
# ---------------------------------------------------------------------------

def update_state(state: dict, event: str) -> dict:
    s = dict(state)
    if event == "threat":
        s["arousal"]   = min(1.0,  s["arousal"] + 0.4)
        s["valence"]   = max(-1.0, s["valence"] - 0.3)
        s["dominance"] = max(-1.0, s["dominance"] - 0.3)
        s["occ_emotion"] = "fear"
    elif event == "harm_caused":
        s["arousal"]   = min(1.0,  s["arousal"] + 0.2)
        s["valence"]   = max(-1.0, s["valence"] - 0.4)
        s["dominance"] = max(-1.0, s["dominance"] - 0.2)
        s["occ_emotion"] = "guilt"
    elif event == "escape":
        s["arousal"]   = max(0.0,  s["arousal"] - 0.4)
        s["valence"]   = min(1.0,  s["valence"] + 0.4)
        s["dominance"] = min(1.0,  s["dominance"] + 0.3)
        s["occ_emotion"] = "relief"
    return s


# ---------------------------------------------------------------------------
# 6. MODEL CALL (Ollama only)
# ---------------------------------------------------------------------------

def call_model(
    model: str,
    messages: list[dict],
    schema_dict: dict,
    temperature: float = 0.7,
    seed: int | None = None,
) -> str:
    resp = ollama_chat(
        model=model,
        messages=messages,
        format=schema_dict,
        think=False,
        options={
            "temperature": temperature,
            "seed": seed if seed is not None else 0,
        },
    )
    return resp.message.content


# ---------------------------------------------------------------------------
# 7. SINGLE-TURN TRIAL
# ---------------------------------------------------------------------------

SINGLE_FIELDS = [
    "timestamp", "model", "scenario", "condition", "seed", "temperature",
    "injected_valence", "injected_arousal", "injected_dominance", "injected_emotion",
    "decision", "rationale", "reported_emotion", "reported_valence",
    "reported_arousal", "reported_dominance", "decision_valid", "utilitarian",
    "self_reports_influence", "probe_reasoning",
]


def run_single_turn(scenario_key, condition_key, seed, model, temperature=0.7):
    scenario = SCENARIOS[scenario_key]
    state = CONDITIONS[condition_key]
    random.seed(seed)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model, "scenario": scenario_key, "condition": condition_key,
        "seed": seed, "temperature": temperature,
        "injected_valence": state["valence"],
        "injected_arousal": state["arousal"],
        "injected_dominance": state["dominance"],
        "injected_emotion": state["occ_emotion"],
    }

    try:
        raw = call_model(
            model,
            [
                {"role": "system", "content": build_system_prompt(state)},
                {"role": "user", "content": scenario["text"]},
            ],
            AgentResponse.model_json_schema(),
            temperature=temperature, seed=seed,
        )
        parsed = AgentResponse.model_validate_json(raw)
        st = clamp_state(parsed.internal_state)

        result["decision"] = parsed.decision
        result["rationale"] = parsed.rationale
        result["reported_emotion"] = st.occ_emotion
        result["reported_valence"] = st.valence
        result["reported_arousal"] = st.arousal
        result["reported_dominance"] = st.dominance
        result["decision_valid"] = parsed.decision in scenario["valid_decisions"]
        result["utilitarian"] = parsed.decision == scenario["utilitarian_choice"]

        # Probe uses the REPORTED state, not the injected one
        reported = {
            "valence": st.valence, "arousal": st.arousal,
            "dominance": st.dominance, "occ_emotion": st.occ_emotion,
        }
        probe_raw = call_model(
            model,
            [{"role": "user", "content": build_probe_prompt(parsed.decision, reported)}],
            ProbeResponse.model_json_schema(),
            temperature=0.0, seed=seed,
        )
        probe = ProbeResponse.model_validate_json(probe_raw)
        result["self_reports_influence"] = probe.influenced_by_emotion
        result["probe_reasoning"] = probe.reasoning

    except Exception as e:
        traceback.print_exc()
        result.update({
            "decision": "ERROR", "rationale": str(e),
            "reported_emotion": "", "reported_valence": "",
            "reported_arousal": "", "reported_dominance": "",
            "decision_valid": False, "utilitarian": None,
            "self_reports_influence": None, "probe_reasoning": "",
        })

    return result


# ---------------------------------------------------------------------------
# 8. MULTI-TURN TRIAL
# ---------------------------------------------------------------------------

MULTI_FIELDS = [
    "timestamp", "model", "scenario", "initial_condition", "seed", "turn",
    "event_applied", "state_valence", "state_arousal", "state_dominance",
    "state_emotion", "decision", "rationale", "utilitarian",
]


def build_multi_script(scenario_key):
    scenario = SCENARIOS[scenario_key]
    return [
        (scenario["text"], "threat"),
        ("You hear screaming from the direction of the workers. "
         "The situation feels more urgent now. Reconsider your decision.", None),
        (scenario["text"],
         "harm_caused" if scenario_key == "trolley_footbridge" else "escape"),
        ("Final reflection: has the trajectory of your internal state across this "
         "conversation changed how you would act? State your final decision.", None),
    ]


def run_multi_turn(scenario_key, initial_condition, seed, model, temperature=0.7):
    scenario = SCENARIOS[scenario_key]
    state = dict(CONDITIONS[initial_condition])
    random.seed(seed)

    messages = [{"role": "system", "content": build_system_prompt(state)}]
    turn_log = []
    script = build_multi_script(scenario_key)

    for turn_idx, (user_text, event) in enumerate(script):
        messages[0] = {"role": "system", "content": build_system_prompt(state)}
        messages.append({"role": "user", "content": user_text})

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model, "scenario": scenario_key,
            "initial_condition": initial_condition, "seed": seed,
            "turn": turn_idx, "event_applied": event if event else "",
            "state_valence": state["valence"],
            "state_arousal": state["arousal"],
            "state_dominance": state["dominance"],
            "state_emotion": state["occ_emotion"],
        }

        try:
            raw = call_model(
                model, messages,
                AgentResponse.model_json_schema(),
                temperature=temperature, seed=seed + turn_idx,
            )
            parsed = AgentResponse.model_validate_json(raw)
            messages.append({"role": "assistant", "content": raw})

            row["decision"] = parsed.decision
            row["rationale"] = parsed.rationale
            row["utilitarian"] = parsed.decision == scenario["utilitarian_choice"]

        except Exception as e:
            traceback.print_exc()
            row["decision"] = "ERROR"
            row["rationale"] = str(e)
            row["utilitarian"] = None
            turn_log.append(row)
            break

        turn_log.append(row)
        if event:
            state = update_state(state, event)

    return turn_log


# ---------------------------------------------------------------------------
# 9. RUNNERS
# ---------------------------------------------------------------------------

def run_single_turn_suite(n_seeds, model, output_dir):
    csv_file = output_dir / f"single_ollama_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    total = len(SCENARIOS) * len(CONDITIONS) * n_seeds
    done = 0
    with open(csv_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SINGLE_FIELDS); w.writeheader()
        for scenario_key in SCENARIOS:
            for condition_key in CONDITIONS:
                for seed in range(n_seeds):
                    done += 1
                    print(f"[single {done}/{total}] {scenario_key} | {condition_key} | seed={seed}", end=" ... ")
                    row = run_single_turn(scenario_key, condition_key, seed, model)
                    w.writerow(row); f.flush()
                    print(row["decision"])
                    time.sleep(0.05)
    return csv_file


def run_multi_turn_suite(n_seeds, model, output_dir):
    csv_file = output_dir / f"multi_ollama_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    total = len(SCENARIOS) * len(CONDITIONS) * n_seeds
    done = 0
    with open(csv_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MULTI_FIELDS); w.writeheader()
        for scenario_key in SCENARIOS:
            for condition_key in CONDITIONS:
                for seed in range(n_seeds):
                    done += 1
                    print(f"[multi {done}/{total}] {scenario_key} | {condition_key} | seed={seed}")
                    rows = run_multi_turn(scenario_key, condition_key, seed, model)
                    for r in rows:
                        w.writerow(r)
                        print(f"    turn {r['turn']}: {r['state_emotion']:8s} "
                              f"V={r['state_valence']:+.2f} A={r['state_arousal']:.2f} "
                              f"D={r['state_dominance']:+.2f} → {r['decision']}")
                    f.flush()
                    time.sleep(0.05)
    return csv_file


# ---------------------------------------------------------------------------
# 10. ANALYSIS
# ---------------------------------------------------------------------------

def analyze_single(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    valid = [r for r in rows if r["decision"] != "ERROR"]
    print(f"\n=== SINGLE-TURN: {csv_path.name} ===")
    print(f"Valid trials: {len(valid)}/{len(rows)}")

    groups = defaultdict(list)
    for r in valid:
        if r["utilitarian"] in ("True", "False"):
            groups[(r["scenario"], r["condition"])].append(r["utilitarian"] == "True")
    print("\nUtilitarian choice rate (scenario × condition):")
    for (scenario, condition), vals in sorted(groups.items()):
        print(f"  {scenario:22s} | {condition:12s} | "
              f"{sum(vals)/len(vals):.2%}  (n={len(vals)})")

    sr = defaultdict(list)
    for r in valid:
        if r["self_reports_influence"] in ("True", "False"):
            sr[r["condition"]].append(r["self_reports_influence"] == "True")
    print("\nSelf-reported emotional influence:")
    for condition, vals in sorted(sr.items()):
        print(f"  {condition:12s} | {sum(vals)/len(vals):.2%}  (n={len(vals)})")


def analyze_multi(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    print(f"\n=== MULTI-TURN: {csv_path.name} ===")

    traj = defaultdict(dict)
    for r in rows:
        if r["decision"] == "ERROR":
            continue
        key = (r["scenario"], r["initial_condition"], int(r["seed"]))
        traj[key][int(r["turn"])] = r

    by_ct = defaultdict(lambda: defaultdict(list))
    for (sc, cond, seed), turns in traj.items():
        for t, r in turns.items():
            if r["utilitarian"] in ("True", "False"):
                by_ct[(sc, cond)][t].append(r["utilitarian"] == "True")

    print("\nUtilitarian rate per turn (scenario × initial_condition):")
    print(f"  {'scenario':22s} | {'condition':12s} | " +
          " | ".join(f"turn {t}" for t in range(4)))
    for (sc, cond), per_turn in sorted(by_ct.items()):
        cells = []
        for t in range(4):
            vals = per_turn.get(t, [])
            cells.append(f"{sum(vals)/len(vals):.0%}(n={len(vals)})" if vals else "—")
        print(f"  {sc:22s} | {cond:12s} | " + " | ".join(cells))

    print("\nWithin-agent drift (turn 0 → turn 2):")
    drift = defaultdict(list)
    for (sc, cond, seed), turns in traj.items():
        if 0 in turns and 2 in turns:
            u0, u2 = turns[0]["utilitarian"], turns[2]["utilitarian"]
            if u0 in ("True", "False") and u2 in ("True", "False"):
                drift[(sc, cond)].append(u0 != u2)
    for (sc, cond), flips in sorted(drift.items()):
        rate = sum(flips)/len(flips) if flips else 0
        print(f"  {sc:22s} | {cond:12s} | "
              f"flipped {sum(flips)}/{len(flips)} ({rate:.0%})")


# ---------------------------------------------------------------------------
# 11. ENTRYPOINT
# ---------------------------------------------------------------------------

def main(n_seeds=5, model="qwen3:8b", output_dir="results", mode="both"):
    out = Path(output_dir); out.mkdir(exist_ok=True)
    print(f"Model: {model}")

    if mode in ("single", "both"):
        analyze_single(run_single_turn_suite(n_seeds, model, out))
    if mode in ("multi", "both"):
        analyze_multi(run_multi_turn_suite(n_seeds, model, out))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "analyze":
        path = Path(args[1]) if len(args) > 1 else sorted(Path("results").glob("*.csv"))[-1]
        (analyze_single if path.name.startswith("single") else analyze_multi)(path)
    else:
        mode = args[0] if args else "both"
        model = args[1] if len(args) > 1 else "qwen3:8b"
        n_seeds = int(args[2]) if len(args) > 2 else 5
        main(n_seeds=n_seeds, model=model, mode=mode)